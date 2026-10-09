use super::{check_f16, check_f32, compiler, support, upload};
use criterion::{Criterion, Throughput};
use h3_hrx::dispatch::{Matmul16, Profile};
use h3_hrx::model::{VHD, VHDP, VHEADS, VHID, VISION_PATCH, VMERGE, VMLP, VOUT};
use half::{bf16, f16};
use hrx::{Buffer, Stream};
use std::time::{Duration, Instant};

struct Fixture {
    stream: Stream,
    gemm: Matmul16,
    profile: Profile,
    stage: &'static str,
    m: usize,
    a: Buffer,
    ring: Vec<Buffer>,
    bias: Buffer,
    lambda: Option<Buffer>,
    initial: Option<Buffer>,
    out: Buffer,
    half_output: bool,
    expected: Vec<u8>,
    index: usize,
}

impl Fixture {
    fn new(stage: &'static str, kind: &str, m: usize, k: usize, n: usize, rotating: bool) -> Self {
        let manager = hrx::residency::ResidencyManager::new(512 << 20).unwrap();
        let mut stream = support::stream(Some(manager.budget()));
        let compiler = compiler();
        let gemm = Matmul16::build(&compiler, &mut stream, kind, k, n).unwrap();
        compiler.flush(&mut stream).unwrap();
        let residual = matches!(kind, "resid" | "resid80");
        let half_output = residual || kind == "gelu_f16";
        let input: Vec<f32> = (0..m * k)
            .map(|i| {
                if kind == "resid80" && i % VHDP >= VHD {
                    0.
                } else {
                    ((i * 17 % 251) as f32 - 125.) / 97.
                }
            })
            .collect();
        let input_bytes = if residual {
            bytemuck::cast_slice::<_, u8>(
                &input.iter().copied().map(f16::from_f32).collect::<Vec<_>>(),
            )
            .to_vec()
        } else {
            bytemuck::cast_slice(&input).to_vec()
        };
        let a = upload(&mut stream, &input_bytes);
        let weights: Vec<_> = (0..n * k)
            .map(|i| bf16::from_f32(((i * 37 % 127) as f32 - 63.) / 509.))
            .collect();
        let weight_bytes = bytemuck::cast_slice(&weights);
        // Exceed the 64 MiB cache without changing the arithmetic
        // between replays. Tower layers have distinct weights.
        let count = if rotating {
            (64 * 1024 * 1024 / weight_bytes.len() + 1).max(2)
        } else {
            1
        };
        let ring: Vec<_> = (0..count)
            .map(|_| upload(&mut stream, weight_bytes))
            .collect();
        let biases: Vec<f32> = (0..n).map(|i| (i % 13) as f32 / 31.).collect();
        let bias = upload(&mut stream, bytemuck::cast_slice(&biases));
        let lambdas: Vec<f32> = (0..n).map(|i| (i % 7 + 1) as f32 / 8.).collect();
        let lambda = residual.then(|| upload(&mut stream, bytemuck::cast_slice(&lambdas)));
        let initial = residual.then(|| {
            upload(
                &mut stream,
                bytemuck::cast_slice(&vec![f16::from_f32(0.25); m * n]),
            )
        });
        let out = stream
            .allocate_zeroed(m * n * if half_output { 2 } else { 4 })
            .unwrap();
        let mut case = Self {
            stream,
            gemm,
            profile: Profile::from_env(),
            stage,
            m,
            a,
            ring,
            bias,
            lambda,
            initial,
            out,
            half_output,
            expected: Vec::new(),
            index: 0,
        };
        case.restore();
        case.run();
        case.expected = case.read();
        // Independent FP64 oracle at tile boundaries and the tail.
        // Match operand rounding; accumulate the oracle in FP64.
        for row in [0, 15, 31, 63, m / 2, m - 1] {
            if row >= m {
                continue;
            }
            for col in [0, 15, 16, 31, 32, 63, n / 2, n - 1] {
                let mut v = f64::from(biases[col]);
                for j in 0..k {
                    let x = input[row * k + j];
                    let x = if residual {
                        f16::from_f32(x).to_f32()
                    } else {
                        x
                    };
                    v += bf16::from_f32(x).to_f64() * weights[col * k + j].to_f64();
                }
                let want = match kind {
                    "gelu_f16" => {
                        0.5 * v * (1. + (0.7978845608028654 * (v + 0.044715 * v.powi(3))).tanh())
                    }
                    "gelu_erf" => 0.5 * v * (1. + libm::erf(v / std::f64::consts::SQRT_2)),
                    "resid" | "resid80" => 0.25 + f64::from(lambdas[col]) * v,
                    _ => v,
                };
                let offset = (row * n + col) * if half_output { 2 } else { 4 };
                let got = if half_output {
                    f16::from_le_bytes(case.expected[offset..offset + 2].try_into().unwrap())
                        .to_f64()
                } else {
                    f64::from(f32::from_le_bytes(
                        case.expected[offset..offset + 4].try_into().unwrap(),
                    ))
                };
                assert!(
                    (got - want).abs() <= 2e-3 + 4e-3 * want.abs(),
                    "{stage}/{m}x{k}x{n} [{row},{col}]: {got} != {want}"
                );
            }
        }
        case
    }

    fn restore(&mut self) {
        if let Some(initial) = &self.initial {
            self.stream
                .copy(self.out.binding(), initial.binding())
                .unwrap();
            self.stream.synchronize().unwrap();
        }
    }

    fn run(&mut self) {
        self.gemm
            .run(
                &mut self.stream,
                Some(&mut self.profile),
                self.stage,
                self.m,
                self.a.binding(),
                self.ring[self.index].binding(),
                self.bias.binding(),
                self.out.binding(),
                self.lambda.as_ref().map(Buffer::binding),
            )
            .unwrap();
        self.stream.synchronize().unwrap();
        self.index = (self.index + 1) % self.ring.len();
    }

    fn read(&mut self) -> Vec<u8> {
        if self.half_output {
            check_f16(&mut self.stream, &self.out)
        } else {
            check_f32(&mut self.stream, &self.out)
        }
    }
}

pub fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("vision_gemm");
    // Complete tower shapes: 64x64, 864x480 and 1344x768 images. Merger
    // projections also serve DeepStack and consume four patches per row.
    for tokens in [16usize, 1620, 4032] {
        for (stage, kind, m, k, n) in [
            ("patch", "bias", tokens, VISION_PATCH, VHID),
            ("qkv", "bias", tokens, VHID, 3 * VHID),
            ("proj", "resid80", tokens, VHEADS * VHDP, VHID),
            ("fc1", "gelu_f16", tokens, VHID, VMLP),
            ("fc2", "resid", tokens, VMLP, VHID),
            ("merge1", "gelu_erf", tokens / 4, VMERGE, VMERGE),
            ("merge2", "bias", tokens / 4, VMERGE, VOUT),
        ] {
            for rotating in [false, true] {
                let storage = if rotating { "rotating" } else { "cached" };
                let active_k = if kind == "resid80" { k / 128 * 80 } else { k };
                group.throughput(Throughput::Elements((2 * m * active_k * n) as u64));
                // Criterion calls this closure again for each sample. Keep both
                // allocations and the ring position across warmup and samples,
                // even when a sample contains only one measured iteration.
                let mut fixture = None;
                group.bench_function(format!("{stage}/{storage}/{m}x{k}x{n}"), |b| {
                    let case =
                        fixture.get_or_insert_with(|| Fixture::new(stage, kind, m, k, n, rotating));
                    b.iter_custom(|iterations| {
                        let mut elapsed = Duration::ZERO;
                        for _ in 0..iterations {
                            // Residual restoration is outside kernel timing.
                            case.restore();
                            let start = Instant::now();
                            case.run();
                            elapsed += start.elapsed();
                        }
                        elapsed
                    });
                    assert_eq!(case.read(), case.expected, "{stage} replay changed");
                });
            }
        }
    }
    group.finish();
}

pub fn normalization(c: &mut Criterion) {
    use h3_hrx::dispatch::LayerNorm16;

    let mut group = c.benchmark_group("vision_layernorm");
    for (rows, width) in [
        (1usize, 32usize),
        (17, 96),
        (16, VHID),
        (1620, VHID),
        (4032, VHID),
        (17, 1184),
        (4, VMERGE),
        (405, VMERGE),
        (1008, VMERGE),
        (17, 8192),
    ] {
        group.throughput(Throughput::Elements((rows * width) as u64));
        let mut fixture = None;
        group.bench_function(format!("{rows}x{width}"), |b| {
            let (stream, norm, input, weights, bias, out, expected, profile) = fixture
                .get_or_insert_with(|| {
                    let manager = hrx::residency::ResidencyManager::new(512 << 20).unwrap();
                    let mut stream = support::stream(Some(manager.budget()));
                    let compiler = compiler();
                    let norm = LayerNorm16::build(&compiler, &mut stream, width, 1e-6).unwrap();
                    compiler.flush(&mut stream).unwrap();
                    let values: Vec<_> = (0..rows * width)
                        .map(|i| f16::from_f32(((i * 73 % 257) as f32 - 128.) / 97.))
                        .collect();
                    let input = upload(&mut stream, bytemuck::cast_slice(&values));
                    let weights = upload(&mut stream, bytemuck::cast_slice(&vec![0.75f32; width]));
                    let bias = upload(&mut stream, bytemuck::cast_slice(&vec![0.125f32; width]));
                    let out = stream.allocate_zeroed(rows * width * 4).unwrap();
                    let mut profile = Profile::from_env();
                    norm.run(
                        &mut stream,
                        Some(&mut profile),
                        rows,
                        input.binding(),
                        weights.binding(),
                        bias.binding(),
                        out.binding(),
                    )
                    .unwrap();
                    let expected = check_f32(&mut stream, &out);
                    (stream, norm, input, weights, bias, out, expected, profile)
                });
            b.iter(|| {
                norm.run(
                    stream,
                    Some(&mut *profile),
                    rows,
                    input.binding(),
                    weights.binding(),
                    bias.binding(),
                    out.binding(),
                )
                .unwrap();
                stream.synchronize().unwrap();
            });
            assert_eq!(
                check_f32(stream, out),
                *expected,
                "LayerNorm replay changed"
            );
        });
    }
    group.finish();
}

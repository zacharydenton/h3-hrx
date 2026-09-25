//! Native GPU regressions against independent scalar CPU oracles.
//! Run `cargo test --test kernels -- --ignored --test-threads=1` on gfx1151.
use half::{bf16, f16};
use hrx::{Buffer, Constants, Stream};
use std::path::Path;

struct Harness {
    compiler: hrx::loom::Compiler,
    stream: Stream,
}
impl Harness {
    fn new() -> Self {
        Self {
            compiler: hrx::loom::Compiler::resolve(None).expect("provisioned Loom compiler"),
            stream: Stream::open().expect("gfx1151 device"),
        }
    }
    fn run(
        &mut self,
        stem: &str,
        cfg: &[(&str, String)],
        grid: [u32; 3],
        threads: u32,
        scalars: &[u64],
        data: &[Vec<u8>],
    ) -> Vec<Vec<u8>> {
        self.run_module(stem, stem, cfg, grid, threads, scalars, data)
    }
    #[allow(clippy::too_many_arguments)]
    fn run_module(
        &mut self,
        module: &str,
        stem: &str,
        cfg: &[(&str, String)],
        grid: [u32; 3],
        threads: u32,
        scalars: &[u64],
        data: &[Vec<u8>],
    ) -> Vec<Vec<u8>> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let path = root.join("kernels").join(format!("{module}.loom"));
        let path = if path.is_file() {
            path
        } else {
            root.join("experiments").join(format!("{stem}.loom"))
        };
        let source = std::fs::read_to_string(path).unwrap();
        let symbol = format!("h3_{stem}");
        let mut request = hrx::loom::Specialization::new(&symbol);
        request.replace_config(
            cfg.iter()
                .map(|(key, value)| (format!("h3.{stem}.{key}"), value.clone()))
                .collect(),
        );
        let path = self.compiler.module(&source).compile(&request).unwrap();
        let baseline = std::env::var_os("H3_KERNEL_BASELINE")
            .filter(|_| module != stem)
            .map(|dir| {
                let source = std::fs::read_to_string(Path::new(&dir).join(format!("{stem}.loom")))
                    .expect("baseline kernel source");
                let mut old = hrx::loom::Specialization::new(&symbol);
                old.replace_config(request.configuration().clone());
                let old_path = self.compiler.module(&source).compile(&old).unwrap();
                eprintln!(
                    "artifact {stem}: {}",
                    if old_path.bytes() == path.bytes() {
                        "identical"
                    } else {
                        "changed"
                    }
                );
                old_path
            });
        // Safety: trusted checked-in source compiled through HRX. Every test below
        // sizes the bindings from the same dimensions passed as kernel configuration.
        let kernel = unsafe { self.stream.load_artifact(&path).unwrap() };
        let buffers: Vec<Buffer> = data
            .iter()
            .map(|bytes| self.stream.allocate(bytes.len()).unwrap())
            .collect();
        for (buffer, bytes) in buffers.iter().zip(data) {
            self.stream.upload(buffer.binding(), bytes).unwrap();
        }
        let indices: Vec<_> = scalars.iter().map(|&v| u32::try_from(v).unwrap()).collect();
        let constants = Constants::indices(&kernel, &indices).unwrap();
        let bindings: Vec<_> = buffers.iter().map(Buffer::binding).collect();
        unsafe {
            self.stream
                .dispatch(&kernel, grid, [threads, 1, 1], &constants, &bindings)
                .unwrap();
        }
        let reads: Vec<_> = bindings
            .iter()
            .map(|&v| self.stream.read(v).unwrap())
            .collect();
        let result: Vec<Vec<u8>> = reads
            .into_iter()
            .map(|r| r.wait(&mut self.stream).unwrap())
            .collect();
        if let Some(path) = baseline {
            let old = unsafe { self.stream.load_artifact(&path).unwrap() };
            for (buffer, bytes) in buffers.iter().zip(data) {
                self.stream.upload(buffer.binding(), bytes).unwrap();
            }
            let old_constants = Constants::indices(&old, &indices).unwrap();
            unsafe {
                self.stream
                    .dispatch(&old, grid, [threads, 1, 1], &old_constants, &bindings)
                    .unwrap();
            }
            for (i, binding) in bindings.iter().enumerate() {
                let actual = self
                    .stream
                    .read(*binding)
                    .unwrap()
                    .wait(&mut self.stream)
                    .unwrap();
                assert!(
                    actual == result[i],
                    "baseline mismatch: {stem}, binding {i}, config {cfg:?}"
                );
            }
            if std::env::var_os("H3_KERNEL_TIMING").is_some() {
                let mut graphs = Vec::new();
                for (k, c) in [(&old, &old_constants), (&kernel, &constants)] {
                    let mut graph = self.stream.graph().unwrap();
                    // Every repetition writes the same bindings, so they are chained: this times
                    // the kernel back to back, not a hundred and twenty-eight copies at once.
                    let mut previous = None;
                    for _ in 0..128 {
                        let after = previous.as_slice();
                        // Safety: the same validated bindings as the numerical comparison.
                        previous = Some(unsafe {
                            graph
                                .dispatch(after, k, grid, [threads, 1, 1], c, &bindings)
                                .unwrap()
                        });
                    }
                    graphs.push(graph.finish().unwrap());
                }
                for graph in &mut graphs {
                    self.stream.launch(graph).unwrap();
                }
                self.stream.synchronize().unwrap();
                let mut times = [Vec::new(), Vec::new()];
                for batch in 0..10 {
                    for order in 0..2 {
                        let version = (batch + order) % 2;
                        for (buffer, bytes) in buffers.iter().zip(data) {
                            self.stream.upload(buffer.binding(), bytes).unwrap();
                        }
                        self.stream.synchronize().unwrap();
                        let start = std::time::Instant::now();
                        for _ in 0..8 {
                            self.stream.launch(&mut graphs[version]).unwrap();
                        }
                        self.stream.synchronize().unwrap();
                        times[version].push(start.elapsed().as_secs_f64());
                    }
                }
                for values in &mut times {
                    values.sort_by(f64::total_cmp);
                }
                let old = (times[0][4] + times[0][5]) / 2.;
                let new = (times[1][4] + times[1][5]) / 2.;
                eprintln!(
                    "timing {stem}: ratio={:.5} old_us={:.3} new_us={:.3} config={cfg:?}",
                    new / old,
                    old * 1e6 / 1024.,
                    new * 1e6 / 1024.
                );
            }
        }
        result
    }
}
fn bytes<T: bytemuck::Pod>(v: &[T]) -> Vec<u8> {
    bytemuck::cast_slice(v).to_vec()
}
fn floats(v: &[u8]) -> Vec<f64> {
    v.as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b) as f64)
        .collect()
}
fn halves(v: &[u8], bf: bool) -> Vec<f64> {
    v.as_chunks::<2>()
        .0
        .iter()
        .map(|b| {
            let n = u16::from_le_bytes(*b);
            if bf {
                bf16::from_bits(n).to_f64()
            } else {
                f16::from_bits(n).to_f64()
            }
        })
        .collect()
}
fn values(n: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 37 % 101) as f32 - 50.) * scale / 50.)
        .collect()
}
fn close(got: &[f64], want: &[f64], abs: f64, rel: f64) {
    assert_eq!(got.len(), want.len());
    for (i, (&a, &b)) in got.iter().zip(want).enumerate() {
        assert!(
            a.is_finite() && (a - b).abs() <= abs + rel * b.abs(),
            "element {i}: {a} vs {b}"
        );
    }
}
fn cfg(v: &[(&'static str, usize)]) -> Vec<(&'static str, String)> {
    v.iter().map(|&(k, v)| (k, v.to_string())).collect()
}

#[test]
#[ignore = "requires gfx1151 and provisioned HRX"]
fn groupnorm_silu_handles_zero_and_small_variance() {
    let mut h = Harness::new();
    let mut config = cfg(&[
        ("channels", 32),
        ("groups", 1),
        ("plane", 2),
        ("rows_bound", 64),
    ]);
    config.push(("eps", "1e-6".into()));
    for amplitude in [0.0005, 0., 0.5] {
        let x: Vec<f16> = (0..64)
            .map(|i| f16::from_f32(if i % 2 == 0 { -amplitude } else { amplitude }))
            .collect();
        let a = x[1].to_f64();
        let stats = [0f32, (64. * a * a) as f32];
        let want: Vec<_> = x
            .iter()
            .map(|v| {
                let y = v.to_f64() / (a * a + 1e-6).sqrt();
                y / (1. + (-y).exp())
            })
            .collect();
        let out = h.run(
            "gn_silu_f16",
            &config,
            [1, 1, 1],
            256,
            &[1],
            &[
                bytes(&x),
                bytes(&stats),
                bytes(&[1f32; 32]),
                bytes(&[0f32; 32]),
                vec![0; 128],
            ],
        );
        close(&halves(&out[4], false), &want, 5e-4, 2e-3);
    }
}

#[test]
#[ignore = "requires gfx1151 and provisioned HRX"]
fn attention_preserves_the_upper_tile_softmax_maximum() {
    let mut h = Harness::new();
    for (stem, waves) in [
        ("attention_mhat32_lds_f16_wmma", 4),
        ("attention_mha8t32_lds_f16_wmma", 8),
        ("attention_mha8h2t32_lds_f16_wmma", 8),
        ("attention_mha8h0t32_lds_f16_wmma", 8),
    ] {
        let (tokens, capacity, width) = (32usize, 128, 128);
        let mut q = vec![f16::ZERO; capacity * width];
        let mut k = q.clone();
        for row in 0..tokens {
            q[row * width] = f16::ONE;
        }
        k[16 * width] = f16::from_f32(15.);
        let mut v: Vec<_> = values(capacity * width, 1.)
            .into_iter()
            .map(f16::from_f32)
            .collect();
        v[tokens * width..].fill(f16::ZERO);
        let config = cfg(&[
            ("q_stride", width),
            ("kv_stride", width),
            ("tokens", tokens),
            ("token_capacity", capacity),
            ("scale", 1),
            ("out_stride", width),
        ]);
        let mut row = vec![0f64; width];
        let denominator = 1. + 31. * (-15f64).exp();
        for key in 0..tokens {
            let p = if key == 16 { 1. } else { (-15f64).exp() } / denominator;
            for c in 0..width {
                row[c] += p * v[key * width + c].to_f64();
            }
        }
        let want: Vec<_> = (0..tokens).flat_map(|_| row.iter().copied()).collect();
        let out = h.run(
            stem,
            &config,
            [1, 1, 1],
            32 * waves,
            &[tokens as u64],
            &[bytes(&q), bytes(&k), bytes(&v), vec![0; tokens * width * 2]],
        );
        close(&halves(&out[3], false), &want, 2e-3, 2e-3);
    }
}

#[test]
#[ignore = "requires gfx1151 and provisioned HRX"]
fn audio_convolution_matches_f64_with_padding_residuals_and_guards() {
    let mut h = Harness::new();
    for n in [1usize, 63, 64, 65, 127, 128, 129, 255, 256, 257, 769] {
        for co in [5usize, 16] {
            for acc in [0usize, 1] {
                let (ci, taps, dilation, pad) = (3, 11, 5, 25);
                let x = values(ci * n, 0.25);
                let w = values(co * ci * taps, 0.03);
                let bias = values(co, 0.01);
                let mut prev = values(co * n + 64, 0.02);
                prev[co * n..].fill(113.);
                let mut want = vec![0f64; co * n];
                for o in 0..co {
                    for t in 0..n {
                        let mut sum =
                            bias[o] as f64 + if acc == 1 { prev[o * n + t] as f64 } else { 0. };
                        for c in 0..ci {
                            for tap in 0..taps {
                                let j = t as isize + (tap * dilation) as isize - pad as isize;
                                if (0..n as isize).contains(&j) {
                                    sum += w[(o * ci + c) * taps + tap] as f64
                                        * x[c * n + j as usize] as f64;
                                }
                            }
                        }
                        want[o * n + t] = sum;
                    }
                }
                let config = cfg(&[
                    ("cin", ci),
                    ("cout", co),
                    ("ksize", taps),
                    ("dilation", dilation),
                    ("pad", pad),
                    ("accumulate", acc),
                    ("len_bound", n.div_ceil(256) * 256),
                ]);
                let mut outputs = Vec::new();
                let mut kernels = vec![("conv1d_f32", 256), ("conv1d4_f32", 64)];
                if co == 16 {
                    kernels.push(("conv1d_block_f32", 64));
                }
                for (stem, threads) in kernels {
                    let blocked = stem == "conv1d_block_f32";
                    let mut weights = w.clone();
                    if blocked {
                        weights.clear();
                        for group in 0..co / 8 {
                            for tap in 0..ci * taps {
                                for channel in 0..8 {
                                    weights.push(w[(group * 8 + channel) * ci * taps + tap]);
                                }
                            }
                        }
                    }
                    let (span, grid_outputs) = if blocked { (128, co / 8) } else { (256, co) };
                    let out = h.run(
                        stem,
                        &config,
                        [n.div_ceil(span) as u32, grid_outputs as u32, 1],
                        threads,
                        &[n as u64],
                        &[bytes(&x), bytes(&weights), bytes(&bias), bytes(&prev)],
                    );
                    assert_eq!(&out[3][co * n * 4..], bytes(&[113f32; 64]));
                    close(&floats(&out[3][..co * n * 4]), &want, 2e-6, 0.);
                    outputs.push(out[3].clone());
                }
                for output in &outputs[1..] {
                    assert_eq!(
                        &outputs[0], output,
                        "convolution accumulation order at n={n}, channels={co}, residual={acc}"
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires gfx1151 and provisioned HRX"]
fn float_matmul_addresses_rows_past_32768() {
    let mut h = Harness::new();
    for (m, k, n) in [(40000usize, 7usize, 9usize), (65, 96, 257)] {
        let x = values(m * k, 0.5);
        let w = values(n * k, 0.1);
        let bias = values(n, 0.1);
        let want: Vec<_> = (0..m * n)
            .map(|i| {
                let (r, c) = (i / n, i % n);
                bias[c] as f64
                    + (0..k)
                        .map(|j| x[r * k + j] as f64 * w[c * k + j] as f64)
                        .sum::<f64>()
            })
            .collect();
        let out = h.run(
            "matmul_f32",
            &cfg(&[("k", k), ("n", n)]),
            [n.div_ceil(256) as u32, m as u32, 1],
            256,
            &[m as u64],
            &[bytes(&x), bytes(&w), bytes(&bias), vec![0; m * n * 4]],
        );
        close(&floats(&out[3]), &want, 1e-5, 1e-5);
    }
}

#[test]
#[ignore = "requires gfx1151 and provisioned HRX"]
fn vision_bf16_matmuls_match_rounded_operands_and_epilogues() {
    let mut h = Harness::new();
    let (m, k, n) = (65usize, 128usize, 64usize);
    let input = values(m * k, 0.5);
    let w: Vec<_> = values(n * k, 0.1).into_iter().map(bf16::from_f32).collect();
    let b = values(n, 0.1);
    for kind in ["bias", "gelu", "gelu_erf", "resid"] {
        eprintln!("vision epilogue: {kind}");
        let a: Vec<_> = input
            .iter()
            .map(|&x| {
                if kind == "resid" {
                    bf16::from_f32(f16::from_f32(x).to_f32()).to_f64()
                } else {
                    bf16::from_f32(x).to_f64()
                }
            })
            .collect();
        let residual: Vec<_> = values(m * n, 0.5).into_iter().map(f16::from_f32).collect();
        let lambda = values(n, 0.3);
        let want: Vec<_> = (0..m * n)
            .map(|i| {
                let (r, c) = (i / n, i % n);
                let v = b[c] as f64
                    + (0..k)
                        .map(|j| a[r * k + j] * w[c * k + j].to_f64())
                        .sum::<f64>();
                match kind {
                    "gelu" => {
                        0.5 * v * (1. + (0.7978845608028654 * (v + 0.044715 * v.powi(3))).tanh())
                    }
                    "gelu_erf" => 0.5 * v * (1. + libm::erf(v / std::f64::consts::SQRT_2)),
                    "resid" => residual[i].to_f64() + lambda[c] as f64 * v,
                    _ => v,
                }
            })
            .collect();
        let mut data = if kind == "resid" {
            vec![
                bytes(&input.iter().copied().map(f16::from_f32).collect::<Vec<_>>()),
                bytes(&w),
                bytes(&b),
                bytes(&residual),
                bytes(&lambda),
            ]
        } else {
            vec![bytes(&input), bytes(&w), bytes(&b), vec![0; m * n * 4]]
        };
        let stem = format!("matmul_{kind}_bf16_wmma");
        let out = h.run_module(
            if kind == "resid" {
                &stem
            } else {
                "matmul_bf16_family"
            },
            &stem,
            &cfg(&[("k_size", k), ("n_size", n)]),
            [1, m.div_ceil(64) as u32, 1],
            256,
            &[m as u64],
            &data,
        );
        let got = if kind == "resid" {
            halves(&out[3], false)
        } else {
            floats(&out[3])
        };
        close(&got, &want, 2e-3, 4e-3);
        data.clear();
    }
}

#[test]
#[ignore = "requires gfx1151 and provisioned HRX"]
fn float_preparation_normalizes_large_values_and_respects_padded_pitch() {
    let mut h = Harness::new();
    for width in [256usize, 2048, 5376] {
        let (tokens, classes, stride, lanes) = (
            3usize,
            2usize,
            width + 64,
            if width % 1024 == 0 { 128usize } else { 32usize },
        );
        let input = values(tokens * width, 1.5e5);
        let weights: Vec<_> = values(width, 0.1).into_iter().map(|x| 1. + x).collect();
        let table = values(classes * 2 * width, 0.2);
        let cls = [0i32, 1, 0];
        for bf in [false, true] {
            for kind in ["norm", "lnorm", "plain"] {
                let dtype = if bf { "bf16" } else { "f16" };
                let stem = format!("prepare_{kind}_{dtype}");
                let mut config = cfg(&[("width", width), ("out_stride", stride), ("lanes", lanes)]);
                let plain: Vec<_> = values(tokens * width, 0.5)
                    .into_iter()
                    .map(f16::from_f32)
                    .collect();
                let mut want = vec![0f64; tokens * width];
                for row in 0..tokens {
                    let slice = &input[row * width..(row + 1) * width];
                    let mean = if kind == "lnorm" {
                        slice.iter().map(|&x| x as f64).sum::<f64>() / width as f64
                    } else {
                        0.
                    };
                    let variance = slice
                        .iter()
                        .map(|&x| (x as f64 - mean).powi(2))
                        .sum::<f64>()
                        / width as f64;
                    for c in 0..width {
                        want[row * width + c] = if kind == "plain" {
                            plain[row * width + c].to_f64()
                        } else {
                            let value = (slice[c] as f64 - mean) / (variance + 1e-5).sqrt()
                                * weights[c] as f64;
                            value * (1. + table[2 * cls[row] as usize * width + c] as f64)
                                + table[(2 * cls[row] as usize + 1) * width + c] as f64
                        };
                    }
                }
                let mut data = if kind == "plain" {
                    vec![bytes(&plain)]
                } else {
                    config.push(("eps", "1e-5".into()));
                    config.push(("classes", classes.to_string()));
                    vec![bytes(&input), bytes(&weights), bytes(&table), bytes(&cls)]
                };
                data.push(vec![0; tokens * stride * 2]);
                let last = data.len() - 1;
                let out = h.run_module(
                    &format!("prepare_{dtype}_family"),
                    &stem,
                    &config,
                    [tokens as u32, 1, 1],
                    lanes as u32,
                    &[tokens as u64],
                    &data,
                );
                let out = halves(&out[last], bf);
                let got: Vec<_> = out
                    .chunks_exact(stride)
                    .flat_map(|r| r[..width].iter().copied())
                    .collect();
                close(
                    &got,
                    &want,
                    if bf { 8e-3 } else { 2e-3 },
                    if bf { 8e-3 } else { 2e-3 },
                );
            }
        }
    }
}

#[test]
#[ignore = "requires gfx1151 and provisioned HRX"]
fn rotary_qk_norm_matches_cpu_for_all_head_layouts_and_copies_v() {
    let mut h = Harness::new();
    for (stem, d, rot) in [
        ("rope_qknorm_f16", 128usize, 96usize),
        ("rope64_qknorm_f16", 64, 48),
        ("rope128_qknorm_f16", 128, 128),
    ] {
        let (tokens, heads, kv) = (3usize, 4usize, 2usize);
        let stride = (heads + 2 * kv) * d;
        let offset = heads * d;
        let fused: Vec<_> = values(tokens * stride, 0.7)
            .into_iter()
            .map(f16::from_f32)
            .collect();
        let qw: Vec<_> = values(d, 0.1).into_iter().map(|x| 1. + x).collect();
        let kw: Vec<_> = qw.iter().map(|x| x + 0.05).collect();
        let angles = values(tokens * rot / 2, 3.);
        let cos: Vec<_> = angles.iter().map(|x| x.cos()).collect();
        let sin: Vec<_> = angles.iter().map(|x| x.sin()).collect();
        let oracle = |start: usize, count: usize, weight: &[f32]| {
            let mut result = Vec::new();
            for t in 0..tokens {
                for head in 0..count {
                    let x =
                        &fused[t * stride + start + head * d..t * stride + start + (head + 1) * d];
                    let norm = (x.iter().map(|x| x.to_f64().powi(2)).sum::<f64>() / d as f64
                        + 1e-5)
                        .sqrt();
                    let x: Vec<_> = x
                        .iter()
                        .zip(weight)
                        .map(|(x, &w)| x.to_f64() / norm * w as f64)
                        .collect();
                    for c in 0..d {
                        let v = if c < rot {
                            let half = rot / 2;
                            let p = c % half;
                            let a = cos[t * half + p] as f64;
                            let b = sin[t * half + p] as f64;
                            if c < half {
                                x[c] * a - x[c + half] * b
                            } else {
                                x[c] * a + x[c - half] * b
                            }
                        } else {
                            x[c]
                        };
                        result.push(v);
                    }
                }
            }
            result
        };
        let q = oracle(0, heads, &qw);
        let k = oracle(offset, kv, &kw);
        let v: Vec<_> = (0..tokens)
            .flat_map(|t| {
                fused[t * stride + offset + kv * d..(t + 1) * stride]
                    .iter()
                    .copied()
            })
            .collect();
        let mut config = cfg(&[
            ("row_stride", stride),
            ("heads", heads),
            ("kv_heads", kv),
            ("k_offset", offset),
        ]);
        config.push(("eps", "1e-5".into()));
        let out = h.run_module(
            if stem == "rope_qknorm_f16" {
                stem
            } else {
                "rope_head_family"
            },
            stem,
            &config,
            [tokens as u32, 1, 1],
            256,
            &[tokens as u64],
            &[
                bytes(&fused),
                bytes(&qw),
                bytes(&kw),
                bytes(&cos),
                bytes(&sin),
                vec![0; q.len() * 2],
                vec![0; k.len() * 2],
                vec![0; v.len() * 2],
            ],
        );
        close(&halves(&out[5], false), &q, 2e-2, 2e-2);
        close(&halves(&out[6], false), &k, 2e-2, 2e-2);
        assert_eq!(out[7], bytes(&v));
    }
}

#[test]
#[ignore = "requires gfx1151 and provisioned HRX"]
fn quantized_gemms_match_integer_dot_products_bias_and_residual_classes() {
    let mut h = Harness::new();
    for bits in [4usize, 8] {
        for tile in [128usize, 256] {
            for biased in [false, true] {
                if tile == 128 && (bits == 8 || biased) {
                    continue;
                }
                for mode in ["plain", "resid", "swiglu"] {
                    for pad in [0usize, 128] {
                        if tile == 128 && pad != 0 {
                            continue;
                        }
                        // Padded cases cross the 256-row workgroup boundary.
                        let (m, k, n) = (if pad == 0 { 17usize } else { 257 }, 128usize, 128usize);
                        let stride = k + pad;
                        let a: Vec<i8> =
                            (0..m * stride).map(|i| ((i * 3 % 15) as i8) - 7).collect();
                        let w: Vec<i8> =
                            (0..n * stride).map(|i| ((i * 7 % 15) as i8) - 7).collect();
                        let pack = |v: &[i8]| -> Vec<u8> {
                            if bits == 8 {
                                v.iter().map(|&x| x as u8).collect()
                            } else {
                                v.as_chunks::<2>()
                                    .0
                                    .iter()
                                    .map(|x| (x[0] as u8 & 15) | ((x[1] as u8 & 15) << 4))
                                    .collect()
                            }
                        };
                        let ws = vec![0.01f32; n];
                        let scales = vec![0.02f32; m];
                        let bias = values(n, 0.1);
                        let residual = values(m * n, 0.5);
                        let gates = values(2 * n, 0.3);
                        let classes: Vec<i32> = (0..m).map(|i| (i % 2) as i32).collect();
                        let full: Vec<f64> = (0..m * n)
                            .map(|i| {
                                let (r, c) = (i / n, i % n);
                                let dot = (0..k)
                                    .map(|j| a[r * stride + j] as i32 * w[c * stride + j] as i32)
                                    .sum::<i32>();
                                dot as f64 * ws[c] as f64 * scales[r] as f64
                                    + if biased { bias[c] as f64 } else { 0. }
                            })
                            .collect();
                        let mut stem = format!(
                            "gemm_i{bits}{}{}{}{}",
                            if mode == "plain" {
                                ""
                            } else if mode == "resid" {
                                "_resid"
                            } else {
                                "_swiglu"
                            },
                            if tile == 256 { "_256" } else { "" },
                            if biased { "b" } else { "" },
                            if biased && mode == "swiglu" {
                                "_gs"
                            } else {
                                ""
                            }
                        );
                        let mut config = cfg(&[
                            ("k_size", k),
                            ("n_size", n),
                            ("k_stride", stride),
                            ("m_group", 1),
                        ]);
                        let mut data = vec![pack(&a), pack(&w), bytes(&ws), bytes(&scales)];
                        let want = if mode == "resid" {
                            config.push(("classes", "2".into()));
                            data.extend([bytes(&residual), bytes(&gates), bytes(&classes)]);
                            full.iter()
                                .enumerate()
                                .map(|(i, &x)| {
                                    residual[i] as f64
                                        + gates[classes[i / n] as usize * n + i % n] as f64 * x
                                })
                                .collect::<Vec<_>>()
                        } else if mode == "swiglu" {
                            data.push(vec![0; m * n]);
                            (0..m * n / 2)
                                .map(|i| {
                                    let (r, c) = (i / (n / 2), i % (n / 2));
                                    let a = full[r * n + (c / 16) * 32 + c % 16];
                                    let b = full[r * n + (c / 16) * 32 + c % 16 + 16];
                                    if biased {
                                        b / (1. + (-b).exp()) * a
                                    } else {
                                        a / (1. + (-a).exp()) * b
                                    }
                                })
                                .collect()
                        } else {
                            data.push(vec![0; m * n * 2]);
                            full
                        };
                        if biased {
                            data.push(bytes(&bias));
                        }
                        let out = h.run_module(
                            if tile == 256 {
                                "gemm_packed_256"
                            } else {
                                &stem
                            },
                            &stem,
                            &config,
                            [1, m.div_ceil(tile) as u32, 1],
                            256,
                            &[m as u64],
                            &data,
                        );
                        let got = if mode == "resid" {
                            floats(&out[4])
                        } else {
                            halves(&out[4], false)
                        };
                        eprintln!("checking {stem} with padding {pad}");
                        close(&got, &want, 2e-3, 2e-3);
                        stem.clear();
                    }
                }
            }
        }
    }
}

/// The attention stems `Stack::build` actually selects, against scaled dot-product attention in f64.
///
/// The four `t32` variants above are experimental; these are the ones that run. Each is one
/// workgroup of `16 * waves` query rows per head, `q`/`k`/`v` contiguous `[capacity][stride]` f16
/// with zero headroom past `tokens`, and `gqa8c` is the text encoder's causal 8-query-per-kv layout.
#[test]
#[ignore = "requires gfx1151 and provisioned HRX"]
fn attention_matches_scaled_dot_product_for_the_shipped_layouts() {
    let mut h = Harness::new();
    for (stem, d, waves, heads, gqa, causal) in [
        (
            "attention_mha_lds_f16_wmma",
            128usize,
            4usize,
            4usize,
            1usize,
            false,
        ),
        ("attention_mha8_lds_f16_wmma", 128, 8, 4, 1, false),
        // Gufo-style score tiles and reduced query residency must pass the same
        // independent oracle and ragged-row cases as production schedules.
        ("attention_mhat32_lds_f16_wmma", 128, 4, 4, 1, false),
        ("attention_mha8t32_lds_f16_wmma", 128, 8, 4, 1, false),
        ("attention_mha8h2t32_lds_f16_wmma", 128, 8, 4, 1, false),
        ("attention_mha8h0t32_lds_f16_wmma", 128, 8, 4, 1, false),
        ("attention_mha64_lds_f16_wmma", 64, 4, 4, 1, false),
        ("attention_mha648_lds_f16_wmma", 64, 8, 4, 1, false),
        ("attention_mha64t32_lds_f16_wmma", 64, 4, 4, 1, false),
        ("attention_gqa8c_lds_f16_wmma", 128, 8, 8, 8, true),
    ] {
        let kv_heads = heads / gqa;
        for tokens in [17usize, 40, 96] {
            // Zero headroom past `tokens`, and enough rows for whole query blocks.
            let block = 16 * waves;
            let capacity = (tokens + 16).div_ceil(32) * 32;
            let capacity = capacity.max(tokens.div_ceil(block) * block);
            let (qs, kvs) = (heads * d, kv_heads * d);
            let pad = |rows: &[f32], stride: usize| {
                let mut out = vec![f16::ZERO; capacity * stride];
                for (i, v) in rows.iter().enumerate() {
                    out[i] = f16::from_f32(*v);
                }
                out
            };
            let q = pad(&values(tokens * qs, 0.5), qs);
            let k = pad(&values(tokens * kvs, 0.45), kvs);
            let v = pad(&values(tokens * kvs, 0.6), kvs);

            let scale = 1.0 / (d as f64).sqrt();
            let mut want = vec![0f64; tokens * qs];
            for row in 0..tokens {
                for head in 0..heads {
                    let kv = head / gqa;
                    let last = if causal { row } else { tokens - 1 };
                    let score = |key: usize| {
                        scale
                            * (0..d)
                                .map(|c| {
                                    q[row * qs + head * d + c].to_f64()
                                        * k[key * kvs + kv * d + c].to_f64()
                                })
                                .sum::<f64>()
                    };
                    let top = (0..=last).map(score).fold(f64::MIN, f64::max);
                    let weights: Vec<f64> = (0..=last).map(|j| (score(j) - top).exp()).collect();
                    let total: f64 = weights.iter().sum();
                    for (key, w) in weights.iter().enumerate() {
                        for c in 0..d {
                            want[row * qs + head * d + c] +=
                                w / total * v[key * kvs + kv * d + c].to_f64();
                        }
                    }
                }
            }

            let config = cfg(&[
                ("q_stride", qs),
                ("kv_stride", kvs),
                ("tokens", tokens),
                ("token_capacity", capacity),
                ("out_stride", qs),
            ]);
            let mut config = config;
            config.push(("scale", format!("{scale:.17}")));
            // gqa8c walks 16 query rows per group over the kv heads; the rest take a whole block.
            let grid = if causal {
                [tokens.div_ceil(16) as u32, kv_heads as u32, 1]
            } else {
                [tokens.div_ceil(block) as u32, heads as u32, 1]
            };
            let module = if causal || stem.contains("t32") {
                stem
            } else if d == 64 {
                "attention_mha64_family"
            } else {
                "attention_mha_family"
            };
            let out = h.run_module(
                module,
                stem,
                &config,
                grid,
                32 * waves as u32,
                &[tokens as u64, kv_heads as u64],
                &[bytes(&q), bytes(&k), bytes(&v), vec![0; tokens * qs * 2]],
            );
            let got = halves(&out[3], false);
            assert_eq!(got.len(), want.len(), "{stem} tokens={tokens}");
            for (i, (&a, &b)) in got.iter().zip(&want).enumerate() {
                assert!(
                    a.is_finite() && (a - b).abs() <= 2e-2 + 2e-2 * b.abs(),
                    "{stem} tokens={tokens} element {i}: {a} vs {b}"
                );
            }
        }
    }
}

/// The unnormalised Sylvester Hadamard of order 128: `H[i][j] = (-1)^popcount(i & j)`.
fn hadamard(i: usize, j: usize) -> f64 {
    if (i & j).count_ones().is_multiple_of(2) {
        1.0
    } else {
        -1.0
    }
}

/// What `prepare_qk_i8` is defined to produce: per (token, head) mean-subtract, rotate by the
/// Hadamard, quantise to int8 against the row maximum, pack four codes per word, and report the
/// scale the attention kernel multiplies back in. Returns (packed words, scales, integer codes).
fn prepared_qk(
    x: &[f16],
    mean: &[f32],
    tokens: usize,
    heads: usize,
    d: usize,
    extra: f64,
) -> (Vec<i32>, Vec<f32>, Vec<f64>) {
    let stride = heads * d;
    let (mut words, mut scales, mut codes) = (
        vec![0i32; tokens * heads * 32],
        vec![0f32; tokens * heads],
        vec![0f64; tokens * stride],
    );
    for t in 0..tokens {
        for head in 0..heads {
            let centred: Vec<f64> = (0..d)
                .map(|c| x[t * stride + head * d + c].to_f64() - f64::from(mean[head * d + c]))
                .collect();
            let rotated: Vec<f64> = (0..d)
                .map(|e| (0..d).map(|c| centred[c] * hadamard(c, e)).sum())
                .collect();
            let amax = rotated.iter().fold(0f64, |m, v| m.max(v.abs())).max(1e-30);
            let step = amax / 127.0;
            for e in 0..d {
                let code = (rotated[e] / step).round().clamp(-127.0, 127.0);
                codes[t * stride + head * d + e] = code;
                // four codes per word, element 4*lane + j in byte j
                let word = &mut words[(t * heads + head) * 32 + e / 4];
                *word |= ((code as i32) & 0xff) << (8 * (e % 4));
            }
            scales[t * heads + head] = (step * extra) as f32;
        }
    }
    (words, scales, codes)
}

/// `prepare_qk_i8` against that definition: the codes the attention kernels consume, and the scales
/// they multiply back in. Ties may round either way, so a handful of differing words is expected;
/// a wrong rotation, packing or scale is not.
#[test]
#[ignore = "requires gfx1151 and provisioned HRX"]
fn prepare_qk_int8_rotates_quantises_and_packs_the_attention_operands() {
    let mut h = Harness::new();
    let (tokens, heads, d) = (37usize, 4usize, 128usize);
    let stride = heads * d;
    let extra = 1.0 / (d as f64).sqrt() / 128.0;
    let x: Vec<f16> = values(tokens * stride, 0.7)
        .into_iter()
        .map(f16::from_f32)
        .collect();
    let mean = values(stride, 0.1);
    let (want_words, want_scales, _) = prepared_qk(&x, &mean, tokens, heads, d, extra);

    let mut config = cfg(&[("row_stride", stride), ("head_offset", 0), ("heads", heads)]);
    config.push(("extra_scale", format!("{extra:.17e}")));
    let out = h.run(
        "prepare_qk_i8",
        &config,
        [tokens as u32, 1, 1],
        256,
        &[tokens as u64],
        &[
            bytes(&x),
            bytes(&mean),
            vec![0; tokens * heads * 32 * 4],
            vec![0; tokens * heads * 4],
        ],
    );
    let words: Vec<i32> = out[2]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| i32::from_le_bytes(*b))
        .collect();
    let differing = words
        .iter()
        .zip(&want_words)
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        differing * 500 < words.len(),
        "{differing} of {} packed words differ, beyond rounding ties",
        words.len()
    );
    close(
        &floats(&out[3]),
        &want_scales
            .iter()
            .map(|&v| f64::from(v))
            .collect::<Vec<_>>(),
        1e-7,
        1e-4,
    );
}

/// `attention_i8qk_mha_lds_f16_wmma`, the int8 QK^T path `attn_qk_bits = 8` selects, against the
/// attention its own operands define: the integer dot product scaled by both rows' scales, softmax,
/// then V in f16. Comparing against exact attention would measure the quantisation, not the kernel.
#[test]
#[ignore = "requires gfx1151 and provisioned HRX"]
fn int8_qk_attention_matches_the_attention_its_operands_define() {
    let mut h = Harness::new();
    let (heads, d) = (4usize, 128usize);
    for head_major in [false, true] {
        let waves = if head_major { 8 } else { 4 };
        let stride = heads * d;
        for tokens in [23usize, 64, 97, 129] {
            let block = 16 * waves;
            let capacity = ((tokens + 16).div_ceil(32) * 32).max(tokens.div_ceil(block) * block);
            let q: Vec<f16> = values(tokens * stride, 0.7)
                .into_iter()
                .map(f16::from_f32)
                .collect();
            let k: Vec<f16> = values(tokens * stride, 0.65)
                .into_iter()
                .map(f16::from_f32)
                .collect();
            let v: Vec<f16> = values(tokens * stride, 0.5)
                .into_iter()
                .map(f16::from_f32)
                .collect();
            // Q folds the softmax scale into its own; K is centred on its column mean, as the host does.
            let extra = 1.0 / (d as f64).sqrt() / 128.0;
            let mut kmean = vec![0f32; stride];
            for (c, m) in kmean.iter_mut().enumerate() {
                *m = (0..tokens).map(|t| k[t * stride + c].to_f32()).sum::<f32>() / tokens as f32;
            }
            let (qw, qs, qc) = prepared_qk(&q, &vec![0f32; stride], tokens, heads, d, extra);
            let (kw, ks, kc) = prepared_qk(&k, &kmean, tokens, heads, d, 1.0);

            let mut want = vec![0f64; tokens * stride];
            for row in 0..tokens {
                for head in 0..heads {
                    let score = |key: usize| {
                        (0..d)
                            .map(|e| {
                                qc[row * stride + head * d + e] * kc[key * stride + head * d + e]
                            })
                            .sum::<f64>()
                            * f64::from(qs[row * heads + head])
                            * f64::from(ks[key * heads + head])
                    };
                    let top = (0..tokens).map(score).fold(f64::MIN, f64::max);
                    let weights: Vec<f64> = (0..tokens).map(|j| (score(j) - top).exp()).collect();
                    let total: f64 = weights.iter().sum();
                    for (key, w) in weights.iter().enumerate() {
                        for c in 0..d {
                            want[row * stride + head * d + c] +=
                                w / total * v[key * stride + head * d + c].to_f64();
                        }
                    }
                }
            }

            // The operands arrive padded to the capacity, and V transposed: [channels][capacity].
            let pad_words = |w: &[i32]| {
                let mut out = vec![0i32; capacity * heads * 32];
                for t in 0..tokens {
                    for head in 0..heads {
                        let source = (t * heads + head) * 32;
                        let destination = if head_major {
                            (head * capacity + t) * 32
                        } else {
                            source
                        };
                        out[destination..destination + 32].copy_from_slice(&w[source..source + 32]);
                    }
                }
                out
            };
            let pad_scales = |s: &[f32]| {
                let mut out = vec![0f32; capacity * heads];
                for t in 0..tokens {
                    for head in 0..heads {
                        let destination = if head_major {
                            head * capacity + t
                        } else {
                            t * heads + head
                        };
                        out[destination] = s[t * heads + head];
                    }
                }
                out
            };
            let mut vt = vec![f16::ZERO; stride * capacity];
            for t in 0..tokens {
                for c in 0..stride {
                    vt[c * capacity + t] = v[t * stride + c];
                }
            }
            let mut config = cfg(&[
                ("q_stride", stride),
                ("kv_stride", stride),
                ("tokens", tokens),
                ("token_capacity", capacity),
                ("out_stride", stride),
            ]);
            config.push(("scale", "1.0".into()));
            let out = h.run_module(
                if head_major {
                    "attention_i8qkhm_mha8_k64_lds_f16_wmma"
                } else {
                    "attention_i8qk_family"
                },
                if head_major {
                    "attention_i8qkhm_mha8_k64_lds_f16_wmma"
                } else {
                    "attention_i8qk_mha_lds_f16_wmma"
                },
                &config,
                [tokens.div_ceil(block) as u32, heads as u32, 1],
                32 * waves as u32,
                &[tokens as u64, heads as u64],
                &[
                    bytes(&pad_words(&qw)),
                    bytes(&pad_scales(&qs)),
                    bytes(&pad_words(&kw)),
                    bytes(&pad_scales(&ks)),
                    bytes(&vt),
                    vec![0; tokens * stride * 2],
                ],
            );
            close(&halves(&out[5], false), &want, 2e-2, 2e-2);
        }
    }
}

/// The f16 and bf16 GEMM families: the video VAE decoder's operands and the refiner's, which the
/// int4/int8 test above does not reach. Same three modes and the same epilogues, but the operands
/// arrive as stored floats with no per-row scale, so the reference rounds through the stored width
/// and accumulates in f64.
#[test]
#[ignore = "requires gfx1151 and provisioned HRX"]
fn float_gemms_match_rounded_operands_across_modes_and_epilogues() {
    let mut h = Harness::new();
    for bf in [false, true] {
        for mode in ["plain", "resid", "swiglu"] {
            for biased in [false, true] {
                let (m, k, n) = (if biased { 257usize } else { 17 }, 128usize, 128usize);
                let stride = k + 64;
                let narrow = |v: f32| {
                    if bf {
                        bf16::from_f32(v).to_f64()
                    } else {
                        f16::from_f32(v).to_f64()
                    }
                };
                let raw_a = values(m * stride, 0.5);
                let raw_w = values(n * stride, 0.3);
                let store = |v: &[f32]| -> Vec<u8> {
                    if bf {
                        bytes(&v.iter().map(|&x| bf16::from_f32(x)).collect::<Vec<_>>())
                    } else {
                        bytes(&v.iter().map(|&x| f16::from_f32(x)).collect::<Vec<_>>())
                    }
                };
                let bias = values(n, 0.1);
                let residual = values(m * n, 0.5);
                let gates = values(2 * n, 0.3);
                let classes: Vec<i32> = (0..m).map(|i| (i % 2) as i32).collect();
                let full: Vec<f64> = (0..m * n)
                    .map(|i| {
                        let (r, c) = (i / n, i % n);
                        (0..k)
                            .map(|j| narrow(raw_a[r * stride + j]) * narrow(raw_w[c * stride + j]))
                            .sum::<f64>()
                            + if biased { f64::from(bias[c]) } else { 0.0 }
                    })
                    .collect();

                let stem = format!(
                    "gemm_{}{}_256{}{}",
                    if bf { "bf16" } else { "f16" },
                    match mode {
                        "plain" => "",
                        "resid" => "_resid",
                        _ => "_swiglu",
                    },
                    if biased { "b" } else { "" },
                    if biased && mode == "swiglu" {
                        "_gs"
                    } else {
                        ""
                    }
                );
                let mut config = cfg(&[
                    ("k_size", k),
                    ("n_size", n),
                    ("k_stride", stride),
                    ("m_group", 1),
                ]);
                let mut data = vec![store(&raw_a), store(&raw_w)];
                let want = match mode {
                    "resid" => {
                        config.push(("classes", "2".into()));
                        data.push(bytes(&residual));
                        data.push(bytes(&gates));
                        data.push(bytes(&classes));
                        if biased {
                            data.push(bytes(&bias));
                        }
                        full.iter()
                            .enumerate()
                            .map(|(i, &x)| {
                                f64::from(residual[i])
                                    + f64::from(gates[classes[i / n] as usize * n + i % n]) * x
                            })
                            .collect()
                    }
                    "swiglu" => {
                        data.push(vec![0; m * n]);
                        if biased {
                            data.push(bytes(&bias));
                        }
                        // gate and up interleave in 16-column groups; `_gs` swaps which one gates
                        (0..m * n / 2)
                            .map(|i| {
                                let (r, c) = (i / (n / 2), i % (n / 2));
                                let a = full[r * n + (c / 16) * 32 + c % 16];
                                let b = full[r * n + (c / 16) * 32 + c % 16 + 16];
                                if biased {
                                    b / (1.0 + (-b).exp()) * a
                                } else {
                                    a / (1.0 + (-a).exp()) * b
                                }
                            })
                            .collect()
                    }
                    _ => {
                        data.push(vec![0; m * n * 2]);
                        if biased {
                            data.push(bytes(&bias));
                        }
                        full
                    }
                };
                let module = format!("gemm_{}_family", if bf { "bf16" } else { "f16" });
                let out = h.run_module(
                    &module,
                    &stem,
                    &config,
                    [1, m.div_ceil(256) as u32, 1],
                    256,
                    &[m as u64],
                    &data,
                );
                let got = if mode == "resid" {
                    floats(&out[2])
                } else {
                    halves(&out[2], false)
                };
                eprintln!("checking {stem}");
                close(&got, &want, 3e-3, 3e-3);
            }
        }
    }
}

#[test]
#[ignore = "requires gfx1151 and provisioned HRX"]
fn packed_attention_families_preserve_wave_layouts_and_skip_decisions() {
    let mut h = Harness::new();
    for bits in [4usize, 8] {
        for waves in [4usize, 8] {
            for skip in [false, true] {
                if skip && bits == 8 {
                    continue;
                }
                let (tokens, heads, d) = (97usize, 2usize, 128usize);
                let block = 16 * waves;
                let capacity = (tokens + 16).div_ceil(block) * block;
                let stride = heads * d;
                let codes = |seed: usize| {
                    (0..capacity * stride)
                        .map(|i| {
                            if i / stride < tokens {
                                ((i * 37 + seed) % 7) as i8 - 3
                            } else {
                                0
                            }
                        })
                        .collect::<Vec<_>>()
                };
                let q = codes(1);
                let k = codes(3);
                let pack = |values: &[i8]| {
                    if bits == 8 {
                        values.iter().map(|&v| v as u8).collect::<Vec<_>>()
                    } else {
                        values
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .map(|v| (v[0] as u8 & 15) | ((v[1] as u8 & 15) << 4))
                            .collect()
                    }
                };
                let scales: Vec<f32> = (0..capacity * heads)
                    .map(|i| if i / heads < tokens { 0.02 } else { 0. })
                    .collect();
                let v: Vec<f16> = values(tokens * stride, 0.3)
                    .into_iter()
                    .map(f16::from_f32)
                    .collect();
                let mut vt = vec![f16::ZERO; stride * capacity];
                for t in 0..tokens {
                    for c in 0..stride {
                        vt[c * capacity + t] = v[t * stride + c];
                    }
                }
                let mut want = vec![0.; tokens * stride];
                for row in 0..tokens {
                    for head in 0..heads {
                        let scores: Vec<f64> = (0..tokens)
                            .map(|col| {
                                let dot: i32 = (0..d)
                                    .map(|j| {
                                        i32::from(q[row * stride + head * d + j])
                                            * i32::from(k[col * stride + head * d + j])
                                    })
                                    .sum();
                                f64::from(dot)
                                    * f64::from(scales[row * heads + head])
                                    * f64::from(scales[col * heads + head])
                            })
                            .collect();
                        let top = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                        let p: Vec<_> = scores.iter().map(|s| (s - top).exp()).collect();
                        let total: f64 = p.iter().sum();
                        for j in 0..d {
                            want[row * stride + head * d + j] = (0..tokens)
                                .map(|col| p[col] / total * v[col * stride + head * d + j].to_f64())
                                .sum();
                        }
                    }
                }
                let kind = format!("i{bits}qk{}", if skip { "s" } else { "" });
                let stem = format!(
                    "attention_{kind}_mha{}_lds_f16_wmma",
                    if waves == 8 { "8" } else { "" }
                );
                let mut config = cfg(&[
                    ("q_stride", stride),
                    ("kv_stride", stride),
                    ("out_stride", stride),
                    ("tokens", tokens),
                    ("token_capacity", capacity),
                ]);
                config.push(("scale", "1.0".into()));
                // A large tau keeps all tiles and permits an independent exact-attention oracle.
                if skip {
                    config.push(("skip_tau", "1000.0".into()));
                }
                let data = vec![
                    pack(&q),
                    bytes(&scales),
                    pack(&k),
                    bytes(&scales),
                    bytes(&vt),
                    vec![0; capacity * stride * 2],
                ];
                let out = h.run_module(
                    &format!("attention_{kind}_family"),
                    &stem,
                    &config,
                    [tokens.div_ceil(block) as u32, heads as u32, 1],
                    32 * waves as u32,
                    &[tokens as u64, heads as u64],
                    &data,
                );
                close(
                    &halves(&out[5], false)[..tokens * stride],
                    &want,
                    3e-4,
                    0.03,
                );
                if skip {
                    // Exercise actual skipping as well; the baseline comparison checks decisions.
                    config.last_mut().unwrap().1 = "0.0".into();
                    let out = h.run_module(
                        &format!("attention_{kind}_family"),
                        &stem,
                        &config,
                        [tokens.div_ceil(block) as u32, heads as u32, 1],
                        32 * waves as u32,
                        &[tokens as u64, heads as u64],
                        &data,
                    );
                    assert!(halves(&out[5], false)[..tokens * stride]
                        .iter()
                        .all(|x| x.is_finite()));
                }
            }
        }
    }
}

#[test]
#[ignore = "requires gfx1151 and provisioned HRX"]
fn convolution_family_matches_causal_reflected_gather_and_residual() {
    let mut h = Harness::new();
    for taps in [1usize, 3] {
        for stride in [1usize, 2] {
            for residual in [false, true] {
                let (frames, height, width, cin, pitch, n) =
                    (3usize, 6usize, 6usize, 8usize, 16usize, 64usize);
                let tstride = if taps == 1 { 1 } else { 2 };
                let tout = (frames - 1) / tstride + 1;
                let ho = height / stride;
                let wo = width / stride;
                let m = tout * ho * wo;
                let k = (9 * taps * cin).div_ceil(32) * 32;
                let input: Vec<_> = values(frames * height * width * pitch, 0.3)
                    .into_iter()
                    .map(f16::from_f32)
                    .collect();
                let weight: Vec<_> = values(n * k, 0.15).into_iter().map(f16::from_f32).collect();
                let bias = values(n, 0.05);
                let prior: Vec<_> = values(m * n, 0.1).into_iter().map(f16::from_f32).collect();
                let reflect = |x: isize, size: usize| {
                    if x < 0 {
                        (-x) as usize
                    } else if x as usize >= size {
                        2 * size - 2 - x as usize
                    } else {
                        x as usize
                    }
                };
                let mut want = vec![0.; m * n];
                for row in 0..m {
                    let t = row / (ho * wo);
                    let y = row / wo % ho;
                    let x = row % wo;
                    for c in 0..n {
                        let mut sum = f64::from(bias[c]);
                        for dt in 0..taps {
                            let ti = (t * tstride + if taps == 1 { 2 } else { dt }) as isize - 2;
                            if ti < 0 {
                                continue;
                            }
                            for dy in 0..3 {
                                for dx in 0..3 {
                                    let yi = reflect(
                                        (y * stride + dy) as isize
                                            - if stride == 1 { 1 } else { 0 },
                                        height,
                                    );
                                    let xi = reflect(
                                        (x * stride + dx) as isize
                                            - if stride == 1 { 1 } else { 0 },
                                        width,
                                    );
                                    for ch in 0..cin {
                                        sum += input[((ti as usize * height + yi) * width + xi)
                                            * pitch
                                            + ch]
                                            .to_f64()
                                            * weight[c * k + ((dt * 3 + dy) * 3 + dx) * cin + ch]
                                                .to_f64();
                                    }
                                }
                            }
                        }
                        want[row * n + c] = sum
                            + if residual {
                                prior[row * n + c].to_f64()
                            } else {
                                0.
                            };
                    }
                }
                let stem = if residual {
                    "conv3d_f16_wmma_add"
                } else {
                    "conv3d_f16_wmma"
                };
                let config = cfg(&[
                    ("frames", frames),
                    ("height", height),
                    ("width", width),
                    ("stride", stride),
                    ("tstride", tstride),
                    ("taps_t", taps),
                    ("cin_pad", cin),
                    ("cin_stride", pitch),
                    ("rows_bound", (frames * height * width).div_ceil(64) * 64),
                    ("k_size", k),
                    ("n_size", n),
                ]);
                let mut data = vec![
                    bytes(&input),
                    bytes(&weight),
                    bytes(&bias),
                    bytes(&vec![f16::from_f32(123.); m * n + 64]),
                ];
                if residual {
                    data.push(bytes(&prior));
                }
                let out = h.run_module(
                    "conv3d_f16_family",
                    stem,
                    &config,
                    [1, m.div_ceil(64) as u32, 1],
                    256,
                    &[m as u64],
                    &data,
                );
                let got = halves(&out[3], false);
                close(&got[..m * n], &want, 0.001, 0.002);
                assert!(got[m * n..].iter().all(|&v| v == 123.));
            }
        }
    }
}

#[test]
#[ignore = "requires gfx1151 and provisioned HRX"]
fn quantized_preparation_matches_group_rotation_and_packing() {
    let mut h = Harness::new();
    for bits in [4usize, 8] {
        for kind in ["plain", "norm", "lnorm"] {
            if bits == 4 && kind == "lnorm" {
                continue;
            }
            let (tokens, width, stride, lanes) = (3usize, 512usize, 640usize, 64usize);
            let input = values(tokens * width, 0.4);
            let weights = vec![1.0f32; width];
            let table = vec![0.0f32; 2 * width];
            let classes = vec![0i32; tokens];
            let mut want = vec![0.; tokens * width];
            for row in 0..tokens {
                let x: Vec<f64> = input[row * width..(row + 1) * width]
                    .iter()
                    .map(|&v| {
                        if kind == "plain" {
                            f16::from_f32(v).to_f64()
                        } else {
                            f64::from(v)
                        }
                    })
                    .collect();
                let mean = if kind == "lnorm" {
                    x.iter().sum::<f64>() / width as f64
                } else {
                    0.
                };
                let variance = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / width as f64;
                for col in 0..width {
                    want[row * width + col] = (0..256)
                        .map(|j| {
                            let sign = (0..4).fold(1., |s, digit| {
                                if ((col % 256) >> (2 * digit) & 3) + (j >> (2 * digit) & 3) == 3 {
                                    -s
                                } else {
                                    s
                                }
                            });
                            let value = x[col / 256 * 256 + j];
                            sign * if kind == "plain" {
                                value
                            } else {
                                (value - mean) / (variance + 1e-5).sqrt()
                            }
                        })
                        .sum::<f64>()
                        / 16.;
                }
            }
            let mut config = cfg(&[("width", width), ("out_stride", stride), ("lanes", lanes)]);
            let mut data = if kind == "plain" {
                vec![bytes(
                    &input.iter().copied().map(f16::from_f32).collect::<Vec<_>>(),
                )]
            } else {
                config.extend([("eps", "1e-5".into()), ("classes", "1".into())]);
                vec![
                    bytes(&input),
                    bytes(&weights),
                    bytes(&table),
                    bytes(&classes),
                ]
            };
            let output = data.len();
            data.push(vec![0x55; tokens * stride * bits / 8]);
            data.push(vec![0; tokens * 4]);
            let stem = format!("prepare_{kind}_i{bits}");
            let module = format!("prepare_i{bits}_family");
            let out = h.run_module(
                &module,
                &stem,
                &config,
                [tokens as u32, 1, 1],
                lanes as u32,
                &[tokens as u64],
                &data,
            );
            let scales = floats(&out[output + 1]);
            let qmax = if bits == 4 { 7. } else { 127. };
            for row in 0..tokens {
                let max = want[row * width..(row + 1) * width]
                    .iter()
                    .map(|v| v.abs())
                    .fold(0., f64::max);
                close(&[scales[row]], &[max / qmax], 1e-7, 1e-4);
                for col in 0..width {
                    let i = row * stride + col;
                    let q = if bits == 8 {
                        i32::from(out[output][i] as i8)
                    } else {
                        let n = (out[output][i / 2] >> ((i % 2) * 4)) & 15;
                        if n >= 8 {
                            i32::from(n) - 16
                        } else {
                            i32::from(n)
                        }
                    };
                    let expected = (want[row * width + col] / scales[row]).clamp(-qmax, qmax);
                    assert!(
                        (f64::from(q) - expected).abs() <= 0.501,
                        "{stem} row {row} col {col}: {q} vs {expected}"
                    );
                }
                assert!(out[output]
                    [(row * stride + width) * bits / 8..(row + 1) * stride * bits / 8]
                    .iter()
                    .all(|&v| v == 0x55));
            }
        }
    }
}

/// A recorded chain replays to the bytes dispatching it produces.
///
/// The launches alternate between two buffers, so each one reads what the one before it wrote: a
/// recording that dropped an edge would let them run together and the last write would not be the
/// last one to land. Both arms start from the same input and are compared byte for byte.
#[test]
#[ignore = "requires gfx1151 and provisioned HRX"]
fn a_recorded_chain_replays_to_what_dispatching_it_produces() {
    use h3_hrx::compile::Compiler;
    use h3_hrx::dispatch::{Prepare, Sink};

    const WIDTH: usize = 256;
    const TOKENS: u32 = 64;
    const BYTES: usize = WIDTH * TOKENS as usize * 2;

    let mut stream = Stream::open().expect("gfx1151 device");
    let compiler = Compiler::new(None, "");
    let prepare = Prepare::build(&compiler, &mut stream, "plain", "f16", WIDTH, 0.0, 1, WIDTH)
        .expect("prepare");
    compiler.flush(&mut stream).expect("build the kernel");

    let source: Vec<u8> = (0..BYTES).map(|i| (i % 251) as u8).collect();
    let buffers: Vec<Buffer> = (0..3)
        .map(|_| stream.allocate(BYTES).expect("allocate"))
        .collect();
    // ping-pong through the middle buffer so every launch depends on the one before it
    let chain = [(0usize, 1usize), (1, 2), (2, 1), (1, 2), (2, 1)];

    let mut outcome = Vec::new();
    for recorded in [false, true] {
        stream
            .upload_blocking(buffers[0].binding(), &source)
            .expect("seed the input");
        for b in &buffers[1..] {
            stream.fill(b.binding(), 0x7f).expect("poison the outputs");
        }
        if recorded {
            // the recording borrows the stream until `finish`, so it is scoped tightly
            let mut replay = {
                let mut graph = stream.graph().expect("graph");
                {
                    let mut sink = Sink::Graph {
                        graph: &mut graph,
                        after: Default::default(),
                    };
                    for (from, to) in chain {
                        prepare
                            .emit(
                                &mut sink,
                                None,
                                "prepare",
                                TOKENS,
                                buffers[from].binding(),
                                None,
                                buffers[to].binding(),
                                None,
                            )
                            .expect("record");
                    }
                }
                graph.finish().expect("instantiate")
            };
            stream.launch(&mut replay).expect("replay");
        } else {
            for (from, to) in chain {
                prepare
                    .run(
                        &mut stream,
                        None,
                        "prepare",
                        TOKENS,
                        buffers[from].binding(),
                        None,
                        buffers[to].binding(),
                        None,
                    )
                    .expect("dispatch");
            }
        }
        let mut got = vec![0u8; BYTES];
        stream
            .read_blocking(buffers[1].binding(), &mut got)
            .expect("read back");
        outcome.push(got);
    }
    assert_ne!(outcome[0], vec![0x7f; BYTES], "the chain wrote nothing");
    assert_eq!(
        outcome[0], outcome[1],
        "the replay differs from the launches"
    );
}

/// Three disjoint branches read a shared producer and feed one consumer. Reuse
/// the recording with different bytes to catch missing producer or fan-in edges.
#[test]
#[ignore = "requires gfx1151 and provisioned HRX"]
fn recorded_branches_feed_their_consumer_on_every_replay() {
    use h3_hrx::compile::Compiler;
    use h3_hrx::dispatch::{Prepare, Sink};
    const ROWS: u32 = 64;
    const PART: usize = 256 * ROWS as usize * 2;
    const BYTES: usize = PART * 3;
    let mut stream = Stream::open().unwrap();
    let compiler = Compiler::new(None, "");
    let narrow = Prepare::build(&compiler, &mut stream, "plain", "f16", 256, 0.0, 1, 256).unwrap();
    let wide = Prepare::build(&compiler, &mut stream, "plain", "f16", 768, 0.0, 1, 768).unwrap();
    compiler.flush(&mut stream).unwrap();
    let buffers: Vec<_> = (0..4).map(|_| stream.allocate(BYTES).unwrap()).collect();
    let mut graph = stream.graph().unwrap();
    {
        let mut sink = Sink::Graph {
            graph: &mut graph,
            after: Default::default(),
        };
        wide.emit(
            &mut sink,
            None,
            "producer",
            ROWS,
            buffers[0].binding(),
            None,
            buffers[1].binding(),
            None,
        )
        .unwrap();
        let before = sink.head();
        let mut ends = [Default::default(); 3];
        for (branch, end) in ends.iter_mut().enumerate() {
            sink.resume(before);
            narrow
                .emit(
                    &mut sink,
                    None,
                    "branch",
                    ROWS,
                    buffers[1].binding().slice(branch * PART, PART).unwrap(),
                    None,
                    buffers[2].binding().slice(branch * PART, PART).unwrap(),
                    None,
                )
                .unwrap();
            *end = sink.head();
        }
        sink.after_branches(ends).unwrap();
        wide.emit(
            &mut sink,
            None,
            "consumer",
            ROWS,
            buffers[2].binding(),
            None,
            buffers[3].binding(),
            None,
        )
        .unwrap();
    }
    let mut replay = graph.finish().unwrap();
    for phase in 0..4 {
        let source: Vec<u8> = (0..BYTES / 2)
            .flat_map(|i| {
                f16::from_f32(((i + phase * 19) % 97) as f32 / 128.0)
                    .to_bits()
                    .to_le_bytes()
            })
            .collect();
        stream.upload(buffers[0].binding(), &source).unwrap();
        for buffer in &buffers[1..] {
            stream.fill(buffer.binding(), 0x7f).unwrap();
        }
        stream.launch(&mut replay).unwrap();
        let mut actual = vec![0u8; BYTES];
        stream
            .read_blocking(buffers[3].binding(), &mut actual)
            .unwrap();
        assert_eq!(actual, source);
    }
}

#[test]
#[ignore = "requires gfx1151 and provisioned Loom"]
fn adapter_finish_adds_before_activation_and_preserves_f32_residuals() {
    let mut h = Harness::new();
    let (rows, width) = (3usize, 128usize);
    for mode in 0..3 {
        let input_width = if mode == 1 { width * 2 } else { width };
        let base: Vec<f32> = (0..rows * input_width)
            .map(|i| ((i % 17) as f32 - 8.0) / 8.0)
            .collect();
        let delta: Vec<f16> = (0..rows * input_width)
            .map(|i| f16::from_f32(((i % 7) as f32 - 3.0) / 16.0))
            .collect();
        let gates: Vec<f32> = (0..2 * width)
            .map(|i| if i < width { 0.5 } else { -0.25 })
            .collect();
        let cls = vec![0i32, 1, 0];
        let initial = vec![100000.0f32; rows * width];
        let data = vec![
            bytes(&base),
            bytes(&delta),
            if mode == 2 {
                bytes(&initial)
            } else {
                vec![0; rows * width * 2]
            },
            bytes(&gates),
            bytes(&cls),
        ];
        let out = h.run(
            "adapter_finish",
            &[
                ("width", width.to_string()),
                ("mode", mode.to_string()),
                ("classes", "2".into()),
            ],
            [(rows * width).div_ceil(256) as u32, 1, 1],
            256,
            &[rows as u64],
            &data,
        );
        for row in 0..rows {
            for col in 0..width {
                let ic = if mode == 1 {
                    (col / 16) * 32 + col % 16
                } else {
                    col
                };
                let index = row * input_width + ic;
                let mut expected = base[index] + delta[index].to_f32();
                if mode == 1 {
                    let up = base[index + 16] + delta[index + 16].to_f32();
                    expected = expected * (1.0 / (1.0 + (-expected).exp())) * up;
                }
                let i = row * width + col;
                let actual = if mode == 2 {
                    expected = initial[i] + expected * gates[cls[row] as usize * width + col];
                    f32::from_le_bytes(out[2][i * 4..i * 4 + 4].try_into().unwrap())
                } else {
                    expected = f16::from_f32(expected).to_f32();
                    f16::from_le_bytes(out[2][i * 2..i * 2 + 2].try_into().unwrap()).to_f32()
                };
                assert!(
                    (actual - expected).abs() <= if mode == 2 { 0.008 } else { 0.002 },
                    "mode {mode} row {row} col {col}: {actual} != {expected}"
                );
            }
        }
    }
}

#[test]
#[ignore = "requires gfx1151 and provisioned Loom"]
fn adapter_f32_gemms_preserve_values_above_f16_range() {
    let mut h = Harness::new();
    let (rows, k, n, stride) = (17usize, 128usize, 128usize, 192usize);
    for integer in [true, false] {
        let input = if integer {
            vec![32u8; rows * stride]
        } else {
            bytes(&vec![bf16::from_f32(32.0); rows * stride])
        };
        let weight = if integer {
            vec![32u8; n * stride]
        } else {
            bytes(&vec![bf16::from_f32(32.0); n * stride])
        };
        let mut data = vec![input, weight];
        if integer {
            data.extend([bytes(&vec![1.0f32; n]), bytes(&vec![1.0f32; rows])]);
        }
        data.push(vec![0; rows * n * 4]);
        let stem = if integer {
            "gemm_i8_f32_256"
        } else {
            "gemm_bf16_f32_256"
        };
        let module = if integer {
            "gemm_packed_256"
        } else {
            "gemm_bf16_family"
        };
        let out = h.run_module(
            module,
            stem,
            &[
                ("k_size", k.to_string()),
                ("n_size", n.to_string()),
                ("k_stride", stride.to_string()),
                ("m_group", "1".into()),
            ],
            [1, 1, 1],
            256,
            &[rows as u64],
            &data,
        );
        for value in floats(out.last().unwrap()) {
            assert_eq!(value, 131072.0, "{stem}");
        }
    }
}

/// Fusion must preserve the FP16 boundary after RoPE, every INT8 code/scale,
/// and transposed V including padded rows. The separate kernels have CPU oracles above.
#[test]
#[ignore = "requires gfx1151 and provisioned HRX"]
fn fused_qkv_operands_are_byte_identical_to_separate_preparation() {
    let mut h = Harness::new();
    for (tokens, heads) in [(3usize, 2usize), (129, 56)] {
        let width = heads * 128;
        let capacity = tokens.div_ceil(256) * 256;
        let fused: Vec<_> = values(tokens * width * 3, 0.7)
            .into_iter()
            .map(f16::from_f32)
            .collect();
        let qw: Vec<_> = values(128, 0.1).into_iter().map(|x| 1.0 + x).collect();
        let kw: Vec<_> = qw.iter().map(|x| x + 0.05).collect();
        let angles = values(tokens * 48, 3.0);
        let cos: Vec<_> = angles.iter().map(|x| x.cos()).collect();
        let sin: Vec<_> = angles.iter().map(|x| x.sin()).collect();
        let mut config = cfg(&[
            ("row_stride", width * 3),
            ("heads", heads),
            ("kv_heads", heads),
            ("k_offset", width),
        ]);
        config.push(("eps", "1e-5".into()));
        let split = h.run(
            "rope_qknorm_f16",
            &config,
            [tokens as u32, 1, 1],
            256,
            &[tokens as u64],
            &[
                bytes(&fused),
                bytes(&qw),
                bytes(&kw),
                bytes(&cos),
                bytes(&sin),
                vec![0; tokens * width * 2],
                vec![0; tokens * width * 2],
                vec![0; tokens * width * 2],
            ],
        );
        for (i, weight, extra) in [(0, &qw, 1.0 / 128f64.sqrt() / 128.0), (1, &kw, 1.0)] {
            let mut config = cfg(&[
                ("row_stride", width),
                ("heads", heads),
                ("head_offset", 0),
                ("token_capacity", capacity),
            ]);
            config.push(("extra_scale", format!("{extra:.17e}")));
            let old = h.run(
                "prepare_qk_i8hm",
                &config,
                [tokens as u32, 1, 1],
                256,
                &[tokens as u64],
                &[
                    split[5 + i].clone(),
                    vec![0; width * 4],
                    vec![0; capacity * width],
                    vec![0; capacity * heads * 4],
                ],
            );
            config[0].1 = (width * 3).to_string();
            config[2].1 = (i * width).to_string();
            config.push(("eps", "1e-5".into()));
            let new = h.run(
                "prepare_qk_rope_i8hm",
                &config,
                [tokens as u32, 1, 1],
                256,
                &[tokens as u64],
                &[
                    bytes(&fused),
                    bytes(weight),
                    bytes(&cos),
                    bytes(&sin),
                    vec![0; capacity * width],
                    vec![0; capacity * heads * 4],
                ],
            );
            assert!(old[2] == new[4], "codes differ: tokens={tokens} head={i}");
            assert!(old[3] == new[5], "scales differ: tokens={tokens} head={i}");
        }
        let config = cfg(&[("width", width), ("row_capacity", capacity)]);
        let old = h.run(
            "transpose_f16",
            &config,
            [tokens.div_ceil(32) as u32, (width / 32) as u32, 1],
            256,
            &[tokens as u64],
            &[split[7].clone(), vec![0; capacity * width * 2]],
        );
        let new = h.run(
            "transpose_qkv_v_f16",
            &config,
            [tokens.div_ceil(32) as u32, (width / 32) as u32, 1],
            256,
            &[tokens as u64],
            &[bytes(&fused), vec![0; capacity * width * 2]],
        );
        assert!(old[1] == new[1], "V transpose differs: tokens={tokens}");
    }
}

#[test]
#[ignore = "requires gfx1151 and provisioned HRX; probes the historical compiler workaround"]
fn subgroup_shuffle_preparation_matches_lds_repeatedly() {
    let mut h = Harness::new();
    let (tokens, heads, capacity) = (257usize, 17usize, 512usize);
    let width = heads * 128;
    let x: Vec<_> = values(tokens * width, 0.7)
        .into_iter()
        .map(f16::from_f32)
        .collect();
    let mut config = cfg(&[
        ("row_stride", width),
        ("head_offset", 0),
        ("heads", heads),
        ("token_capacity", capacity),
    ]);
    config.push(("extra_scale", "0.0006905339660024879".into()));
    let inputs = [
        bytes(&x),
        bytes(&values(width, 0.1)),
        vec![0; capacity * width],
        vec![0; capacity * heads * 4],
    ];
    let expected = h.run(
        "prepare_qk_i8hm",
        &config,
        [tokens as u32, 1, 1],
        256,
        &[tokens as u64],
        &inputs,
    );
    for iteration in 0..64 {
        let actual = h.run(
            "prepare_qk_i8hm_shuffle",
            &config,
            [tokens as u32, 1, 1],
            256,
            &[tokens as u64],
            &inputs,
        );
        assert!(
            actual[2] == expected[2],
            "shuffle codes differ at repetition {iteration}"
        );
        assert!(
            actual[3] == expected[3],
            "shuffle scales differ at repetition {iteration}"
        );
    }
}

/// Reused projection scratch is not initially zero. Transposing across the
/// complete capacity must overwrite every poisoned tail element with zero.
#[test]
#[ignore = "requires gfx1151 and provisioned HRX"]
fn transposed_values_clear_reused_capacity() {
    let mut h = Harness::new();
    for tokens in [1usize, 31, 32, 33, 129, 4096, 4097] {
        let width = 256;
        let capacity = (tokens + 32).div_ceil(256) * 256;
        let input: Vec<f16> = values(tokens * width, 0.7)
            .into_iter()
            .map(f16::from_f32)
            .collect();
        let actual = h.run(
            "transpose_f16",
            &cfg(&[("width", width), ("row_capacity", capacity)]),
            [(capacity / 32) as u32, (width / 32) as u32, 1],
            256,
            &[tokens as u64],
            &[bytes(&input), vec![0xff; capacity * width * 2]],
        );
        for (i, word) in actual[1].as_chunks::<2>().0.iter().enumerate() {
            let (channel, row) = (i / capacity, i % capacity);
            let expected = if row < tokens {
                input[row * width + channel].to_le_bytes()
            } else {
                [0, 0]
            };
            assert_eq!(
                *word, expected,
                "tokens={tokens} row={row} channel={channel}"
            );
        }
    }
}

#[test]
#[ignore = "requires gfx1151 and provisioned HRX"]
fn transposed_audio_convolution_matches_f64_with_holes_tails_and_guards() {
    let mut h = Harness::new();
    for n in [1usize, 2, 5, 25, 63, 64, 65, 127, 128, 129] {
        for (taps, stride, pad) in [(9, 5, 2), (4, 2, 1), (3, 5, 0), (5, 2, 2), (1, 1, 0)] {
            for co in [5usize, 32] {
                let ci = 3;
                let olen = (n - 1) * stride + taps - 2 * pad;
                let x = values(ci * n, 0.25);
                let w = values(ci * co * taps, 0.03);
                let bias = values(co, 0.01);
                let mut want: Vec<f64> = bias
                    .iter()
                    .flat_map(|&b| std::iter::repeat_n(b as f64, olen))
                    .collect();
                // Independent scatter formula: each input contributes through
                // each tap, instead of selecting taps from an output phase.
                for i in 0..ci {
                    for input in 0..n {
                        for tap in 0..taps {
                            let output = (input * stride + tap) as isize - pad as isize;
                            if (0..olen as isize).contains(&output) {
                                for o in 0..co {
                                    want[o * olen + output as usize] += x[i * n + input] as f64
                                        * w[(i * co + o) * taps + tap] as f64;
                                }
                            }
                        }
                    }
                }
                let config = cfg(&[
                    ("cin", ci),
                    ("cout", co),
                    ("ksize", taps),
                    ("stride", stride),
                    ("pad", pad),
                    ("len_bound", olen.div_ceil(256) * 256),
                ]);
                let mut kernels = vec![("convt1d_f32", 256)];
                if co == 32 {
                    kernels.push(("convt1d_block_f32", 64));
                }
                let mut outputs = Vec::new();
                for (stem, threads) in kernels {
                    let blocked = stem == "convt1d_block_f32";
                    let mut weights = w.clone();
                    if blocked {
                        weights.clear();
                        for group in 0..co / 16 {
                            for i in 0..ci {
                                for tap in 0..taps {
                                    for o in 0..16 {
                                        weights.push(w[(i * co + group * 16 + o) * taps + tap]);
                                    }
                                }
                            }
                        }
                    }
                    let out = h.run(
                        stem,
                        &config,
                        [
                            olen.div_ceil(threads as usize) as u32,
                            (if blocked { co / 16 } else { co }) as u32,
                            1,
                        ],
                        threads,
                        &[n as u64, olen as u64],
                        &[
                            bytes(&x),
                            bytes(&weights),
                            bytes(&bias),
                            bytes(&vec![113f32; co * olen + 64]),
                        ],
                    );
                    assert_eq!(&out[3][co * olen * 4..], bytes(&[113f32; 64]));
                    close(&floats(&out[3][..co * olen * 4]), &want, 2e-6, 0.);
                    outputs.push(out[3].clone());
                }
                if outputs.len() == 2 {
                    assert_eq!(outputs[0], outputs[1], "transposed convolution at n={n}, kernel={taps}, stride={stride}, pad={pad}");
                }
            }
        }
    }
}

#[test]
#[ignore = "requires gfx1151 and provisioned HRX"]
fn prefetched_audio_upsampling_preserves_fmas_edges_and_guards() {
    let mut h = Harness::new();
    for (ci, co, taps, stride, pad, n) in [
        (4usize, 16usize, 9usize, 5usize, 2usize, 1usize),
        (4, 16, 9, 5, 2, 2),
        (8, 32, 9, 5, 2, 5),
        (20, 32, 9, 5, 2, 13),
        (64, 32, 9, 5, 2, 102),
        (64, 32, 9, 5, 2, 103),
        (4, 16, 4, 2, 1, 31),
        (8, 32, 4, 2, 1, 32),
        (20, 32, 4, 2, 1, 33),
        (32, 16, 4, 2, 1, 256),
        (4, 16, 3, 5, 0, 5),
        (4, 16, 1, 1, 0, 7),
        (1024, 512, 9, 5, 2, 1),
    ] {
        let olen = (n - 1) * stride + taps - 2 * pad;
        let x = values(ci * n, 0.25);
        let w = values(ci * co * taps, 0.03);
        let bias = values(co, 0.01);
        let mut want: Vec<f64> = bias
            .iter()
            .flat_map(|&b| std::iter::repeat_n(b as f64, olen))
            .collect();
        // Independent input-scatter oracle, rather than the kernel's phase lookup.
        for i in 0..ci {
            for input in 0..n {
                for tap in 0..taps {
                    let output = (input * stride + tap) as isize - pad as isize;
                    if (0..olen as isize).contains(&output) {
                        for o in 0..co {
                            want[o * olen + output as usize] +=
                                x[i * n + input] as f64 * w[(i * co + o) * taps + tap] as f64;
                        }
                    }
                }
            }
        }
        let mut packed = Vec::with_capacity(w.len());
        for group in 0..co / 16 {
            for i in 0..ci {
                for tap in 0..taps {
                    for o in 0..16 {
                        packed.push(w[(i * co + group * 16 + o) * taps + tap]);
                    }
                }
            }
        }
        let config = cfg(&[
            ("cin", ci),
            ("cout", co),
            ("ksize", taps),
            ("stride", stride),
            ("pad", pad),
            ("len_bound", olen.div_ceil(256) * 256),
        ]);
        let mut outputs = Vec::new();
        for stem in ["convt1d_block_f32", "convt1d_prefetch_f32"] {
            let out = h.run(
                stem,
                &config,
                [olen.div_ceil(64) as u32, (co / 16) as u32, 1],
                64,
                &[n as u64, olen as u64],
                &[
                    bytes(&x),
                    bytes(&packed),
                    bytes(&bias),
                    bytes(&vec![113f32; co * olen + 64]),
                ],
            );
            assert_eq!(&out[3][co * olen * 4..], bytes(&[113f32; 64]));
            close(&floats(&out[3][..co * olen * 4]), &want, 2e-6, 0.);
            outputs.push(out[3].clone());
        }
        assert_eq!(
            outputs[0], outputs[1],
            "ci={ci}, co={co}, n={n}, taps={taps}"
        );
    }
}

#[test]
#[ignore = "requires gfx1151 and provisioned HRX"]
fn prefetched_audio_convolution_preserves_fmas_padding_and_residuals() {
    let mut h = Harness::new();
    for (ci, co, taps, dilation, pad, n) in [
        (4usize, 8usize, 11usize, 5usize, 25usize, 1usize),
        (4, 16, 11, 5, 25, 2),
        (8, 32, 11, 5, 25, 25),
        (20, 16, 7, 3, 9, 63),
        (20, 16, 7, 3, 9, 64),
        (20, 16, 7, 3, 9, 65),
        (32, 32, 11, 1, 5, 125),
        (32, 16, 3, 5, 5, 127),
        (32, 16, 3, 5, 5, 128),
        (32, 16, 3, 5, 5, 129),
        (8, 16, 1, 1, 0, 250),
        (8, 16, 2, 3, 0, 255),
        (8, 16, 2, 3, 2, 256),
        (8, 16, 2, 3, 8, 257),
        (32, 16, 11, 5, 25, 511),
        (32, 16, 11, 5, 25, 512),
        (32, 16, 11, 5, 25, 513),
        (512, 16, 11, 5, 25, 5),
    ] {
        let x = values(ci * n, 0.25);
        let w = values(co * ci * taps, 0.03);
        let bias = values(co, 0.01);
        let mut packed = Vec::with_capacity(w.len());
        for group in 0..co / 8 {
            for tap in 0..ci * taps {
                for channel in 0..8 {
                    packed.push(w[(group * 8 + channel) * ci * taps + tap]);
                }
            }
        }
        for acc in [0usize, 1] {
            let mut prev = values(co * n + 64, 0.02);
            prev[co * n..].fill(113.);
            let mut want = vec![0f64; co * n];
            // Independent padded-window dot products over the original layout.
            for o in 0..co {
                for t in 0..n {
                    let mut sum =
                        bias[o] as f64 + if acc == 1 { prev[o * n + t] as f64 } else { 0. };
                    for c in 0..ci {
                        for tap in 0..taps {
                            let j = t as isize + (tap * dilation) as isize - pad as isize;
                            if (0..n as isize).contains(&j) {
                                sum += w[(o * ci + c) * taps + tap] as f64
                                    * x[c * n + j as usize] as f64;
                            }
                        }
                    }
                    want[o * n + t] = sum;
                }
            }
            let config = cfg(&[
                ("cin", ci),
                ("cout", co),
                ("ksize", taps),
                ("dilation", dilation),
                ("pad", pad),
                ("accumulate", acc),
                ("len_bound", n.div_ceil(256) * 256),
            ]);
            let mut outputs = Vec::new();
            for (stem, span) in [("conv1d_block_f32", 128), ("conv1d_prefetch_f32", 64)] {
                let out = h.run(
                    stem,
                    &config,
                    [n.div_ceil(span) as u32, (co / 8) as u32, 1],
                    64,
                    &[n as u64],
                    &[bytes(&x), bytes(&packed), bytes(&bias), bytes(&prev)],
                );
                assert_eq!(&out[3][co * n * 4..], bytes(&[113f32; 64]));
                close(&floats(&out[3][..co * n * 4]), &want, 2e-6, 0.);
                outputs.push(out[3].clone());
            }
            assert_eq!(outputs[0], outputs[1],
                "ci={ci}, co={co}, n={n}, taps={taps}, dilation={dilation}, pad={pad}, residual={acc}");
        }
    }
}

#[test]
#[ignore = "requires gfx1151 and provisioned HRX"]
fn channel_lanes_preserve_audio_convolution_fmas_and_edges() {
    let mut h = Harness::new();
    for (ci, co, taps, dilation, pad, n) in [
        (16usize, 8usize, 11usize, 5usize, 25usize, 1usize),
        (16, 16, 11, 5, 25, 2),
        (32, 32, 11, 5, 25, 5),
        (48, 16, 7, 3, 9, 7),
        (48, 16, 7, 3, 9, 8),
        (48, 16, 7, 3, 9, 9),
        (32, 32, 3, 1, 1, 15),
        (32, 32, 3, 1, 1, 16),
        (32, 32, 3, 1, 1, 17),
        (64, 16, 11, 1, 5, 25),
        (32, 16, 3, 5, 5, 31),
        (32, 16, 3, 5, 5, 32),
        (32, 16, 3, 5, 5, 33),
        (32, 16, 3, 5, 5, 63),
        (32, 16, 3, 5, 5, 64),
        (32, 16, 3, 5, 5, 65),
        (16, 16, 1, 1, 0, 125),
        (16, 16, 2, 3, 0, 127),
        (16, 16, 2, 3, 2, 128),
        (16, 16, 2, 3, 8, 129),
        (512, 16, 11, 5, 25, 5),
    ] {
        let x = values(ci * n, 0.25);
        let w = values(co * ci * taps, 0.03);
        let bias = values(co, 0.01);
        let mut packed = Vec::with_capacity(w.len());
        for group in 0..co / 8 {
            for tap in 0..ci * taps {
                for channel in 0..8 {
                    packed.push(w[(group * 8 + channel) * ci * taps + tap]);
                }
            }
        }
        for acc in [0usize, 1] {
            let mut prev = values(co * n + 64, 0.02);
            prev[co * n..].fill(113.);
            let mut want = vec![0f64; co * n];
            // Independent padded-window dot products over the original layout.
            for o in 0..co {
                for t in 0..n {
                    let mut sum =
                        bias[o] as f64 + if acc == 1 { prev[o * n + t] as f64 } else { 0. };
                    for c in 0..ci {
                        for tap in 0..taps {
                            let j = t as isize + (tap * dilation) as isize - pad as isize;
                            if (0..n as isize).contains(&j) {
                                sum += w[(o * ci + c) * taps + tap] as f64
                                    * x[c * n + j as usize] as f64;
                            }
                        }
                    }
                    want[o * n + t] = sum;
                }
            }
            let config = cfg(&[
                ("cin", ci),
                ("cout", co),
                ("ksize", taps),
                ("dilation", dilation),
                ("pad", pad),
                ("accumulate", acc),
                ("len_bound", n.div_ceil(256) * 256),
            ]);
            let mut outputs = Vec::new();
            for (stem, span) in [("conv1d_prefetch_f32", 64), ("conv1d_lane_f32", 8)] {
                let out = h.run(
                    stem,
                    &config,
                    [n.div_ceil(span) as u32, (co / 8) as u32, 1],
                    64,
                    &[n as u64],
                    &[bytes(&x), bytes(&packed), bytes(&bias), bytes(&prev)],
                );
                assert_eq!(&out[3][co * n * 4..], bytes(&[113f32; 64]));
                close(&floats(&out[3][..co * n * 4]), &want, 2e-6, 0.);
                outputs.push(out[3].clone());
            }
            assert_eq!(outputs[0], outputs[1],
                "ci={ci}, co={co}, n={n}, taps={taps}, dilation={dilation}, pad={pad}, residual={acc}");
        }
    }
}

//! Native GPU regressions against independent scalar CPU oracles.
//! Run `cargo test -p h3 --test kernels -- --ignored --test-threads=1` on gfx1151.
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
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let path = root.join("kernels").join(format!("{stem}.loom"));
        let path = if path.is_file() {
            path
        } else {
            root.join("../experiments").join(format!("{stem}.loom"))
        };
        let source = std::fs::read_to_string(path).unwrap();
        let symbol = format!("h3_{stem}");
        let mut request = hrx::loom::Request::new(&source, &symbol);
        request.config = cfg
            .iter()
            .map(|(key, value)| (format!("h3.{stem}.{key}"), value.clone()))
            .collect();
        let path = self
            .compiler
            .compile(
                &request,
                &hrx::bundle::cache_root().unwrap().join("kernels"),
            )
            .unwrap();
        // Safety: trusted checked-in source compiled through HRX. Every test below
        // sizes the bindings from the same dimensions passed as kernel configuration.
        let kernel = unsafe { self.stream.load(&path, &symbol).unwrap() };
        let buffers: Vec<Buffer> = data
            .iter()
            .map(|bytes| self.stream.allocate(bytes.len()).unwrap())
            .collect();
        for (buffer, bytes) in buffers.iter().zip(data) {
            self.stream.upload_queued(buffer, 0, bytes).unwrap();
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
            .map(|&v| self.stream.read_queued(v).unwrap())
            .collect();
        reads
            .into_iter()
            .map(|r| r.wait(&mut self.stream).unwrap())
            .collect()
    }
}
fn bytes<T: bytemuck::Pod>(v: &[T]) -> Vec<u8> {
    bytemuck::cast_slice(v).to_vec()
}
fn floats(v: &[u8]) -> Vec<f64> {
    v.chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()) as f64)
        .collect()
}
fn halves(v: &[u8], bf: bool) -> Vec<f64> {
    v.chunks_exact(2)
        .map(|b| {
            let n = u16::from_le_bytes(b.try_into().unwrap());
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
        for acc in [0usize, 1] {
            let (ci, co, taps, dilation, pad) = (3, 5, 11, 5, 25);
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
            for (stem, threads) in [("conv1d_f32", 256), ("conv1d4_f32", 64)] {
                let out = h.run(
                    stem,
                    &config,
                    [n.div_ceil(256) as u32, co as u32, 1],
                    threads,
                    &[n as u64],
                    &[bytes(&x), bytes(&w), bytes(&bias), bytes(&prev)],
                );
                assert_eq!(&out[3][co * n * 4..], bytes(&[113f32; 64]));
                close(&floats(&out[3][..co * n * 4]), &want, 2e-6, 0.);
                outputs.push(out[3].clone());
            }
            assert_eq!(
                outputs[0], outputs[1],
                "convolution accumulation order at n={n}, residual={acc}"
            );
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
        let out = h.run(
            &format!("matmul_{kind}_bf16_wmma"),
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
                let out = h.run(
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
        let out = h.run(
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
                        let (m, k, n) = (17usize, 128usize, 128usize);
                        let stride = k + pad;
                        let a: Vec<i8> =
                            (0..m * stride).map(|i| ((i * 3 % 15) as i8) - 7).collect();
                        let w: Vec<i8> =
                            (0..n * stride).map(|i| ((i * 7 % 15) as i8) - 7).collect();
                        let pack = |v: &[i8]| -> Vec<u8> {
                            if bits == 8 {
                                v.iter().map(|&x| x as u8).collect()
                            } else {
                                v.chunks_exact(2)
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
                        let out = h.run(
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

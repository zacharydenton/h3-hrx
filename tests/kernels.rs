//! Native GPU regressions against independent scalar CPU oracles.
//! Run `cargo test --test kernels -- --test-threads=1` on gfx1151.
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
                let source_path = Path::new(&dir).join(format!("{stem}.loom"));
                let source_path = if source_path.is_file() {
                    source_path
                } else {
                    Path::new(&dir).join(format!("{module}.loom"))
                };
                let source = std::fs::read_to_string(source_path).expect("baseline kernel source");
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
#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires provisioned HRX")]
fn groupnorm_apply_preserves_groups_channels_and_output_guards() {
    let mut harness = Harness::new();
    for (frames, plane, channels, groups) in [
        (1usize, 1usize, 32usize, 32usize),
        (2, 31, 64, 32),
        (3, 33, 128, 32),
        (2, 65, 160, 32),
        (2, 513, 192, 32),
        (1, 127, 512, 32),
        (1, 128, 512, 32),
        (2, 129, 512, 32),
        (3, 17, 256, 64),
        (2, 17, 1024, 16),
        (1, 4096, 128, 32),
    ] {
        let count = frames * plane * channels;
        let n = (plane * (channels / groups)) as f32;
        for variance in [0., 0.5, -0.001] {
            let input: Vec<f16> = values(count, 2.).into_iter().map(f16::from_f32).collect();
            let stats: Vec<f32> = (0..frames * groups)
                .flat_map(|i| {
                    let mean = (i % 7) as f32 / 8.;
                    [mean * n, (mean * mean + variance) * n]
                })
                .collect();
            let gamma = values(channels, 0.7);
            let beta = values(channels, 0.1);
            let expected: Vec<f64> = (0..count)
                .map(|i| {
                    let ch = i % channels;
                    let g = i / (plane * channels) * groups + ch / (channels / groups);
                    let mean = stats[2 * g] / n;
                    let var = (stats[2 * g + 1] / n - mean * mean).max(0.);
                    let normalized = (input[i].to_f32() - mean) / (var + 1e-6).sqrt();
                    let y = normalized.mul_add(gamma[ch], beta[ch]);
                    f16::from_f32(y * (1. / (1. + (-y).exp()))).to_f64()
                })
                .collect();
            let mut config = cfg(&[
                ("channels", channels),
                ("groups", groups),
                ("plane", plane),
                ("rows_bound", (frames * plane).div_ceil(64) * 64),
            ]);
            config.push(("eps", "1e-6".into()));
            let tile = if (channels / groups).is_multiple_of(4) {
                1024
            } else {
                256
            };
            let guard = f16::from_bits(0x3555);
            let out = harness.run(
                "gn_silu_f16",
                &config,
                [count.div_ceil(tile) as u32, 1, 1],
                256,
                &[frames as u64],
                &[
                    bytes(&input),
                    bytes(&stats),
                    bytes(&gamma),
                    bytes(&beta),
                    bytes(&vec![guard; count + 19]),
                ],
            );
            close(
                &halves(&out[4][..count * 2], false),
                &expected,
                0.001,
                0.002,
            );
            assert_eq!(out[4][count * 2..], bytes(&[guard; 19]));
        }
    }
}

#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires gfx1151 and provisioned HRX"
)]
fn groupnorm_statistics_match_cpu_with_channel_and_plane_tails() {
    let mut h = Harness::new();
    for (frames, plane, channels, groups) in [
        (1usize, 1usize, 32usize, 32usize),
        (2, 31, 64, 32),
        (3, 33, 96, 32),
        (2, 65, 128, 32),
        (2, 37, 160, 32),
        (1, 127, 192, 32),
        (1, 129, 224, 32),
        (2, 257, 256, 32),
        (1, 4096, 128, 32),
        (2, 32, 512, 32),
        (2, 16, 1024, 32),
        (1, 33, 4096, 1),
    ] {
        let per_group = channels / groups;
        for amplitude in [0., 0.0005, 1., 512.] {
            let input: Vec<f16> = (0..frames * plane * channels)
                .map(|i| f16::from_f32(((i * 37 % 101) as f32 - 50.) * amplitude / 50.))
                .collect();
            let mut expected = vec![0f64; frames * groups * 2];
            for t in 0..frames {
                for p in 0..plane {
                    for c in 0..channels {
                        let v = input[(t * plane + p) * channels + c].to_f64();
                        let i = 2 * (t * groups + c / per_group);
                        expected[i] += v;
                        expected[i + 1] += v * v;
                    }
                }
            }
            let guard = [0xa5u8; 28];
            let mut output = vec![0; expected.len() * 4];
            output.extend_from_slice(&guard);
            let result = h.run(
                "gn_stats_f16",
                &cfg(&[
                    ("channels", channels),
                    ("groups", groups),
                    ("plane", plane),
                    ("rows_bound", (frames * plane).div_ceil(64) * 64),
                ]),
                [frames as u32, groups as u32, 1],
                32,
                &[frames as u64],
                &[bytes(&input), output],
            );
            let actual = &result[1];
            close(
                &floats(&actual[..expected.len() * 4]),
                &expected,
                2e-5 * amplitude.max(1.) as f64,
                2e-5,
            );
            assert_eq!(&actual[expected.len() * 4..], &guard);
        }
    }
}

#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires gfx1151 and provisioned HRX"
)]
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
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires gfx1151 and provisioned HRX"
)]
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
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires gfx1151 and provisioned HRX"
)]
fn packed_fp32_projection_preserves_ordered_dots_and_guards() {
    let mut h = Harness::new();
    for (m, k, n) in [(1usize, 1usize, 32usize), (3, 129, 288), (5, 2048, 64)] {
        let x = values(m * k, 0.25);
        let w = values(n * k, 0.125);
        let bias = values(n, 0.01);
        let mut packed = Vec::with_capacity(w.len());
        for group in w.chunks(32 * k) {
            for i in 0..k {
                for column in 0..32 {
                    packed.push(group[column * k + i]);
                }
            }
        }
        let mut expected = vec![113f32; m * n + 64];
        for row in 0..m {
            for col in 0..n {
                let mut acc = bias[col];
                for i in 0..k {
                    acc = x[row * k + i].mul_add(w[col * k + i], acc);
                }
                expected[row * n + col] = acc;
            }
        }
        for (stem, weights) in [("matmul_f32", &w), ("matmul_packed_f32", &packed)] {
            let out = h.run(
                stem,
                &cfg(&[("k", k), ("n", n)]),
                [n.div_ceil(256) as u32, m as u32, 1],
                256,
                &[m as u64],
                &[
                    bytes(&x),
                    bytes(weights),
                    bytes(&bias),
                    bytes(&vec![113f32; m * n + 64]),
                ],
            );
            assert_eq!(out[3], bytes(&expected), "{stem} {m}x{k}x{n}");
        }
    }
}

#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires gfx1151 and provisioned HRX"
)]
fn strided_audio_convolution_preserves_ordered_dots_and_guards() {
    let manager = hrx::residency::ResidencyManager::new(1 << 30).unwrap();
    let mut h = Harness::new();
    h.stream = h.stream.with_memory_budget(manager.budget());
    for len in [1usize, 63, 64, 65, 255, 256, 257, 1023, 1024, 1025, 1285] {
        for stride in [1usize, 2, 4, 5] {
            let (ci, co) = (3, 5);
            let taps = if stride == 1 { 3 } else { 2 * stride };
            let dilation = if stride == 1 { 3 } else { 1 };
            let pad = if stride == 1 { 3 } else { stride.div_ceil(2) };
            let out_len = (len / stride).max(1);
            let x = values(ci * len, 0.25);
            let w = values(co * ci * taps, 0.03);
            let bias = values(co, 0.01);
            let mut expected = vec![113f32; co * out_len + 64];
            for o in 0..co {
                for t in 0..out_len {
                    let mut acc = bias[o];
                    for c in 0..ci {
                        for tap in 0..taps {
                            let j = (t * stride + tap * dilation) as isize - pad as isize;
                            if (0..len as isize).contains(&j) {
                                acc = w[(o * ci + c) * taps + tap]
                                    .mul_add(x[c * len + j as usize], acc);
                            }
                        }
                    }
                    expected[o * out_len + t] = acc;
                }
            }
            let out = h.run(
                "conv1d_s_f32",
                &cfg(&[
                    ("cin", ci),
                    ("cout", co),
                    ("ksize", taps),
                    ("dilation", dilation),
                    ("pad", pad),
                    ("stride", stride),
                    ("in_bound", len.max(256).next_power_of_two()),
                    ("out_bound", out_len.max(256).next_power_of_two()),
                ]),
                [out_len.div_ceil(256) as u32, co as u32, 1],
                256,
                &[out_len as u64, len as u64],
                &[
                    bytes(&x),
                    bytes(&w),
                    bytes(&bias),
                    bytes(&vec![113f32; co * out_len + 64]),
                ],
            );
            assert_eq!(out[3], bytes(&expected), "length {len}, stride {stride}");
        }
    }
}

#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires gfx1151 and provisioned HRX"
)]
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
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires gfx1151 and provisioned HRX"
)]
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
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires gfx1151 and provisioned HRX"
)]
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
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires gfx1151 and provisioned HRX"
)]
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
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires gfx1151 and provisioned HRX"
)]
fn rotary_qk_norm_matches_cpu_for_all_head_layouts_and_copies_v() {
    let mut h = Harness::new();
    for (stem, d, rot, heads, kv) in [
        ("rope_qknorm_f16", 128usize, 96usize, 4usize, 2usize),
        ("rope_qknorm_f16", 128, 96, 56, 56),
        ("rope_qknorm_f16", 128, 96, 12, 3),
        ("rope64_qknorm_f16", 64, 48, 4, 2),
        ("rope128_qknorm_f16", 128, 128, 4, 2),
    ] {
        let tokens = 3usize;
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
        if stem == "rope_qknorm_f16" {
            config.push(("copy_v", "1".into()));
        }
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
        if stem == "rope_qknorm_f16" {
            config.iter_mut().find(|(k, _)| *k == "copy_v").unwrap().1 = "0".into();
            let mut inputs = out.clone();
            inputs[5].fill(0xff);
            inputs[6].fill(0xff);
            inputs[7].fill(0x55);
            let direct = h.run(
                stem,
                &config,
                [tokens as u32, 1, 1],
                256,
                &[tokens as u64],
                &inputs,
            );
            assert_eq!(direct[5], out[5], "Q changed when skipping V");
            assert_eq!(direct[6], out[6], "K changed when skipping V");
            assert_eq!(direct[7], inputs[7], "skipped V was written");
        }
    }
}

#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires gfx1151 and provisioned HRX"
)]
fn quantized_gemms_match_integer_dot_products_bias_and_residual_classes() {
    let mut h = Harness::new();
    for bits in [4usize, 8] {
        for biased in [false, true] {
            for mode in ["plain", "resid", "swiglu"] {
                for (m, n, pad, m_group) in [
                    (17usize, 128usize, 0usize, 1usize),
                    (257, 128, 128, 1),
                    // Two column tiles and a raster group containing a whole empty row tile.
                    (1025, 256, 128, 3),
                ] {
                    let k = 128;
                    let stride = k + pad;
                    let a: Vec<i8> = (0..m * stride).map(|i| ((i * 3 % 15) as i8) - 7).collect();
                    let w: Vec<i8> = (0..n * stride).map(|i| ((i * 7 % 15) as i8) - 7).collect();
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
                    let stem = format!(
                        "gemm_i{bits}{}_256{}{}",
                        if mode == "plain" {
                            ""
                        } else if mode == "resid" {
                            "_resid"
                        } else {
                            "_swiglu"
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
                        ("m_group", m_group),
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
                    let output_bytes = data[4].len();
                    data[4].extend([0xa5; 64]);
                    let out = h.run_module(
                        "gemm_packed_256",
                        &stem,
                        &config,
                        [
                            (n / 128) as u32,
                            (m.div_ceil(256).div_ceil(m_group) * m_group) as u32,
                            1,
                        ],
                        256,
                        &[m as u64],
                        &data,
                    );
                    let got = if mode == "resid" {
                        floats(&out[4][..output_bytes])
                    } else {
                        halves(&out[4][..output_bytes], false)
                    };
                    assert_eq!(&out[4][output_bytes..], &[0xa5; 64], "{stem} output guard");
                    eprintln!("checking {stem} with rows {m}, padding {pad}, raster {m_group}");
                    close(&got, &want, 2e-3, 2e-3);
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
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires gfx1151 and provisioned HRX"
)]
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

#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires gfx1151 and provisioned HRX"
)]
fn dit_down_projection_matches_integer_dots_with_a_partial_tile() {
    use h3_hrx::dispatch::{Classes, Gemm, Tile};
    use h3_hrx::model::{gemm_pitch, FFN, HID};
    let manager = hrx::residency::ResidencyManager::new(256 << 20).unwrap();
    let mut stream = Stream::open().unwrap().with_memory_budget(manager.budget());
    let compiler =
        h3_hrx::compile::Compiler::new(None, Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels"));
    let (m, stride) = (257usize, gemm_pitch(FFN, 8));
    let op = Gemm::build(
        &compiler,
        &mut stream,
        "resid",
        "i8",
        false,
        true,
        FFN,
        HID,
        m,
        2,
        stride,
        Tile::Plain,
        0,
    )
    .unwrap();
    compiler.flush(&mut stream).unwrap();
    let a: Vec<i8> = (0..m * stride).map(|i| (i * 37 % 15) as i8 - 7).collect();
    let w: Vec<i8> = (0..HID * stride).map(|i| (i * 13 % 17) as i8 - 8).collect();
    let residual: Vec<f32> = (0..m * HID + 16)
        .map(|i| (i % 17) as f32 / 16.0 - 0.5)
        .collect();
    let gates: Vec<f32> = (0..2 * HID)
        .map(|i| if i < HID { 0.5 } else { -0.25 })
        .collect();
    let aq = stream.allocate_from(&bytes(&a)).unwrap();
    let wq = stream.allocate_from(&bytes(&w)).unwrap();
    let ws = stream
        .allocate_from(&bytes(&vec![1.0f32 / 64.0; HID]))
        .unwrap();
    let as_ = stream
        .allocate_from(&bytes(&vec![1.0f32 / 32.0; m]))
        .unwrap();
    let gate = stream.allocate_from(&bytes(&gates)).unwrap();
    let out = stream.allocate_from(&bytes(&residual)).unwrap();
    let mut classes = Classes::zeroed(&mut stream, m).unwrap();
    classes
        .write(
            &mut stream,
            &(0..m).map(|i| (i % 2) as i32).collect::<Vec<_>>(),
            2,
        )
        .unwrap();
    op.run(
        &mut stream,
        None,
        "down projection",
        m as u32,
        aq.binding(),
        wq.binding(),
        Some((ws.binding(), as_.binding())),
        out.binding(),
        Some((gate.binding(), classes.all())),
        None,
    )
    .unwrap();
    let mut actual = vec![0.0f32; residual.len()];
    stream
        .read_blocking(out.binding(), bytemuck::cast_slice_mut(&mut actual))
        .unwrap();
    for row in [0, 1, 127, 255, 256] {
        for col in [0, 1, 63, 64, 127, 128, HID - 1] {
            let dot: i32 = (0..FFN)
                .map(|k| i32::from(a[row * stride + k]) * i32::from(w[col * stride + k]))
                .sum();
            let expected =
                residual[row * HID + col] + dot as f32 / 2048.0 * gates[(row % 2) * HID + col];
            assert_eq!(actual[row * HID + col], expected, "row={row} col={col}");
        }
    }
    assert_eq!(&actual[m * HID..], &residual[m * HID..]);
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
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires gfx1151 and provisioned HRX"
)]
fn prepare_qk_int8_rotates_quantises_and_packs_the_attention_operands() {
    let mut h = Harness::new();
    for (tokens, heads) in [(37usize, 4usize), (3, 9), (3, 56)] {
        let d = 128;
        let stride = heads * d;
        let extra = 1.0 / (d as f64).sqrt() / 128.0;
        let mut x: Vec<f16> = values(tokens * stride, 0.7)
            .into_iter()
            .map(f16::from_f32)
            .collect();
        let mut mean = values(stride, 0.1);
        mean[..d].fill(0.0);
        for row in x.chunks_mut(stride) {
            row[..d].fill(f16::ZERO);
            if heads == 56 {
                row[d..2 * d].fill(f16::MAX);
            }
        }
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
        let capacity = tokens.div_ceil(32) * 32;
        config.push(("token_capacity", capacity.to_string()));
        let head_major = h.run(
            "prepare_qk_i8hm",
            &config,
            [tokens as u32, 1, 1],
            256,
            &[tokens as u64],
            &[
                bytes(&x),
                bytes(&mean),
                vec![0; capacity * stride],
                vec![0; capacity * heads * 4],
            ],
        );
        for head in 0..heads {
            for token in 0..capacity {
                let dst = head * capacity + token;
                if token < tokens {
                    let src = token * heads + head;
                    assert_eq!(
                        &head_major[2][dst * d..(dst + 1) * d],
                        &out[2][src * d..(src + 1) * d]
                    );
                    assert_eq!(
                        &head_major[3][dst * 4..(dst + 1) * 4],
                        &out[3][src * 4..(src + 1) * 4]
                    );
                } else {
                    assert!(head_major[2][dst * d..(dst + 1) * d]
                        .iter()
                        .all(|&v| v == 0));
                    assert_eq!(&head_major[3][dst * 4..(dst + 1) * 4], &[0; 4]);
                }
            }
        }
    }
}

/// `attention_i8qk_mha_lds_f16_wmma`, the int8 QK^T path `attn_qk_bits = 8` selects, against the
/// attention its own operands define: the integer dot product scaled by both rows' scales, softmax,
/// then V in f16. Comparing against exact attention would measure the quantisation, not the kernel.
#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires gfx1151 and provisioned HRX"
)]
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

/// Check the head-major crossover, long key loops, partial final tiles and both query-wave halves
/// against scalar attention; very long cases execute 128 query rows against the
/// entire key sequence to cover production lengths without quadratic test cost.
#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires gfx1151 and provisioned HRX"
)]
fn head_major_int8_attention_handles_long_reference_sequences() {
    check_head_major_attention(&[4096, 4097, 8193, 65537, 119585, 478340]);
}

#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires gfx1151 and provisioned HRX"
)]
fn head_major_int8_attention_handles_short_sequences_and_partial_tiles() {
    check_head_major_attention(&[
        1, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1024, 1025, 2048, 2049, 4095,
    ]);
}

fn check_head_major_attention(token_counts: &[usize]) {
    use rand::{Rng, SeedableRng};
    let mut h = Harness::new();
    let (heads, d) = (3usize, 128usize);
    let stride = heads * d;
    for &tokens in token_counts {
        let queries = if tokens > 65536 { 128 } else { tokens };
        let capacity = (tokens + 16).div_ceil(256) * 256;
        let mut rng = rand_chacha::ChaCha12Rng::seed_from_u64(tokens as u64);
        let mut normal = || rng.sample::<f32, _>(rand_distr::StandardNormal);
        let mut q = vec![0i8; heads * capacity * d];
        let mut k = q.clone();
        let mut qs = vec![0f32; heads * capacity];
        let mut ks = qs.clone();
        let mut vt = vec![f16::ZERO; stride * capacity];
        for head in 0..heads {
            for t in 0..tokens {
                qs[head * capacity + t] = (1 + t % 7) as f32 / 512.;
                ks[head * capacity + t] = (1 + t % 3) as f32 / 128.;
                for c in 0..d {
                    let i = (head * capacity + t) * d + c;
                    q[i] = (normal() * 24.).round().clamp(-127., 127.) as i8;
                    k[i] = (normal() * 24.).round().clamp(-127., 127.) as i8;
                    vt[(head * d + c) * capacity + t] = f16::from_f32(normal() * 0.3);
                }
            }
        }
        // A strong match in the final key tile exercises rescaling earlier
        // accumulators when the softmax maximum rises late.
        for head in 0..heads {
            let src = (head * capacity + 31) * d;
            let dst = (head * capacity + tokens - 1) * d;
            k[dst..dst + d].copy_from_slice(&q[src..src + d]);
        }
        let mut config = cfg(&[
            ("q_stride", stride),
            ("kv_stride", stride),
            ("tokens", tokens),
            ("token_capacity", capacity),
            ("out_stride", stride),
        ]);
        config.push(("scale", "1.0".into()));
        let mut previous = None;
        for _ in 0..3 {
            let out = h.run(
                "attention_i8qkhm_mha8_k64_lds_f16_wmma",
                &config,
                [queries.div_ceil(128) as u32, heads as u32, 1],
                256,
                &[queries as u64, heads as u64],
                &[
                    bytes(&q),
                    bytes(&qs),
                    bytes(&k),
                    bytes(&ks),
                    bytes(&vt),
                    vec![0xff; (queries * stride + 19) * 2],
                ],
            );
            let output_bytes = queries * stride * 2;
            assert!(
                out[5][output_bytes..].iter().all(|&byte| byte == 0xff),
                "attention wrote past the output at {tokens} tokens"
            );
            let got = halves(&out[5][..output_bytes], false);
            assert!(got.iter().all(|x| x.is_finite()), "tokens={tokens}");
            if let Some(previous) = &previous {
                assert!(previous == &out[5], "unstable attention at {tokens} tokens");
            } else {
                for row in [0, 15, 16, 31, 63, 64, 127, 128, 4095, tokens - 1]
                    .into_iter()
                    .filter(|&row| row < queries)
                {
                    for head in 0..heads {
                        let score = |key| {
                            let a = (head * capacity + row) * d;
                            let b = (head * capacity + key) * d;
                            let dot: i32 = (0..d)
                                .map(|c| i32::from(q[a + c]) * i32::from(k[b + c]))
                                .sum();
                            f64::from(dot)
                                * f64::from(qs[head * capacity + row])
                                * f64::from(ks[head * capacity + key])
                        };
                        let top = (0..tokens).map(score).fold(f64::NEG_INFINITY, f64::max);
                        let weights: Vec<_> = (0..tokens).map(|j| (score(j) - top).exp()).collect();
                        let total: f64 = weights.iter().sum();
                        let mut want = vec![0.; d];
                        for (c, value) in want.iter_mut().enumerate() {
                            *value = weights
                                .iter()
                                .enumerate()
                                .map(|(j, w)| w * vt[(head * d + c) * capacity + j].to_f64())
                                .sum::<f64>()
                                / total;
                        }
                        let start = row * stride + head * d;
                        close(&got[start..start + d], &want, 1e-3, 2e-3);
                    }
                }
            }
            previous = Some(out[5].clone());
        }
    }
}

/// The f16 and bf16 GEMM families: the video VAE decoder's operands and the refiner's, which the
/// int4/int8 test above does not reach. Same three modes and the same epilogues, but the operands
/// arrive as stored floats with no per-row scale, so the reference rounds through the stored width
/// and accumulates in f64.
#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires gfx1151 and provisioned HRX"
)]
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
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires gfx1151 and provisioned HRX"
)]
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
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires gfx1151 and provisioned HRX"
)]
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
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires gfx1151 and provisioned HRX"
)]
fn quantized_preparation_matches_group_rotation_and_packing() {
    let mut h = Harness::new();
    for bits in [4usize, 8] {
        for kind in ["plain", "norm", "lnorm"] {
            if bits == 4 && kind == "lnorm" {
                continue;
            }
            let shapes = if bits == 8 && kind == "plain" {
                vec![(512usize, 640usize, 64usize), (7168, 7232, 448)]
            } else if bits == 8 && kind == "norm" {
                vec![
                    (512, 640, 64),
                    (5376, 5440, 96),
                    (5376, 5440, 224),
                    (5376, 5440, 672),
                ]
            } else {
                vec![(512, 640, 64)]
            };
            for (width, stride, lanes) in shapes {
                let tokens = 6;
                let mut input = values(tokens * width, 0.4);
                if kind == "norm" {
                    // Include the large residuals seen in late DiT blocks,
                    // epsilon-dominated rows, and exact zero variance.
                    for (row, scale) in input
                        .chunks_exact_mut(width)
                        .zip([1., 1e6, 1e-4, 0., 17., 0.01])
                    {
                        for v in row {
                            *v *= scale;
                        }
                    }
                }
                let weights: Vec<_> = values(width, 0.2).into_iter().map(|v| 1. + v).collect();
                let table = values(3 * 2 * width, 0.1);
                let classes: Vec<i32> = (0..tokens).map(|r| (r % 3) as i32).collect();
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
                    let normalized: Vec<_> = x
                        .iter()
                        .enumerate()
                        .map(|(col, &v)| {
                            if kind == "plain" {
                                v
                            } else {
                                let scale_row = classes[row] as usize * 2 * width;
                                (v - mean) / (variance + 1e-5).sqrt()
                                    * f64::from(weights[col])
                                    * (1. + f64::from(table[scale_row + col]))
                                    + f64::from(table[scale_row + width + col])
                            }
                        })
                        .collect();
                    for col in 0..width {
                        want[row * width + col] = (0..256)
                            .map(|j| {
                                let sign = (0..4).fold(1., |s, digit| {
                                    if ((col % 256) >> (2 * digit) & 3) + (j >> (2 * digit) & 3)
                                        == 3
                                    {
                                        -s
                                    } else {
                                        s
                                    }
                                });
                                sign * normalized[col / 256 * 256 + j]
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
                    config.extend([("eps", "1e-5".into()), ("classes", "3".into())]);
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
}

/// A recorded chain replays to the bytes dispatching it produces.
///
/// The launches alternate between two buffers, so each one reads what the one before it wrote: a
/// recording that dropped an edge would let them run together and the last write would not be the
/// last one to land. Both arms start from the same input and are compared byte for byte.
#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires gfx1151 and provisioned HRX"
)]
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
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires gfx1151 and provisioned HRX"
)]
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
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires gfx1151 and provisioned Loom"
)]
fn adapter_finish_adds_before_activation_and_preserves_f32_residuals() {
    let mut h = Harness::new();
    let (rows, width) = (3usize, 128usize);
    for mode in 0..3 {
        let input_width = if mode == 1 { width * 2 } else { width };
        let base: Vec<f32> = (0..rows * input_width)
            .map(|i| ((i % 17) as f32 - 8.0) / 8.0)
            .collect();
        let delta: Vec<f32> = (0..rows * input_width)
            .map(|i| ((i % 7) as f32 - 3.0) / 16.0)
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
                vec![0; rows * width * if mode == 0 { 2 } else { 4 }]
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
                let mut expected = base[index] + delta[index];
                if mode == 1 {
                    let up = base[index + 16] + delta[index + 16];
                    expected = expected * (1.0 / (1.0 + (-expected).exp())) * up;
                }
                let i = row * width + col;
                let actual = if mode == 2 {
                    expected = initial[i] + expected * gates[cls[row] as usize * width + col];
                    f32::from_le_bytes(out[2][i * 4..i * 4 + 4].try_into().unwrap())
                } else if mode == 1 {
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
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires gfx1151 and provisioned Loom"
)]
fn adapter_f32_gemms_preserve_values_above_f16_range() {
    let mut h = Harness::new();
    for integer in [true, false] {
        let (rows, k, n, stride, group) = if integer {
            (1025usize, 128usize, 256usize, 192usize, 3usize)
        } else {
            (17, 128, 128, 192, 1)
        };
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
        data.last_mut().unwrap().extend([0xa5; 64]);
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
                ("m_group", group.to_string()),
            ],
            [
                (n / 128) as u32,
                (rows.div_ceil(256).div_ceil(group) * group) as u32,
                1,
            ],
            256,
            &[rows as u64],
            &data,
        );
        let output = out.last().unwrap();
        assert_eq!(&output[rows * n * 4..], &[0xa5; 64]);
        for value in floats(&output[..rows * n * 4]) {
            assert_eq!(value, 131072.0, "{stem}");
        }
    }
}

/// Fusion must preserve the FP16 boundary after RoPE, every INT8 code/scale,
/// and transposed V including padded rows. The separate kernels have CPU oracles above.
#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires gfx1151 and provisioned HRX"
)]
fn fused_qkv_operands_are_byte_identical_to_separate_preparation() {
    let mut h = Harness::new();
    for (tokens, heads) in [(3usize, 2usize), (129, 56), (257, 17), (4097, 56)] {
        let width = heads * 128;
        let capacity = tokens.div_ceil(256) * 256;
        let mut fused: Vec<_> = values(tokens * width * 3, 0.7)
            .into_iter()
            .map(f16::from_f32)
            .collect();
        for row in fused.chunks_exact_mut(width * 3) {
            for offset in [0, width] {
                row[offset..offset + 128].fill(f16::ZERO);
                row[offset + 128..offset + 256].fill(f16::MAX);
            }
        }
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
            ("copy_v", 1),
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
                    vec![0x55; capacity * width],
                    vec![0x55; capacity * heads * 4],
                ],
            );
            config[0].1 = (width * 3).to_string();
            config[2].1 = (i * width).to_string();
            config.push(("eps", "1e-5".into()));
            for repetition in 0..8 {
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
                        vec![0x55; capacity * width],
                        vec![0x55; capacity * heads * 4],
                    ],
                );
                assert!(
                    old[2] == new[4],
                    "codes differ: tokens={tokens} head={i} repetition={repetition}"
                );
                assert!(
                    old[3] == new[5],
                    "scales differ: tokens={tokens} head={i} repetition={repetition}"
                );
            }
        }
        let config = cfg(&[("width", width), ("row_capacity", capacity)]);
        let old = h.run(
            "transpose_f16",
            &config,
            [(width / 32) as u32, tokens.div_ceil(32) as u32, 1],
            256,
            &[tokens as u64],
            &[split[7].clone(), vec![0; capacity * width * 2]],
        );
        let new = h.run(
            "transpose_qkv_v_f16",
            &config,
            [
                (width / h3_hrx::model::transpose_qkv_tile(width)) as u32,
                tokens.div_ceil(h3_hrx::model::transpose_qkv_tile(width)) as u32,
                1,
            ],
            256,
            &[tokens as u64],
            &[bytes(&fused), vec![0; capacity * width * 2]],
        );
        assert!(old[1] == new[1], "V transpose differs: tokens={tokens}");
    }
}

#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires gfx1151 and provisioned HRX; probes the historical compiler workaround"
)]
fn subgroup_shuffle_preparation_matches_lds_repeatedly() {
    let mut h = Harness::new();
    // Exercise partial waves, repeated head rounds and production head counts.
    // Nonzero offsets and poisoned capacity also catch addressing regressions.
    for (tokens, heads, offset, extra_scale) in [
        (1usize, 1usize, 0usize, "1.0"),
        (257, 17, 128, "0.0006905339660024879"),
        (3, 256, 128, "1.0"),
        (2049, 56, 128, "0.0006905339660024879"),
        (4097, 56, 0, "1.0"),
    ] {
        let capacity = tokens.div_ceil(256) * 256;
        let width = heads * 128;
        let stride = offset + width + 128;
        let mut x: Vec<_> = values(tokens * stride, 0.7)
            .into_iter()
            .map(f16::from_f32)
            .collect();
        let mut mean = values(stride, 0.1);
        mean[offset..offset + 128].fill(0.0);
        for row in x.chunks_exact_mut(stride) {
            row[offset..offset + 128].fill(f16::ZERO);
            if heads > 1 {
                row[offset + 128..offset + 256].fill(f16::MAX);
            }
        }
        let mut config = cfg(&[
            ("row_stride", stride),
            ("head_offset", offset),
            ("heads", heads),
            ("token_capacity", capacity),
        ]);
        config.push(("extra_scale", extra_scale.into()));
        let inputs = [
            bytes(&x),
            bytes(&mean),
            vec![0x55; capacity * width],
            vec![0x55; capacity * heads * 4],
        ];
        let expected = h.run(
            "prepare_qk_i8hm_lds",
            &config,
            [tokens as u32, 1, 1],
            256,
            &[tokens as u64],
            &inputs,
        );
        let mut token_codes = inputs[2].clone();
        let mut token_scales = inputs[3].clone();
        for token in 0..tokens {
            for head in 0..heads {
                let src = head * capacity + token;
                let dst = token * heads + head;
                token_codes[dst * 128..(dst + 1) * 128]
                    .copy_from_slice(&expected[2][src * 128..(src + 1) * 128]);
                token_scales[dst * 4..(dst + 1) * 4]
                    .copy_from_slice(&expected[3][src * 4..(src + 1) * 4]);
            }
        }
        for (stem, codes, scales) in [
            ("prepare_qk_i8hm", &expected[2], &expected[3]),
            ("prepare_qk_i8", &token_codes, &token_scales),
        ] {
            if stem == "prepare_qk_i8" {
                config.retain(|(key, _)| *key != "token_capacity");
            }
            for iteration in 0..64 {
                let actual = h.run(
                    stem,
                    &config,
                    [tokens as u32, 1, 1],
                    256,
                    &[tokens as u64],
                    &inputs,
                );
                assert!(
                    actual[2] == *codes,
                    "{stem} codes differ: tokens={tokens}, heads={heads}, repetition={iteration}"
                );
                assert!(
                    actual[3] == *scales,
                    "{stem} scales differ: tokens={tokens}, heads={heads}, repetition={iteration}"
                );
            }
        }
    }
}

/// Reused projection scratch is not initially zero. Transposing across the
/// complete capacity must overwrite every poisoned tail element with zero.
#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires gfx1151 and provisioned HRX"
)]
fn transposed_values_clear_reused_capacity() {
    let mut h = Harness::new();
    for width in [32, 96, 256, 7168] {
        for tokens in [1usize, 31, 32, 33, 63, 64, 65, 129, 4096, 4097, 8193] {
            for capacity in [tokens.div_ceil(32) * 32, (tokens + 32).div_ceil(256) * 256] {
                let output_bytes = capacity * width * 2;
                let mut output = vec![0xff; output_bytes];
                output.extend([0xa5; 19]);
                // Move raw half bits, including NaNs, infinities and signed zero.
                let input: Vec<u16> = (0..tokens * width)
                    .map(|i| i.wrapping_mul(37) as u16)
                    .collect();
                let actual = h.run(
                    "transpose_f16",
                    &cfg(&[("width", width), ("row_capacity", capacity)]),
                    [(width / 32) as u32, (capacity / 32) as u32, 1],
                    256,
                    &[tokens as u64],
                    &[bytes(&input), output.clone()],
                );
                let mut fused = vec![0xffffu16; tokens * width * 3];
                for (row, v) in fused
                    .chunks_exact_mut(width * 3)
                    .zip(input.chunks_exact(width))
                {
                    row[2 * width..].copy_from_slice(v);
                }
                let direct = h.run(
                    "transpose_qkv_v_f16",
                    &cfg(&[("width", width), ("row_capacity", capacity)]),
                    [
                        (width / h3_hrx::model::transpose_qkv_tile(width)) as u32,
                        capacity.div_ceil(h3_hrx::model::transpose_qkv_tile(width)) as u32,
                        1,
                    ],
                    256,
                    &[tokens as u64],
                    &[bytes(&fused), output],
                );
                assert!(
                    direct[1] == actual[1],
                    "direct V transpose changed bits or padding"
                );
                assert_eq!(&actual[1][output_bytes..], &[0xa5; 19]);
                assert_eq!(&direct[1][output_bytes..], &[0xa5; 19]);
                for (i, word) in actual[1][..output_bytes]
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .enumerate()
                {
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
    }
}

#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires gfx1151 and provisioned HRX"
)]
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
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires gfx1151 and provisioned HRX"
)]
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
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires gfx1151 and provisioned HRX"
)]
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
        (32, 32, 11, 5, 25, 1000),
        (64, 64, 11, 5, 25, 2048),
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
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires gfx1151 and provisioned HRX"
)]
fn channel_lanes_preserve_audio_convolution_fmas_and_edges() {
    let mut h = Harness::new();
    for (ci, co, taps, dilation, pad, n) in [
        (16usize, 8usize, 11usize, 5usize, 25usize, 1usize),
        (16, 16, 11, 5, 25, 2),
        (32, 32, 11, 5, 25, 5),
        (16, 16, 7, 1, 3, 1),
        (16, 16, 7, 3, 9, 2),
        (32, 16, 7, 5, 15, 5),
        (32, 16, 7, 1, 3, 25),
        (32, 16, 7, 5, 15, 31),
        (32, 16, 7, 3, 9, 32),
        (32, 16, 7, 3, 9, 33),
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
            let mut variants = vec![("conv1d_prefetch_f32", 64), ("conv1d_lane_f32", 8)];
            if taps <= 3 {
                variants.push(("conv1d_k3_f32", 8));
            }
            if taps == 7 {
                variants.push(("conv1d_k7_f32", 8));
            }
            for (stem, span) in variants {
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
            for output in &outputs[1..] {
                assert_eq!(&outputs[0], output,
                    "ci={ci}, co={co}, n={n}, taps={taps}, dilation={dilation}, pad={pad}, residual={acc}");
            }
        }
    }
}

#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires gfx1151 and provisioned HRX"
)]
fn audio_snake_fir_preserves_phases_padding_and_silence() {
    let mut h = Harness::new();
    // FP32 Kaiser-sinc coefficients used by the audio decoder.
    let fir: [f32; 12] = [
        0.0020289647,
        0.009389466,
        -0.025543459,
        -0.057657383,
        0.12857258,
        0.4432098,
        0.4432098,
        0.12857258,
        -0.057657383,
        -0.025543459,
        0.009389466,
        0.0020289647,
    ];
    for (channels, n) in [
        (1usize, 1usize),
        (3, 2),
        (3, 3),
        (3, 7),
        (3, 8),
        (3, 9),
        (512, 25),
        (3, 31),
        (3, 32),
        (3, 33),
        (3, 63),
        (3, 64),
        (3, 65),
        (3, 125),
        (3, 127),
        (3, 128),
        (3, 129),
        (3, 255),
        (3, 256),
        (3, 257),
        (8, 1035),
    ] {
        let alpha: Vec<f32> = (0..channels).map(|c| [0.173, 1., 5.62][c % 3]).collect();
        let beta: Vec<f32> = (0..channels).map(|c| [0.293, 1., 5.62][c % 3]).collect();
        for amplitude in [0., 0.0005, 0.25, 2.] {
            let x = values(channels * n, amplitude);
            let mut want_up = vec![0f64; channels * 2 * n];
            // Independent original transposed-convolution definition: visit all
            // twelve taps, including the zero-insertion and padded-domain tests.
            for c in 0..channels {
                for m in 0..2 * n {
                    let mut u = 0.;
                    for (k, &weight) in fir.iter().enumerate() {
                        let raw = m as isize + 15 - k as isize;
                        if raw >= 0 && raw % 2 == 0 && raw / 2 < (n + 10) as isize {
                            let src = (raw / 2 - 5).clamp(0, n as isize - 1) as usize;
                            u += 2. * weight as f64 * x[c * n + src] as f64;
                        }
                    }
                    want_up[c * 2 * n + m] =
                        u + (alpha[c] as f64 * u).sin().powi(2) / beta[c] as f64;
                }
            }
            let up = h.run(
                "up2_snake_f32",
                &cfg(&[("channels", channels), ("len_bound", n.div_ceil(256) * 256)]),
                [(2 * n).div_ceil(256) as u32, channels as u32, 1],
                256,
                &[n as u64],
                &[
                    bytes(&x),
                    bytes(&fir),
                    bytes(&alpha),
                    bytes(&beta),
                    bytes(&vec![113f32; channels * 2 * n + 64]),
                ],
            );
            assert_eq!(&up[4][channels * 2 * n * 4..], bytes(&[113f32; 64]));
            close(&floats(&up[4][..channels * 2 * n * 4]), &want_up, 3e-5, 0.);
            // Downsample receives an exactly sized input allocation, including
            // the final channel's right edge; no oversized scratch hides reads.
            let up_input = up[4][..channels * 2 * n * 4].to_vec();
            let got_up = floats(&up_input);
            let mut want_down = vec![0f64; channels * n];
            let mut want_complete = want_down.clone();
            for c in 0..channels {
                for t in 0..n {
                    for (k, &weight) in fir.iter().enumerate() {
                        let src = (2 * t as isize + k as isize - 5).clamp(0, (2 * n - 1) as isize)
                            as usize;
                        want_down[c * n + t] += weight as f64 * got_up[c * 2 * n + src];
                        want_complete[c * n + t] += weight as f64 * want_up[c * 2 * n + src];
                    }
                }
            }
            let down = h.run(
                "down2_f32",
                &cfg(&[
                    ("channels", channels),
                    ("len_bound", (2 * n).div_ceil(256) * 256),
                ]),
                [n.div_ceil(256) as u32, channels as u32, 1],
                256,
                &[(2 * n) as u64],
                &[
                    up_input,
                    bytes(&fir),
                    bytes(&vec![113f32; channels * n + 64]),
                ],
            );
            assert_eq!(&down[2][channels * n * 4..], bytes(&[113f32; 64]));
            let got = floats(&down[2][..channels * n * 4]);
            close(&got, &want_down, 3e-5, 0.);
            close(&got, &want_complete, 3e-5, 0.);
            let fused = h.run(
                "snake_fused_f32",
                &cfg(&[("channels", channels), ("len_bound", n.div_ceil(256) * 256)]),
                [n.div_ceil(64) as u32, channels as u32, 1],
                64,
                &[n as u64],
                &[
                    bytes(&x),
                    bytes(&fir),
                    bytes(&alpha),
                    bytes(&beta),
                    bytes(&vec![113f32; channels * n + 64]),
                ],
            );
            assert_eq!(&fused[4][channels * n * 4..], bytes(&[113f32; 64]));
            close(
                &floats(&fused[4][..channels * n * 4]),
                &want_complete,
                3e-5,
                0.,
            );
            assert_eq!(
                fused[4], down[2],
                "channels={channels}, len={n}, amplitude={amplitude}"
            );
            if amplitude == 0. {
                assert!(got.iter().chain(&got_up).all(|&v| v == 0.));
            }
        }
    }
}

#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires gfx1151 and provisioned HRX"
)]
fn narrow_audio_prefetch_preserves_padding_residuals_and_tails() {
    let mut h = Harness::new();
    for (ci, co, taps, dilation, pad, n) in [
        (4usize, 1usize, 1usize, 1usize, 0usize, 1usize),
        (4, 3, 2, 3, 0, 2),
        (8, 8, 11, 5, 128, 2),
        (8, 8, 3, 1, 1, 7),
        (8, 16, 7, 3, 9, 63),
        (12, 3, 11, 5, 25, 64),
        (16, 16, 11, 5, 25, 65),
        (16, 8, 7, 1, 3, 127),
        (8, 8, 11, 3, 15, 128),
        (8, 1, 7, 1, 3, 129),
        (16, 16, 11, 5, 25, 255),
        (16, 16, 11, 1, 5, 256),
        (8, 8, 7, 5, 15, 257),
        (16, 16, 11, 1, 5, 2000),
        (8, 8, 11, 5, 25, 4000),
    ] {
        let x = values(ci * n, 0.25);
        let w = values(co * ci * taps, 0.03);
        let bias = values(co, 0.01);
        for acc in [0usize, 1] {
            let mut prev = values(co * n + 64, 0.02);
            prev[co * n..].fill(113.);
            let mut want = vec![0f64; co * n];
            // Independent padded convolution over the original unpacked weights.
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
            for (stem, span) in [("conv1d4_f32", 256), ("conv1d_narrow_f32", 64)] {
                let out = h.run(
                    stem,
                    &config,
                    [n.div_ceil(span) as u32, co as u32, 1],
                    64,
                    &[n as u64],
                    &[bytes(&x), bytes(&w), bytes(&bias), bytes(&prev)],
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
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires gfx1151 and provisioned HRX"
)]
fn seven_tap_audio_convolution_preserves_ordered_fp32_dots() {
    let mut h = Harness::new();
    for (ci, co, n, dilation) in [
        (2048usize, 16usize, 1usize, 1usize),
        (2048, 16, 2, 1),
        (2048, 1024, 5, 1),
        (2048, 16, 8, 1),
        (512, 16, 5, 1),
        (512, 16, 25, 3),
        (512, 16, 32, 5),
    ] {
        let taps = 7;
        let pad = 3 * dilation;
        let x = values(ci * n, 0.25);
        let w = values(co * ci * taps, 0.03);
        let bias = values(co, 0.01);
        let mut packed = Vec::with_capacity(w.len());
        for group in 0..co / 8 {
            for i in 0..ci * taps {
                for o in 0..8 {
                    packed.push(w[(group * 8 + o) * ci * taps + i]);
                }
            }
        }
        for acc in [0usize, 1] {
            let mut prev = values(co * n + 64, 0.02);
            prev[co * n..].fill(113.);
            let mut want = vec![0f32; co * n];
            // Scalar oracle for the original ordered FP32 contract, with no
            // packing, channel chunks or workgroup tap pruning. This is exact
            // accumulation-order coverage; FP64/model accuracy is a separate gate.
            for o in 0..co {
                for t in 0..n {
                    let mut sum = (if acc == 1 { prev[o * n + t] } else { 0. }) + bias[o];
                    for i in 0..ci {
                        for k in 0..taps {
                            let j = t as isize + (k * dilation) as isize - pad as isize;
                            if (0..n as isize).contains(&j) {
                                sum =
                                    w[(o * ci + i) * taps + k].mul_add(x[i * n + j as usize], sum);
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
            for stem in ["conv1d_lane_f32", "conv1d_k7_f32"] {
                let out = h.run(
                    stem,
                    &config,
                    [n.div_ceil(8) as u32, (co / 8) as u32, 1],
                    64,
                    &[n as u64],
                    &[bytes(&x), bytes(&packed), bytes(&bias), bytes(&prev)],
                );
                assert_eq!(&out[3][co * n * 4..], bytes(&[113f32; 64]));
                assert_eq!(
                    &out[3][..co * n * 4],
                    bytes(&want),
                    "{stem} ci={ci} co={co} n={n} d={dilation} acc={acc}"
                );
            }
        }
    }
}

#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires gfx1151")]
fn world_attention_matches_directed_cpu_oracle() {
    let mut h = Harness::new();
    for waves in [4usize, 8] {
        let stem = if waves == 4 {
            "attention_world_lds_f16_wmma"
        } else {
            "attention_world8_lds_f16_wmma"
        };
        let (tokens, capacity, d) = (97usize, 256usize, 128usize);
        let pad = |scale| {
            let mut x = vec![f16::ZERO; capacity * d];
            for (a, b) in x.iter_mut().zip(values(tokens * d, scale)) {
                *a = f16::from_f32(b);
            }
            x
        };
        let (q, k, v) = (pad(0.5), pad(0.45), pad(0.6));
        for swap in [false, true] {
            let mut routing = vec![0i32; capacity * 2];
            for row in 5..33 {
                routing[row * 2] = if row < 17 { 1 } else { 2 };
            }
            for row in 41..tokens {
                routing[row * 2 + 1] = if (row < 63) ^ swap { 1 } else { 2 };
            }
            let allowed = |r: usize, j: usize| {
                let (a, b, c, e) = (
                    routing[r * 2],
                    routing[j * 2],
                    routing[r * 2 + 1],
                    routing[j * 2 + 1],
                );
                (b == 0 || a == b || c == b) && (a == 0 || e == 0 || a == e)
            };
            let scale = 1.0 / (d as f64).sqrt();
            let mut want = vec![0.0; tokens * d];
            for row in 0..tokens {
                let scores: Vec<_> = (0..tokens)
                    .map(|j| {
                        if allowed(row, j) {
                            scale
                                * (0..d)
                                    .map(|c| q[row * d + c].to_f64() * k[j * d + c].to_f64())
                                    .sum::<f64>()
                        } else {
                            f64::NEG_INFINITY
                        }
                    })
                    .collect();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<_> = scores.iter().map(|s| (s - max).exp()).collect();
                let sum: f64 = weights.iter().sum();
                for c in 0..d {
                    want[row * d + c] = (0..tokens)
                        .map(|j| weights[j] * v[j * d + c].to_f64() / sum)
                        .sum();
                }
            }
            let mut config = cfg(&[
                ("q_stride", d),
                ("kv_stride", d),
                ("out_stride", d),
                ("tokens", tokens),
                ("token_capacity", capacity),
            ]);
            config.push(("scale", format!("{scale:.17}")));
            let out = h.run_module(
                "attention_world_family",
                stem,
                &config,
                [tokens.div_ceil(16 * waves) as u32, 1, 1],
                (32 * waves) as u32,
                &[tokens as u64, 1],
                &[
                    bytes(&q),
                    bytes(&k),
                    bytes(&v),
                    vec![0; tokens * d * 2],
                    bytes(&routing),
                ],
            );
            for (i, (a, b)) in halves(&out[3], false).iter().zip(&want).enumerate() {
                assert!(
                    a.is_finite() && (a - b).abs() < 0.002,
                    "waves={waves} swap={swap} element={i}: {a} != {b}"
                );
            }
        }
    }
}

#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires gfx1151 and provisioned HRX"
)]
fn wide_bf16_preparation_does_not_need_large_lds() {
    let mut h = Harness::new();
    for width in [5120usize, 25600] {
        let input: Vec<f16> = values(width, 0.6).into_iter().map(f16::from_f32).collect();
        let stride = width + 128;
        let out = h.run_module(
            "prepare_bf16_family",
            "prepare_plain_bf16",
            &cfg(&[("width", width), ("lanes", 256), ("out_stride", stride)]),
            [1, 1, 1],
            256,
            &[1],
            &[bytes(&input), vec![0; stride * 2]],
        );
        let expected: Vec<_> = input.iter().map(|x| bf16::from_f32(x.to_f32())).collect();
        assert_eq!(&out[1][..width * 2], bytes(&expected));
    }
}

#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires gfx1151 and provisioned HRX"
)]
fn wide_hadamard_preserves_finite_range() {
    let mut h = Harness::new();
    let (tokens, width, stride) = (2usize, 25600usize, 25664usize);
    let mut x = vec![f16::ZERO; tokens * width];
    for (row, amp) in [2048.0f32, 8192.0].into_iter().enumerate() {
        for c in 0usize..256 {
            let sign = if (0..4).filter(|&j| ((c >> (2 * j)) & 3) == 3).count() % 2 == 0 {
                1.0
            } else {
                -1.0
            };
            x[row * width + c] = f16::from_f32(amp * sign);
        }
    }
    let out = h.run_module(
        "prepare_plain_tiled",
        "prepare_plain_tiled_i8",
        &cfg(&[("width", width), ("lanes", 64), ("out_stride", stride)]),
        [tokens as u32, 1, 1],
        64,
        &[tokens as u64],
        &[bytes(&x), vec![0; tokens * stride], vec![0; tokens * 4]],
    );
    let scales = floats(&out[2]);
    eprintln!(
        "FINITE INPUT Hadamard: expected scales {:?}, actual {:?}",
        [32768.0 / 127.0, 131072.0 / 127.0],
        scales
    );
    assert!(scales[0].is_finite());
    assert!((scales[1] - 131072.0 / 127.0).abs() < 0.001);
}

#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires gfx1151 and provisioned HRX"
)]
fn tiled_preparation_matches_single_pass_across_formats() {
    let mut h = Harness::new();
    for width in [4352usize, 14336] {
        let (rows, stride) = (5usize, width + 64);
        let mut input = vec![0.0f32; rows * width];
        for (i, value) in input[..width].iter_mut().enumerate() {
            *value = ((i * 37 % 101) as f32 - 50.0) * 128.0;
            if i >= width - 256 {
                *value *= 4.0; // Put the largest values in the partial final tile.
            }
        }
        input[2 * width] = f32::NAN;
        input[4 * width - 1] = f32::INFINITY;
        input[4 * width + 13] = f32::NEG_INFINITY;
        for bits in [4, 8] {
            for f32_input in [false, true] {
                let suffix = if f32_input { "_f32" } else { "" };
                let encoded = if f32_input {
                    bytes(&input)
                } else {
                    bytes(&input.iter().copied().map(f16::from_f32).collect::<Vec<_>>())
                };
                let data = [
                    encoded,
                    vec![0xa5; rows * stride * bits / 8],
                    vec![0xa5; rows * 4],
                ];
                let mut expected = None;
                for (tiled, lanes) in [
                    (false, h3_hrx::model::lanes_for(width).unwrap()),
                    (true, 64),
                    (true, 256),
                    (true, 1024),
                ] {
                    let (module, stem) = if tiled {
                        (
                            "prepare_plain_tiled".into(),
                            format!("prepare_plain_tiled{suffix}_i{bits}"),
                        )
                    } else {
                        (
                            format!("prepare_i{bits}_family"),
                            format!("prepare_plain{suffix}_i{bits}"),
                        )
                    };
                    let out = h.run_module(
                        &module,
                        &stem,
                        &cfg(&[("width", width), ("lanes", lanes), ("out_stride", stride)]),
                        [rows as u32, 1, 1],
                        lanes as u32,
                        &[rows as u64],
                        &data,
                    );
                    if let Some(expected) = &expected {
                        assert_eq!(&out[1..], expected, "{stem}, width={width}, lanes={lanes}");
                    } else {
                        expected = Some(out[1..].to_vec());
                    }
                }
            }
        }
    }
}

#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires gfx1151 and provisioned HRX"
)]
fn swiglu_preserves_f32_range() {
    let mut h = Harness::new();
    let (rows, k, n) = (1025usize, 128usize, 256usize);
    let output_bytes = rows * n / 2 * 4;
    let mut output = vec![0; output_bytes];
    output.extend([0xa5; 64]);
    let out = h.run_module(
        "gemm_packed_256",
        "gemm_i8_swiglu_f32_256",
        &cfg(&[
            ("k_size", k),
            ("n_size", n),
            ("k_stride", k),
            ("m_group", 3),
        ]),
        [2, 6, 1],
        256,
        &[rows as u64],
        &[
            vec![1u8; rows * k],
            vec![2u8; n * k],
            bytes(&vec![2.0f32; n]),
            bytes(&vec![1.0f32; rows]),
            output,
        ],
    );
    assert_eq!(&out[4][output_bytes..], &[0xa5; 64]);
    let got = floats(&out[4][..output_bytes]);
    eprintln!(
        "FINITE SwiGLU: gate=512 up=512 expected product=262144 actual={}",
        got[0]
    );
    assert!(got.iter().all(|&v| v == 262144.0));
}

#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires gfx1151 and provisioned HRX"
)]
fn gemm_preserves_nonfinite_results() {
    let mut h = Harness::new();
    let (k, n) = (128usize, 128usize);
    let out = h.run_module(
        "gemm_packed_256",
        "gemm_i8_256",
        &cfg(&[
            ("k_size", k),
            ("n_size", n),
            ("k_stride", k),
            ("m_group", 1),
        ]),
        [1, 1, 1],
        256,
        &[1u64],
        &[
            vec![0u8; 256 * k],
            vec![1u8; n * k],
            bytes(&vec![1.0f32; n]),
            bytes(&vec![f32::INFINITY; 256]),
            vec![0; n * 2],
        ],
    );
    let got = halves(&out[4], false);
    eprintln!("Invalid scale preserved: zero integer dot * infinite scale gives {} in all {} output channels",got[0],got.len());
    assert!(got.iter().all(|v| v.is_nan()));
}

#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires gfx1151 and provisioned HRX"
)]
fn f32_feedforward_variants_and_adapter_finish_preserve_range() {
    let mut h = Harness::new();
    let (rows, capacity, k, n) = (257usize, 512usize, 128usize, 128usize);
    for (module, stem, quant, bias) in [
        ("gemm_packed_256", "gemm_i8_swiglu_f32_256b_gs", true, true),
        ("gemm_bf16_family", "gemm_bf16_swiglu_f32_256", false, false),
    ] {
        let mut data = if quant {
            vec![
                vec![1; capacity * k],
                vec![2; n * k],
                bytes(&vec![2.0f32; n]),
                bytes(&vec![1.0f32; capacity]),
                vec![0; rows * n / 2 * 4],
            ]
        } else {
            vec![
                bytes(&vec![bf16::from_f32(2.0); capacity * k]),
                bytes(&vec![bf16::from_f32(2.0); n * k]),
                vec![0; rows * n / 2 * 4],
            ]
        };
        let output = data.len() - 1;
        if bias {
            data.push(bytes(&vec![1.0f32; n]));
        }
        let out = h.run_module(
            module,
            stem,
            &cfg(&[
                ("k_size", k),
                ("n_size", n),
                ("k_stride", k),
                ("m_group", 1),
            ]),
            [1, 2, 1],
            256,
            &[rows as u64],
            &data,
        );
        let expected = if bias { 513.0 * 513.0 } else { 512.0 * 512.0 };
        assert!(
            floats(&out[output]).iter().all(|&v| v == expected),
            "{stem}"
        );
    }
    let width = 128;
    let mut base = vec![0.0f32; 2 * width];
    for i in 0..width {
        base[(i / 16) * 32 + i % 16] = 512.0;
        base[(i / 16) * 32 + i % 16 + 16] = if i % 2 == 0 { 512.0 } else { -512.0 };
    }
    let out = h.run(
        "adapter_finish",
        &cfg(&[("width", width), ("mode", 1), ("classes", 1)]),
        [1, 1, 1],
        256,
        &[1],
        &[
            bytes(&base),
            bytes(&vec![0.0f32; 2 * width]),
            vec![0; width * 4],
            vec![0; 4],
            vec![0; 4],
        ],
    );
    for (i, v) in floats(&out[2]).iter().enumerate() {
        assert_eq!(*v, if i % 2 == 0 { 262144.0 } else { -262144.0 });
    }
}

#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires gfx1151 and provisioned HRX"
)]
fn f32_preparation_matches_dense_hadamard_and_marks_invalid_rows() {
    let mut h = Harness::new();
    let base = [
        [1.0f32, 1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0, 1.0],
    ];
    let matrix: Vec<f32> = (0usize..256 * 256)
        .map(|i| {
            (0..4)
                .map(|b| base[((i / 256) >> (2 * b)) & 3][((i % 256) >> (2 * b)) & 3])
                .product::<f32>()
                / 16.0
        })
        .collect();
    for width in [256usize, 4352, 8192, 14336, 25600] {
        let rows = 7;
        let stride = width + 64;
        let mut x = vec![0.0f32; rows * width];
        for (i, v) in x[..width].iter_mut().enumerate() {
            *v = ((i * 37 % 997) as f32 - 498.0) * 512.0;
        }
        for (i, v) in x[width..2 * width].iter_mut().enumerate() {
            *v = ((i * 17 % 199) as f32 - 99.0) / 127.0;
        }
        x[3 * width] = f32::NAN;
        x[4 * width + width - 1] = f32::INFINITY;
        x[5 * width + 13] = f32::NEG_INFINITY;
        // Finite inputs can still overflow inside a butterfly; mark that row invalid too.
        x[6 * width..].fill(f32::MAX);
        let mut untiled = None;
        let mut variants = vec![(true, 256)];
        if width <= 16384 {
            variants.insert(0, (false, h3_hrx::model::lanes_for(width).unwrap()));
        }
        if width == 14336 {
            variants.splice(1..1, [(false, 448), (false, 896)]);
        }
        if width == 8192 {
            variants.splice(
                1..1,
                [(false, 32), (false, 64), (false, 512), (false, 1024)],
            );
        }
        for (tiled, lanes) in variants {
            let (module, stem) = if tiled {
                ("prepare_plain_tiled", "prepare_plain_tiled_f32_i8")
            } else {
                ("prepare_i8_family", "prepare_plain_f32_i8")
            };
            let mut codes = vec![0; rows * stride + 19];
            codes[rows * stride..].fill(0x55);
            let mut scale_bytes = vec![0; rows * 4 + 20];
            scale_bytes[rows * 4..].fill(0x55);
            let out = h.run_module(
                module,
                stem,
                &cfg(&[("width", width), ("lanes", lanes), ("out_stride", stride)]),
                [rows as u32, 1, 1],
                lanes as u32,
                &[rows as u64],
                &[bytes(&x), codes, scale_bytes],
            );
            assert!(out[1][rows * stride..].iter().all(|&byte| byte == 0x55));
            assert!(out[2][rows * 4..].iter().all(|&byte| byte == 0x55));
            let scales = floats(&out[2][..rows * 4]);
            let finite = (&out[1][..3 * stride], &out[2][..3 * 4]);
            if let Some((codes, scales)) = &untiled {
                assert_eq!(
                    finite.0, codes,
                    "codes differ: width={width}, lanes={lanes}, tiled={tiled}"
                );
                assert_eq!(
                    finite.1, scales,
                    "scales differ: width={width}, lanes={lanes}, tiled={tiled}"
                );
            } else if !tiled {
                untiled = Some((finite.0.to_vec(), finite.1.to_vec()));
            }
            if width == 14336 && !tiled && lanes == 256 {
                use h3_hrx::dispatch::{ActivationType, Prepare};
                let compiler = h3_hrx::compile::Compiler::new(
                    None,
                    Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels"),
                );
                let prepare = Prepare::build_with_input(
                    &compiler,
                    &mut h.stream,
                    "plain",
                    "i8",
                    width,
                    1e-5,
                    1,
                    stride,
                    ActivationType::F32,
                )
                .unwrap();
                for tokens in [1, 31, 32, 33] {
                    let input = h
                        .stream
                        .allocate_from(&bytes(&x[..width].repeat(tokens)))
                        .unwrap();
                    let codes = h.stream.allocate_zeroed(tokens * stride).unwrap();
                    let scales = h.stream.allocate_zeroed(tokens * 4).unwrap();
                    prepare
                        .run(
                            &mut h.stream,
                            None,
                            "prepare FFN",
                            tokens as u32,
                            input.binding(),
                            None,
                            codes.binding(),
                            Some(scales.binding()),
                        )
                        .unwrap();
                    assert_eq!(
                        h.stream
                            .read(codes.binding())
                            .unwrap()
                            .wait(&mut h.stream)
                            .unwrap(),
                        out[1][..stride].repeat(tokens),
                        "dispatch codes for {tokens} rows",
                    );
                    assert_eq!(
                        h.stream
                            .read(scales.binding())
                            .unwrap()
                            .wait(&mut h.stream)
                            .unwrap(),
                        out[2][..4].repeat(tokens),
                        "dispatch scales for {tokens} rows",
                    );
                }
            }
            let rotated: Vec<f32> = x[..width]
                .as_chunks::<256>()
                .0
                .iter()
                .flat_map(|group| {
                    (0..256)
                        .map(|row| {
                            (0..256)
                                .map(|col| group[col] * matrix[row * 256 + col])
                                .sum::<f32>()
                        })
                        .collect::<Vec<_>>()
                })
                .collect();
            let max = rotated.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            assert!(
                (scales[0] - f64::from(max) / 127.0).abs() < 0.01,
                "{stem} width={width}"
            );
            for (i, v) in rotated.iter().enumerate() {
                let expected = (f64::from(*v) / scales[0])
                    .round_ties_even()
                    .clamp(-127.0, 127.0) as i8;
                assert!(
                    (out[1][i] as i8 as i16 - expected as i16).abs() <= 1,
                    "{stem} code {i}"
                );
            }
            assert!(scales[2].is_finite() && scales[2] > 0.0);
            assert!(out[1][2 * stride..3 * stride].iter().all(|&v| v == 0));
            for row in 3..rows {
                assert!(!scales[row].is_finite());
                assert!(out[1][row * stride..(row + 1) * stride]
                    .iter()
                    .all(|&v| v == 0));
            }
        }
    }
    let x = [262144.0f32, -262144.0, 65536.0, -65536.0].repeat(64);
    let out = h.run_module(
        "prepare_bf16_family",
        "prepare_plain_f32_bf16",
        &cfg(&[("width", 256), ("lanes", 64), ("out_stride", 320)]),
        [1, 1, 1],
        64,
        &[1],
        &[bytes(&x), vec![0; 320 * 2]],
    );
    for (i, v) in x.iter().enumerate() {
        assert_eq!(
            bf16::from_le_bytes(out[1][i * 2..i * 2 + 2].try_into().unwrap()).to_f32(),
            *v
        );
    }
}

#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires gfx1151 and provisioned HRX"
)]
fn f32_dispatch_rejects_half_sized_activation_bindings() {
    use h3_hrx::dispatch::{ActivationType, Gemm, Prepare, Tile};
    let mut stream = Stream::open().unwrap();
    let c = h3_hrx::compile::Compiler::new(None, std::path::PathBuf::new());
    let gemm = Gemm::build_with_output(
        &c,
        &mut stream,
        "swiglu",
        "i8",
        false,
        true,
        128,
        128,
        1,
        1,
        128,
        Tile::Plain,
        0,
        ActivationType::F32,
    )
    .unwrap();
    let a = stream.allocate_zeroed(256 * 128).unwrap();
    let w = stream.allocate_zeroed(128 * 128).unwrap();
    let scale = stream.allocate_zeroed(256 * 4).unwrap();
    let out = stream.allocate(64 * 2).unwrap();
    assert!(gemm
        .run(
            &mut stream,
            None,
            "f32 output extent",
            1,
            a.binding(),
            w.binding(),
            Some((scale.binding(), scale.binding())),
            out.binding(),
            None,
            None
        )
        .is_err());
    let prep = Prepare::build_with_input(
        &c,
        &mut stream,
        "plain",
        "i8",
        256,
        1e-5,
        1,
        256,
        ActivationType::F32,
    )
    .unwrap();
    let half = stream.allocate(256 * 2).unwrap();
    assert!(prep
        .run(
            &mut stream,
            None,
            "f32 input extent",
            1,
            half.binding(),
            None,
            a.binding(),
            Some(scale.binding())
        )
        .is_err());
}

#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires gfx1151 and provisioned HRX"
)]
fn upscale_convolution_uses_symmetric_zero_padding() {
    let mut harness = Harness::new();
    for (t, h, w, ci, co) in [
        (3usize, 4usize, 4usize, 24usize, 64usize),
        (3, 10, 10, 24, 128),
        (2, 10, 18, 32, 192),
        (5, 10, 10, 24, 192),
        (3, 4, 6, 512, 64),
    ] {
        let k = (27 * ci).div_ceil(32) * 32;
        let rows = t * h * w;
        let x = values(rows * ci, 0.3)
            .into_iter()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let weights = values(co * k, 0.2)
            .into_iter()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let bias = values(co, 0.1);
        let mut want = vec![0f64; rows * co];
        for z in 0..t {
            for y in 0..h {
                for xx in 0..w {
                    for o in 0..co {
                        let mut sum = bias[o] as f64;
                        for dz in 0..3 {
                            for dy in 0..3 {
                                for dx in 0..3 {
                                    let (iz, iy, ix) = (
                                        z as isize + dz as isize - 1,
                                        y as isize + dy as isize - 1,
                                        xx as isize + dx as isize - 1,
                                    );
                                    if iz < 0
                                        || iy < 0
                                        || ix < 0
                                        || iz >= t as isize
                                        || iy >= h as isize
                                        || ix >= w as isize
                                    {
                                        continue;
                                    }
                                    for c in 0..ci {
                                        sum += x[((iz as usize * h + iy as usize) * w
                                            + ix as usize)
                                            * ci
                                            + c]
                                            .to_f64()
                                            * weights[o * k + ((dz * 3 + dy) * 3 + dx) * ci + c]
                                                .to_f64();
                                    }
                                }
                            }
                        }
                        want[((z * h + y) * w + xx) * co + o] = sum;
                    }
                }
            }
        }
        let config = cfg(&[
            ("frames", t),
            ("height", h),
            ("width", w),
            ("stride", 1),
            ("tstride", 1),
            ("taps_t", 3),
            ("cin_pad", ci),
            ("cin_stride", ci),
            ("rows_bound", rows.div_ceil(64) * 64),
            ("k_size", k),
            ("n_size", co),
        ]);
        let out = harness.run_module(
            "conv3d_f16_family",
            "upscale_conv3d",
            &config,
            [(co / 64) as u32, rows.div_ceil(64) as u32, 1],
            256,
            &[rows as u64],
            &[
                bytes(&x),
                bytes(&weights),
                bytes(&bias),
                bytes(&vec![f16::from_f32(123.0); rows * co + 64]),
            ],
        );
        let got = halves(&out[3], false);
        close(&got[..rows * co], &want, 0.002, 0.003);
        assert!(got[rows * co..].iter().all(|v| *v == 123.0));
    }
}

#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires provisioned HRX")]
fn upscale_groupnorm_centered_variance_handles_large_offsets() {
    let mut harness = Harness::new();
    for (frames, rows, channels, lanes, offset, scale) in [
        (1usize, 128usize, 64usize, 32usize, 0.0, 8.0),
        (1, 128, 64, 32, 10000.0, 8.0),
        (2, 513, 512, 128, 10000.0, 8.0),
        (3, 1031, 512, 256, 0.0, 8000.0),
        (2, 129, 512, 512, 10000.0, 0.0),
        (1, 12961, 512, 1024, 0.0, 0.001),
        (1, 272161, 512, 256, 10000.0, 8.0),
    ] {
        let input = (0..frames * rows * channels)
            .map(|i| f16::from_f32(offset + (((i * 7) % 17) as f32 - 8.0) * scale))
            .collect::<Vec<_>>();
        let config = cfg(&[
            ("lanes", lanes),
            ("channels", channels),
            ("groups", 32),
            ("plane", rows),
            ("rows_bound", (frames * rows).div_ceil(64) * 64),
        ]);
        let stats = harness.run(
            "upscale_gn_stats",
            &config,
            [frames as u32, 32, 1],
            lanes as u32,
            &[frames as u64],
            &[bytes(&input), bytes(&vec![0.0f32; frames * 64])],
        );
        let got = stats[1]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect::<Vec<_>>();
        for tg in 0..frames * 32 {
            let t = tg / 32;
            let g = tg % 32;
            let per_group = channels / 32;
            let data = (t * rows..(t + 1) * rows)
                .flat_map(|r| {
                    input[r * channels + g * per_group..r * channels + (g + 1) * per_group]
                        .iter()
                        .map(|v| v.to_f64())
                })
                .collect::<Vec<_>>();
            let mean = data.iter().sum::<f64>() / data.len() as f64;
            let variance = data.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / data.len() as f64;
            assert!((got[tg * 2] as f64 - mean).abs() < 0.002 + mean.abs() * 1e-6);
            assert!((got[tg * 2 + 1] as f64 - variance).abs() < 1e-8 + variance * 1e-5);
        }
    }
}

#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires provisioned HRX")]
fn upscale_temporal_depthwise_keeps_channels_separate() {
    let mut harness = Harness::new();
    for (frames, plane, channels, taps) in [
        (1usize, 1usize, 64usize, 1usize),
        (1, 7, 128, 15),
        (3, 4, 64, 5),
        (3, 7, 192, 3),
        (3, 17, 512, 5),
        (42, 17, 512, 5),
        (64, 3, 1024, 15),
        (3, 2051, 64, 5),
        (2, 1, 64, 31),
    ] {
        let count = frames * plane * channels;
        let input = values(count, 0.2)
            .into_iter()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let weight = values(channels * taps, 0.3)
            .into_iter()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let bias = values(channels, 0.1);
        let mut want = vec![0f64; count];
        let mut ordered = vec![f16::ZERO; count];
        for t in 0..frames {
            for p in 0..plane {
                for c in 0..channels {
                    let mut sum = bias[c] as f64;
                    let mut fp32 = bias[c];
                    for k in 0..taps {
                        let ti = t as isize + k as isize - (taps / 2) as isize;
                        if ti >= 0 && ti < frames as isize {
                            let x = input[(ti as usize * plane + p) * channels + c];
                            let w = weight[c * taps + k];
                            sum += x.to_f64() * w.to_f64();
                            fp32 = x.to_f32().mul_add(w.to_f32(), fp32);
                        }
                    }
                    let i = (t * plane + p) * channels + c;
                    want[i] = sum;
                    ordered[i] = f16::from_f32(fp32);
                }
            }
        }
        // Partial workgroups must leave the suffix untouched.
        let guard = f16::from_f32(-123.0);
        let out = harness.run(
            "upscale_temporal",
            &cfg(&[
                ("frames", frames),
                ("plane", plane),
                ("channels", channels),
                ("taps", taps),
            ]),
            [count.div_ceil(1024) as u32, 1, 1],
            256,
            &[count as u64],
            &[
                bytes(&input),
                bytes(&weight),
                bytes(&bias),
                bytes(&vec![guard; count + 19]),
            ],
        );
        assert_eq!(out[3][..count * 2], bytes(&ordered));
        assert_eq!(out[3][count * 2..], bytes(&[guard; 19]));
        close(&halves(&out[3][..count * 2], false), &want, 0.001, 0.002);
    }
}

#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires provisioned HRX")]
fn upscale_groupnorm_apply_preserves_modulation_and_padding() {
    let mut harness = Harness::new();
    let half = |x| f16::from_f32(x).to_f32();
    for (frames, plane, channels, groups, offset, spread, variance) in [
        (1usize, 1usize, 32usize, 32usize, 0.0f32, 2.0f32, 1.0f32),
        (2, 7, 64, 32, 0.0, 8.0, 2.0),
        (3, 17, 192, 32, 0.0, 2.0, 0.5),
        (2, 513, 192, 32, 0.0, 2.0, 0.5),
        (3, 19, 128, 32, 0.0, 8.0, 0.0),
        (2, 513, 128, 32, 0.0, 2.0, 1.0),
        (1, 127, 512, 32, 0.0, 2.0, 1.0),
        (1, 128, 512, 32, 0.0, 2.0, 1.0),
        (2, 129, 512, 32, 10000.0, 8.0, 64.0),
        (1, 513, 512, 32, 65504.0, 0.0, 0.0),
        (3, 17, 256, 64, 0.0, 0.00001, 1e-12),
        (2, 17, 1024, 16, 0.0, 0.0, -0.001),
    ] {
        let count = frames * plane * channels;
        let input = values(count, spread)
            .into_iter()
            .map(|v| f16::from_f32(offset + v))
            .collect::<Vec<_>>();
        let stats = (0..frames * groups)
            .flat_map(|tg| [offset + (tg % 7) as f32 * spread * 0.125, variance])
            .collect::<Vec<_>>();
        let gamma = values(channels, 0.7);
        let beta = values(channels, 0.1);
        let modulation = values(channels * 2, 1.25);
        let expected = (0..count)
            .map(|i| {
                let ch = i % channels;
                let tg = (i / (channels * plane)) * groups + ch / (channels / groups);
                let mean = stats[2 * tg];
                let inv = 1.0 / (stats[2 * tg + 1].max(0.0) + 1e-5).sqrt();
                let norm = half(((input[i].to_f32() - mean) * inv).mul_add(gamma[ch], beta[ch]));
                // The upstream node rounds normalization, factor, product and shift separately.
                let factor = half(1.0 + modulation[ch]);
                let y = half(half(norm * factor) + modulation[channels + ch]);
                half(y * (1.0 / (1.0 + (-y).exp()))) as f64
            })
            .collect::<Vec<_>>();
        let mut config = cfg(&[
            ("channels", channels),
            ("groups", groups),
            ("plane", plane),
            ("rows_bound", (frames * plane).div_ceil(64) * 64),
        ]);
        config.push(("eps", "0.00001".into()));
        let tile = if plane * channels >= 65536 && (channels / groups).is_multiple_of(4) {
            1024
        } else {
            256
        };
        let guard = f16::from_f32(-123.0);
        let out = harness.run(
            "upscale_gn_silu",
            &config,
            [count.div_ceil(tile) as u32, 1, 1],
            256,
            &[frames as u64],
            &[
                bytes(&input),
                bytes(&stats),
                bytes(&gamma),
                bytes(&beta),
                bytes(&vec![guard; count + 19]),
                bytes(&modulation),
            ],
        );
        assert_eq!(out[4][count * 2..], bytes(&[guard; 19]));
        close(
            &halves(&out[4][..count * 2], false),
            &expected,
            0.0001,
            0.002,
        );
    }
}

#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires provisioned HRX")]
fn upscale_fused_residual_preserves_convolution_rounding_and_guards() {
    let mut harness = Harness::new();
    let compiler =
        h3_hrx::compile::Compiler::new(None, Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels"));
    for (frames, height, width, ci, co) in [
        (1usize, 2usize, 2usize, 8usize, 64usize),
        (3, 4, 6, 24, 64),
        (2, 10, 18, 32, 192),
        (3, 4, 6, 512, 64),
    ] {
        let rows = frames * height * width;
        let k = (27 * ci).div_ceil(32) * 32;
        let kernels = [false, true].map(|add| {
            h3_hrx::dispatch::Conv3d::build_padding(
                &compiler,
                &mut harness.stream,
                add,
                frames,
                height,
                width,
                1,
                1,
                3,
                ci,
                ci,
                k,
                co,
                true,
            )
            .unwrap()
        });
        for zero_weights in [false, true] {
            let input = values(rows * ci, 0.2)
                .into_iter()
                .map(f16::from_f32)
                .collect::<Vec<_>>();
            let weights = values(co * k, if zero_weights { 0.0 } else { 0.03 })
                .into_iter()
                .map(f16::from_f32)
                .collect::<Vec<_>>();
            // These biases differ from their FP16 values. With zero weights and a cancelling
            // residual, a fused path that skips the convolution's FP16 boundary returns nonzero.
            let bias = (0..co)
                .map(|ch| if ch % 2 == 0 { 1.0003f32 } else { -1.0003 })
                .collect::<Vec<_>>();
            let guard = f16::from_f32(-123.0);
            let mut residual = (0..rows * co)
                .map(|i| {
                    if zero_weights {
                        f16::from_f32(-f16::from_f32(bias[i % co]).to_f32())
                    } else {
                        f16::from_f32(((i * 7 % 101) as f32 - 50.0) * 0.03)
                    }
                })
                .collect::<Vec<_>>();
            residual.extend([guard; 19]);
            let x = harness.stream.allocate_from(&bytes(&input)).unwrap();
            let w = harness.stream.allocate_from(&bytes(&weights)).unwrap();
            let b = harness.stream.allocate_from(&bytes(&bias)).unwrap();
            let res = harness.stream.allocate_from(&bytes(&residual)).unwrap();
            let out = harness
                .stream
                .allocate_from(&bytes(&vec![guard; rows * co + 19]))
                .unwrap();
            kernels[0]
                .run(
                    &mut harness.stream,
                    None,
                    "upscale conv",
                    x.binding(),
                    w.binding(),
                    b.binding(),
                    out.binding(),
                    None,
                )
                .unwrap();
            let separate = harness
                .stream
                .read(out.binding())
                .unwrap()
                .wait(&mut harness.stream)
                .unwrap();
            let expected = separate[..rows * co * 2]
                .as_chunks::<2>()
                .0
                .iter()
                .zip(&residual)
                .map(|(v, r)| f16::from_f32(f16::from_le_bytes(*v).to_f32() + r.to_f32()))
                .collect::<Vec<_>>();
            if zero_weights {
                assert!(expected.iter().all(|v| v.to_f32() == 0.0));
            }
            harness
                .stream
                .upload(out.binding(), &bytes(&vec![guard; rows * co + 19]))
                .unwrap();
            kernels[1]
                .run(
                    &mut harness.stream,
                    None,
                    "upscale conv + residual",
                    x.binding(),
                    w.binding(),
                    b.binding(),
                    out.binding(),
                    Some(res.binding()),
                )
                .unwrap();
            let actual = harness
                .stream
                .read(out.binding())
                .unwrap()
                .wait(&mut harness.stream)
                .unwrap();
            assert_eq!(actual[..rows * co * 2], bytes(&expected));
            assert_eq!(actual[rows * co * 2..], bytes(&[guard; 19]));
            assert_eq!(
                harness
                    .stream
                    .read(res.binding())
                    .unwrap()
                    .wait(&mut harness.stream)
                    .unwrap(),
                bytes(&residual)
            );
        }
    }
}

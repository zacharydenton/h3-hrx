//! Resident GPU workloads. Compilation, allocation and readback are outside timing.
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use h3_hrx::{
    compile::Compiler,
    dispatch::{ActivationType, Gemm, MatmulF32, Prepare, Sink, Tile},
};
use half::{bf16, f16};
use hrx::{Buffer, Stream};
use std::{path::PathBuf, time::Duration};

fn compiler() -> Compiler {
    Compiler::new(
        None,
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("kernels"),
    )
}

fn upload(stream: &mut Stream, bytes: &[u8]) -> Buffer {
    stream.allocate_from(bytes).unwrap()
}

fn check_f32(stream: &mut Stream, buffer: &Buffer) -> Vec<u8> {
    let mut bytes = vec![0; buffer.binding().len()];
    stream.read_blocking(buffer.binding(), &mut bytes).unwrap();
    assert!(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .all(|v| f32::from_le_bytes(*v).is_finite()));
    assert!(bytes.iter().any(|&v| v != 0), "empty output");
    bytes
}

fn preparation(c: &mut Criterion) {
    let mut group = c.benchmark_group("prepare_f32_i8");
    for (tokens, width) in [
        (1usize, 14336usize),
        (32, 14336),
        (256, 14336),
        (256, 25600),
        (2048, 14336),
    ] {
        group.throughput(Throughput::Elements((tokens * width) as u64));
        group.bench_function(
            BenchmarkId::new("plain", format!("{tokens}x{width}")),
            |b| {
                let manager = hrx::residency::ResidencyManager::new(512 << 20).unwrap();
                let mut stream = Stream::open().unwrap().with_memory_budget(manager.budget());
                let compiler = compiler();
                let prepare = Prepare::build_with_input(
                    &compiler,
                    &mut stream,
                    "plain",
                    "i8",
                    width,
                    1e-5,
                    1,
                    width,
                    ActivationType::F32,
                )
                .unwrap();
                compiler.flush(&mut stream).unwrap();
                let input: Vec<f32> = (0..tokens * width)
                    .map(|i| ((i * 17 % 127) as f32 - 63.) * 4096.)
                    .collect();
                let x = upload(&mut stream, bytemuck::cast_slice(&input));
                let out = stream.allocate_zeroed(tokens * width).unwrap();
                let scales = stream.allocate_zeroed(tokens * 4).unwrap();
                let run = |stream: &mut Stream| {
                    prepare
                        .run(
                            stream,
                            None,
                            "bench",
                            tokens as u32,
                            x.binding(),
                            None,
                            out.binding(),
                            Some(scales.binding()),
                        )
                        .unwrap();
                    stream.synchronize().unwrap();
                };
                run(&mut stream);
                let expected = stream
                    .read(out.binding())
                    .unwrap()
                    .wait(&mut stream)
                    .unwrap();
                let expected_scales = check_f32(&mut stream, &scales);
                b.iter(|| run(&mut stream));
                assert_eq!(
                    stream
                        .read(out.binding())
                        .unwrap()
                        .wait(&mut stream)
                        .unwrap(),
                    expected
                );
                assert_eq!(check_f32(&mut stream, &scales), expected_scales);
            },
        );
    }
    group.finish();
}

fn operand(count: usize, elem: &str, seed: usize) -> Vec<u8> {
    let value = |i: usize| ((i.wrapping_mul(37).wrapping_add(seed) % 15) as i8) - 7;
    if elem == "i8" {
        (0..count).map(|i| value(i) as u8).collect()
    } else {
        (0..count)
            .flat_map(|i| bf16::from_f32(f32::from(value(i)) / 128.).to_le_bytes())
            .collect()
    }
}

fn attention_preparation(c: &mut Criterion) {
    use h3_hrx::dispatch::{emit, Profile};
    use h3_hrx::model::{HEADS, INNER};
    let mut group = c.benchmark_group("prepare_qk_i8");
    for tokens in [1usize, 257, 2048, 8192] {
        for head_major in [false, true] {
            let layout = if head_major {
                "head_major"
            } else {
                "token_major"
            };
            group.throughput(Throughput::Elements((tokens * INNER) as u64));
            group.bench_function(BenchmarkId::new(layout, tokens), |b| {
                let manager = hrx::residency::ResidencyManager::new(512 << 20).unwrap();
                let mut stream = Stream::open().unwrap().with_memory_budget(manager.budget());
                let compiler = compiler();
                let capacity = tokens.div_ceil(32) * 32;
                let stem = if head_major {
                    "prepare_qk_i8hm"
                } else {
                    "prepare_qk_i8"
                };
                let mut cfg = vec![
                    (format!("h3.{stem}.row_stride"), INNER.to_string()),
                    (format!("h3.{stem}.head_offset"), "0".into()),
                    (format!("h3.{stem}.heads"), HEADS.to_string()),
                    (
                        format!("h3.{stem}.extra_scale"),
                        h3_hrx::compile::num(1.0 / 128f64.sqrt() / 128.0),
                    ),
                ];
                if head_major {
                    cfg.push((format!("h3.{stem}.token_capacity"), capacity.to_string()));
                }
                let kernel = compiler
                    .get(&mut stream, stem, &format!("h3_{stem}"), &cfg)
                    .unwrap();
                compiler.flush(&mut stream).unwrap();
                let input: Vec<_> = (0..tokens * INNER)
                    .map(|i| f16::from_f32(((i * 37 % 997) as f32 - 498.0) / 128.0))
                    .collect();
                let x = upload(&mut stream, bytemuck::cast_slice(&input));
                let mean = stream.allocate_zeroed(INNER * 4).unwrap();
                let rows = if head_major { capacity } else { tokens };
                let codes = stream.allocate_zeroed(rows * INNER).unwrap();
                let scales = stream.allocate_zeroed(rows * HEADS * 4).unwrap();
                let bindings = [
                    x.binding(),
                    mean.binding(),
                    codes.binding(),
                    scales.binding(),
                ];
                let required = bindings.map(|view| view.len());
                let mut profile = Profile::from_env();
                let mut run = |stream: &mut Stream| {
                    emit(
                        &mut Sink::Stream(stream),
                        &kernel,
                        Some(&mut profile),
                        "prepare attention operands",
                        [tokens as u32, 1, 1],
                        [256, 1, 1],
                        &[tokens as u32],
                        &bindings,
                        &required,
                    )
                    .unwrap();
                    stream.synchronize().unwrap();
                };
                run(&mut stream);
                let mut expected = vec![0; rows * INNER];
                stream
                    .read_blocking(codes.binding(), &mut expected)
                    .unwrap();
                let expected_scales = check_f32(&mut stream, &scales);
                b.iter(|| run(&mut stream));
                let mut actual = vec![0; expected.len()];
                stream.read_blocking(codes.binding(), &mut actual).unwrap();
                assert_eq!(actual, expected);
                assert_eq!(check_f32(&mut stream, &scales), expected_scales);
            });
        }
    }
    group.finish();
}

fn gemm(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm");
    for elem in ["i8", "bf16"] {
        let mut shapes = vec![("f32", 2048, 2048, 6144), ("swiglu", 2048, 2048, 16384)];
        if elem == "i8" {
            use h3_hrx::model::{FFN, HID, QKV};
            for m in [256, 2048] {
                shapes.extend([
                    ("f32", m, HID, QKV),
                    ("swiglu", m, HID, 2 * FFN),
                    ("f32", m, FFN, HID),
                ]);
            }
        }
        for (mode, m, k, n) in shapes {
            for rotating in [false, true] {
                let storage = if rotating { "rotating" } else { "cached" };
                group.throughput(Throughput::Elements((2 * m * k * n) as u64));
                group.bench_function(format!("{elem}/{mode}/{storage}/{m}x{k}x{n}"), |b| {
                    let manager = hrx::residency::ResidencyManager::new(512 << 20).unwrap();
                    let mut stream = Stream::open().unwrap().with_memory_budget(manager.budget());
                    let compiler = compiler();
                    let stride = h3_hrx::model::gemm_pitch(k, h3_hrx::model::elem_bits(elem));
                    let gemm = Gemm::build_with_output(
                        &compiler,
                        &mut stream,
                        mode,
                        elem,
                        false,
                        true,
                        k,
                        n,
                        m,
                        1,
                        stride,
                        Tile::Plain,
                        0,
                        ActivationType::F32,
                    )
                    .unwrap();
                    compiler.flush(&mut stream).unwrap();
                    let a = upload(&mut stream, &operand(m * stride, elem, 3));
                    let weights = operand(n * stride, elem, 13);
                    // A rotating ring exceeds the 64 MiB cache, even for small matrices.
                    let count = if rotating {
                        (64 * 1024 * 1024 / weights.len() + 1).max(2)
                    } else {
                        1
                    };
                    let ring: Vec<_> = (0..count).map(|_| upload(&mut stream, &weights)).collect();
                    let ws = upload(&mut stream, bytemuck::cast_slice(&vec![0.01f32; n]));
                    let as_ = upload(&mut stream, bytemuck::cast_slice(&vec![0.01f32; m]));
                    let width = if mode == "swiglu" { n / 2 } else { n };
                    let out = stream.allocate_zeroed(m * width * 4).unwrap();
                    let mut profile = h3_hrx::dispatch::Profile::from_env();
                    let mut run = |stream: &mut Stream, index: usize| {
                        gemm.run(
                            stream,
                            Some(&mut profile),
                            "bench",
                            m as u32,
                            a.binding(),
                            ring[index].binding(),
                            (elem == "i8").then(|| (ws.binding(), as_.binding())),
                            out.binding(),
                            None,
                            None,
                        )
                        .unwrap();
                        stream.synchronize().unwrap();
                    };
                    run(&mut stream, 0);
                    let expected = check_f32(&mut stream, &out);
                    let mut index = 0;
                    b.iter(|| {
                        run(&mut stream, index);
                        index = (index + 1) % count;
                    });
                    assert_eq!(check_f32(&mut stream, &out), expected);
                });
            }
        }
    }
    group.finish();
}

fn attention_transpose(c: &mut Criterion) {
    use h3_hrx::dispatch::{emit, Profile};
    use h3_hrx::model::INNER;
    let mut group = c.benchmark_group("transpose_v_f16");
    for tokens in [1usize, 257, 2048, 8192] {
        let capacity = ((tokens + 16).div_ceil(32) * 32).max(tokens.div_ceil(256) * 256);
        group.throughput(Throughput::Bytes(((tokens + capacity) * INNER * 2) as u64));
        group.bench_function(BenchmarkId::from_parameter(tokens), |b| {
            let manager = hrx::residency::ResidencyManager::new(512 << 20).unwrap();
            let mut stream = Stream::open().unwrap().with_memory_budget(manager.budget());
            let compiler = compiler();
            let kernel = compiler
                .get(
                    &mut stream,
                    "transpose_f16",
                    "h3_transpose_f16",
                    &vec![
                        ("h3.transpose_f16.width".into(), INNER.to_string()),
                        ("h3.transpose_f16.row_capacity".into(), capacity.to_string()),
                    ],
                )
                .unwrap();
            compiler.flush(&mut stream).unwrap();
            // Include every half bit pattern: transpose must preserve NaNs and signed zero too.
            let input: Vec<u16> = (0..tokens * INNER)
                .map(|i| i.wrapping_mul(37) as u16)
                .collect();
            let x = upload(&mut stream, bytemuck::cast_slice(&input));
            let out = upload(&mut stream, &vec![0xff; capacity * INNER * 2]);
            let bindings = [x.binding(), out.binding()];
            let required = bindings.map(|view| view.len());
            let mut profile = Profile::from_env();
            let mut run = |stream: &mut Stream| {
                emit(
                    &mut Sink::Stream(stream),
                    &kernel,
                    Some(&mut profile),
                    "transpose V",
                    [(INNER / 32) as u32, (capacity / 32) as u32, 1],
                    [256, 1, 1],
                    &[tokens as u32],
                    &bindings,
                    &required,
                )
                .unwrap();
                stream.synchronize().unwrap();
            };
            run(&mut stream);
            b.iter(|| run(&mut stream));
            let mut actual = vec![0u16; capacity * INNER];
            stream
                .read_blocking(out.binding(), bytemuck::cast_slice_mut(&mut actual))
                .unwrap();
            for (i, value) in actual.into_iter().enumerate() {
                let (column, row) = (i / capacity, i % capacity);
                let expected = if row < tokens {
                    (row * INNER + column).wrapping_mul(37) as u16
                } else {
                    0
                };
                assert_eq!(value, expected, "row={row} column={column}");
            }
        });
    }
    group.finish();
}

fn audio_qkv(c: &mut Criterion) {
    let mut group = c.benchmark_group("audio_qkv_f32");
    let (k, n) = (2048, 6144);
    for m in [1usize, 207] {
        for packed in [false, true] {
            let layout = if packed { "packed" } else { "row_major" };
            group.bench_function(format!("{layout}/{m}x{k}x{n}"), |b| {
                let mut stream = Stream::open().unwrap();
                let compiler = compiler();
                let plain = MatmulF32::build(&compiler, &mut stream, k, n).unwrap();
                let op = if packed {
                    MatmulF32::build_packed(&compiler, &mut stream, k, n).unwrap()
                } else {
                    MatmulF32::build(&compiler, &mut stream, k, n).unwrap()
                };
                compiler.flush(&mut stream).unwrap();
                let values = |len| {
                    (0..len)
                        .map(|i| ((i * 37 % 101) as f32 - 50.) / 200.)
                        .collect::<Vec<_>>()
                };
                let x = upload(&mut stream, bytemuck::cast_slice(&values(m * k)));
                let w = values(n * k);
                let reference = upload(&mut stream, bytemuck::cast_slice(&w));
                let bias = upload(&mut stream, bytemuck::cast_slice(&values(n)));
                let out = stream.allocate(m * n * 4).unwrap();
                plain
                    .run(
                        &mut stream,
                        None,
                        "qkv",
                        m,
                        x.binding(),
                        reference.binding(),
                        bias.binding(),
                        out.binding(),
                    )
                    .unwrap();
                let expected = check_f32(&mut stream, &out);
                let weights = if packed {
                    let mut permuted = Vec::with_capacity(w.len());
                    for tile in w.chunks(32 * k) {
                        for i in 0..k {
                            for col in 0..32 {
                                permuted.push(tile[col * k + i]);
                            }
                        }
                    }
                    upload(&mut stream, bytemuck::cast_slice(&permuted))
                } else {
                    reference
                };
                b.iter(|| {
                    op.run(
                        &mut stream,
                        None,
                        "qkv",
                        m,
                        x.binding(),
                        weights.binding(),
                        bias.binding(),
                        out.binding(),
                    )
                    .unwrap();
                    stream.synchronize().unwrap();
                });
                assert_eq!(check_f32(&mut stream, &out), expected);
            });
        }
    }
    group.finish();
}

fn audio_convolution(c: &mut Criterion) {
    let mut group = c.benchmark_group("audio_encoder_conv_f32");
    // Full-soundtrack encoder shapes, including dilation, stride and partial tiles.
    let levels = [
        (64usize, 165600usize, 2usize),
        (128, 82800, 4),
        (256, 20700, 4),
        (512, 5175, 5),
        (1024, 1035, 5),
    ];
    let residuals = levels.into_iter().flat_map(|(cin, len, stride)| {
        [
            (1, 1, 1),
            (7, 1, 1),
            (7, 3, 1),
            (7, 9, 1),
            (2 * stride, 1, stride),
        ]
        .map(move |(ksize, dilation, step)| (cin, len, ksize, dilation, step))
    });
    let projections = [1, 4, 207, 500].map(|len| (2048, len, 3, 1, 1));
    for (cin, len, ksize, dilation, step) in residuals.chain(projections) {
        let down = step > 1;
        let cout = if down { 2 * cin } else { cin };
        let out_len = len / step;
        let pad = if down {
            step.div_ceil(2)
        } else {
            ksize / 2 * dilation
        };
        for layout in [
            "scalar",
            "four_samples",
            "packed_channels",
            "packed_prefetch",
            "packed_k3",
        ] {
            if down && layout != "scalar" {
                continue; // The tiled kernels require stride one.
            }
            if matches!(layout, "packed_prefetch" | "packed_k3") && ksize != 3 {
                continue; // Compare the projection's short-window dispatch choices.
            }
            group.bench_function(
                format!("{layout}/{cin}x{cout}x{len}/k{ksize}_d{dilation}_s{step}"),
                |b| {
                    let manager = hrx::residency::ResidencyManager::new(1 << 30).unwrap();
                    let mut stream = Stream::open().unwrap().with_memory_budget(manager.budget());
                    let compiler = compiler();
                    let build = |stream: &mut Stream, layout: &str| {
                        let stem = match layout {
                            "packed_channels" => "conv1d_block_f32",
                            "packed_prefetch" => "conv1d_prefetch_f32",
                            "packed_k3" => "conv1d_k3_f32",
                            "four_samples" => "conv1d4_f32",
                            _ => "conv1d_s_f32",
                        };
                        let mut config = vec![
                            ("cin", cin),
                            ("cout", cout),
                            ("ksize", ksize),
                            ("dilation", dilation),
                            ("pad", pad),
                        ];
                        if layout != "scalar" {
                            config.extend([
                                ("accumulate", 0),
                                ("len_bound", len.max(256).next_power_of_two()),
                            ]);
                        } else {
                            config.extend([
                                ("stride", step),
                                ("in_bound", len.max(256).next_power_of_two()),
                                ("out_bound", out_len.max(256).next_power_of_two()),
                            ]);
                        }
                        let config = config
                            .into_iter()
                            .map(|(key, value)| (format!("h3.{stem}.{key}"), value.to_string()))
                            .collect();
                        compiler
                            .get(stream, stem, &format!("h3_{stem}"), &config)
                            .unwrap()
                    };
                    let reference = build(&mut stream, "scalar");
                    let kernel = build(&mut stream, layout);
                    compiler.flush(&mut stream).unwrap();
                    let values = |count| {
                        (0..count)
                            .map(|i| ((i * 37 % 101) as f32 - 50.) / 200.)
                            .collect::<Vec<_>>()
                    };
                    let x = upload(&mut stream, bytemuck::cast_slice(&values(cin * len)));
                    let weights = values(cout * cin * ksize);
                    let w = upload(&mut stream, bytemuck::cast_slice(&weights));
                    let packed = layout.starts_with("packed_").then(|| {
                        let mut packed = Vec::with_capacity(weights.len());
                        for group in weights.chunks(8 * cin * ksize) {
                            for tap in 0..cin * ksize {
                                for channel in 0..8 {
                                    packed.push(group[channel * cin * ksize + tap]);
                                }
                            }
                        }
                        upload(&mut stream, bytemuck::cast_slice(&packed))
                    });
                    let bias = upload(&mut stream, bytemuck::cast_slice(&values(cout)));
                    let out = stream.allocate(cout * out_len * 4).unwrap();
                    let run = |stream: &mut Stream,
                               kernel: &h3_hrx::compile::Kernel,
                               layout: &str,
                               w: &Buffer| {
                        let tiled = layout != "scalar";
                        let (span, outputs) = match layout {
                            "packed_channels" => (128, cout / 8),
                            "packed_prefetch" => (64, cout / 8),
                            "packed_k3" => (8, cout / 8),
                            _ => (256, cout),
                        };
                        let scalars = [out_len as u32, len as u32];
                        h3_hrx::dispatch::emit(
                            &mut Sink::Stream(stream),
                            kernel,
                            None,
                            "audio convolution",
                            [out_len.div_ceil(span) as u32, outputs as u32, 1],
                            [if tiled { 64 } else { 256 }, 1, 1],
                            &scalars[..if tiled { 1 } else { 2 }],
                            &[x.binding(), w.binding(), bias.binding(), out.binding()],
                            &[
                                cin * len * 4,
                                cout * cin * ksize * 4,
                                cout * 4,
                                cout * out_len * 4,
                            ],
                        )
                        .unwrap();
                        stream.synchronize().unwrap();
                    };
                    run(&mut stream, &reference, "scalar", &w);
                    let expected = check_f32(&mut stream, &out);
                    let w = packed.as_ref().unwrap_or(&w);
                    run(&mut stream, &kernel, layout, w);
                    assert_eq!(check_f32(&mut stream, &out), expected);
                    b.iter(|| run(&mut stream, &kernel, layout, w));
                    assert_eq!(check_f32(&mut stream, &out), expected);
                },
            );
        }
    }
    group.finish();
}

fn dispatch(c: &mut Criterion) {
    const LAUNCHES: usize = 256;
    let mut group = c.benchmark_group("dispatch_256");
    group.throughput(Throughput::Elements(LAUNCHES as u64));
    for (tokens, width) in [(1usize, 256usize), (1024, 4096)] {
        for graph_mode in [false, true] {
            let mode = if graph_mode { "graph" } else { "eager" };
            group.bench_function(BenchmarkId::new(mode, format!("{tokens}x{width}")), |b| {
                let mut stream = Stream::open().unwrap();
                let compiler = compiler();
                let prepare = Prepare::build(
                    &compiler,
                    &mut stream,
                    "plain",
                    "f16",
                    width,
                    1e-5,
                    1,
                    width,
                )
                .unwrap();
                compiler.flush(&mut stream).unwrap();
                let input = vec![half::f16::from_f32(0.5); tokens * width];
                let x = upload(&mut stream, bytemuck::cast_slice(&input));
                let out = stream.allocate_zeroed(input.len() * 2).unwrap();
                let mut graph = if graph_mode {
                    let mut recording = stream.graph().unwrap();
                    let mut sink = Sink::Graph {
                        graph: &mut recording,
                        after: Default::default(),
                    };
                    for _ in 0..LAUNCHES {
                        prepare
                            .emit(
                                &mut sink,
                                None,
                                "bench",
                                tokens as u32,
                                x.binding(),
                                None,
                                out.binding(),
                                None,
                            )
                            .unwrap();
                    }
                    Some(recording.finish().unwrap())
                } else {
                    None
                };
                let mut run = |stream: &mut Stream| {
                    if let Some(graph) = &mut graph {
                        stream.launch(graph).unwrap();
                    } else {
                        for _ in 0..LAUNCHES {
                            prepare
                                .run(
                                    stream,
                                    None,
                                    "bench",
                                    tokens as u32,
                                    x.binding(),
                                    None,
                                    out.binding(),
                                    None,
                                )
                                .unwrap();
                        }
                    }
                    stream.synchronize().unwrap();
                };
                run(&mut stream);
                b.iter(|| run(&mut stream));
                let actual = stream
                    .read(out.binding())
                    .unwrap()
                    .wait(&mut stream)
                    .unwrap();
                assert_eq!(actual, bytemuck::cast_slice::<_, u8>(&input));
            });
        }
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10).warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3));
    targets = preparation, attention_preparation, attention_transpose, gemm, audio_qkv, audio_convolution, dispatch
}
criterion_main!(benches);

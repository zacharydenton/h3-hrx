//! Resident GPU workloads. Compilation, allocation and readback are outside timing.
mod support;
#[path = "kernels/vision.rs"]
mod vision;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use h3_hrx::dispatch::{ActivationType, Gemm, MatmulF32, Prepare, Sink, Tile};
use half::{bf16, f16};
use hrx::{Buffer, Stream};
use std::time::Duration;

use support::compiler;

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

fn check_f16(stream: &mut Stream, buffer: &Buffer) -> Vec<u8> {
    let mut bytes = vec![0; buffer.binding().len()];
    stream.read_blocking(buffer.binding(), &mut bytes).unwrap();
    assert!(bytes
        .as_chunks::<2>()
        .0
        .iter()
        .all(|v| f16::from_le_bytes(*v).is_finite()));
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
        (4096, 14336),
        (8193, 14336),
    ] {
        group.throughput(Throughput::Elements((tokens * width) as u64));
        group.bench_function(
            BenchmarkId::new("plain", format!("{tokens}x{width}")),
            |b| {
                let budget = if tokens > 4096 { 1 << 30 } else { 512 << 20 };
                let manager = hrx::residency::ResidencyManager::new(budget).unwrap();
                let mut stream = support::stream(Some(manager.budget()));
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
                let mut profile = h3_hrx::dispatch::Profile::from_env();
                let mut run = |stream: &mut Stream| {
                    prepare
                        .run(
                            stream,
                            Some(&mut profile),
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
                let mut expected = vec![0; tokens * width];
                stream.read_blocking(out.binding(), &mut expected).unwrap();
                let expected_scales = check_f32(&mut stream, &scales);
                b.iter(|| run(&mut stream));
                let mut actual = vec![0; expected.len()];
                stream.read_blocking(out.binding(), &mut actual).unwrap();
                assert_eq!(actual, expected);
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

fn vision_rotary(c: &mut Criterion) {
    use h3_hrx::dispatch::{emit, Profile};
    use h3_hrx::model::{VHD, VHDP, VHEADS};
    let mut group = c.benchmark_group("vision_rotary");
    for tokens in [1usize, 16, 257, 1620, 4032, 8193] {
        group.throughput(Throughput::Elements((tokens * 3 * VHEADS * VHD) as u64));
        group.bench_function(BenchmarkId::from_parameter(tokens), |b| {
            let manager = hrx::residency::ResidencyManager::new(512 << 20).unwrap();
            let mut stream = support::stream(Some(manager.budget()));
            let compiler = compiler();
            let kernel = compiler
                .get(
                    &mut stream,
                    "rope2d_qkv_f16",
                    "h3_rope2d_qkv_f16",
                    &vec![
                        ("h3.rope2d_qkv_f16.heads".into(), VHEADS.to_string()),
                        ("h3.rope2d_qkv_f16.hd".into(), VHD.to_string()),
                        ("h3.rope2d_qkv_f16.hd_pad".into(), VHDP.to_string()),
                    ],
                )
                .unwrap();
            compiler.flush(&mut stream).unwrap();
            let input = support::values(tokens * 3 * VHEADS * VHD, 2.);
            let angles = support::values(tokens * VHD / 2, 3.);
            let cos: Vec<_> = angles.iter().map(|v| v.cos()).collect();
            let sin: Vec<_> = angles.iter().map(|v| v.sin()).collect();
            let input = upload(&mut stream, bytemuck::cast_slice(&input));
            let cos = upload(&mut stream, bytemuck::cast_slice(&cos));
            let sin = upload(&mut stream, bytemuck::cast_slice(&sin));
            let outputs: Vec<_> = (0..3)
                .map(|_| stream.allocate(tokens * VHEADS * VHDP * 2).unwrap())
                .collect();
            let bindings = [
                input.binding(),
                cos.binding(),
                sin.binding(),
                outputs[0].binding(),
                outputs[1].binding(),
                outputs[2].binding(),
            ];
            let required = bindings.map(|v| v.len());
            let mut profile = Profile::from_env();
            let mut run = |stream: &mut Stream| {
                emit(
                    &mut Sink::Stream(stream),
                    &kernel,
                    Some(&mut profile),
                    "vision rope",
                    &[tokens as u32],
                    &[tokens as u32],
                    &bindings,
                    &required,
                )
                .unwrap();
                stream.synchronize().unwrap();
            };
            run(&mut stream);
            let expected: Vec<_> = outputs.iter().map(|v| check_f16(&mut stream, v)).collect();
            b.iter(|| run(&mut stream));
            for (output, expected) in outputs.iter().zip(expected) {
                assert_eq!(support::read(&mut stream, output.binding()), expected);
            }
        });
    }
    group.finish();
}

fn rotary_preparation(c: &mut Criterion) {
    use h3_hrx::dispatch::{emit, Profile};
    use h3_hrx::model::{HEADS, INNER, QKV, ROPE_HALF};
    let mut group = c.benchmark_group("rope_qknorm_f16");
    for tokens in [1usize, 257, 2048, 4096, 8193] {
        for copy_v in [true, false] {
            group.throughput(Throughput::Elements(
                (tokens * INNER * if copy_v { 3 } else { 2 }) as u64,
            ));
            group.bench_function(
                BenchmarkId::new(if copy_v { "qkv" } else { "qk" }, tokens),
                |b| {
                    // Long QKV inputs and the three FP16 outputs exceed 512 MiB.
                    let budget = if tokens > 4096 { 1 << 30 } else { 512 << 20 };
                    let manager = hrx::residency::ResidencyManager::new(budget).unwrap();
                    let mut stream = support::stream(Some(manager.budget()));
                    let compiler = compiler();
                    let kernel = compiler
                        .get(
                            &mut stream,
                            "rope_qknorm_f16",
                            "h3_rope_qknorm_f16",
                            &vec![
                                ("h3.rope_qknorm_f16.row_stride".into(), QKV.to_string()),
                                ("h3.rope_qknorm_f16.heads".into(), HEADS.to_string()),
                                ("h3.rope_qknorm_f16.kv_heads".into(), HEADS.to_string()),
                                ("h3.rope_qknorm_f16.k_offset".into(), INNER.to_string()),
                                ("h3.rope_qknorm_f16.eps".into(), "1e-5".into()),
                                (
                                    "h3.rope_qknorm_f16.copy_v".into(),
                                    usize::from(copy_v).to_string(),
                                ),
                            ],
                        )
                        .unwrap();
                    compiler.flush(&mut stream).unwrap();
                    let input: Vec<_> = (0..tokens * QKV)
                        .map(|i| f16::from_f32(((i * 37 % 997) as f32 - 498.) / 512.))
                        .collect();
                    let weight: Vec<_> = (0..128).map(|i| 1. + (i % 7) as f32 / 100.).collect();
                    let angles: Vec<_> = (0..tokens * ROPE_HALF)
                        .map(|i| (i % 271) as f32 / 271.)
                        .collect();
                    let cos: Vec<_> = angles.iter().map(|v| v.cos()).collect();
                    let sin: Vec<_> = angles.iter().map(|v| v.sin()).collect();
                    let x = upload(&mut stream, bytemuck::cast_slice(&input));
                    let weight = upload(&mut stream, bytemuck::cast_slice(&weight));
                    let cos = upload(&mut stream, bytemuck::cast_slice(&cos));
                    let sin = upload(&mut stream, bytemuck::cast_slice(&sin));
                    let output: Vec<_> = (0..3)
                        .map(|_| stream.allocate_zeroed(tokens * INNER * 2).unwrap())
                        .collect();
                    let bindings = [
                        x.binding(),
                        weight.binding(),
                        weight.binding(),
                        cos.binding(),
                        sin.binding(),
                        output[0].binding(),
                        output[1].binding(),
                        output[2].binding(),
                    ];
                    let required = bindings.map(|v| v.len());
                    let mut profile = Profile::from_env();
                    let mut run = |stream: &mut Stream| {
                        emit(
                            &mut Sink::Stream(stream),
                            &kernel,
                            Some(&mut profile),
                            "qk norm + rope",
                            &[tokens as u32],
                            &[tokens as u32],
                            &bindings,
                            &required,
                        )
                        .unwrap();
                        stream.synchronize().unwrap();
                    };
                    run(&mut stream);
                    let expected: Vec<_> = output
                        .iter()
                        .map(|v| support::read(&mut stream, v.binding()))
                        .collect();
                    assert!(expected.iter().all(|v| v
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .all(|&v| f16::from_le_bytes(v).is_finite())));
                    b.iter(|| run(&mut stream));
                    for (output, expected) in output.iter().zip(&expected) {
                        let actual = support::read(&mut stream, output.binding());
                        assert_eq!(&actual, expected);
                    }
                },
            );
        }
    }
    group.finish();
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
                let mut stream = support::stream(Some(manager.budget()));
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
                        &[tokens as u32],
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

fn fused_qk_preparation(c: &mut Criterion) {
    use h3_hrx::dispatch::{emit, Profile};
    use h3_hrx::model::{HEADS, INNER, QKV, ROPE_HALF};
    let mut group = c.benchmark_group("qk_rotary_quantization");
    for tokens in [1usize, 257, 4096, 8192, 8193] {
        for fused in [false, true] {
            group.throughput(Throughput::Elements((tokens * INNER * 2) as u64));
            group.bench_function(
                BenchmarkId::new(if fused { "fused" } else { "separate" }, tokens),
                |b| {
                    // The reference needs both full-width FP16 intermediates.
                    let manager = hrx::residency::ResidencyManager::new(1 << 30).unwrap();
                    let mut stream = support::stream(Some(manager.budget()));
                    let compiler = compiler();
                    let capacity = (tokens + 16).div_ceil(32) * 32;
                    let rope = compiler
                        .get(
                            &mut stream,
                            "rope_qknorm_f16",
                            "h3_rope_qknorm_f16",
                            &vec![
                                ("h3.rope_qknorm_f16.row_stride".into(), QKV.to_string()),
                                ("h3.rope_qknorm_f16.heads".into(), HEADS.to_string()),
                                ("h3.rope_qknorm_f16.kv_heads".into(), HEADS.to_string()),
                                ("h3.rope_qknorm_f16.k_offset".into(), INNER.to_string()),
                                ("h3.rope_qknorm_f16.eps".into(), "1e-5".into()),
                                ("h3.rope_qknorm_f16.copy_v".into(), "0".into()),
                            ],
                        )
                        .unwrap();
                    let prepare = |stream: &mut Stream, fused: bool| {
                        let stem = if fused {
                            "prepare_qk_rope_i8hm"
                        } else {
                            "prepare_qk_i8hm"
                        };
                        [0, 1].map(|head| {
                            let mut cfg = vec![
                                (
                                    format!("h3.{stem}.row_stride"),
                                    if fused { QKV } else { INNER }.to_string(),
                                ),
                                (
                                    format!("h3.{stem}.head_offset"),
                                    if fused { head * INNER } else { 0 }.to_string(),
                                ),
                                (format!("h3.{stem}.heads"), HEADS.to_string()),
                                (format!("h3.{stem}.token_capacity"), capacity.to_string()),
                                (
                                    format!("h3.{stem}.extra_scale"),
                                    h3_hrx::compile::num(if head == 0 {
                                        1. / 128f64.sqrt() / 128.
                                    } else {
                                        1.
                                    }),
                                ),
                            ];
                            if fused {
                                cfg.push((format!("h3.{stem}.eps"), "1e-5".into()));
                            }
                            compiler
                                .get(stream, stem, &format!("h3_{stem}"), &cfg)
                                .unwrap()
                        })
                    };
                    let separate = prepare(&mut stream, false);
                    let combined = prepare(&mut stream, true);
                    compiler.flush(&mut stream).unwrap();
                    let input: Vec<_> = (0..tokens * QKV)
                        .map(|i| f16::from_f32(((i * 37 % 997) as f32 - 498.) / 512.))
                        .collect();
                    let x = upload(&mut stream, bytemuck::cast_slice(&input));
                    let weights: Vec<_> = [0, 1]
                        .into_iter()
                        .map(|head| {
                            let weight: Vec<_> = (0..128)
                                .map(|i| 1. + (i % 7 + head) as f32 / 100.)
                                .collect();
                            upload(&mut stream, bytemuck::cast_slice(&weight))
                        })
                        .collect();
                    let angles: Vec<_> = (0..tokens * ROPE_HALF)
                        .map(|i| (i % 271) as f32 / 271.)
                        .collect();
                    let cos: Vec<_> = angles.iter().map(|v| v.cos()).collect();
                    let sin: Vec<_> = angles.iter().map(|v| v.sin()).collect();
                    let cos = upload(&mut stream, bytemuck::cast_slice(&cos));
                    let sin = upload(&mut stream, bytemuck::cast_slice(&sin));
                    let qk: Vec<_> = (0..2)
                        .map(|_| stream.allocate(tokens * INNER * 2).unwrap())
                        .collect();
                    let codes: Vec<_> = (0..2)
                        .map(|_| stream.allocate_zeroed(capacity * INNER).unwrap())
                        .collect();
                    let scales: Vec<_> = (0..2)
                        .map(|_| stream.allocate_zeroed(capacity * HEADS * 4).unwrap())
                        .collect();
                    let mean = stream.allocate_zeroed(INNER * 4).unwrap();
                    let mut profile = Profile::from_env();
                    let mut run = |stream: &mut Stream, fused: bool| {
                        if !fused {
                            let views = [
                                x.binding(),
                                weights[0].binding(),
                                weights[1].binding(),
                                cos.binding(),
                                sin.binding(),
                                qk[0].binding(),
                                qk[1].binding(),
                                x.binding(),
                            ];
                            emit(
                                &mut Sink::Stream(stream),
                                &rope,
                                Some(&mut profile),
                                "qk norm + rope",
                                &[tokens as u32],
                                &[tokens as u32],
                                &views,
                                &views.map(|v| v.len()),
                            )
                            .unwrap();
                        }
                        for head in 0..2 {
                            if fused {
                                let views = [
                                    x.binding(),
                                    weights[head].binding(),
                                    cos.binding(),
                                    sin.binding(),
                                    codes[head].binding(),
                                    scales[head].binding(),
                                ];
                                emit(
                                    &mut Sink::Stream(stream),
                                    &combined[head],
                                    Some(&mut profile),
                                    "fused QK preparation",
                                    &[tokens as u32],
                                    &[tokens as u32],
                                    &views,
                                    &views.map(|v| v.len()),
                                )
                                .unwrap();
                            } else {
                                let views = [
                                    qk[head].binding(),
                                    mean.binding(),
                                    codes[head].binding(),
                                    scales[head].binding(),
                                ];
                                emit(
                                    &mut Sink::Stream(stream),
                                    &separate[head],
                                    Some(&mut profile),
                                    "separate QK preparation",
                                    &[tokens as u32],
                                    &[tokens as u32],
                                    &views,
                                    &views.map(|v| v.len()),
                                )
                                .unwrap();
                            }
                        }
                        stream.synchronize().unwrap();
                    };
                    run(&mut stream, false);
                    let expected: Vec<_> = codes
                        .iter()
                        .chain(&scales)
                        .map(|v| support::read(&mut stream, v.binding()))
                        .collect();
                    run(&mut stream, fused);
                    b.iter(|| run(&mut stream, fused));
                    for (buffer, expected) in codes.iter().chain(&scales).zip(expected) {
                        let actual = support::read(&mut stream, buffer.binding());
                        assert!(
                            actual == expected,
                            "Q/K codes or scales differ: tokens={tokens}, fused={fused}"
                        );
                    }
                },
            );
        }
    }
    group.finish();
}

fn quantized_attention(c: &mut Criterion) {
    use h3_hrx::dispatch::{emit, Profile};
    use h3_hrx::model::{HEADS, INNER};
    let mut group = c.benchmark_group("attention_i8qkhm");
    for tokens in [
        1usize, 129, 256, 257, 2048, 4095, 4096, 4097, 8192, 8193, 15666, 16384, 16385, 32768,
        37977,
    ] {
        group.throughput(Throughput::Elements((4 * tokens * tokens * INNER) as u64));
        group.bench_function(BenchmarkId::from_parameter(tokens), |b| {
            // Long attention inputs and outputs exceed 512 MiB; keep each case bounded.
            let budget = if tokens > 16384 {
                2 << 30
            } else if tokens > 8192 {
                1 << 30
            } else {
                512 << 20
            };
            let manager = hrx::residency::ResidencyManager::new(budget).unwrap();
            let mut stream = support::stream(Some(manager.budget()));
            let compiler = compiler();
            let capacity = (tokens + 16).div_ceil(128) * 128;
            let stem = "attention_i8qkhm_mha8_k64_lds_f16_wmma";
            let cfg = [
                ("q_stride", INNER.to_string()),
                ("kv_stride", INNER.to_string()),
                ("out_stride", INNER.to_string()),
                ("tokens", tokens.to_string()),
                ("token_capacity", capacity.to_string()),
                ("scale", "1.0".into()),
            ]
            .into_iter()
            .map(|(key, value)| (format!("h3.{stem}.{key}"), value))
            .collect();
            let kernel = compiler
                .get(&mut stream, stem, &format!("h3_{stem}"), &cfg)
                .unwrap();
            compiler.flush(&mut stream).unwrap();
            let q = upload(&mut stream, &operand(capacity * INNER, "i8", 3));
            let k = upload(&mut stream, &operand(capacity * INNER, "i8", 7));
            // Distinct head and token scales exercise attention operand addressing.
            let scales = |period: usize| {
                (0..capacity * HEADS)
                    .map(|i| (1 + (i % capacity + 3 * (i / capacity)) % period) as f32 / 512.0)
                    .collect::<Vec<_>>()
            };
            let q_scales = upload(&mut stream, bytemuck::cast_slice(&scales(7)));
            let k_scales = upload(&mut stream, bytemuck::cast_slice(&scales(11)));
            // V is channel-major, matching the runtime's transposed operand.
            let values: Vec<_> = (0..capacity * INNER)
                .map(|i| f16::from_f32(((i * 37 % 997) as f32 - 498.0) / 512.0))
                .collect();
            let v = upload(&mut stream, bytemuck::cast_slice(&values));
            let out = stream.allocate_zeroed(tokens * INNER * 2).unwrap();
            let bindings = [
                q.binding(),
                q_scales.binding(),
                k.binding(),
                k_scales.binding(),
                v.binding(),
                out.binding(),
            ];
            let required = bindings.map(|view| view.len());
            let mut profile = Profile::from_env();
            let mut run = |stream: &mut Stream| {
                emit(
                    &mut Sink::Stream(stream),
                    &kernel,
                    Some(&mut profile),
                    "attention",
                    &[tokens as u32, HEADS as u32],
                    &[tokens as u32],
                    &bindings,
                    &required,
                )
                .unwrap();
                stream.synchronize().unwrap();
            };
            run(&mut stream);
            let mut expected = vec![0; tokens * INNER * 2];
            stream.read_blocking(out.binding(), &mut expected).unwrap();
            assert!(expected
                .as_chunks::<2>()
                .0
                .iter()
                .all(|&v| f16::from_le_bytes(v).is_finite()));
            assert!(expected.iter().any(|&v| v != 0), "empty attention output");
            support::report_digest(&format!("attention_i8qkhm/{tokens}"), &expected);
            b.iter(|| run(&mut stream));
            let mut actual = vec![0; expected.len()];
            stream.read_blocking(out.binding(), &mut actual).unwrap();
            assert_eq!(actual, expected);
        });
    }
    group.finish();
}

fn attention_output_preparation(c: &mut Criterion) {
    use h3_hrx::dispatch::{emit, Profile};
    use h3_hrx::model::{gemm_pitch, INNER};
    let mut group = c.benchmark_group("prepare_attention_output_i8");
    for tokens in [1usize, 256, 2048, 8192, 8193] {
        for lanes in [128usize, 224, 448] {
            group.throughput(Throughput::Elements((tokens * INNER) as u64));
            group.bench_function(format!("{tokens}/lanes_{lanes}"), |b| {
                let manager = hrx::residency::ResidencyManager::new(512 << 20).unwrap();
                let mut stream = support::stream(Some(manager.budget()));
                let compiler = compiler();
                let stride = gemm_pitch(INNER, 8);
                let build = |stream: &mut Stream, lanes: usize| {
                    compiler
                        .get(
                            stream,
                            "prepare_i8_family",
                            "h3_prepare_plain_i8",
                            &vec![
                                ("h3.prepare_plain_i8.width".into(), INNER.to_string()),
                                ("h3.prepare_plain_i8.lanes".into(), lanes.to_string()),
                                ("h3.prepare_plain_i8.out_stride".into(), stride.to_string()),
                            ],
                        )
                        .unwrap()
                };
                let reference = build(&mut stream, 128);
                let kernel = build(&mut stream, lanes);
                compiler.flush(&mut stream).unwrap();
                let input: Vec<_> = (0..tokens * INNER)
                    .map(|i| f16::from_f32(((i * 17 % 127) as f32 - 63.0) * 0.25))
                    .collect();
                let x = upload(&mut stream, bytemuck::cast_slice(&input));
                let codes = upload(&mut stream, &vec![0x55; tokens * stride]);
                let scales = stream.allocate_zeroed(tokens * 4).unwrap();
                let bindings = [x.binding(), codes.binding(), scales.binding()];
                let required = bindings.map(|view| view.len());
                emit(
                    &mut Sink::Stream(&mut stream),
                    &reference,
                    None,
                    "reference",
                    &[tokens as u32],
                    &[tokens as u32],
                    &bindings,
                    &required,
                )
                .unwrap();
                let mut expected = vec![0; tokens * stride];
                stream
                    .read_blocking(codes.binding(), &mut expected)
                    .unwrap();
                let expected_scales = check_f32(&mut stream, &scales);
                let mut profile = Profile::from_env();
                let mut run = |stream: &mut Stream| {
                    emit(
                        &mut Sink::Stream(stream),
                        &kernel,
                        Some(&mut profile),
                        "prepare out input",
                        &[tokens as u32],
                        &[tokens as u32],
                        &bindings,
                        &required,
                    )
                    .unwrap();
                    stream.synchronize().unwrap();
                };
                run(&mut stream);
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

fn normalization_preparation(c: &mut Criterion) {
    use h3_hrx::dispatch::{emit, Profile};
    use h3_hrx::model::{gemm_pitch, CLASSES, HID};
    let mut group = c.benchmark_group("prepare_norm_i8");
    for tokens in [1usize, 256, 2048, 8192, 8193] {
        for lanes in [96usize, 224, 672] {
            group.throughput(Throughput::Elements((tokens * HID) as u64));
            group.bench_function(format!("{tokens}/lanes_{lanes}"), |b| {
                let manager = hrx::residency::ResidencyManager::new(512 << 20).unwrap();
                let mut stream = support::stream(Some(manager.budget()));
                let compiler = compiler();
                let stride = gemm_pitch(HID, 8);
                let build = |stream: &mut Stream, lanes: usize| {
                    compiler
                        .get(
                            stream,
                            "prepare_i8_family",
                            "h3_prepare_norm_i8",
                            &vec![
                                ("h3.prepare_norm_i8.width".into(), HID.to_string()),
                                ("h3.prepare_norm_i8.lanes".into(), lanes.to_string()),
                                ("h3.prepare_norm_i8.out_stride".into(), stride.to_string()),
                                ("h3.prepare_norm_i8.classes".into(), CLASSES.to_string()),
                                ("h3.prepare_norm_i8.eps".into(), "0.00001".into()),
                            ],
                        )
                        .unwrap()
                };
                let reference = build(&mut stream, 96);
                let kernel = build(&mut stream, lanes);
                compiler.flush(&mut stream).unwrap();
                // Exact binary fractions keep the square sum independent of reduction grouping.
                let input: Vec<f32> = (0..tokens * HID)
                    .map(|i| ((i * 17 % 127) as f32 - 63.) / 64.)
                    .collect();
                let weight: Vec<f32> = (0..HID).map(|i| 1. + (i % 7) as f32 / 128.).collect();
                let table: Vec<f32> = (0..CLASSES * 2 * HID)
                    .map(|i| (i % 13) as f32 / 128.)
                    .collect();
                let classes: Vec<i32> = (0..tokens).map(|i| (i % CLASSES) as i32).collect();
                let x = upload(&mut stream, bytemuck::cast_slice(&input));
                let weight = upload(&mut stream, bytemuck::cast_slice(&weight));
                let table = upload(&mut stream, bytemuck::cast_slice(&table));
                let classes = upload(&mut stream, bytemuck::cast_slice(&classes));
                let codes = upload(&mut stream, &vec![0x55; tokens * stride]);
                let scales = stream.allocate_zeroed(tokens * 4).unwrap();
                let bindings = [
                    x.binding(),
                    weight.binding(),
                    table.binding(),
                    classes.binding(),
                    codes.binding(),
                    scales.binding(),
                ];
                let required = bindings.map(|view| view.len());
                emit(
                    &mut Sink::Stream(&mut stream),
                    &reference,
                    None,
                    "reference",
                    &[tokens as u32],
                    &[tokens as u32],
                    &bindings,
                    &required,
                )
                .unwrap();
                let mut expected = vec![0; tokens * stride];
                stream
                    .read_blocking(codes.binding(), &mut expected)
                    .unwrap();
                let expected_scales = check_f32(&mut stream, &scales);
                let mut profile = Profile::from_env();
                let mut run = |stream: &mut Stream| {
                    emit(
                        &mut Sink::Stream(stream),
                        &kernel,
                        Some(&mut profile),
                        "prepare norm",
                        &[tokens as u32],
                        &[tokens as u32],
                        &bindings,
                        &required,
                    )
                    .unwrap();
                    stream.synchronize().unwrap();
                };
                let check = |stream: &mut Stream| {
                    let mut actual = vec![0; expected.len()];
                    stream.read_blocking(codes.binding(), &mut actual).unwrap();
                    assert_eq!(actual, expected);
                    assert_eq!(check_f32(stream, &scales), expected_scales);
                };
                run(&mut stream);
                check(&mut stream);
                b.iter(|| run(&mut stream));
                check(&mut stream);
            });
        }
    }
    group.finish();
}

fn decoder_feed_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("decoder_feed_forward");
    let (k, n) = (2048, 16384);
    let stride = h3_hrx::model::gemm_pitch(k, 16);
    let output_stride = h3_hrx::model::gemm_pitch(n / 2, 16);
    for m in [117usize, 773, 1797, 2049] {
        for rotating in [false, true] {
            let storage = if rotating { "rotating" } else { "cached" };
            group.bench_function(format!("{storage}/{m}"), |b| {
                let manager = hrx::residency::ResidencyManager::new(512 << 20).unwrap();
                let mut stream = support::stream(Some(manager.budget()));
                let compiler = compiler();
                let op = Gemm::build(
                    &compiler,
                    &mut stream,
                    "swiglu",
                    "f16",
                    true,
                    false,
                    k,
                    n,
                    m,
                    1,
                    stride,
                    Tile::Fast,
                    output_stride,
                )
                .unwrap();
                compiler.flush(&mut stream).unwrap();
                let values = |count, seed| {
                    (0..count)
                        .flat_map(|i| {
                            f16::from_f32(((i * 37 + seed) % 127) as f32 / 1024. - 0.0625)
                                .to_le_bytes()
                        })
                        .collect::<Vec<_>>()
                };
                let a = upload(&mut stream, &values(m * stride, 3));
                let w = values(n * stride, 17);
                let ring: Vec<_> = (0..if rotating { 2 } else { 1 })
                    .map(|_| upload(&mut stream, &w))
                    .collect();
                let bias = upload(&mut stream, bytemuck::cast_slice(&vec![0.01f32; n]));
                let out = stream.allocate_zeroed(m * output_stride * 2).unwrap();
                let run = |stream: &mut Stream, index: usize| {
                    op.run(
                        stream,
                        None,
                        "decoder ff",
                        m as u32,
                        a.binding(),
                        ring[index].binding(),
                        None,
                        out.binding(),
                        None,
                        Some(bias.binding()),
                    )
                    .unwrap();
                    stream.synchronize().unwrap();
                };
                run(&mut stream, 0);
                let expected = check_f16(&mut stream, &out);
                let mut index = 0;
                b.iter(|| {
                    run(&mut stream, index);
                    index = (index + 1) % ring.len();
                });
                assert_eq!(check_f16(&mut stream, &out), expected);
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
            use h3_hrx::model::{FFN, HID, INNER, QKV};
            for m in [256, 2048] {
                shapes.extend([
                    ("f32", m, HID, QKV),
                    ("swiglu", m, HID, 2 * FFN),
                    ("f32", m, FFN, HID),
                    ("resid", m, FFN, HID),
                ]);
            }
            shapes.push(("resid", 4096, FFN, HID));
            shapes.extend([
                ("swiglu", 4096, HID, 2 * FFN),
                ("swiglu", 8192, HID, 2 * FFN),
            ]);
            // Actual FP16 QKV outputs, including raster groups with empty row tiles.
            for m in [1024, 1025, 4096, 4097] {
                shapes.push(("plain", m, HID, QKV));
            }
            for m in [1025, 4097] {
                shapes.extend([("swiglu", m, HID, 2 * FFN), ("resid", m, FFN, HID)]);
            }
            // Full-clip activation traffic for all four DiT projections.
            for m in [15666, 37977] {
                shapes.extend([
                    ("plain", m, HID, QKV),
                    ("resid", m, INNER, HID),
                    ("swiglu", m, HID, 2 * FFN),
                    ("resid", m, FFN, HID),
                ]);
            }
        }
        for (mode, m, k, n) in shapes {
            for rotating in [false, true] {
                let storage = if rotating { "rotating" } else { "cached" };
                group.throughput(Throughput::Elements((2 * m * k * n) as u64));
                group.bench_function(format!("{elem}/{mode}/{storage}/{m}x{k}x{n}"), |b| {
                    // Long FP32 SwiGLU outputs plus the rotating weights exceed 512 MiB.
                    let budget = if m >= 15666 {
                        4 << 30
                    } else if mode == "swiglu" && m >= 4096 {
                        1 << 30
                    } else {
                        512 << 20
                    };
                    let manager = hrx::residency::ResidencyManager::new(budget).unwrap();
                    let mut stream = support::stream(Some(manager.budget()));
                    let compiler = compiler();
                    let stride = h3_hrx::model::gemm_pitch(k, h3_hrx::model::elem_bits(elem));
                    let output_type = if mode == "plain" {
                        ActivationType::F16
                    } else {
                        ActivationType::F32
                    };
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
                        output_type,
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
                    let out = stream
                        .allocate_zeroed(m * width * if mode == "plain" { 2 } else { 4 })
                        .unwrap();
                    let check = if mode == "plain" {
                        check_f16
                    } else {
                        check_f32
                    };
                    let residual = (mode == "resid").then(|| {
                        let gate = upload(&mut stream, bytemuck::cast_slice(&vec![0.75f32; n]));
                        let classes = h3_hrx::dispatch::Classes::zeroed(&mut stream, m).unwrap();
                        (gate, classes)
                    });
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
                            residual
                                .as_ref()
                                .map(|(gate, classes)| (gate.binding(), classes.all())),
                            None,
                        )
                        .unwrap();
                        stream.synchronize().unwrap();
                    };
                    run(&mut stream, 0);
                    let expected = check(&mut stream, &out);
                    let mut index = 0;
                    if mode == "resid" {
                        b.iter_custom(|iterations| {
                            let mut elapsed = Duration::ZERO;
                            for _ in 0..iterations {
                                // Restore the residual outside timing, as in the block benchmark.
                                stream.fill(out.binding(), 0).unwrap();
                                stream.synchronize().unwrap();
                                let start = std::time::Instant::now();
                                run(&mut stream, index);
                                elapsed += start.elapsed();
                                index = (index + 1) % count;
                            }
                            elapsed
                        });
                    } else {
                        b.iter(|| {
                            run(&mut stream, index);
                            index = (index + 1) % count;
                        });
                    }
                    assert!(
                        check(&mut stream, &out) == expected,
                        "GEMM replay changed: {elem}/{mode}/{storage}/{m}x{k}x{n}"
                    );
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
    for tokens in [1usize, 257, 2048, 8192, 8193] {
        for direct in [false, true] {
            let capacity = ((tokens + 16).div_ceil(32) * 32).max(tokens.div_ceil(256) * 256);
            group.throughput(Throughput::Bytes(((tokens + capacity) * INNER * 2) as u64));
            group.bench_function(
                BenchmarkId::new(if direct { "qkv" } else { "v" }, tokens),
                |b| {
                    let manager = hrx::residency::ResidencyManager::new(512 << 20).unwrap();
                    let mut stream = support::stream(Some(manager.budget()));
                    let compiler = compiler();
                    let stem = if direct {
                        "transpose_qkv_v_f16"
                    } else {
                        "transpose_f16"
                    };
                    let kernel = compiler
                        .get(
                            &mut stream,
                            stem,
                            &format!("h3_{stem}"),
                            &vec![
                                (format!("h3.{stem}.width"), INNER.to_string()),
                                (format!("h3.{stem}.row_capacity"), capacity.to_string()),
                            ],
                        )
                        .unwrap();
                    compiler.flush(&mut stream).unwrap();
                    // Include every half bit pattern: transpose must preserve NaNs and signed zero too.
                    let stride = if direct { 3 * INNER } else { INNER };
                    let offset = if direct { 2 * INNER } else { 0 };
                    let input: Vec<u16> = (0..tokens * stride)
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
                            &[tokens as u32],
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
                            (row * stride + offset + column).wrapping_mul(37) as u16
                        } else {
                            0
                        };
                        assert_eq!(value, expected, "row={row} column={column}");
                    }
                },
            );
        }
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
                let mut stream = support::stream(None);
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
                    let mut stream = support::stream(Some(manager.budget()));
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
                        let scalars = [out_len as u32, len as u32];
                        h3_hrx::dispatch::emit(
                            &mut Sink::Stream(stream),
                            kernel,
                            None,
                            "audio convolution",
                            &scalars[..if tiled { 1 } else { 2 }],
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
                let mut stream = support::stream(None);
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
                let actual = support::read(&mut stream, out.binding());
                assert_eq!(actual, bytemuck::cast_slice::<_, u8>(&input));
            });
        }
    }
    group.finish();
}

fn groupnorm_statistics(c: &mut Criterion) {
    use h3_hrx::dispatch::{emit, Profile};
    let mut group = c.benchmark_group("video_groupnorm_stats");
    for (frames, plane, channels) in [
        (1usize, 31usize, 32usize),
        (1, 33, 128),
        (17, 4096, 128),
        (9, 1024, 256),
        (5, 256, 512),
        (5, 16, 1024),
        (1, 65536, 128),
    ] {
        group.throughput(Throughput::Elements((frames * plane * channels) as u64));
        group.bench_function(format!("{frames}x{plane}x{channels}"), |b| {
            let manager = hrx::residency::ResidencyManager::new(512 << 20).unwrap();
            let mut stream = support::stream(Some(manager.budget()));
            let compiler = compiler();
            let cfg = [
                ("channels", channels),
                ("groups", 32),
                ("plane", plane),
                ("rows_bound", (frames * plane).div_ceil(64) * 64),
            ]
            .map(|(key, value)| (format!("h3.gn_stats_f16.{key}"), value.to_string()))
            .to_vec();
            let kernel = compiler
                .get(&mut stream, "gn_stats_f16", "h3_gn_stats_f16", &cfg)
                .unwrap();
            compiler.flush(&mut stream).unwrap();
            let input: Vec<f16> = (0..frames * plane * channels)
                .map(|i| f16::from_f32(((i * 17 % 127) as f32 - 63.) / 32.))
                .collect();
            let x = upload(&mut stream, bytemuck::cast_slice(&input));
            let out = stream.allocate_zeroed(frames * 32 * 2 * 4).unwrap();
            let mut profile = Profile::from_env();
            let mut run = |stream: &mut Stream| {
                emit(
                    &mut Sink::Stream(stream),
                    &kernel,
                    Some(&mut profile),
                    "video groupnorm stats",
                    &[frames as u32],
                    &[frames as u32],
                    &[x.binding(), out.binding()],
                    &[input.len() * 2, frames * 32 * 2 * 4],
                )
                .unwrap();
                stream.synchronize().unwrap();
            };
            run(&mut stream);
            let expected = check_f32(&mut stream, &out);
            b.iter(|| run(&mut stream));
            assert_eq!(check_f32(&mut stream, &out), expected);
        });
    }
    group.finish();
}

fn groupnorm_apply(c: &mut Criterion) {
    use h3_hrx::dispatch::{emit, Profile};
    let mut group = c.benchmark_group("video_groupnorm_apply");
    for (frames, plane, channels) in [
        (1usize, 31usize, 32usize),
        (1, 33, 128),
        (17, 4096, 128),
        (9, 1024, 256),
        (5, 256, 512),
        (5, 16, 1024),
        (1, 65536, 128),
        (1, 127, 512),
        (1, 128, 512),
        (2, 513, 192),
    ] {
        let count = frames * plane * channels;
        group.throughput(Throughput::Elements(count as u64));
        group.bench_function(format!("{frames}x{plane}x{channels}"), |b| {
            let manager = hrx::residency::ResidencyManager::new(512 << 20).unwrap();
            let mut stream = support::stream(Some(manager.budget()));
            let compiler = compiler();
            let mut cfg = [
                ("channels", channels),
                ("groups", 32),
                ("plane", plane),
                ("rows_bound", (frames * plane).div_ceil(64) * 64),
            ]
            .map(|(key, value)| (format!("h3.gn_silu_f16.{key}"), value.to_string()))
            .to_vec();
            cfg.push(("h3.gn_silu_f16.eps".into(), "1e-6".into()));
            let kernel = compiler
                .get(&mut stream, "gn_silu_f16", "h3_gn_silu_f16", &cfg)
                .unwrap();
            compiler.flush(&mut stream).unwrap();
            let input: Vec<f16> = (0..count)
                .map(|i| f16::from_f32(((i * 17 % 127) as f32 - 63.) / 32.))
                .collect();
            let n = (plane * channels / 32) as f32;
            let stats: Vec<f32> = (0..frames * 32)
                .flat_map(|i| {
                    let mean = (i % 7) as f32 / 8.;
                    [mean * n, (mean * mean + 0.5) * n]
                })
                .collect();
            let gamma: Vec<f32> = (0..channels).map(|i| (i % 13) as f32 / 13.).collect();
            let beta: Vec<f32> = (0..channels).map(|i| (i % 7) as f32 / 14.).collect();
            let x = upload(&mut stream, bytemuck::cast_slice(&input));
            let stats = upload(&mut stream, bytemuck::cast_slice(&stats));
            let gamma = upload(&mut stream, bytemuck::cast_slice(&gamma));
            let beta = upload(&mut stream, bytemuck::cast_slice(&beta));
            let out = stream.allocate_zeroed(count * 2).unwrap();
            let mut profile = Profile::from_env();
            let mut run = |stream: &mut Stream| {
                emit(
                    &mut Sink::Stream(stream),
                    &kernel,
                    Some(&mut profile),
                    "video groupnorm apply",
                    &[frames as u32],
                    &[frames as u32],
                    &[
                        x.binding(),
                        stats.binding(),
                        gamma.binding(),
                        beta.binding(),
                        out.binding(),
                    ],
                    &[
                        count * 2,
                        frames * 32 * 2 * 4,
                        channels * 4,
                        channels * 4,
                        count * 2,
                    ],
                )
                .unwrap();
                stream.synchronize().unwrap();
            };
            run(&mut stream);
            let expected = check_f16(&mut stream, &out);
            b.iter(|| run(&mut stream));
            assert_eq!(check_f16(&mut stream, &out), expected);
        });
    }
    group.finish();
}

fn video_convolution(c: &mut Criterion) {
    use h3_hrx::dispatch::{Conv3d, Profile};
    let mut group = c.benchmark_group("video_encoder_conv_f16");
    for (frames, size, cin, cout, stride, tstride) in [
        (1usize, 32usize, 8usize, 128usize, 1usize, 1usize),
        (1, 32, 128, 128, 1, 1),
        (1, 16, 128, 128, 1, 1),
        (1, 18, 128, 128, 1, 1),
        (17, 32, 128, 128, 1, 1),
        (17, 64, 128, 128, 1, 1),
        (17, 128, 128, 128, 1, 1),
        (3, 16, 128, 128, 1, 1),
        (3, 18, 128, 128, 1, 1),
        (17, 64, 128, 128, 2, 1),
        (17, 32, 256, 256, 2, 2),
        (17, 64, 256, 256, 2, 2),
        (9, 128, 256, 256, 2, 2),
        (1, 32, 256, 256, 2, 2),
        (9, 16, 256, 256, 2, 2),
        (5, 8, 512, 512, 2, 1),
        (17, 32, 128, 256, 2, 2),
        (9, 16, 256, 256, 1, 1),
        (9, 32, 256, 256, 1, 1),
        (1, 34, 128, 128, 1, 1),
        (5, 32, 512, 512, 1, 1),
        (9, 16, 256, 512, 2, 2),
        (5, 8, 512, 512, 1, 1),
        (5, 4, 512, 1024, 1, 1),
        (5, 4, 1024, 1024, 1, 1),
        (5, 4, 1024, 64, 1, 1),
        (1, 4, 1024, 1024, 1, 1),
        (1, 8, 1024, 1024, 1, 1),
        (1, 16, 1024, 1024, 1, 1),
        (5, 8, 1024, 1024, 1, 1),
        (5, 16, 1024, 1024, 1, 1),
        (5, 18, 1024, 1024, 1, 1),
    ] {
        let taps = if frames == 1 { 1 } else { 3 };
        let k = (9 * taps * cin).div_ceil(32) * 32;
        for residual in [false, true] {
            group.bench_function(
                format!("{frames}x{size}x{cin}_{cout}/s{stride}t{tstride}/add_{residual}"),
                |b| {
                    let manager = hrx::residency::ResidencyManager::new(512 << 20).unwrap();
                    let mut stream = support::stream(Some(manager.budget()));
                    let compiler = compiler();
                    let conv = Conv3d::build(
                        &compiler,
                        &mut stream,
                        residual,
                        frames,
                        size,
                        size,
                        stride,
                        tstride,
                        taps,
                        cin,
                        cin,
                        k,
                        cout,
                    )
                    .unwrap();
                    compiler.flush(&mut stream).unwrap();
                    let rows = conv.rows();
                    let half_values = |count: usize, seed: usize| {
                        (0..count)
                            .flat_map(|i| {
                                f16::from_f32(((i * 37 + seed) % 101) as f32 / 200. - 0.25)
                                    .to_le_bytes()
                            })
                            .collect::<Vec<_>>()
                    };
                    let input = half_values(frames * size * size * cin, 17);
                    let weights = half_values(cout * k, 31);
                    let bias: Vec<f32> = (0..cout).map(|i| (i % 7) as f32 / 100.).collect();
                    let x = upload(&mut stream, &input);
                    let w = upload(&mut stream, &weights);
                    let bias = upload(&mut stream, bytemuck::cast_slice(&bias));
                    let prior = upload(&mut stream, &half_values(rows * cout, 43));
                    let out = stream.allocate_zeroed(rows * cout * 2).unwrap();
                    let mut profile = Profile::from_env();
                    let mut run = |stream: &mut Stream| {
                        conv.run(
                            stream,
                            Some(&mut profile),
                            "video encoder convolution",
                            x.binding(),
                            w.binding(),
                            bias.binding(),
                            out.binding(),
                            residual.then(|| prior.binding()),
                        )
                        .unwrap();
                        stream.synchronize().unwrap();
                    };
                    run(&mut stream);
                    let expected = check_f16(&mut stream, &out);
                    b.iter(|| run(&mut stream));
                    assert_eq!(check_f16(&mut stream, &out), expected);
                },
            );
        }
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = support::criterion();
    targets = vision_rotary, decoder_feed_forward, video_convolution, groupnorm_apply, groupnorm_statistics, preparation, rotary_preparation, attention_preparation, fused_qk_preparation, quantized_attention, attention_output_preparation, normalization_preparation, attention_transpose, gemm, vision::bench, vision::normalization, audio_qkv, audio_convolution, dispatch
}
criterion_main!(benches);

//! Resident GPU workloads. Compilation, allocation and readback are outside timing.
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use h3_hrx::{
    compile::Compiler,
    dispatch::{ActivationType, Gemm, Prepare, Sink, Tile},
};
use half::bf16;
use hrx::{Buffer, Stream};
use std::{path::PathBuf, time::Duration};

fn compiler() -> Compiler {
    Compiler::new(
        None,
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("kernels"),
    )
}

fn upload(stream: &mut Stream, bytes: &[u8]) -> Buffer {
    let buffer = stream.allocate(bytes.len()).unwrap();
    stream.upload_blocking(buffer.binding(), bytes).unwrap();
    buffer
}

fn check_f32(stream: &mut Stream, buffer: &Buffer) -> Vec<u8> {
    let bytes = stream.read(buffer.binding()).unwrap().wait(stream).unwrap();
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
    for (tokens, width) in [(256usize, 14336usize), (256, 25600), (2048, 14336)] {
        group.throughput(Throughput::Elements((tokens * width) as u64));
        group.bench_function(
            BenchmarkId::new("plain", format!("{tokens}x{width}")),
            |b| {
                let mut stream = Stream::open().unwrap();
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

fn gemm(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm");
    for elem in ["i8", "bf16"] {
        for (mode, n) in [("f32", 6144usize), ("swiglu", 16384)] {
            for rotating in [false, true] {
                let (m, k) = (2048usize, 2048usize);
                let storage = if rotating { "rotating" } else { "cached" };
                group.throughput(Throughput::Elements((2 * m * k * n) as u64));
                group.bench_function(format!("{elem}/{mode}/{storage}/{m}x{k}x{n}"), |b| {
                    let mut stream = Stream::open().unwrap();
                    let compiler = compiler();
                    let stride = k + 64;
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
                    let run = |stream: &mut Stream, index: usize| {
                        gemm.run(
                            stream,
                            None,
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
    targets = preparation, gemm, dispatch
}
criterion_main!(benches);

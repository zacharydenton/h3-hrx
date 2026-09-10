//! Compare production vision shapes against a source snapshot, with resident weights.
//! cargo run --release --example compare_vision -- BASELINE_DIR
use half::bf16;
use hrx::{Buffer, Constants, Kernel, Stream};
use std::{path::Path, time::Instant};

fn values(count: usize, scale: f32) -> Vec<f32> {
    (0..count)
        .map(|i| ((i * 37 % 101) as f32 - 50.) * scale / 50.)
        .collect()
}

fn upload(stream: &mut Stream, bytes: &[u8]) -> Buffer {
    let buffer = stream.allocate(bytes.len()).unwrap();
    stream.upload(buffer.binding(), bytes).unwrap();
    buffer
}

fn launch(
    stream: &mut Stream,
    kernel: &Kernel,
    constants: &Constants,
    data: &[Buffer],
    weight: &Buffer,
    grid: [u32; 3],
) {
    let bindings = [
        data[0].binding(),
        weight.binding(),
        data[1].binding(),
        data[2].binding(),
    ];
    // Safety: trusted vision sources; dimensions size both bindings and launch geometry.
    unsafe {
        stream
            .dispatch(kernel, grid, [256, 1, 1], constants, &bindings)
            .unwrap()
    };
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.
}

fn main() {
    let baseline = std::env::args().nth(1).expect("BASELINE_DIR");
    let batches: usize = std::env::var("H3_COMPARE_BATCHES")
        .map(|v| v.parse().expect("H3_COMPARE_BATCHES must be an integer"))
        .unwrap_or(10);
    assert!(batches >= 2 && batches.is_multiple_of(2));
    let compiler = hrx::loom::Compiler::resolve(None).unwrap();
    let source = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/matmul_bf16_family.loom"),
    )
    .unwrap();
    let mut stream = Stream::open().unwrap();
    println!("stem,M,K,N,weights,artifact_identical,baseline_ms,candidate_ms,ratio");
    for (kind, k, n) in [
        ("bias", 1536usize, 1152usize),
        ("bias", 1152, 3456),
        ("gelu", 1152, 4352),
        ("gelu_erf", 4608, 4608),
        ("bias", 4608, 5120),
    ] {
        let stem = format!("matmul_{kind}_bf16_wmma");
        let symbol = format!("h3_{stem}");
        let original =
            std::fs::read_to_string(Path::new(&baseline).join(format!("{stem}.loom"))).unwrap();
        let mut request = hrx::loom::Specialization::new(&symbol);
        request.config = [("k_size", k), ("n_size", n)]
            .into_iter()
            .map(|(key, value)| (format!("h3.{stem}.{key}"), value.to_string()))
            .collect();
        let old = compiler.module(&original).compile(&request).unwrap();
        let new = compiler.module(&source).compile(&request).unwrap();
        let identical = old.bytes() == new.bytes();
        // Safety: both artifacts were compiled from the trusted sources above.
        let kernels = unsafe {
            [
                stream.load_artifact(&old).unwrap(),
                stream.load_artifact(&new).unwrap(),
            ]
        };
        let weights: Vec<_> = values(k * n, 0.1).into_iter().map(bf16::from_f32).collect();
        let weight_bytes = bytemuck::cast_slice(&weights);
        let count = (64 * 1024 * 1024 / weight_bytes.len() + 1).max(2);
        let ring: Vec<_> = (0..count)
            .map(|_| upload(&mut stream, weight_bytes))
            .collect();
        for m in [65usize, 256, 1024] {
            let data = [
                upload(&mut stream, bytemuck::cast_slice(&values(m * k, 0.5))),
                upload(&mut stream, bytemuck::cast_slice(&values(n, 0.1))),
                upload(&mut stream, &vec![0; m * n * 4]),
            ];
            let constants = kernels
                .each_ref()
                .map(|kernel| Constants::indices(kernel, &[m as u32]).unwrap());
            let grid = [n.div_ceil(64) as u32, m.div_ceil(64) as u32, 1];
            let outputs: Vec<_> = (0..2)
                .map(|version| {
                    launch(
                        &mut stream,
                        &kernels[version],
                        &constants[version],
                        &data,
                        &ring[0],
                        grid,
                    );
                    stream
                        .read(data[2].binding())
                        .unwrap()
                        .wait(&mut stream)
                        .unwrap()
                })
                .collect();
            assert_eq!(outputs[0], outputs[1], "{stem}: M={m} K={k} N={n}");
            for rotation in [1, count] {
                let launches = rotation.max(12);
                for version in 0..2 {
                    for j in 0..launches {
                        launch(
                            &mut stream,
                            &kernels[version],
                            &constants[version],
                            &data,
                            &ring[j % rotation],
                            grid,
                        );
                    }
                }
                stream.synchronize().unwrap();
                let mut times = [Vec::new(), Vec::new()];
                for batch in 0..batches {
                    for order in 0..2 {
                        let version = (batch + order) % 2;
                        let start = Instant::now();
                        for j in 0..launches {
                            launch(
                                &mut stream,
                                &kernels[version],
                                &constants[version],
                                &data,
                                &ring[j % rotation],
                                grid,
                            );
                        }
                        stream.synchronize().unwrap();
                        times[version]
                            .push(start.elapsed().as_secs_f64() * 1000. / launches as f64);
                    }
                }
                let old = median(&mut times[0]);
                let new = median(&mut times[1]);
                println!(
                    "{stem},{m},{k},{n},{rotation},{identical},{old:.6},{new:.6},{:.5}",
                    new / old
                );
            }
        }
    }
}

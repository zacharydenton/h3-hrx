//! Paired resident GEMM comparisons against a source snapshot.
//! cargo run --release -p h3 --example compare_gemm -- BASELINE_DIR [FILTER]
//! Weights rotate through more than 64 MiB as well as repeatedly using one matrix.
//! H3_COMPARE_BATCHES sets the number of paired timing batches (default 10, even and >= 2).
use half::{bf16, f16};
use hrx::{Buffer, Constants, Kernel, Stream};
use std::{path::Path, time::Instant};

fn operand(count: usize, dtype: &str, seed: usize) -> Vec<u8> {
    let value = |i: usize| ((i.wrapping_mul(37).wrapping_add(seed) % 15) as i8) - 7;
    match dtype {
        "i4" => (0..count / 2)
            .map(|i| (value(2 * i) as u8 & 15) | ((value(2 * i + 1) as u8 & 15) << 4))
            .collect(),
        "i8" => (0..count).map(|i| value(i) as u8).collect(),
        _ => (0..count)
            .flat_map(|i| {
                let f = f32::from(value(i)) / 128.;
                if dtype == "f16" {
                    f16::from_f32(f).to_bits()
                } else {
                    bf16::from_f32(f).to_bits()
                }
                .to_le_bytes()
            })
            .collect(),
    }
}
fn uploaded(stream: &mut Stream, bytes: &[u8]) -> Buffer {
    let b = stream.allocate(bytes.len()).unwrap();
    stream.upload(&b, bytes).unwrap();
    b
}
fn launch(
    stream: &mut Stream,
    kernel: &Kernel,
    constants: &Constants,
    buffers: &[Buffer],
    weight: &Buffer,
    grid: [u32; 3],
) {
    let mut bindings: Vec<_> = buffers.iter().map(Buffer::binding).collect();
    bindings[1] = weight.binding();
    // Safety: trusted GEMMs, with buffers and launch geometry sized from the same config.
    unsafe {
        stream
            .dispatch(kernel, grid, [256, 1, 1], constants, &bindings)
            .unwrap();
    }
}
fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    (v[v.len() / 2 - 1] + v[v.len() / 2]) / 2.
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let baseline = Path::new(args.get(1).expect("BASELINE_DIR [FILTER]"));
    let filter = args.get(2).map(String::as_str).unwrap_or("");
    let batches: usize = std::env::var("H3_COMPARE_BATCHES")
        .map(|v| v.parse().expect("H3_COMPARE_BATCHES must be an integer"))
        .unwrap_or(10);
    assert!(
        batches >= 2 && batches.is_multiple_of(2),
        "H3_COMPARE_BATCHES must be even and >= 2"
    );
    let compiler = hrx::loom::Compiler::resolve(None).unwrap();
    let cache = hrx::bundle::cache_root().unwrap().join("kernels");
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels");
    let mut stream = Stream::open().unwrap();
    println!("stem,M,K,N,weights,artifact_identical,baseline_ms,candidate_ms,ratio");
    for dtype in ["i4", "i8", "f16", "bf16"] {
        for mode in ["plain", "resid", "swiglu"] {
            for biased in [false, true] {
                let stem = format!(
                    "gemm_{dtype}{}_256{}{}",
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
                if !stem.contains(filter) {
                    continue;
                }
                let module = if dtype.starts_with('i') {
                    "gemm_packed_256".into()
                } else {
                    format!("gemm_{dtype}_family")
                };
                for (k, n) in [(2048usize, 6144usize), (2048, 16384), (8192, 2048)] {
                    let m = 2048usize;
                    let stride = k + if dtype == "i4" { 128 } else { 64 };
                    let cfg = [
                        ("k_size", k),
                        ("n_size", n),
                        ("k_stride", stride),
                        ("m_group", 4),
                        ("classes", 2),
                    ]
                    .into_iter()
                    .filter(|(name, _)| *name != "classes" || mode == "resid")
                    .map(|(name, value)| (format!("h3.{stem}.{name}"), value.to_string()))
                    .collect();
                    let symbol = format!("h3_{stem}");
                    let old =
                        std::fs::read_to_string(baseline.join(format!("{stem}.loom"))).unwrap();
                    let new = std::fs::read_to_string(root.join(format!("{module}.loom"))).unwrap();
                    let mut request = hrx::loom::Request::new(&old, &symbol);
                    request.config = cfg;
                    let old_path = compiler.compile(&request, &cache).unwrap();
                    request.source = &new;
                    let new_path = compiler.compile(&request, &cache).unwrap();
                    let identical =
                        std::fs::read(&old_path).unwrap() == std::fs::read(&new_path).unwrap();
                    // Safety: trusted baseline and candidate sources compiled through HRX.
                    let kernels = [
                        unsafe { stream.load(&old_path, &symbol).unwrap() },
                        unsafe { stream.load(&new_path, &symbol).unwrap() },
                    ];
                    let constants = kernels
                        .each_ref()
                        .map(|kernel| Constants::indices(kernel, &[m as u32]).unwrap());
                    let weights = operand(n * stride, dtype, 13);
                    let count = (64 * 1024 * 1024 / weights.len() + 1).max(2);
                    let mut weight_ring = Vec::new();
                    for _ in 0..count {
                        weight_ring.push(uploaded(&mut stream, &weights));
                    }
                    let mut data = vec![operand(m * stride, dtype, 3), weights];
                    if dtype.starts_with('i') {
                        data.push(bytemuck::cast_slice(&vec![0.01f32; n]).to_vec());
                        data.push(bytemuck::cast_slice(&vec![0.01f32; m]).to_vec());
                    }
                    let output = data.len();
                    let output_bytes = match mode {
                        "resid" => 4,
                        "swiglu" => 1, // N/2 stored f16 values per row.
                        _ => 2,
                    };
                    data.push(vec![0; m * n * output_bytes]);
                    if mode == "resid" {
                        data.push(bytemuck::cast_slice(&vec![0.1f32; 2 * n]).to_vec());
                        data.push(
                            bytemuck::cast_slice(
                                &(0..m).map(|i| (i % 2) as i32).collect::<Vec<_>>(),
                            )
                            .to_vec(),
                        );
                    }
                    if biased {
                        data.push(bytemuck::cast_slice(&vec![0.01f32; n]).to_vec());
                    }
                    let buffers: Vec<_> = data.iter().map(|v| uploaded(&mut stream, v)).collect();
                    let grid = [
                        n.div_ceil(128) as u32,
                        m.div_ceil(256).div_ceil(4) as u32 * 4,
                        1,
                    ];
                    let mut results = Vec::new();
                    for version in 0..2 {
                        stream.upload(&buffers[output], &data[output]).unwrap();
                        launch(
                            &mut stream,
                            &kernels[version],
                            &constants[version],
                            &buffers,
                            &weight_ring[0],
                            grid,
                        );
                        results.push(
                            stream
                                .read_queued(buffers[output].binding())
                                .unwrap()
                                .wait(&mut stream)
                                .unwrap(),
                        );
                    }
                    assert!(
                        results[0] == results[1],
                        "numerical mismatch: {stem} K={k} N={n}"
                    );
                    for rotation in [1, count] {
                        for version in 0..2 {
                            for j in 0..12 {
                                launch(
                                    &mut stream,
                                    &kernels[version],
                                    &constants[version],
                                    &buffers,
                                    &weight_ring[j % rotation],
                                    grid,
                                );
                            }
                        }
                        stream.synchronize().unwrap();
                        let mut times = [Vec::new(), Vec::new()];
                        let launches = 12.max(rotation);
                        for batch in 0..batches {
                            for order in 0..2 {
                                let version = (batch + order) % 2;
                                stream.upload(&buffers[output], &data[output]).unwrap();
                                stream.synchronize().unwrap();
                                let start = Instant::now();
                                for j in 0..launches {
                                    launch(
                                        &mut stream,
                                        &kernels[version],
                                        &constants[version],
                                        &buffers,
                                        &weight_ring[j % rotation],
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
    }
}

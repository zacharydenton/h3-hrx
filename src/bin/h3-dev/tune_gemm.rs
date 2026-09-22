//! Bounded production-shape INT8 GEMM traversal screening with real checkpoint rows.
//! `h3-dev tune-gemm [38048|40047] [qkv|gu|out|down]` emits JSONL.
//! Compare groups 2/3/4/8 in alternating order; four actual layer weights rotate.
use h3_hrx::{model::*, weights::Weights};
use hrx::{Buffer, Constants, Stream};
use std::{path::Path, time::Instant};

fn upload(stream: &mut Stream, data: &[u8]) -> Buffer {
    let b = stream.allocate(data.len()).unwrap();
    stream.upload(b.binding(), data).unwrap();
    b
}
fn median(v: &[f64]) -> f64 {
    let mut v = v.to_vec();
    v.sort_by(f64::total_cmp);
    (v[v.len() / 2 - 1] + v[v.len() / 2]) / 2.0
}
fn resources(path: &Path) {
    if let Ok(output) = std::process::Command::new("llvm-readelf")
        .arg("--notes")
        .arg(path)
        .output()
    {
        if output.status.success() {
            for line in String::from_utf8_lossy(&output.stdout)
                .lines()
                .filter(|line| {
                    [
                        ".sgpr_count:",
                        ".vgpr_count:",
                        ".group_segment_fixed_size:",
                        ".private_segment_fixed_size:",
                    ]
                    .iter()
                    .any(|field| line.contains(field))
                })
            {
                eprintln!("{}", line.trim());
            }
        }
    }
}

pub fn run(args: Vec<String>) {
    let shapes = match args.first().map(String::as_str) {
        None => vec![38048usize, 40047],
        Some("38048") => vec![38048],
        Some("40047") => vec![40047],
        _ => panic!("tune-gemm [38048|40047] [qkv|gu|out|down]"),
    };
    let filter = args.get(1).map(String::as_str);
    assert!(filter.is_none() || matches!(filter, Some("qkv" | "gu" | "out" | "down")));
    let compiler = hrx::loom::Compiler::resolve(None).unwrap();
    let source = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/gemm_packed_256.loom"),
    )
    .unwrap();
    let path = h3_hrx::models::Resolver::new()
        .offline(true)
        .find(h3_hrx::models::DIT_FL2VA)
        .unwrap();
    let mut stream = Stream::open().unwrap();
    for m in shapes {
        for (op, k, n, mode) in [
            ("qkv", HID, QKV, ""),
            ("gu", HID, 2 * FFN, "_swiglu"),
            ("out", INNER, HID, "_resid"),
            ("down", FFN, HID, "_resid"),
        ] {
            if filter.is_some_and(|f| f != op) {
                continue;
            }
            // Drop the weight owner between shapes/projections; only four matrices are resident.
            // Safety: this diagnostic never mutates checkpoint files.
            let weights = unsafe { Weights::open(&path, h3_hrx::plan::dit::plan) }.unwrap();
            let stride = gemm_pitch(k, 8);
            let cap = m.div_ceil(2048) * 2048;
            let a = upload(
                &mut stream,
                &(0..cap * stride)
                    .map(|i| {
                        (if i / stride < m && i % stride < k {
                            ((i * 37 + 3) % 15) as i8 - 7
                        } else {
                            0
                        }) as u8
                    })
                    .collect::<Vec<_>>(),
            );
            let scale_a = upload(&mut stream, bytemuck::cast_slice(&vec![0.01f32; cap]));
            let out_bytes = cap
                * n
                * if mode == "_resid" {
                    4
                } else if mode == "_swiglu" {
                    1
                } else {
                    2
                };
            let out = stream.allocate(out_bytes).unwrap();
            let gates = upload(&mut stream, bytemuck::cast_slice(&vec![0.1f32; 4 * n]));
            let classes = upload(
                &mut stream,
                bytemuck::cast_slice(&(0..cap).map(|i| (i % 4) as i32).collect::<Vec<_>>()),
            );
            let ring: Vec<_> = (0..4)
                .map(|layer| {
                    let p = format!("blocks.{layer}.{op}");
                    (
                        weights
                            .rows(&mut stream, &format!("{p}.q"), n, k, stride)
                            .unwrap(),
                        weights.at(&mut stream, &format!("{p}.s"), n * 4).unwrap(),
                    )
                })
                .collect();
            let stem = format!("gemm_i8{mode}_256");
            let groups = [2u32, 3, 4, 8];
            let baseline = gemm_m_group_for(m, k, n, 8);
            let mut kernels = Vec::new();
            for &group in &groups {
                let mut req = hrx::loom::Specialization::new(format!("h3_{stem}"));
                req.set_report(hrx::loom::ReportMode::Summary);
                req.replace_config(
                    [
                        ("k_size", k),
                        ("n_size", n),
                        ("k_stride", stride),
                        ("m_group", group as usize),
                        ("classes", 4),
                    ]
                    .into_iter()
                    .filter(|(key, _)| *key != "classes" || mode == "_resid")
                    .map(|(key, value)| (format!("h3.{stem}.{key}"), value.to_string()))
                    .collect(),
                );
                let artifact = compiler.module(&source).compile(&req).unwrap();
                println!("{{\"kind\":\"artifact\",\"op\":\"{op}\",\"m\":{m},\"group\":{group},\"sha256\":\"{}\",\"compiler\":\"{}\",\"report\":{}}}",
                    hrx::bundle::digest(artifact.bytes()), artifact.compiler_identity(), artifact.report().map_or("null".into(), |report| report.json().to_string()));
                eprintln!(
                    "resources {op} M={m} group={group} {}",
                    artifact.path().display()
                );
                resources(artifact.path());
                // Safety: checked-in source with dimensions and buffer sizes specified above.
                let kernel = unsafe { stream.load_artifact(&artifact).unwrap() };
                let constants = Constants::indices(&kernel, &[m as u32]).unwrap();
                kernels.push((kernel, constants));
            }
            let launch = |stream: &mut Stream, version: usize, layer: usize| {
                let (weight, scales) = &ring[layer];
                let mut b = vec![
                    a.binding(),
                    weight.binding(),
                    scales.binding(),
                    scale_a.binding(),
                    out.binding(),
                ];
                if mode == "_resid" {
                    b.extend([gates.binding(), classes.binding()]);
                }
                let (kernel, constants) = &kernels[version];
                // Safety: same configured pitches, padded rows and output type as the compiled GEMM.
                unsafe {
                    stream
                        .dispatch(
                            kernel,
                            [
                                n.div_ceil(128) as u32,
                                gemm_grid_y(m, groups[version], 256),
                                1,
                            ],
                            [256, 1, 1],
                            constants,
                            &b,
                        )
                        .unwrap();
                }
            };
            let mut digest = None;
            for (v, group) in groups.iter().enumerate() {
                stream.fill(out.binding(), 0).unwrap();
                launch(&mut stream, v, 0);
                let bytes = stream
                    .read(out.binding())
                    .unwrap()
                    .wait(&mut stream)
                    .unwrap();
                let got = hrx::bundle::digest(&bytes);
                if let Some(expected) = &digest {
                    assert_eq!(&got, expected, "{op} group {group} parity");
                } else {
                    digest = Some(got);
                }
            }
            // Warm all traversal variants with the same rotating real matrices before timing.
            for v in 0..groups.len() {
                for layer in 0..ring.len() {
                    launch(&mut stream, v, layer);
                }
            }
            stream.synchronize().unwrap();
            let mut samples = vec![Vec::new(); groups.len()];
            for round in 0..8 {
                for order in 0..groups.len() {
                    let v = if round % 2 == 0 {
                        order
                    } else {
                        groups.len() - 1 - order
                    };
                    stream.fill(out.binding(), 0).unwrap();
                    stream.synchronize().unwrap();
                    let start = Instant::now();
                    for layer in 0..ring.len() {
                        launch(&mut stream, v, layer);
                    }
                    stream.synchronize().unwrap();
                    samples[v].push(start.elapsed().as_secs_f64() * 1000.0 / ring.len() as f64);
                }
            }
            let baseline_ms = median(&samples[groups.iter().position(|g| *g == baseline).unwrap()]);
            for (i, group) in groups.iter().enumerate() {
                let ms = median(&samples[i]);
                println!("{{\"kind\":\"timing\",\"op\":\"{op}\",\"m\":{m},\"k\":{k},\"n\":{n},\"group\":{group},\"baseline_group\":{baseline},\"real_weight_layers\":[0,1,2,3],\"byte_identical\":true,\"samples_ms\":{:?},\"median_ms\":{ms},\"baseline_over_candidate\":{}}}", samples[i], baseline_ms / ms);
            }
        }
    }
}

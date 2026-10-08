mod support;
use criterion::{criterion_group, criterion_main, Criterion};
use h3_hrx::{
    upscale::{Upscaler, CHECKPOINT},
    DenoiseParams, Latents, Noise, RefinementSettings, SessionOptions, UpscaleSettings,
    UpscaleTarget,
};

fn upscale(c: &mut Criterion) {
    let settings = UpscaleSettings {
        target: UpscaleTarget::Scale(2.0),
        ..Default::default()
    };
    let shape = h3_hrx::shape_for(64, 64, 5).unwrap();
    let input = Latents {
        video: support::values(24 * shape.latent_t as usize * 16, 0.2),
        audio: support::values(64 * shape.audio_t as usize, 0.1),
    };
    let mut path = None;
    c.bench_function("upscale/checkpoint_metadata", |b| {
        let path = path.get_or_insert_with(|| {
            h3_hrx::models::Resolver::new()
                .repository("LBH-123-AI", "Minimax_h3_latent_Upscaler")
                .find(CHECKPOINT)
                .unwrap()
        });
        b.iter(|| {
            let model = unsafe { Upscaler::open(&*path) }.unwrap();
            std::hint::black_box(model.device_bytes());
        });
    });
    let (network_profile, network_params) = support::params();
    let network_shape = h3_hrx::shape_for(
        network_params.height,
        network_params.width,
        network_params.frames,
    )
    .unwrap();
    let network_input = Latents {
        video: support::values(
            24 * network_shape.latent_t as usize
                * network_shape.lat_h as usize
                * network_shape.lat_w as usize,
            0.2,
        ),
        audio: support::values(64 * network_shape.audio_t as usize, 0.1),
    };
    let target_shape = settings.output_shape(&network_shape).unwrap();
    for cold in [false, true] {
        let mut fixture = None;
        c.bench_function(
            &format!(
                "upscale/network/{}/{network_profile}",
                if cold { "cold_weights" } else { "warm_weights" }
            ),
            |b| {
                let (runtime, session) = fixture.get_or_insert_with(|| {
                    let runtime = support::Runtime::new();
                    let session = runtime.session(
                        h3_hrx::Config {
                            kernel_sources: support::sources(),
                            ..Default::default()
                        },
                        SessionOptions {
                            residency: if cold {
                                h3_hrx::ResidencyPolicy::StageScoped
                            } else {
                                h3_hrx::ResidencyPolicy::Retain
                            },
                            ..Default::default()
                        },
                    );
                    (runtime, session)
                });
                let _ = runtime;
                support::measure(
                    b,
                    || {
                        session
                            .upscale_latents(&network_input, &network_shape, &settings, None)
                            .unwrap()
                    },
                    |out| {
                        assert_eq!(out.shape.size(), target_shape.size());
                        assert_eq!(out.latents.audio, network_input.audio);
                        std::hint::black_box(support::digest(&out.latents.video));
                    },
                );
            },
        );
    }
    for full in [false, true] {
        let mut fixture = None;
        c.bench_function(
            &format!(
                "{}/{:?}/{}",
                if full {
                    "upscale/pipeline/generate_upscale_refine_decode"
                } else {
                    "upscale/refinement/references"
                },
                support::residency(),
                support::execution_id()
            ),
            |b| {
                let (_runtime, session, ids, initial, refs) = fixture.get_or_insert_with(|| {
                    let runtime = support::Runtime::new();
                    let session = runtime.session(
                        support::config(true),
                        SessionOptions {
                            residency: support::residency(),
                            ..Default::default()
                        },
                    );
                    let ids = h3_hrx::Tokenizer::new()
                        .unwrap()
                        .encode(support::PROMPT)
                        .unwrap();
                    let target = settings.output_shape(&shape).unwrap();
                    let initial = Latents {
                        video: support::values(24 * target.latent_t as usize * 64, 0.2),
                        audio: input.audio.clone(),
                    };
                    let refs = support::values(24 * 4, 0.2);
                    (runtime, session, ids, initial, refs)
                });
                let references = [h3_hrx::Reference::Image {
                    latents: refs,
                    grid: h3_hrx::LatentGrid {
                        frames: 1,
                        height: 2,
                        width: 2,
                    },
                    presented: None,
                }];
                let refinement = RefinementSettings::default();
                support::measure(
                    b,
                    || {
                        let mut p = DenoiseParams {
                            width: 64,
                            height: 64,
                            frames: 5,
                            steps: 4,
                            seed: 7,
                            ..Default::default()
                        };
                        if full {
                            let low = session
                                .denoise(ids, &p, Noise::default(), &references, &[], None)
                                .unwrap();
                            let up = session
                                .upscale_latents(&low, &shape, &settings, None)
                                .unwrap();
                            (p.width, p.height) = up.shape.size();
                            let result = session
                                .refine(
                                    ids,
                                    None,
                                    &p,
                                    &up.latents,
                                    &refinement,
                                    &references,
                                    &[],
                                    None,
                                )
                                .unwrap();
                            assert_eq!(result.audio, low.audio);
                            let mut video = vec![0u8; up.shape.video_bytes()];
                            let mut audio = vec![0f32; up.shape.audio_samples()];
                            session
                                .decode_video(&up.shape, &result.video, &mut video)
                                .unwrap();
                            session
                                .decode_audio(&result.audio, up.shape.audio_t as usize, &mut audio)
                                .unwrap();
                            std::hint::black_box((video, audio));
                            result
                        } else {
                            p.width = 128;
                            p.height = 128;
                            session
                                .refine(ids, None, &p, initial, &refinement, &references, &[], None)
                                .unwrap()
                        }
                    },
                    |out| {
                        std::hint::black_box(support::digest(&out.video));
                    },
                );
            },
        );
    }
}
criterion_group! {name=benches;config=support::criterion();targets=packing,components,groupnorm_stats,convolutions,temporal_convolutions,groupnorm_apply,upscale}
criterion_main!(benches);

fn packing(c: &mut Criterion) {
    let mut group = c.benchmark_group("upscale/pack");
    for name in [
        "conv_in.weight",
        "in_blocks.0.in_layers.2.weight",
        "conv_out.weight",
    ] {
        group.bench_function(name, |b| {
            let path = h3_hrx::models::Resolver::new()
                .repository("LBH-123-AI", "Minimax_h3_latent_Upscaler")
                .find(CHECKPOINT)
                .unwrap();
            // SAFETY: benchmark checkpoints remain immutable throughout the run.
            let ck = unsafe { h3_hrx::checkpoint::Checkpoint::open(path) }.unwrap();
            let entry = ck.at(name).unwrap();
            let (co, ci) = (entry.shape[0], entry.shape[1]);
            let recipe = h3_hrx::weights::conv3d_taps(&ck, name, co, ci, 27).unwrap();
            let cp = ci.div_ceil(8) * 8;
            let k = (27 * cp).div_ceil(32) * 32;
            let src = ck.bytes(entry);
            let mut expected = vec![0; recipe.device_bytes()];
            for o in 0..co {
                for i in 0..ci {
                    for tap in 0..27 {
                        let from = ((o * ci + i) * 27 + tap) * 2;
                        let to = (o * k + tap * cp + i) * 2;
                        expected[to..to + 2].copy_from_slice(&src[from..from + 2]);
                    }
                }
            }
            support::measure(
                b,
                || recipe.assemble(&ck).unwrap(),
                |out| assert_eq!(*out, expected),
            );
        });
    }
    group.finish();
}

fn components(c: &mut Criterion) {
    use half::f16;
    let (frames, plane, channels) = (8usize, 64usize, 512usize);
    let rows = frames * plane;
    for stem in [
        "upscale_temporal",
        "upscale_resize",
        "upscale_gn_stats",
        "upscale_gn_silu",
        "upscale_add",
    ] {
        c.bench_function(&format!("upscale/components/{stem}"), |b| {
            let manager = hrx::residency::ResidencyManager::new(256 << 20).unwrap();
            let mut stream = support::stream(Some(manager.budget()));
            let compiler = hrx::loom::Compiler::resolve(None).unwrap();
            let source =
                std::fs::read_to_string(support::sources().join(format!("{stem}.loom"))).unwrap();
            let mut specialization = hrx::loom::Specialization::new(format!("h3_{stem}"));
            let mut config = std::collections::BTreeMap::new();
            let mut set = |key: &str, value: usize| {
                config.insert(format!("h3.{stem}.{key}"), value.to_string());
            };
            let input = support::values(rows * channels, 0.2)
                .into_iter()
                .flat_map(|v| f16::from_f32(v).to_le_bytes())
                .collect::<Vec<_>>();
            let (data, grid, threads, scalar) = match stem {
                "upscale_temporal" => {
                    set("frames", frames);
                    set("plane", plane);
                    set("channels", channels);
                    set("taps", 5);
                    (
                        vec![
                            input,
                            vec![0; channels * 5 * 2],
                            vec![0; channels * 4],
                            vec![0; rows * channels * 2],
                        ],
                        [(rows * channels).div_ceil(1024) as u32, 1, 1],
                        256,
                        rows * channels,
                    )
                }
                "upscale_resize" => {
                    set("frames", frames);
                    set("in_plane", plane);
                    set("out_plane", plane * 4);
                    set("channels", channels);
                    let indices = (0..plane * 4)
                        .flat_map(|p| std::iter::repeat_n((p / 4) as i32, 4))
                        .flat_map(i32::to_le_bytes)
                        .collect();
                    let weights = vec![0.25f32; plane * 16];
                    (
                        vec![
                            input,
                            indices,
                            bytemuck::cast_slice(&weights).to_vec(),
                            vec![0; rows * 4 * channels * 2],
                        ],
                        [(rows * 4 * channels).div_ceil(256) as u32, 1, 1],
                        256,
                        rows * 4 * channels,
                    )
                }
                "upscale_gn_stats" | "upscale_gn_silu" => {
                    set("channels", channels);
                    set("groups", 32);
                    set("plane", rows);
                    set("rows_bound", rows);
                    if stem == "upscale_gn_stats" {
                        set("lanes", 256);
                        (vec![input, vec![0; 256]], [1, 32, 1], 256, 1)
                    } else {
                        config.insert(format!("h3.{stem}.eps"), "0.00001".into());
                        let stats = (0..64).map(|i| (i % 2) as f32).collect::<Vec<_>>();
                        let gamma = vec![1.0f32; channels];
                        (
                            vec![
                                input,
                                bytemuck::cast_slice(&stats).to_vec(),
                                bytemuck::cast_slice(&gamma).to_vec(),
                                vec![0; channels * 4],
                                vec![0; rows * channels * 2],
                                vec![0; channels * 8],
                            ],
                            [(rows * channels).div_ceil(1024) as u32, 1, 1],
                            256,
                            1,
                        )
                    }
                }
                _ => (
                    vec![input, vec![0; rows * channels * 2]],
                    [(rows * channels).div_ceil(256) as u32, 1, 1],
                    256,
                    rows * channels,
                ),
            };
            specialization.replace_config(config);
            let artifact = compiler.module(&source).compile(&specialization).unwrap();
            // SAFETY: checked-in sources, exact configuration-derived bindings and launch geometry.
            let kernel = unsafe { stream.load_artifact(&artifact) }.unwrap();
            let buffers = data
                .iter()
                .map(|bytes| stream.allocate_from(bytes).unwrap())
                .collect::<Vec<_>>();
            let views = buffers
                .iter()
                .map(|buffer| buffer.binding())
                .collect::<Vec<_>>();
            let constants = hrx::Constants::indices(&kernel, &[scalar as u32]).unwrap();
            let mut run = || {
                unsafe { stream.dispatch(&kernel, grid, [threads, 1, 1], &constants, &views) }
                    .unwrap();
                stream.synchronize().unwrap();
            };
            run();
            b.iter(&mut run);
        });
    }
    c.bench_function("upscale/components/conv3d", |b| {
        let manager = hrx::residency::ResidencyManager::new(256 << 20).unwrap();
        let mut stream = support::stream(Some(manager.budget()));
        let compiler = support::compiler();
        let k = 27 * channels;
        let conv = h3_hrx::dispatch::Conv3d::build_padding(
            &compiler,
            &mut stream,
            false,
            frames,
            8,
            8,
            1,
            1,
            3,
            channels,
            channels,
            k,
            channels,
            true,
        )
        .unwrap();
        let x = stream.allocate_zeroed(rows * channels * 2).unwrap();
        let w = stream.allocate_zeroed(channels * k * 2).unwrap();
        let bias = stream.allocate_zeroed(channels * 4).unwrap();
        let out = stream.allocate_zeroed(rows * channels * 2).unwrap();
        let mut profile = h3_hrx::dispatch::Profile::from_env();
        let mut run = || {
            conv.run(
                &mut stream,
                Some(&mut profile),
                "upscale conv3d",
                x.binding(),
                w.binding(),
                bias.binding(),
                out.binding(),
                None,
            )
            .unwrap();
            stream.synchronize().unwrap();
        };
        run();
        b.iter(&mut run);
    });
}

fn groupnorm_stats(c: &mut Criterion) {
    use h3_hrx::dispatch::{emit, Profile, Sink};
    use half::f16;
    let channels = 512usize;
    let groups = 32usize;
    // Whole-volume rows, including long clips and a partial workgroup.
    for rows in [512usize, 12960, 51840, 129025] {
        for lanes in [32usize, 128, 256, 512, 1024] {
            c.bench_function(&format!("upscale/groupnorm_stats/{rows}/{lanes}"), |b| {
                let manager = hrx::residency::ResidencyManager::new(256 << 20).unwrap();
                let mut stream = support::stream(Some(manager.budget()));
                let compiler = support::compiler();
                let cfg = [
                    ("channels", channels),
                    ("groups", groups),
                    ("plane", rows),
                    ("rows_bound", rows.div_ceil(64) * 64),
                    ("lanes", lanes),
                ]
                .map(|(key, value)| (format!("h3.upscale_gn_stats.{key}"), value.to_string()))
                .to_vec();
                let kernel = compiler
                    .get(&mut stream, "upscale_gn_stats", "h3_upscale_gn_stats", &cfg)
                    .unwrap();
                let input = support::values(rows * channels, 0.2)
                    .into_iter()
                    .map(f16::from_f32)
                    .collect::<Vec<_>>();
                let mut means = [0f64; 32];
                let mut variances = [0f64; 32];
                for (i, value) in input.iter().enumerate() {
                    means[i % channels / (channels / groups)] += value.to_f64();
                }
                let count = (rows * channels / groups) as f64;
                means.iter_mut().for_each(|v| *v /= count);
                for (i, value) in input.iter().enumerate() {
                    let g = i % channels / (channels / groups);
                    variances[g] += (value.to_f64() - means[g]).powi(2);
                }
                variances.iter_mut().for_each(|v| *v /= count);
                let x = stream.allocate_from(bytemuck::cast_slice(&input)).unwrap();
                let out = stream.allocate(256).unwrap();
                let mut profile = Profile::from_env();
                let mut run = |stream: &mut hrx::Stream| {
                    emit(
                        &mut Sink::Stream(stream),
                        &kernel,
                        Some(&mut profile),
                        "upscale groupnorm stats",
                        &[1],
                        &[1],
                        &[x.binding(), out.binding()],
                        &[rows * channels * 2, 256],
                    )
                    .unwrap();
                    stream.synchronize().unwrap();
                };
                run(&mut stream);
                let expected = support::read(&mut stream, out.binding());
                for (g, pair) in expected.as_chunks::<8>().0.iter().enumerate() {
                    let mean = f32::from_le_bytes(pair[..4].try_into().unwrap()) as f64;
                    let variance = f32::from_le_bytes(pair[4..].try_into().unwrap()) as f64;
                    assert!(
                        (mean - means[g]).abs() < 5e-5,
                        "mean group {g}: {mean} != {}",
                        means[g]
                    );
                    assert!(
                        (variance - variances[g]).abs() < 1e-4 * variances[g] + 1e-7,
                        "variance group {g}: {variance} != {}",
                        variances[g]
                    );
                }
                b.iter(|| run(&mut stream));
                let actual = support::read(&mut stream, out.binding());
                assert!(
                    actual == expected,
                    "GroupNorm replay changed for rows={rows}, lanes={lanes}"
                );
            });
        }
    }
}

fn convolutions(c: &mut Criterion) {
    for add in [false, true] {
        convolution_cases(c, add);
    }
}

fn convolution_cases(c: &mut Criterion, add: bool) {
    use half::f16;
    for (frames, height, width) in [
        (8usize, 8usize, 8usize),
        (2, 60, 108),
        (10, 60, 108),
        (42, 60, 108),
    ] {
        let rows = frames * height * width;
        let kind = if add { "conv3d_residual" } else { "conv3d" };
        c.bench_function(&format!("upscale/{kind}/{frames}x{height}x{width}"), |b| {
            let manager =
                hrx::residency::ResidencyManager::new((if add { 2 } else { 1 }) << 30).unwrap();
            let mut stream = support::stream(Some(manager.budget()));
            let compiler = support::compiler();
            let channels = 512usize;
            let k = channels * 27;
            let conv = h3_hrx::dispatch::Conv3d::build_padding(
                &compiler,
                &mut stream,
                add,
                frames,
                height,
                width,
                1,
                1,
                3,
                channels,
                channels,
                k,
                channels,
                true,
            )
            .unwrap();
            let input = support::values(rows * channels, 0.2)
                .into_iter()
                .map(f16::from_f32)
                .collect::<Vec<_>>();
            let weights = support::values(channels * k, 0.03)
                .into_iter()
                .map(f16::from_f32)
                .collect::<Vec<_>>();
            let bias = support::values(channels, 0.1);
            let x = stream.allocate_from(bytemuck::cast_slice(&input)).unwrap();
            let w = stream
                .allocate_from(bytemuck::cast_slice(&weights))
                .unwrap();
            let bias_buffer = stream.allocate_from(bytemuck::cast_slice(&bias)).unwrap();
            let out = stream.allocate(rows * channels * 2).unwrap();
            let residual_values = add.then(|| {
                support::values(rows * channels, 0.3)
                    .into_iter()
                    .map(f16::from_f32)
                    .collect::<Vec<_>>()
            });
            let residual = residual_values
                .as_ref()
                .map(|v| stream.allocate_from(bytemuck::cast_slice(v)).unwrap());
            let mut profile = h3_hrx::dispatch::Profile::from_env();
            let mut run = |stream: &mut hrx::Stream| {
                conv.run(
                    stream,
                    Some(&mut profile),
                    "upscale conv3d",
                    x.binding(),
                    w.binding(),
                    bias_buffer.binding(),
                    out.binding(),
                    residual.as_ref().map(hrx::Buffer::binding),
                )
                .unwrap();
                stream.synchronize().unwrap();
            };
            run(&mut stream);
            let expected = support::read(&mut stream, out.binding());
            // Sample interiors, temporal/spatial boundaries, and the final tile against FP64.
            for row in [0, width - 1, height * width, rows / 2 + width + 1, rows - 1] {
                let (t, y, x) = (row / (height * width), row / width % height, row % width);
                for co in [0, 255, 511] {
                    let mut sum = bias[co] as f64;
                    for dt in -1isize..=1 {
                        for dy in -1isize..=1 {
                            for dx in -1isize..=1 {
                                let (it, iy, ix) =
                                    (t as isize + dt, y as isize + dy, x as isize + dx);
                                if it < 0
                                    || iy < 0
                                    || ix < 0
                                    || it >= frames as isize
                                    || iy >= height as isize
                                    || ix >= width as isize
                                {
                                    continue;
                                }
                                let ir = (it as usize * height + iy as usize) * width + ix as usize;
                                let tap = ((dt + 1) * 9 + (dy + 1) * 3 + dx + 1) as usize;
                                for ci in 0..channels {
                                    sum += input[ir * channels + ci].to_f64()
                                        * weights[co * k + tap * channels + ci].to_f64();
                                }
                            }
                        }
                    }
                    if let Some(residual) = &residual_values {
                        sum = f16::from_f32(sum as f32).to_f64()
                            + residual[row * channels + co].to_f64();
                    }
                    let pos = (row * channels + co) * 2;
                    let got =
                        f16::from_le_bytes(expected[pos..pos + 2].try_into().unwrap()).to_f64();
                    assert!(
                        (got - sum).abs() < 0.001 + 0.001 * sum.abs(),
                        "conv3d row={row} co={co}: {got} != {sum}"
                    );
                }
            }
            assert!(expected
                .as_chunks::<2>()
                .0
                .iter()
                .all(|b| f16::from_le_bytes(*b).is_finite()));
            b.iter(|| run(&mut stream));
            let actual = support::read(&mut stream, out.binding());
            assert!(
                actual == expected,
                "convolution replay changed for {frames}x{height}x{width}"
            );
        });
    }
}

fn temporal_convolutions(c: &mut Criterion) {
    use half::f16;
    for (frames, plane) in [(8usize, 64usize), (42, 1620), (10, 6480), (42, 6480)] {
        c.bench_function(&format!("upscale/temporal/{frames}x{plane}"), |b| {
            let (channels, taps) = (512usize, 5usize);
            let count = frames * plane * channels;
            let manager = hrx::residency::ResidencyManager::new(1 << 30).unwrap();
            let mut stream = support::stream(Some(manager.budget()));
            let compiler = hrx::loom::Compiler::resolve(None).unwrap();
            let source =
                std::fs::read_to_string(support::sources().join("upscale_temporal.loom")).unwrap();
            let mut spec = hrx::loom::Specialization::new("h3_upscale_temporal");
            spec.replace_config(
                [
                    ("frames", frames),
                    ("plane", plane),
                    ("channels", channels),
                    ("taps", taps),
                ]
                .map(|(key, value)| (format!("h3.upscale_temporal.{key}"), value.to_string()))
                .into_iter()
                .collect(),
            );
            let artifact = compiler.module(&source).compile(&spec).unwrap();
            // SAFETY: checked-in source, configuration-derived bindings and launch geometry.
            let kernel = unsafe { stream.load_artifact(&artifact) }.unwrap();
            let input = support::values(count, 0.2)
                .into_iter()
                .map(f16::from_f32)
                .collect::<Vec<_>>();
            let weights = support::values(channels * taps, 0.3)
                .into_iter()
                .map(f16::from_f32)
                .collect::<Vec<_>>();
            let bias = support::values(channels, 0.1);
            let x = stream.allocate_from(bytemuck::cast_slice(&input)).unwrap();
            let w = stream
                .allocate_from(bytemuck::cast_slice(&weights))
                .unwrap();
            let bias_buffer = stream.allocate_from(bytemuck::cast_slice(&bias)).unwrap();
            let out = stream.allocate(count * 2).unwrap();
            let constants = hrx::Constants::indices(&kernel, &[count as u32]).unwrap();
            let run = |stream: &mut hrx::Stream| {
                // SAFETY: complete channel packets, exact tensor extents, disjoint output.
                unsafe {
                    stream.dispatch(
                        &kernel,
                        [count.div_ceil(1024) as u32, 1, 1],
                        [256, 1, 1],
                        &constants,
                        &[
                            x.binding(),
                            w.binding(),
                            bias_buffer.binding(),
                            out.binding(),
                        ],
                    )
                }
                .unwrap();
                stream.synchronize().unwrap();
            };
            run(&mut stream);
            let expected = support::read(&mut stream, out.binding());
            // Cover temporal padding, interior frames, spatial endpoints and packet boundaries.
            for t in [0, 1, frames / 2, frames - 2, frames - 1] {
                for p in [0, plane / 2, plane - 1] {
                    for ch in [0, 3, 4, channels / 2, channels - 1] {
                        let mut sum = bias[ch] as f64;
                        for k in 0..taps {
                            let ti = t as isize + k as isize - (taps / 2) as isize;
                            if (0..frames as isize).contains(&ti) {
                                sum += input[(ti as usize * plane + p) * channels + ch].to_f64()
                                    * weights[ch * taps + k].to_f64();
                            }
                        }
                        let i = ((t * plane + p) * channels + ch) * 2;
                        let got =
                            f16::from_le_bytes(expected[i..i + 2].try_into().unwrap()).to_f64();
                        assert!(
                            (got - sum).abs() < 0.001 + 0.001 * sum.abs(),
                            "temporal t={t} p={p} ch={ch}: {got} != {sum}"
                        );
                    }
                }
            }
            assert!(expected
                .as_chunks::<2>()
                .0
                .iter()
                .all(|v| f16::from_le_bytes(*v).is_finite()));
            drop(input);
            b.iter(|| run(&mut stream));
            let actual = support::read(&mut stream, out.binding());
            assert!(
                actual == expected,
                "temporal replay changed for {frames}x{plane}"
            );
        });
    }
}

fn groupnorm_apply(c: &mut Criterion) {
    use half::f16;
    for rows in [32usize, 64, 127, 128, 256, 512, 12960, 64800, 272160] {
        c.bench_function(&format!("upscale/groupnorm_apply/{rows}"), |b| {
            let channels = 512usize;
            let count = rows * channels;
            let tile = if count >= 65536 { 1024 } else { 256 };
            let manager = hrx::residency::ResidencyManager::new(1 << 30).unwrap();
            let mut stream = support::stream(Some(manager.budget()));
            let compiler = hrx::loom::Compiler::resolve(None).unwrap();
            let source =
                std::fs::read_to_string(support::sources().join("upscale_gn_silu.loom")).unwrap();
            let mut spec = hrx::loom::Specialization::new("h3_upscale_gn_silu");
            let mut config = [
                ("channels", channels),
                ("groups", 32),
                ("plane", rows),
                ("rows_bound", rows.div_ceil(64) * 64),
            ]
            .map(|(k, v)| (format!("h3.upscale_gn_silu.{k}"), v.to_string()))
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>();
            config.insert("h3.upscale_gn_silu.eps".into(), "0.00001".into());
            spec.replace_config(config);
            let artifact = compiler.module(&source).compile(&spec).unwrap();
            // SAFETY: checked-in source, configuration-derived bindings and launch geometry.
            let kernel = unsafe { stream.load_artifact(&artifact) }.unwrap();
            let input = support::values(count, 8.0)
                .into_iter()
                .map(f16::from_f32)
                .collect::<Vec<_>>();
            let stats = (0..32)
                .flat_map(|g| [((g * 17 % 31) as f32 - 15.0) * 0.1, 0.2 + (g % 7) as f32])
                .collect::<Vec<_>>();
            let gamma = support::values(channels, 0.3);
            let beta = support::values(channels, 0.1);
            let modulation = support::values(channels * 2, 0.2);
            let x = stream.allocate_from(bytemuck::cast_slice(&input)).unwrap();
            let st = stream.allocate_from(bytemuck::cast_slice(&stats)).unwrap();
            let ga = stream.allocate_from(bytemuck::cast_slice(&gamma)).unwrap();
            let be = stream.allocate_from(bytemuck::cast_slice(&beta)).unwrap();
            let mods = stream
                .allocate_from(bytemuck::cast_slice(&modulation))
                .unwrap();
            let out = stream.allocate(count * 2).unwrap();
            let constants = hrx::Constants::indices(&kernel, &[1]).unwrap();
            let run = |stream: &mut hrx::Stream| {
                // SAFETY: complete channel packets, exact tensor extents, disjoint output.
                unsafe {
                    stream.dispatch(
                        &kernel,
                        [count.div_ceil(tile) as u32, 1, 1],
                        [256, 1, 1],
                        &constants,
                        &[
                            x.binding(),
                            st.binding(),
                            ga.binding(),
                            be.binding(),
                            out.binding(),
                            mods.binding(),
                        ],
                    )
                }
                .unwrap();
                stream.synchronize().unwrap();
            };
            run(&mut stream);
            let expected = support::read(&mut stream, out.binding());
            let half = |v| f16::from_f32(v).to_f32();
            for row in [0, 1, rows / 2, rows - 1] {
                for ch in [0, 3, 4, 15, 16, channels / 2, channels - 1] {
                    let g = ch / 16;
                    let inv = 1.0 / (stats[g * 2 + 1] + 1e-5).sqrt();
                    let norm = half(
                        ((input[row * channels + ch].to_f32() - stats[g * 2]) * inv)
                            .mul_add(gamma[ch], beta[ch]),
                    );
                    let y =
                        half(half(norm * half(1.0 + modulation[ch])) + modulation[channels + ch]);
                    let want = half(y * (1.0 / (1.0 + (-y).exp())));
                    let i = (row * channels + ch) * 2;
                    let got = f16::from_le_bytes(expected[i..i + 2].try_into().unwrap()).to_f32();
                    assert!(
                        (got - want).abs() < 0.0001 + 0.002 * want.abs(),
                        "groupnorm apply row={row} ch={ch}: {got} != {want}"
                    );
                }
            }
            assert!(expected
                .as_chunks::<2>()
                .0
                .iter()
                .all(|v| f16::from_le_bytes(*v).is_finite()));
            drop(input);
            b.iter(|| run(&mut stream));
            let actual = support::read(&mut stream, out.binding());
            assert!(
                actual == expected,
                "groupnorm apply replay changed for {rows} rows"
            );
        });
    }
}

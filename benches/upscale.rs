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
    for cold in [false, true] {
        let mut fixture = None;
        c.bench_function(
            if cold {
                "upscale/network/cold_weights"
            } else {
                "upscale/network/warm_weights"
            },
            |b| {
                let (runtime, session) = fixture.get_or_insert_with(|| {
                    let runtime = support::Runtime::new();
                    let session = runtime.session(
                        h3_hrx::Config::default(),
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
                            .upscale_latents(&input, &shape, &settings, None)
                            .unwrap()
                    },
                    |out| {
                        assert_eq!(out.shape.size(), (128, 128));
                        assert_eq!(out.latents.audio, input.audio);
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
criterion_group! {name=benches;config=support::criterion();targets=packing,components,upscale}
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
            let mut stream = hrx::Stream::open()
                .unwrap()
                .with_memory_budget(manager.budget());
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
                        [(rows * channels).div_ceil(256) as u32, 1, 1],
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
                        (vec![input, vec![0; 256]], [1, 32, 1], 32, 1)
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
                            [(rows * channels).div_ceil(256) as u32, 1, 1],
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
        let mut stream = hrx::Stream::open()
            .unwrap()
            .with_memory_budget(manager.budget());
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

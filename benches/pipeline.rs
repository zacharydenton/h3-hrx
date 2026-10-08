#[allow(dead_code, unused_imports)]
#[path = "../src/bin/h3/media.rs"]
mod media;
#[allow(dead_code)]
#[path = "support/render.rs"]
mod render;
mod support;
use criterion::{criterion_group, criterion_main, Criterion};
use h3_hrx::ResidencyPolicy;
use std::time::{Duration, Instant};

fn pipeline(c: &mut Criterion) {
    let (profile, params) = support::params();
    let mut group = c.benchmark_group(format!(
        "render/{profile}/{:?}/{}",
        support::residency(),
        support::execution_id()
    ));
    for case in render::cases() {
        for cold in [false, true] {
            let mut state = None;
            let mode = if cold { "cold_session" } else { "warm_session" };
            group.bench_function(format!("{}/{mode}", case.name), |b| {
                let (render, runtime, session, expected) = state.get_or_insert_with(|| {
                    let render = render::Render::new(case, params, support::residency());
                    let runtime = support::Runtime::new();
                    let mut session = runtime.session(render.config.clone(), render.options);
                    let expected = render.run(&mut session).digest();
                    render.check_media();
                    (
                        render,
                        runtime,
                        if cold {
                            drop(session);
                            None
                        } else {
                            Some(session)
                        },
                        expected,
                    )
                });
                b.iter_custom(|iterations| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iterations {
                        let start = Instant::now();
                        let output = if cold {
                            let mut session =
                                runtime.session(render.config.clone(), render.options);
                            let output = render.run(&mut session);
                            drop(session); // Include fencing, model release and teardown.
                            output
                        } else {
                            render.run(session.as_mut().unwrap())
                        };
                        total += start.elapsed();
                        assert_eq!(output.digest(), *expected, "complete render replay changed");
                        render.check_media();
                        if cold {
                            assert_eq!(runtime.manager.statistics().reserved_bytes, 0);
                        }
                    }
                    total
                });
            });
        }
    }
    group.finish();
}

fn output(c: &mut Criterion) {
    let (profile, p) = support::params();
    let shape = h3_hrx::shape_for(p.height, p.width, p.frames).unwrap();
    c.bench_function(&format!("output/{profile}/wav"), |b| {
        let samples = support::values(shape.audio_samples(), 0.2);
        b.iter(|| media::wav_bytes(&samples, (samples.len() / 2) as u32));
    });
    let mut fixture = None;
    c.bench_function(&format!("output/{profile}/h264_aac_mux"), |b| {
        let (rgb, _directory, wav, mp4) = fixture.get_or_insert_with(|| {
            let directory = tempfile::tempdir().unwrap();
            let wav = directory.path().join("audio.wav");
            let mp4 = directory.path().join("clip.mp4");
            let rgb: Vec<u8> = (0..shape.video_bytes())
                .map(|i| ((i * 17 + i / 251) % 251) as u8)
                .collect();
            let samples = support::values(shape.audio_samples(), 0.2);
            std::fs::write(&wav, media::wav_bytes(&samples, (samples.len() / 2) as u32)).unwrap();
            (rgb, directory, wav, mp4)
        });
        support::measure(
            b,
            || media::mux(mp4, wav, rgb, p.width, p.height).unwrap(),
            |_| {
                assert_eq!(media::probe_size(mp4).unwrap(), (p.width, p.height));
            },
        );
    });
}

fn cli(c: &mut Criterion) {
    let (profile, params) = support::params();
    for references in [false, true] {
        let kind = if references {
            "image_audio_references"
        } else {
            "text"
        };
        let mut fixture = None;
        c.bench_function(
            &format!(
                "cli/{profile}/{}/{kind}/cold_process",
                support::execution_id()
            ),
            |b| {
                let (render, command) = fixture.get_or_insert_with(|| {
                    let case = render::cases()
                        .into_iter()
                        .find(|c| {
                            c.name
                                == if references {
                                    "image_audio_references"
                                } else {
                                    "text/res_multistep"
                                }
                        })
                        .unwrap();
                    let render = render::Render::new(case, params, ResidencyPolicy::StageScoped);
                    let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_h3"));
                    command
                        .args([
                            "--offline",
                            "--residency",
                            "stage-scoped",
                            "--memory-budget-mib",
                        ])
                        .arg((support::budget_bytes() >> 20).to_string())
                        .arg("--dit")
                        .arg(render.config.dit.as_ref().unwrap())
                        .arg("--te")
                        .arg(render.config.te.as_ref().unwrap())
                        .arg("--video-vae")
                        .arg(render.config.video_vae.as_ref().unwrap())
                        .arg("--audio-vae")
                        .arg(render.config.audio_vae.as_ref().unwrap())
                        .arg("--root")
                        .arg(env!("CARGO_MANIFEST_DIR"))
                        .arg("--width")
                        .arg(params.width.to_string())
                        .arg("--height")
                        .arg(params.height.to_string())
                        .arg("--frames")
                        .arg(params.frames.to_string())
                        .arg("--steps")
                        .arg(params.steps.to_string())
                        .arg("--seed")
                        .arg(params.seed.to_string())
                        .args(["--sampler", "res_multistep", "--attn"])
                        .arg(match support::attention().bits() {
                            8 => "i8",
                            4 => "i4",
                            _ => "f16",
                        })
                        .arg("-p")
                        .arg(support::prompt())
                        .arg("--out")
                        .arg(&render.mp4);
                    if references {
                        let image = render.directory.path().join("reference.png");
                        let rgb: Vec<u8> =
                            support::pixels(params.width as usize, params.height as usize, 1)
                                .into_iter()
                                .map(|v| (v * 255.) as u8)
                                .collect();
                        media::write_still(&image, &rgb, params.width, params.height).unwrap();
                        let audio = render.directory.path().join("reference.wav");
                        std::fs::write(&audio, media::wav_bytes(&support::values(6400, 0.2), 3200))
                            .unwrap();
                        command.arg(image).arg(audio);
                    }
                    (render, command)
                });
                support::measure(
                    b,
                    || command.output().unwrap(),
                    |output| {
                        assert!(
                            output.status.success(),
                            "{}",
                            String::from_utf8_lossy(&output.stderr)
                        );
                        render.check_media();
                    },
                );
            },
        );
    }
}
criterion_group! { name = benches; config = support::criterion(); targets = pipeline, output, cli }
criterion_main!(benches);

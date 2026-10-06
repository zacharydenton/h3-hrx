//! Complete native render, shared by latency and memory benchmarks.
use crate::{media, support};
use h3_hrx::{
    adapter::{Lora, TurboPreset},
    media_context::{Frame, Media, MediaEntry},
    CachePolicy, CacheThresholds, Clip, Config, DenoiseParams, Keyframe, LatentGrid, Latents,
    Noise, PreparedPresentation, Reference, ResidencyPolicy, Sampler, Session, SessionOptions,
    Shape, Tokenizer,
};
use std::path::PathBuf;

#[derive(Clone, Copy, PartialEq)]
pub enum Kind {
    Text,
    Keyframes,
    ImageAudio,
    VideoAudio,
    Refmod,
    Lora,
    TurboFour,
    TurboEight,
    World,
}
#[derive(Clone, Copy)]
pub struct Case {
    pub name: &'static str,
    pub kind: Kind,
    pub sampler: Sampler,
    pub cache: CachePolicy,
}
pub fn cases() -> Vec<Case> {
    let case = |name, kind| Case {
        name,
        kind,
        sampler: Sampler::ResMultistep,
        cache: CachePolicy::Off,
    };
    vec![
        case("text/res_multistep", Kind::Text),
        Case {
            sampler: Sampler::Euler,
            ..case("text/euler", Kind::Text)
        },
        case("first_last_frames", Kind::Keyframes),
        case("image_audio_references", Kind::ImageAudio),
        case("video_audio_reference", Kind::VideoAudio),
        case("refmod", Kind::Refmod),
        Case {
            cache: CachePolicy::Observe,
            ..case("cache/observe", Kind::Text)
        },
        Case {
            cache: CachePolicy::Conservative(CacheThresholds {
                conditioning: 1e6,
                audio: 1e6,
                video: 1e6,
            }),
            ..case("cache/forced_reuse", Kind::Text)
        },
        case("lora", Kind::Lora),
        case("turbo4", Kind::TurboFour),
        case("turbo8", Kind::TurboEight),
        case("world", Kind::World),
    ]
}

pub struct Render {
    pub config: Config,
    pub options: SessionOptions,
    pub params: DenoiseParams,
    pub shape: Shape,
    pub case: Case,
    tokenizer: Tokenizer,
    pixels: Vec<f32>,
    audio: Vec<f32>,
    pub directory: tempfile::TempDir,
    pub wav: PathBuf,
    pub mp4: PathBuf,
}
pub struct Output {
    pub latents: Latents,
    pub rgb: Vec<u8>,
    pub audio: Vec<f32>,
}
impl Output {
    pub fn digest(&self) -> [String; 4] {
        [
            support::digest(&self.latents.video),
            support::digest(&self.latents.audio),
            hrx::bundle::digest(&self.rgb),
            support::digest(&self.audio),
        ]
    }
}
impl Render {
    pub fn new(case: Case, mut params: DenoiseParams, residency: ResidencyPolicy) -> Self {
        params.sampler = case.sampler;
        if case.cache != CachePolicy::Off {
            params.steps = params.steps.max(9);
        }
        let references = matches!(
            case.kind,
            Kind::ImageAudio | Kind::VideoAudio | Kind::Refmod
        );
        let mut config = support::config(references);
        let mut options = SessionOptions {
            residency,
            cache: case.cache,
            ..Default::default()
        };
        if case.kind == Kind::Lora || case.kind == Kind::World {
            let key = if case.kind == Kind::World {
                "H3_BENCH_WORLD_ADAPTER"
            } else {
                "H3_BENCH_LORA"
            };
            let path = std::env::var_os(key).map(PathBuf::from).unwrap_or_else(|| {
                let (owner, repo, revision, file) = if case.kind == Kind::World {
                    (
                        "DANNY621",
                        "H3-World",
                        h3_hrx::world::ADAPTER_REVISION,
                        h3_hrx::world::ADAPTER_FILE,
                    )
                } else {
                    (
                        "pablodawson",
                        "MiniMax-H3-360-Orbit-LoRA",
                        "5ddbc2dbbe95edbbdaf5017c3e934b1d01791697",
                        "minimax_h3_flf2v_lora_v1.safetensors",
                    )
                };
                h3_hrx::models::Resolver::new()
                    .repository(owner, repo)
                    .revision(Some(revision.into()))
                    .find(file)
                    .unwrap()
            });
            assert!(path.is_file(), "missing adapter {}", path.display());
            config.loras = vec![Lora::new(path, 1.0)];
        }
        if matches!(case.kind, Kind::TurboFour | Kind::TurboEight) {
            let preset = if case.kind == Kind::TurboFour {
                TurboPreset::Four
            } else {
                TurboPreset::Eight
            };
            preset
                .resolve(false)
                .expect("resolve the pinned Turbo adapter before benchmarking");
            preset.configure(&mut params);
            options.turbo = Some(preset);
        }
        if case.kind == Kind::World {
            params.sampler = Sampler::Euler;
        }
        let shape = h3_hrx::shape_for(params.height, params.width, params.frames).unwrap();
        let directory = tempfile::tempdir().unwrap();
        let wav = directory.path().join("render.wav");
        let mp4 = directory.path().join("render.mp4");
        let pixels = support::pixels(
            params.width as usize,
            params.height as usize,
            if case.kind == Kind::VideoAudio { 5 } else { 1 },
        );
        let audio = support::values(6400, 0.2);
        Self {
            config,
            options,
            params,
            shape,
            case,
            tokenizer: Tokenizer::new().unwrap(),
            pixels,
            audio,
            directory,
            wav,
            mp4,
        }
    }

    pub fn run(&self, session: &mut Session) -> Output {
        let (w, h) = (self.params.width as usize, self.params.height as usize);
        let frame = || Frame {
            pixels: self.pixels[..w * h * 3].to_vec().into(),
            width: w,
            height: h,
        };
        let entry = |media, role: &str| MediaEntry {
            media,
            role: role.into(),
            metadata: Default::default(),
        };
        let mut entries = Vec::new();
        let mut keyframes = Vec::new();
        let mut references = Vec::new();
        let mut visual = None;
        let mut audio = None;
        if matches!(
            self.case.kind,
            Kind::Keyframes | Kind::ImageAudio | Kind::VideoAudio | Kind::Lora | Kind::World
        ) {
            visual = Some(
                session
                    .encode_video(Clip {
                        pixels: &self.pixels,
                        width: w,
                        height: h,
                        frames: if self.case.kind == Kind::VideoAudio {
                            5
                        } else {
                            1
                        },
                    })
                    .unwrap(),
            );
        }
        if matches!(self.case.kind, Kind::ImageAudio | Kind::VideoAudio) {
            audio = Some(
                session
                    .encode_audio(&self.audio, self.audio.len() / 2)
                    .unwrap(),
            );
        }
        if let Some((z, t)) = &visual {
            let grid = LatentGrid {
                frames: *t,
                height: h / 16,
                width: w / 16,
            };
            match self.case.kind {
                Kind::ImageAudio => {
                    references.push(Reference::Image {
                        latents: z,
                        grid,
                        presented: None,
                    });
                    entries.push(entry(Media::Picture(frame()), "reference"));
                    let (audio, t) = audio.as_ref().unwrap();
                    references.push(Reference::Audio {
                        latents: audio,
                        frames: *t,
                    });
                    entries.push(entry(Media::Audio(self.audio.clone().into()), "reference"));
                }
                Kind::VideoAudio => {
                    let (audio, t) = audio.as_ref().unwrap();
                    references.push(Reference::Video {
                        latents: z,
                        grid,
                        audio: Some((audio, *t)),
                    });
                    let second = Frame {
                        pixels: self.pixels[4 * w * h * 3..5 * w * h * 3].to_vec().into(),
                        width: w,
                        height: h,
                    };
                    entries.push(entry(
                        Media::Video(vec![(0., frame()), (4. / 24., second)]),
                        "reference",
                    ));
                }
                _ => {
                    keyframes.push(Keyframe {
                        frame_index: 0,
                        latents: z,
                        presented: None,
                        audio: None,
                    });
                    entries.push(entry(Media::Picture(frame()), "first_frame"));
                    if matches!(self.case.kind, Kind::Keyframes | Kind::Lora) {
                        keyframes.push(Keyframe {
                            frame_index: self.shape.frames - 1,
                            latents: z,
                            presented: None,
                            audio: None,
                        });
                        entries.push(entry(Media::Picture(frame()), "last_frame"));
                    }
                }
            }
        }
        let refmod = if self.case.kind == Kind::Refmod {
            Some(
                h3_hrx::refmod::RefMod::load(
                    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                        .join("tests/fixtures/refmod/combined.safetensors"),
                )
                .unwrap(),
            )
        } else {
            None
        };
        let prepared = refmod
            .as_ref()
            .map(|r| r.prepare(h3_hrx::refmod::ApplyOptions::default()).unwrap());
        if let Some(r) = &prepared {
            for reference in r.references() {
                match reference {
                    Reference::Video { latents, grid, .. } => {
                        let (pixels, frames) =
                            session.decode_reference_visual(grid, latents).unwrap();
                        let width = grid.width * 16;
                        let height = grid.height * 16;
                        let stride = width * height * 3;
                        let sampled = h3_hrx::media_context::video_sample_indices(frames, 24.)
                            .unwrap()
                            .into_iter()
                            .map(|(time, index)| {
                                (
                                    time,
                                    Frame {
                                        pixels: pixels[index * stride..(index + 1) * stride]
                                            .to_vec()
                                            .into(),
                                        width,
                                        height,
                                    },
                                )
                            })
                            .collect();
                        entries.push(entry(Media::Video(sampled), "reference"));
                    }
                    Reference::Audio { latents, frames } => {
                        let mut samples = vec![0.; 2 * frames * h3_hrx::avae::HOP];
                        session.decode_audio(latents, frames, &mut samples).unwrap();
                        entries.push(entry(Media::Audio(samples.into()), "reference"));
                    }
                    Reference::Image { .. } => {
                        unreachable!("the fixture contains a video and audio member")
                    }
                }
                references.push(reference);
            }
        }
        let presentation =
            PreparedPresentation::new(&self.tokenizer, &entries, support::PROMPT, &self.shape)
                .unwrap();
        let latents = if self.case.kind == Kind::World {
            let schedule =
                h3_hrx::ActionSchedule::preset("pan-left", self.shape.frames as usize).unwrap();
            let request =
                h3_hrx::WorldRequest::new(&self.tokenizer, &schedule, self.shape.frames as usize)
                    .unwrap();
            session
                .denoise_world(
                    &presentation,
                    &request,
                    &self.params,
                    Noise::default(),
                    &keyframes[0],
                    None,
                )
                .unwrap()
        } else {
            session
                .denoise_presented(
                    &presentation,
                    &self.params,
                    Noise::default(),
                    &references,
                    &keyframes,
                    None,
                )
                .unwrap()
        };
        let mut rgb = vec![0; self.shape.video_bytes()];
        let mut audio = vec![0.; self.shape.audio_samples()];
        session
            .decode_video(&self.shape, &latents.video, &mut rgb)
            .unwrap();
        session
            .decode_audio(&latents.audio, self.shape.audio_t as usize, &mut audio)
            .unwrap();
        std::fs::write(
            &self.wav,
            media::wav_bytes(&audio, (audio.len() / 2) as u32),
        )
        .unwrap();
        media::mux(
            &self.mp4,
            &self.wav,
            &rgb,
            self.params.width,
            self.params.height,
        )
        .unwrap();
        Output {
            latents,
            rgb,
            audio,
        }
    }
    pub fn check_media(&self) {
        let result = std::process::Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-count_frames",
                "-show_streams",
                "-of",
                "json",
            ])
            .arg(&self.mp4)
            .output()
            .unwrap();
        assert!(result.status.success());
        let info: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
        let streams = info["streams"].as_array().unwrap();
        let video = streams
            .iter()
            .find(|s| s["codec_type"] == "video")
            .expect("missing video stream");
        assert_eq!(video["width"], self.params.width);
        assert_eq!(video["height"], self.params.height);
        assert_eq!(
            video["nb_read_frames"]
                .as_str()
                .unwrap()
                .parse::<i32>()
                .unwrap(),
            // The production muxer uses -shortest. Audio lengths are rounded to
            // 800-sample hops; a five-frame clip has only four complete 24 fps
            // frame intervals within its 6400 samples.
            self.shape.frames.min(
                (self.shape.audio_t as u64 * h3_hrx::avae::HOP as u64 * media::FPS as u64
                    / media::RATE as u64) as i32
            )
        );
        assert!(streams.iter().any(|s| s["codec_type"] == "audio"));
    }
}

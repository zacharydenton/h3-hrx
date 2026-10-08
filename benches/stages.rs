mod support;
use criterion::{criterion_group, criterion_main, Criterion};
use h3_hrx::{Clip, Session, Shape, Tokenizer};
use support::{measure, Runtime};

#[derive(Clone, Copy)]
enum Stage {
    Text,
    Vision,
    VideoEncode,
    VideoDecode,
    AudioEncode,
    AudioDecode,
}
enum Output {
    Floats(Vec<f32>),
    Bytes(Vec<u8>),
}
impl Output {
    fn digest(&self) -> String {
        match self {
            Self::Floats(v) => support::digest(v),
            Self::Bytes(v) => {
                assert!(!v.is_empty());
                hrx::bundle::digest(v)
            }
        }
    }
}
struct Fixture {
    session: Session,
    _runtime: Runtime,
    stage: Stage,
    shape: Shape,
    input: Vec<f32>,
    ids: Vec<i32>,
    length: usize,
}
impl Fixture {
    fn new(stage: Stage, width: i32, height: i32, length: usize) -> Self {
        let runtime = Runtime::new();
        let session = runtime.warm_session(false);
        let shape = h3_hrx::shape_for(height, width, length.min(124) as i32).unwrap();
        let input = match stage {
            Stage::Text => vec![],
            Stage::Vision => support::pixels(width as usize, height as usize, 1),
            Stage::VideoEncode => support::pixels(width as usize, height as usize, length),
            Stage::VideoDecode => support::values(
                24 * shape.latent_t as usize * shape.lat_h as usize * shape.lat_w as usize,
                0.5,
            ),
            Stage::AudioEncode => support::values(2 * length, 0.2),
            Stage::AudioDecode => support::values(64 * length, 0.5),
        };
        let tokens = Tokenizer::new().unwrap().encode(support::prompt()).unwrap();
        let ids = if matches!(stage, Stage::Text) {
            tokens.into_iter().cycle().take(length).collect()
        } else {
            Vec::new()
        };
        Self {
            session,
            _runtime: runtime,
            stage,
            shape,
            input,
            ids,
            length,
        }
    }
    fn run(&mut self) -> Output {
        let (w, h) = self.shape.size();
        let result = match self.stage {
            Stage::Text => {
                let mut out = vec![0.; self.ids.len() * h3_hrx::model::HID];
                self.session.text_in(&self.ids, &mut out).unwrap();
                out
            }
            Stage::Vision => {
                let mut embedding = self
                    .session
                    .vision_embed(&self.input, h as usize, w as usize)
                    .unwrap();
                assert_eq!(embedding.tokens, (w * h / 1024) as usize);
                embedding.merged.append(&mut embedding.deepstack);
                embedding.merged
            }
            Stage::VideoEncode => {
                self.session
                    .encode_video(Clip {
                        pixels: &self.input,
                        width: w as usize,
                        height: h as usize,
                        frames: self.length,
                    })
                    .unwrap()
                    .0
            }
            Stage::VideoDecode => {
                let mut out = vec![0; self.shape.video_bytes()];
                self.session
                    .decode_video(&self.shape, &self.input, &mut out)
                    .unwrap();
                return Output::Bytes(out);
            }
            Stage::AudioEncode => {
                self.session
                    .encode_audio(&self.input, self.length)
                    .unwrap()
                    .0
            }
            Stage::AudioDecode => {
                let mut out = vec![0.; 2 * self.length * h3_hrx::avae::HOP];
                self.session
                    .decode_audio(&self.input, self.length, &mut out)
                    .unwrap();
                out
            }
        };
        Output::Floats(result)
    }
}
fn stages(c: &mut Criterion) {
    let (profile, params) = support::params();
    let mut cases = vec![
        ("text_refiner/32_tokens".into(), Stage::Text, 64, 64, 32),
        ("text_refiner/512_tokens".into(), Stage::Text, 64, 64, 512),
        ("vision/64x64".into(), Stage::Vision, 64, 64, 1),
        (
            format!("vision/{}x{}", params.width, params.height),
            Stage::Vision,
            params.width,
            params.height,
            1,
        ),
    ];
    for frames in [1, 5, 22] {
        cases.push((
            format!("video_encode/64x64x{frames}"),
            Stage::VideoEncode,
            64,
            64,
            frames,
        ));
    }
    for (w, h, frames) in [
        (64, 64, 5),
        (320, 320, 22),
        (320, 320, 39),
        (320, 320, 56),
        (512, 256, 22),
        (256, 512, 22),
        (params.width, params.height, params.frames as usize),
    ] {
        cases.push((
            format!("video_decode/{w}x{h}x{frames}"),
            Stage::VideoDecode,
            w,
            h,
            frames,
        ));
    }
    for n in [800, 801, 3200, 165600] {
        cases.push((format!("audio_encode/{n}"), Stage::AudioEncode, 64, 64, n));
    }
    for n in [1, 207, 255, 256, 257] {
        cases.push((format!("audio_decode/{n}"), Stage::AudioDecode, 64, 64, n));
    }
    // Profile smoke duplicates the minimum decoder case; keep unique benchmark IDs.
    cases.sort_by(|a, b| a.0.cmp(&b.0));
    cases.dedup_by(|a, b| a.0 == b.0);
    let mut group = c.benchmark_group(format!(
        "stages/{profile}/{:?}/{}",
        support::residency(),
        support::execution_id()
    ));
    for (name, stage, w, h, length) in cases {
        let mut fixture = None;
        group.bench_function(&name, |b| {
            let (fixture, expected) = fixture.get_or_insert_with(|| {
                let mut fixture = Fixture::new(stage, w, h, length);
                let expected = fixture.run().digest();
                if std::env::var_os("H3_BENCH_DETAILS").is_some() {
                    eprintln!("{name} output sha256: {expected}");
                }
                (fixture, expected)
            });
            measure(
                b,
                || fixture.run(),
                |out| assert_eq!(out.digest(), *expected, "stage replay changed"),
            );
        });
    }
    group.finish();
}
fn denoise(c: &mut Criterion) {
    let (profile, mut params) = support::params();
    let mut group = c.benchmark_group(format!(
        "denoise/{profile}/{:?}/{}",
        support::residency(),
        support::execution_id()
    ));
    for (name, sampler) in [
        ("euler", h3_hrx::Sampler::Euler),
        ("res_multistep", h3_hrx::Sampler::ResMultistep),
    ] {
        params.sampler = sampler;
        let mut state = None;
        group.bench_function(name, |b| {
            let (_runtime, session, ids, expected) = state.get_or_insert_with(|| {
                let runtime = Runtime::new();
                let mut session = runtime.warm_session(false);
                let ids = Tokenizer::new().unwrap().encode(support::prompt()).unwrap();
                let out = session
                    .denoise(&ids, &params, h3_hrx::Noise::default(), &[], &[], None)
                    .unwrap();
                let expected = (support::digest(&out.video), support::digest(&out.audio));
                (runtime, session, ids, expected)
            });
            measure(
                b,
                || {
                    session
                        .denoise(ids, &params, h3_hrx::Noise::default(), &[], &[], None)
                        .unwrap()
                },
                |out| {
                    assert_eq!(
                        (support::digest(&out.video), support::digest(&out.audio)),
                        *expected
                    );
                },
            );
        });
    }
    group.finish();
}
criterion_group! { name = benches; config = support::criterion(); targets = stages, denoise }
criterion_main!(benches);

//! Runs one or more denoising cases from a manifest, in a single process.
//!
//!   denoise_cases <dit> <te> <manifest>
//!
//! Opening the two checkpoints costs far more than a short run, so the manifest exists to amortise it
//! across a comparison sweep. It is one `key value` pair per line; a `case` line starts a new case and
//! a blank line is ignored:
//!
//!   ids <path>            int32 token ids; before any case it is the default, inside one it is
//!                         that case's own — a prompt with image placeholders is not the same prompt
//!   case <name>           starts a case, and names its outputs <name>.video / <name>.audio
//!   size <h> <w> <frames>
//!   steps <n> <sampler> <seed> <cache_threshold>
//!   attn <16|8|4>         the QK operands' width, 8 by default
//!   noise <video.f32> <audio.f32>
//!   ref <kind> <video.f32|-> <t> <h> <w> <audio.f32|-> <audio_t> <pixels.f32|-> <ph> <pw>
//!   keyframe <index> <video.f32> <audio.f32|-> <audio_t> <pixels.f32|-> <ph> <pw>
//!   out <dir>
use h3_hrx::compile::Compiler;
use h3_hrx::dispatch::Profile;
use h3_hrx::dit::{
    Attention, DenoiseParams, Dit, Keyframe, LatentGrid, Noise, Presented, Reference, Sampler,
};
use h3_hrx::te::TextEncoder;
use std::io::Write;

fn read(path: &str) -> Vec<f32> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("{path}: {e}"))
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// `-` means absent, anything else is a path.
fn maybe(path: &str) -> Option<Vec<f32>> {
    (path != "-").then(|| read(path))
}

fn write(path: &str, v: &[f32]) {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).expect("output"));
    for x in v {
        f.write_all(&x.to_le_bytes()).unwrap();
    }
    f.flush().unwrap();
}

/// A reference as the manifest gives it: `RefInput`'s fields, owning their buffers.
struct OwnedRef {
    kind: i32,
    video: Option<Vec<f32>>,
    latent_t: i32,
    lat_h: i32,
    lat_w: i32,
    audio: Option<Vec<f32>>,
    audio_t: i32,
    pixels: Option<Vec<f32>>,
    height: i32,
    width: i32,
}

/// A keyframe as the manifest gives it.
struct OwnedKeyframe {
    frame_index: i32,
    video: Vec<f32>,
    audio: Option<Vec<f32>>,
    audio_t: i32,
    pixels: Option<Vec<f32>>,
    height: i32,
    width: i32,
}

#[derive(Default)]
struct Case {
    name: String,
    ids: Vec<i32>,
    /// int8 by default, as `Attention::default()` is
    attention: Attention,
    p: DenoiseParams,
    noise: Option<(Vec<f32>, Vec<f32>)>,
    refs: Vec<OwnedRef>,
    kfs: Vec<OwnedKeyframe>,
}

pub fn run(args: Vec<String>) {
    let a = args;
    let text = std::fs::read_to_string(&a[2]).expect("manifest");

    let mut ids: Vec<i32> = Vec::new();
    let mut out_dir = ".".to_string();
    let mut cases: Vec<Case> = Vec::new();
    for line in text.lines() {
        let f: Vec<&str> = line.split_ascii_whitespace().collect();
        if f.is_empty() {
            continue;
        }
        match f[0] {
            "ids" => {
                let v: Vec<i32> = std::fs::read(f[1])
                    .expect("ids")
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| i32::from_le_bytes(*c))
                    .collect();
                match cases.last_mut() {
                    Some(c) => c.ids = v,
                    None => ids = v,
                }
            }
            "out" => out_dir = f[1].to_string(),
            "attn" => {
                cases.last_mut().expect("a case first").attention =
                    Attention::from_bits(f[1].parse().unwrap()).expect("16, 8 or 4")
            }
            "case" => cases.push(Case {
                name: f[1].into(),
                ids: ids.clone(),
                ..Case::default()
            }),
            "size" => {
                let c = cases.last_mut().expect("a case first");
                c.p.height = f[1].parse().unwrap();
                c.p.width = f[2].parse().unwrap();
                c.p.frames = f[3].parse().unwrap();
            }
            "steps" => {
                let c = cases.last_mut().expect("a case first");
                c.p.steps = f[1].parse().unwrap();
                c.p.sampler = if f[2] == "0" {
                    Sampler::Euler
                } else {
                    Sampler::ResMultistep
                };
                c.p.seed = f[3].parse().unwrap();
                c.p.cache_threshold = f[4].parse().unwrap();
            }
            "noise" => {
                let c = cases.last_mut().expect("a case first");
                c.noise = Some((read(f[1]), read(f[2])));
            }
            "ref" => {
                let c = cases.last_mut().expect("a case first");
                c.refs.push(OwnedRef {
                    kind: f[1].parse().unwrap(),
                    video: maybe(f[2]),
                    latent_t: f[3].parse().unwrap(),
                    lat_h: f[4].parse().unwrap(),
                    lat_w: f[5].parse().unwrap(),
                    audio: maybe(f[6]),
                    audio_t: f[7].parse().unwrap(),
                    pixels: maybe(f[8]),
                    height: f[9].parse().unwrap(),
                    width: f[10].parse().unwrap(),
                });
            }
            "keyframe" => {
                let c = cases.last_mut().expect("a case first");
                c.kfs.push(OwnedKeyframe {
                    frame_index: f[1].parse().unwrap(),
                    video: read(f[2]),
                    audio: maybe(f[3]),
                    audio_t: f[4].parse().unwrap(),
                    pixels: maybe(f[5]),
                    height: f[6].parse().unwrap(),
                    width: f[7].parse().unwrap(),
                });
            }
            other => panic!("unknown manifest key {other}"),
        }
    }

    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let exe = std::env::var_os("HRX_LOOM_LIBRARY").map(std::path::PathBuf::from);
    let mut stream = hrx::Stream::open().expect("stream");
    let compiler = Compiler::new(exe, root.join("kernels"));
    // Safety: a diagnostic run over checkpoints the operator named and is not writing to.
    let mut dit = unsafe { Dit::open(&mut stream, &a[0]) }.expect("DiT checkpoint");
    let mut te = unsafe { TextEncoder::open(&mut stream, &a[1]) }.expect("text encoder checkpoint");
    let mut prof = Profile::from_env();

    for case in &cases {
        let refs: Vec<Reference<'_>> = case
            .refs
            .iter()
            .map(|r| {
                let grid = LatentGrid {
                    frames: r.latent_t.max(0) as usize,
                    height: r.lat_h.max(0) as usize,
                    width: r.lat_w.max(0) as usize,
                };
                let presented = r.pixels.as_deref().map(|pixels| Presented {
                    pixels,
                    height: r.height as usize,
                    width: r.width as usize,
                });
                let audio = r.audio.as_deref().map(|z| (z, r.audio_t.max(0) as usize));
                match r.kind {
                    0 => Reference::Image {
                        latents: r.video.as_deref().expect("image latents"),
                        grid: LatentGrid { frames: 1, ..grid },
                        presented,
                    },
                    1 => {
                        let (latents, frames) = audio.expect("audio latents");
                        Reference::Audio { latents, frames }
                    }
                    _ => Reference::Video {
                        latents: r.video.as_deref().expect("video latents"),
                        grid,
                        audio,
                    },
                }
            })
            .collect();
        let kfs: Vec<Keyframe<'_>> = case
            .kfs
            .iter()
            .map(|k| Keyframe {
                frame_index: k.frame_index,
                latents: &k.video,
                audio: k.audio.as_deref().map(|z| (z, k.audio_t.max(0) as usize)),
                presented: k.pixels.as_deref().map(|pixels| Presented {
                    pixels,
                    height: k.height as usize,
                    width: k.width as usize,
                }),
            })
            .collect();
        let noise = match &case.noise {
            Some((v, a)) => Noise {
                video: Some(v),
                audio: Some(a),
            },
            None => Noise::default(),
        };
        prof.take();
        let start = std::time::Instant::now();
        let out = dit
            .denoise(
                &mut stream,
                &compiler,
                &mut prof,
                &mut te,
                &case.ids,
                &case.p,
                case.attention.bits(),
                noise,
                &refs,
                &kfs,
                None,
            )
            .unwrap_or_else(|e| panic!("{}: {e}", case.name));
        eprintln!(
            "{}: {} steps at {}x{} in {:.1}s",
            case.name,
            case.p.steps,
            case.p.height,
            case.p.width,
            start.elapsed().as_secs_f64()
        );
        if prof.on {
            eprintln!("{}", prof.report(0.01));
        }
        write(&format!("{out_dir}/{}.video", case.name), &out.video);
        write(&format!("{out_dir}/{}.audio", case.name), &out.audio);
    }
}

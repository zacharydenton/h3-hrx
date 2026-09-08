//! Runs one or more denoising cases from a manifest, in a single process.
//!
//!   denoise_cases <dit> <te> <manifest>
//!
//! Opening the two checkpoints costs far more than a short run, so the manifest exists to amortise it
//! across a comparison sweep. It is one `key value` pair per line; a `case` line starts a new case and
//! a blank line is ignored:
//!
//!   ids <path>            int32 token ids
//!   case <name>           starts a case, and names its outputs <name>.video / <name>.audio
//!   size <h> <w> <frames>
//!   steps <n> <sampler> <seed> <cache_threshold>
//!   noise <video.f32> <audio.f32>
//!   ref <kind> <video.f32|-> <t> <h> <w> <audio.f32|-> <audio_t> <pixels.f32|-> <ph> <pw>
//!   keyframe <index> <video.f32> <audio.f32|-> <audio_t> <pixels.f32|-> <ph> <pw>
//!   out <dir>
use h3::compile::Compiler;
use h3::dispatch::Profile;
use h3::dit::{DenoiseParams, Dit, KeyframeInput, Noise, RefInput};
use h3::te::TextEncoder;
use std::io::Write;

fn read(path: &str) -> Vec<f32> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("{path}: {e}"))
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
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
    p: DenoiseParams,
    noise: Option<(Vec<f32>, Vec<f32>)>,
    refs: Vec<OwnedRef>,
    kfs: Vec<OwnedKeyframe>,
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
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
                ids = std::fs::read(f[1])
                    .expect("ids")
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                    .collect()
            }
            "out" => out_dir = f[1].to_string(),
            "case" => cases.push(Case {
                name: f[1].into(),
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
                c.p.sampler = f[2].parse().unwrap();
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

    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let exe = std::env::var("LOOM_COMPILE").unwrap_or_else(|_| "loom-compile".into());
    let gpu = hrx::Gpu::open().expect("gpu");
    let compiler = Compiler::new(exe, root.join("kernels"), root.join("build/kernel_cache"));
    let mut dit = Dit::open(&gpu, &a[0]).expect("DiT checkpoint");
    let mut te = TextEncoder::open(&gpu, &a[1]).expect("text encoder checkpoint");
    let mut prof = Profile::from_env();

    for case in &cases {
        let refs: Vec<RefInput<'_>> = case
            .refs
            .iter()
            .map(|r| RefInput {
                kind: r.kind,
                video_latent: r.video.as_deref(),
                latent_t: r.latent_t,
                lat_h: r.lat_h,
                lat_w: r.lat_w,
                audio_latent: r.audio.as_deref(),
                audio_t: r.audio_t,
                pixels: r.pixels.as_deref(),
                height: r.height,
                width: r.width,
            })
            .collect();
        let kfs: Vec<KeyframeInput<'_>> = case
            .kfs
            .iter()
            .map(|k| KeyframeInput {
                frame_index: k.frame_index,
                video_latent: &k.video,
                audio_latent: k.audio.as_deref(),
                audio_t: k.audio_t,
                pixels: k.pixels.as_deref(),
                height: k.height,
                width: k.width,
            })
            .collect();
        let noise = match &case.noise {
            Some((v, a)) => Noise {
                video: Some(v),
                audio: Some(a),
            },
            None => Noise::default(),
        };
        let start = std::time::Instant::now();
        let out = dit
            .denoise(
                &gpu, &compiler, &mut prof, &mut te, &ids, &case.p, noise, &refs, &kfs, None,
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
        write(&format!("{out_dir}/{}.video", case.name), &out.video);
        write(&format!("{out_dir}/{}.audio", case.name), &out.audio);
    }
}

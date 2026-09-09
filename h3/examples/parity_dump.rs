//! Host-side artefacts for `scripts/parity.py`, written as plain files.
//!
//! The parity checks compare this host against MiniMax's released weights through diffusers and
//! transformers, so the oracle has to be Python. The host does not: this produces everything those
//! checks need — token ids, refined text rows, latents, decoded frames, and the per-block dumps
//! `H3_DUMP_BLOCKS` writes — and Python only reads files. That keeps the comparison honest while
//! leaving no foreign-function boundary to drift: the ABI mismatch that used to break the harness
//! was a ctypes struct tracking a C ABI, and there is no longer either.
//!
//!   parity_dump shape   --height H --width W --frames F
//!   parity_dump text    --prompt P --out DIR
//!   parity_dump encode  --pixels F --height H --width W --out DIR
//!   parity_dump decode  --latents F --height H --width W --frames F --out DIR
//!   parity_dump denoise --prompt P --height H --width W --frames F --steps N --seed S
//!                       [--sampler euler|res_multistep] [--noise-video F --noise-audio F]
//!                       [--keyframe F --keyframe-latents F --keyframe-latent-t N] --out DIR
//!
//! Scalars go to `<out>/shape.json` so Python needs no arithmetic of its own; everything else is
//! little-endian f32 except `ids.i32` and `frames.rgb`.
use h3::{shape_for, Clip, Config, DenoiseParams, Keyframe, Noise, Sampler, Session, Tokenizer};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const VISION_START: i32 = 151652;
const VISION_END: i32 = 151653;

fn fail(message: &str) -> ! {
    eprintln!("parity_dump: {message}");
    std::process::exit(2);
}

/// `--flag value` pairs; every command below takes its arguments this way.
struct Args(HashMap<String, String>);

impl Args {
    fn parse(argv: &[String]) -> Self {
        let mut map = HashMap::new();
        let mut i = 0;
        while i < argv.len() {
            let Some(key) = argv[i].strip_prefix("--") else {
                fail(&format!("expected --flag, found {:?}", argv[i]));
            };
            let Some(value) = argv.get(i + 1) else {
                fail(&format!("--{key} needs a value"));
            };
            map.insert(key.to_string(), value.clone());
            i += 2;
        }
        Self(map)
    }
    fn get(&self, key: &str) -> &str {
        self.0
            .get(key)
            .unwrap_or_else(|| fail(&format!("--{key} is required")))
    }
    fn opt(&self, key: &str) -> Option<&str> {
        self.0.get(key).map(String::as_str)
    }
    fn num<T: std::str::FromStr>(&self, key: &str) -> T {
        self.get(key)
            .parse()
            .unwrap_or_else(|_| fail(&format!("--{key} is not a number")))
    }
    fn num_or<T: std::str::FromStr>(&self, key: &str, default: T) -> T {
        match self.opt(key) {
            Some(v) => v
                .parse()
                .unwrap_or_else(|_| fail(&format!("--{key} is not a number"))),
            None => default,
        }
    }
}

fn read_f32(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| fail(&format!("{path}: {e}")));
    if !bytes.len().is_multiple_of(4) {
        fail(&format!("{path} is not a whole number of f32"));
    }
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().expect("four bytes")))
        .collect()
}

fn write(dir: &Path, name: &str, bytes: &[u8]) {
    std::fs::create_dir_all(dir).unwrap_or_else(|e| fail(&format!("{}: {e}", dir.display())));
    std::fs::write(dir.join(name), bytes)
        .unwrap_or_else(|e| fail(&format!("{}/{name}: {e}", dir.display())));
}

fn write_f32(dir: &Path, name: &str, values: &[f32]) {
    write(dir, name, bytemuck::cast_slice(values));
}

/// The presentation the model is trained on, built the way `cli/` builds it: a vision span per
/// keyframe and reference image, then the audio labels, then the prompt. Doing it here rather than
/// in Python means the checks exercise the host's own presentation.
fn presentation(prompt: &str, spans: &[(i32, i32)], audios: usize) -> h3::Result<Vec<i32>> {
    let tok = Tokenizer::new()?;
    let mut ids: Vec<i32> = Vec::new();
    for (i, (height, width)) in spans.iter().enumerate() {
        tok.encode_into(&format!("<Picture {}>: ", i + 1), &mut ids)?;
        ids.push(VISION_START);
        ids.extend(std::iter::repeat_n(
            -1,
            (height / 32) as usize * (width / 32) as usize,
        ));
        ids.push(VISION_END);
    }
    for j in 0..audios {
        tok.encode_into(&format!("<Audio {}>: ", j + 1), &mut ids)?;
    }
    tok.encode_into(prompt, &mut ids)?;
    Ok(ids)
}

fn config(a: &Args) -> Config {
    let mut config = Config::default();
    if let Some(dit) = a.opt("dit") {
        config.dit = Some(PathBuf::from(dit));
    }
    if let Some(te) = a.opt("te") {
        config.te = Some(PathBuf::from(te));
    }
    if let Some(bits) = a.opt("attn") {
        config.attention = match bits {
            "f16" | "16" => h3::Attention::F16,
            "i4" | "4" => h3::Attention::I4,
            _ => h3::Attention::I8,
        };
    }
    config
}

/// Safety: the caller owns these checkpoints and does not write to them while the session lives.
unsafe fn open(a: &Args) -> Session {
    unsafe { Session::new(config(a)) }.unwrap_or_else(|e| fail(&format!("session: {e}")))
}

fn shape_json(p: &DenoiseParams) -> String {
    let s = shape_for(p.height, p.width, p.frames)
        .unwrap_or_else(|| fail("no shape for that canvas and frame count"));
    format!(
        "{{\"frames\": {}, \"latent_t\": {}, \"lat_h\": {}, \"lat_w\": {}, \"audio_t\": {}, \"text_rows_max\": {}}}\n",
        s.frames, s.latent_t, s.lat_h, s.lat_w, s.audio_t, s.text_rows_max
    )
}

fn params(a: &Args) -> DenoiseParams {
    DenoiseParams {
        height: a.num("height"),
        width: a.num("width"),
        frames: a.num("frames"),
        steps: a.num_or("steps", 2),
        seed: a.num_or("seed", 1),
        sampler: match a.opt("sampler") {
            Some("euler") | Some("0") => Sampler::Euler,
            _ => Sampler::ResMultistep,
        },
        ..DenoiseParams::default()
    }
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let Some(command) = argv.first().cloned() else {
        fail("a command is required: shape, text, encode, decode or denoise");
    };
    let a = Args::parse(&argv[1..]);
    let out = || PathBuf::from(a.get("out"));
    // `Stack::dump` discards its write error, so a missing directory loses the dumps silently.
    if let Some(dir) = std::env::var_os("H3_DUMP_BLOCKS") {
        std::fs::create_dir_all(&dir)
            .unwrap_or_else(|e| fail(&format!("{}: {e}", Path::new(&dir).display())));
    }

    match command.as_str() {
        // Sizes, so Python needs none of the model's arithmetic.
        "shape" => print!("{}", shape_json(&params(&a))),

        // The refined text rows, and the `te_*` dumps when H3_DUMP_BLOCKS is set.
        "text" => {
            let dir = out();
            let ids =
                presentation(a.get("prompt"), &[], 0).unwrap_or_else(|e| fail(&format!("{e}")));
            let mut session = unsafe { open(&a) };
            // the refined rows are at the DiT's hidden width, not the encoder's
            let mut rows = vec![0f32; ids.len() * h3::model::HID];
            session
                .text_in(&ids, &mut rows)
                .unwrap_or_else(|e| fail(&format!("text_in: {e}")));
            write(&dir, "ids.i32", bytemuck::cast_slice(&ids));
            write_f32(&dir, "text_in.f32", &rows);
        }

        // One frame of pixels to its video latents, for a keyframe or a reference image.
        "encode" => {
            let dir = out();
            let (height, width): (i32, i32) = (a.num("height"), a.num("width"));
            let pixels = read_f32(a.get("pixels"));
            let mut session = unsafe { open(&a) };
            let (latents, latent_t) = session
                .encode_video(Clip {
                    pixels: &pixels,
                    frames: 1,
                    height: height as usize,
                    width: width as usize,
                })
                .unwrap_or_else(|e| fail(&format!("encode_video: {e}")));
            write_f32(&dir, "latents.f32", &latents);
            write(&dir, "latent_t.txt", format!("{latent_t}\n").as_bytes());
        }

        // Model-space latents to RGB8, which is what the VAE check compares against diffusers.
        "decode" => {
            let dir = out();
            let p = params(&a);
            let shape = shape_for(p.height, p.width, p.frames)
                .unwrap_or_else(|| fail("no shape for that canvas and frame count"));
            let latents = read_f32(a.get("latents"));
            let mut session = unsafe { open(&a) };
            let mut frames =
                vec![0u8; shape.frames as usize * p.height as usize * p.width as usize * 3];
            session
                .decode_video(&shape, &latents, &mut frames)
                .unwrap_or_else(|e| fail(&format!("decode_video: {e}")));
            write(&dir, "frames.rgb", &frames);
            write(&dir, "shape.json", shape_json(&p).as_bytes());
        }

        // One evaluation or a whole trajectory, with the noise the caller supplies so the run is
        // reproducible, and the `dit_*` dumps when H3_DUMP_BLOCKS is set.
        "denoise" => {
            let dir = out();
            let p = params(&a);
            let shape = shape_for(p.height, p.width, p.frames)
                .unwrap_or_else(|| fail("no shape for that canvas and frame count"));

            // A keyframe contributes a vision span to the presentation and its own latents.
            let keyframe_pixels = a.opt("keyframe").map(read_f32);
            let spans: Vec<(i32, i32)> = if keyframe_pixels.is_some() {
                vec![(p.height, p.width)]
            } else {
                Vec::new()
            };
            let ids =
                presentation(a.get("prompt"), &spans, 0).unwrap_or_else(|e| fail(&format!("{e}")));

            let keyframe_latents = a.opt("keyframe-latents").map(read_f32);
            let keyframes: Vec<Keyframe<'_>> = match (&keyframe_pixels, &keyframe_latents) {
                (Some(pixels), Some(latents)) => vec![Keyframe {
                    frame_index: 0,
                    latents,
                    audio: None,
                    presented: Some(h3::Presented {
                        pixels,
                        height: p.height as usize,
                        width: p.width as usize,
                    }),
                }],
                _ => Vec::new(),
            };

            let video = a.opt("noise-video").map(read_f32);
            let audio = a.opt("noise-audio").map(read_f32);
            let noise = Noise {
                video: video.as_deref(),
                audio: audio.as_deref(),
            };

            let mut session = unsafe { open(&a) };
            let latents = session
                .denoise(&ids, &p, noise, &[], &keyframes, None)
                .unwrap_or_else(|e| fail(&format!("denoise: {e}")));
            write(&dir, "ids.i32", bytemuck::cast_slice(&ids));
            write_f32(&dir, "video.f32", &latents.video);
            write_f32(&dir, "audio.f32", &latents.audio);
            write(&dir, "shape.json", shape_json(&p).as_bytes());
            let _ = shape;
        }

        other => fail(&format!("unknown command {other:?}")),
    }
}

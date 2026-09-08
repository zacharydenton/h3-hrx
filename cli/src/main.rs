//! `h3`: the whole MiniMax H3 pipeline from the shell, through `libh3pipe` alone (every kernel in Loom).
//!
//!   h3 [ref1.jpg ref2.png voice.wav ...] [-p "prompt"] [options] < prompt
//!
//! Positional files are references by extension: images are presented as `<Picture i>` and encoded by the
//! video VAE's encoder, audio as `<Audio j>` through the audio VAE's encoder. The prompt is read from stdin
//! unless `-p` is given; `docs/prompting.md` is the format the model expects. ffmpeg decodes the inputs and
//! muxes the output.
//!
//! This is also the worked example of the C ABI: `ffi.rs` declares it by hand and `pipe.rs` wraps it, so a
//! client in any language can be read off these two files. `examples/rust` is the minimal version.
mod ffi;
mod media;
mod pipe;
mod resize;

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

const VISION_START: i32 = 151652;
const VISION_END: i32 = 151653;

/// Usage errors exit 64, as the C CLI did; runtime failures exit 1.
const EXIT_USAGE: u8 = 64;

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum Attn {
    F16,
    I8,
    I4,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum Sampler {
    /// ComfyUI's stock workflow sampler; the underscore spelling is the documented one
    #[value(name = "res_multistep", alias = "res-multistep")]
    ResMultistep,
    Euler,
}

#[derive(Parser)]
#[command(
    name = "h3",
    about = "MiniMax H3 text/image/audio -> video with sound, every kernel in Loom",
    disable_help_subcommand = true
)]
struct Cli {
    /// Reference files, by extension: images become <Picture i>, audio becomes <Audio j>
    #[arg(value_name = "FILE")]
    files: Vec<PathBuf>,

    /// The prompt (otherwise read from stdin)
    #[arg(short = 'p', value_name = "TEXT", allow_hyphen_values = true)]
    prompt: Option<String>,

    /// fl2va keyframe: the generated clip starts from this image
    #[arg(long, value_name = "IMG")]
    first_frame: Option<PathBuf>,

    /// Output clip; <out>.wav is kept next to it
    #[arg(long, default_value = "h3_out.mp4", value_name = "CLIP")]
    out: PathBuf,

    #[arg(long, default_value_t = 124, value_parser = clap::value_parser!(i32).range(1..=1 << 20))]
    frames: i32,

    /// Sigma grid points; one fewer than this many model evaluations
    #[arg(long, default_value_t = 31, value_parser = clap::value_parser!(i32).range(2..=1000))]
    steps: i32,

    #[arg(long, default_value_t = 864, value_parser = clap::value_parser!(i32).range(32..=8192))]
    width: i32,

    #[arg(long, default_value_t = 480, value_parser = clap::value_parser!(i32).range(32..=8192))]
    height: i32,

    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// The DiT attention's QK^T operands (i8 is the parity path)
    #[arg(long, value_enum, default_value = "i8")]
    attn: Attn,

    #[arg(long, value_enum, default_value = "res_multistep")]
    sampler: Sampler,

    /// ComfyUI's models directory (default $H3_MODELS, else ~/comfy-models)
    #[arg(long, value_name = "DIR")]
    models: Option<PathBuf>,

    #[arg(long, value_name = "FILE")]
    dit: Option<PathBuf>,
    #[arg(long, value_name = "FILE")]
    te: Option<PathBuf>,
    #[arg(long, value_name = "FILE")]
    video_vae: Option<PathBuf>,
    #[arg(long, value_name = "FILE")]
    audio_vae: Option<PathBuf>,

    /// The repository (default: the binary's parent's parent)
    #[arg(long, value_name = "DIR")]
    root: Option<PathBuf>,

    /// Denoise only, decode nothing
    #[arg(long)]
    no_decode: bool,

    /// Voice and sound only: skip the video decoder and write <out>.wav (use a 32x32 canvas)
    #[arg(long)]
    audio_only: bool,

    /// Write one frame as an image; any format ffmpeg writes by extension
    #[arg(long, value_name = "IMG")]
    still: Option<PathBuf>,

    #[arg(long, default_value_t = 0, value_parser = clap::value_parser!(i32).range(0..=1 << 20))]
    still_frame: i32,

    /// Write the raw latents as <prefix>.video.f32 and <prefix>.audio.f32
    #[arg(long, value_name = "PREFIX")]
    latents: Option<PathBuf>,

    /// Run reference files on the base checkpoint when the ref2va one is absent
    #[arg(long)]
    base_weights: bool,
}

fn lower_ext(path: &Path) -> String {
    path.extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default()
}

fn is_image(e: &str) -> bool {
    matches!(
        e,
        "jpg" | "jpeg" | "png" | "webp" | "bmp" | "gif" | "tif" | "tiff"
    )
}

fn is_audio(e: &str) -> bool {
    matches!(e, "wav" | "mp3" | "flac" | "ogg" | "m4a" | "aac" | "opus")
}

/// The directory a path would be written into exists and is writable, checked before the long run begins.
fn dir_writable(path: &Path) -> bool {
    let dir = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    dir.metadata().map(|m| m.is_dir()).unwrap_or(false) && tempfile_probe(dir)
}

/// `access(dir, W_OK)` without libc: creating and removing a uniquely named entry.
fn tempfile_probe(dir: &Path) -> bool {
    let probe = dir.join(format!(".h3-write-probe-{}", std::process::id()));
    match std::fs::File::create(&probe) {
        Ok(_) => {
            let _ = std::fs::remove_file(&probe);
            true
        }
        Err(_) => false,
    }
}

/// The repository root: the binary's parent's parent, so `build/h3` finds `kernels/` beside it.
fn exe_root() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().and_then(|p| p.parent()).map(Path::to_path_buf))
        .unwrap_or_else(|| PathBuf::from("."))
}

struct Image {
    pixels: Vec<f32>,
    w: i32,
    h: i32,
}

fn main() -> ExitCode {
    let cli = match Cli::try_parse() {
        Ok(c) => c,
        Err(e) => {
            let help = matches!(
                e.kind(),
                clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion
            );
            let _ = e.print();
            return if help {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(EXIT_USAGE)
            };
        }
    };
    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("h3: {e:#}");
            ExitCode::from(if e.downcast_ref::<UsageError>().is_some() {
                EXIT_USAGE
            } else {
                1
            })
        }
    }
}

/// Marks the failures that are the caller's mistake rather than the run's, so they exit 64.
#[derive(Debug)]
struct UsageError(String);
impl std::fmt::Display for UsageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for UsageError {}

macro_rules! usage {
    ($($arg:tt)*) => { return Err(UsageError(format!($($arg)*)).into()) };
}

fn run(cli: Cli) -> Result<()> {
    if unsafe { ffi::h3pipe_abi_version() } != ffi::ABI_VERSION {
        bail!(
            "libh3pipe is ABI {}, this h3 was built for {} (rebuild: scripts/build_host.sh)",
            unsafe { ffi::h3pipe_abi_version() },
            ffi::ABI_VERSION
        );
    }

    let (mut image_files, mut audio_files) = (Vec::new(), Vec::new());
    for f in &cli.files {
        let e = lower_ext(f);
        if is_image(&e) {
            image_files.push(f.clone());
        } else if is_audio(&e) {
            audio_files.push(f.clone());
        } else {
            usage!("{}: not an image or audio file by extension", f.display());
        }
    }

    let mut prompt = match &cli.prompt {
        Some(p) => p.clone(),
        None => std::io::read_to_string(std::io::stdin())
            .context("cannot read the prompt from stdin")?,
    };
    while prompt.ends_with(['\n', '\r', ' ']) {
        prompt.pop();
    }
    if prompt.is_empty() {
        usage!("no prompt (give -p \"...\" or pipe it on stdin)");
    }

    if cli.height % 32 != 0 || cli.width % 32 != 0 {
        usage!("--width and --height must be multiples of 32");
    }
    if cli.audio_only && cli.still.is_some() {
        usage!("--audio-only decodes no frames; it cannot be combined with --still");
    }
    if cli.no_decode && (cli.still.is_some() || cli.audio_only) {
        usage!("--no-decode decodes nothing; it cannot be combined with --still or --audio-only");
    }

    let root = cli.root.clone().unwrap_or_else(exe_root);
    let mut out = cli.out.clone();
    if lower_ext(&out) != "mp4" {
        out.set_extension("mp4");
    }
    let out_stem = out.with_extension("");
    let wav_path = out.with_extension("wav");
    if !cli.no_decode && !dir_writable(&out) {
        usage!(
            "cannot write {}: the directory is missing or not writable",
            out.display()
        );
    }
    if let Some(prefix) = &cli.latents {
        if !dir_writable(prefix) {
            usage!(
                "cannot write {}.video.f32: the directory is missing or not writable",
                prefix.display()
            );
        }
    }
    if let Some(still) = &cli.still {
        if !dir_writable(still) {
            usage!(
                "cannot write {}: the directory is missing or not writable",
                still.display()
            );
        }
    }

    // ComfyUI's models directory: the four checkpoints, read as they are
    let models = cli.models.clone().unwrap_or_else(|| {
        std::env::var_os("H3_MODELS")
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(std::env::var_os("HOME").unwrap_or_else(|| ".".into()))
                    .join("comfy-models")
            })
    });
    let dit_base = models.join("diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors");
    let dit_ref2va =
        models.join("diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors");
    let want_refs = !image_files.is_empty() || !audio_files.is_empty();
    let have_ref2va = dit_ref2va.exists();
    let ref2va = want_refs && have_ref2va && !cli.base_weights;
    if want_refs && !have_ref2va && !cli.base_weights && cli.dit.is_none() {
        usage!(
            "reference files need the ref2va checkpoint, {} (README, Weights); --base-weights runs the base checkpoint anyway",
            dit_ref2va.display()
        );
    }
    let dit = cli
        .dit
        .clone()
        .unwrap_or(if ref2va { dit_ref2va } else { dit_base });
    let te = cli.te.clone().unwrap_or_else(|| {
        models.join("text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors")
    });
    let video_vae = cli
        .video_vae
        .clone()
        .unwrap_or_else(|| models.join("vae/minimax_h3_video_vae_fp16.safetensors"));
    let audio_vae = cli
        .audio_vae
        .clone()
        .unwrap_or_else(|| models.join("vae/minimax_h3_audio_vae_fp32.safetensors"));
    if !dit.exists() {
        usage!(
            "{} not found (README, Weights; --models or --dit)",
            dit.display()
        );
    }

    let params = ffi::Params {
        height: cli.height,
        width: cli.width,
        frames: cli.frames,
        steps: cli.steps,
        seed: cli.seed,
        video_shift: 0.0,
        audio_shift: 0.0,
        sampler: match cli.sampler {
            Sampler::Euler => 0,
            Sampler::ResMultistep => 1,
        },
        cache_threshold: 0.0,
    };
    let shape = pipe::Session::shape_for(&params).map_err(|e| UsageError(e.to_string()))?;

    // inputs first: they are cheap and they fail fast. The keyframe fills the canvas; references are
    // scaled to at most the canvas' pixel count on a 32-pixel grid.
    let keyframe = match &cli.first_frame {
        Some(ff) => {
            let (rgb, w, h) = media::decode_image(ff)
                .with_context(|| format!("cannot decode {}", ff.display()))?;
            Some(Image {
                pixels: resize::pil_bilinear(&rgb, w, h, params.width, params.height),
                w: params.width,
                h: params.height,
            })
        }
        None => None,
    };
    let mut ref_images = Vec::new();
    for path in &image_files {
        let (rgb, w, h) = media::decode_image(path)
            .with_context(|| format!("cannot decode {}", path.display()))?;
        let scale = (f64::from(params.width) * f64::from(params.height)
            / (f64::from(w) * f64::from(h)))
        .sqrt()
        .min(1.0);
        let tw = 32.max(((f64::from(w) * scale / 32.0).round() as i32) * 32);
        let th = 32.max(((f64::from(h) * scale / 32.0).round() as i32) * 32);
        ref_images.push(Image {
            pixels: resize::pil_bilinear(&rgb, w, h, tw, th),
            w: tw,
            h: th,
        });
    }
    let mut ref_audio = Vec::new();
    for path in &audio_files {
        ref_audio.push(
            media::decode_audio(path)
                .with_context(|| format!("cannot decode {}", path.display()))?,
        );
    }

    // the presentation: keyframe, then reference images ("<Picture i>: " + a vision span), then
    // "<Audio j>: ", then the prompt
    let tok = pipe::Tokenizer::new(None)?; // the tokenizer compiled into libh3pipe (H3_TOKENIZER overrides it)
    let mut ids: Vec<i32> = Vec::new();
    let mut picture = 0;
    let mut vision_span = |ids: &mut Vec<i32>, w: i32, h: i32| -> Result<()> {
        picture += 1;
        tok.encode_into(&format!("<Picture {picture}>: "), ids)?;
        ids.push(VISION_START);
        ids.extend(std::iter::repeat_n(
            -1,
            (h / 32) as usize * (w / 32) as usize,
        ));
        ids.push(VISION_END);
        Ok(())
    };
    if let Some(k) = &keyframe {
        vision_span(&mut ids, k.w, k.h)?;
    }
    for im in &ref_images {
        vision_span(&mut ids, im.w, im.h)?;
    }
    for j in 0..ref_audio.len() {
        tok.encode_into(&format!("<Audio {}>: ", j + 1), &mut ids)?;
    }
    tok.encode_into(&prompt, &mut ids)
        .context("cannot tokenize the prompt")?;
    if ids.len() > shape.text_rows_max as usize {
        usage!(
            "the presentation is {} tokens; the model takes at most {}",
            ids.len(),
            shape.text_rows_max
        );
    }

    eprintln!(
        "{} frames at {}x{}: {}x{}x{} latents, {} audio latents; {} prompt tokens ({} keyframe, {} reference images, {} reference audio){}",
        shape.frames, params.width, params.height, shape.latent_t, shape.lat_h, shape.lat_w, shape.audio_t,
        ids.len(), keyframe.is_some() as i32, ref_images.len(), ref_audio.len(),
        if ref2va { ", ref2va weights" } else { "" }
    );

    let sources = root.join("kernels");
    let cache = root.join("build/kernel_cache");
    let loom_compile = std::env::var("LOOM_COMPILE").unwrap_or_else(|_| "loom-compile".into());
    let c = |p: &Path| std::ffi::CString::new(p.as_os_str().as_encoded_bytes()).unwrap();
    let (c_dit, c_te, c_vv, c_av) = (c(&dit), c(&te), c(&video_vae), c(&audio_vae));
    let (c_src, c_cache) = (c(&sources), c(&cache));
    let c_loom = std::ffi::CString::new(loom_compile)?;
    let config = ffi::Config {
        dit_file: c_dit.as_ptr(),
        te_file: c_te.as_ptr(),
        video_vae_file: c_vv.as_ptr(),
        audio_vae_file: c_av.as_ptr(),
        kernel_sources: c_src.as_ptr(),
        cache_dir: c_cache.as_ptr(),
        loom_compile: c_loom.as_ptr(),
        attn_qk_bits: match cli.attn {
            Attn::F16 => 16,
            Attn::I8 => 8,
            Attn::I4 => 4,
        },
    };

    let t0 = Instant::now();
    let session = pipe::Session::new(&config)?;
    eprintln!(
        "session in {:.1} s ({}, {} attention)",
        t0.elapsed().as_secs_f64(),
        dit.file_name().unwrap_or_default().to_string_lossy(),
        match cli.attn {
            Attn::F16 => "f16",
            Attn::I8 => "i8",
            Attn::I4 => "i4",
        }
    );

    // encoders: latents for the keyframe and the references. The latent buffers must outlive the raw
    // pointers the ABI structs hold, so they are all owned here.
    let mut latent_store: Vec<Vec<f32>> =
        Vec::with_capacity(1 + ref_images.len() + ref_audio.len());
    let mut keyframes: Vec<ffi::Keyframe> = Vec::new();
    let mut refs: Vec<ffi::Ref> = Vec::new();
    if let Some(k) = &keyframe {
        let mut latents = vec![0.0f32; 24 * (k.h / 16) as usize * (k.w / 16) as usize];
        session
            .encode_video(&k.pixels, 1, k.h, k.w, &mut latents)
            .context("encode keyframe")?;
        latent_store.push(latents);
        keyframes.push(ffi::Keyframe {
            frame_index: 0,
            video_latent: latent_store.last().unwrap().as_ptr(),
            pixels: k.pixels.as_ptr(),
            height: k.h,
            width: k.w,
            audio_latent: std::ptr::null(),
            audio_t: 0,
        });
    }
    for im in &ref_images {
        let mut latents = vec![0.0f32; 24 * (im.h / 16) as usize * (im.w / 16) as usize];
        session
            .encode_video(&im.pixels, 1, im.h, im.w, &mut latents)
            .context("encode reference image")?;
        latent_store.push(latents);
        refs.push(ffi::Ref {
            kind: 0,
            video_latent: latent_store.last().unwrap().as_ptr(),
            latent_t: 1,
            lat_h: im.h / 16,
            lat_w: im.w / 16,
            pixels: im.pixels.as_ptr(),
            height: im.h,
            width: im.w,
            ..Default::default()
        });
    }
    for (samples, n) in &ref_audio {
        let t = (n + 799) / 800;
        let mut latents = vec![0.0f32; 2 * 32 * t as usize];
        let audio_t = session
            .encode_audio(samples, *n, &mut latents)
            .context("encode reference audio")?;
        latent_store.push(latents);
        refs.push(ffi::Ref {
            kind: 1,
            audio_latent: latent_store.last().unwrap().as_ptr(),
            audio_t,
            ..Default::default()
        });
    }

    let mut video =
        vec![0.0f32; 24 * shape.latent_t as usize * shape.lat_h as usize * shape.lat_w as usize];
    let mut audio = vec![0.0f32; 2 * 32 * shape.audio_t as usize];
    let t0 = Instant::now();
    session.denoise(&ids, &params, &keyframes, &refs, &mut video, &mut audio)?;
    eprintln!("denoised in {:.1} s", t0.elapsed().as_secs_f64());
    drop(latent_store); // the pointers in keyframes/refs are dead from here

    if let Some(prefix) = &cli.latents {
        for (suffix, data) in [("video.f32", &video), ("audio.f32", &audio)] {
            let path = prefix.with_extension(suffix);
            let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(&path, &bytes)
                .with_context(|| format!("cannot write {}", path.display()))?;
        }
        eprintln!(
            "wrote {p}.video.f32 and {p}.audio.f32",
            p = prefix.display()
        );
    }
    if cli.no_decode {
        return Ok(());
    }

    let t0 = Instant::now();
    let mut frames = vec![
        0u8;
        if cli.audio_only {
            0
        } else {
            shape.frames as usize * params.height as usize * params.width as usize * 3
        }
    ];
    if !cli.audio_only {
        session.decode_video(&params, &video, &mut frames)?;
    }
    let mut samples = vec![0.0f32; 2 * shape.audio_t as usize * 800];
    session.decode_audio(&audio, shape.audio_t, &mut samples)?;
    drop(session);
    eprintln!("decoded in {:.1} s", t0.elapsed().as_secs_f64());

    let n = shape.audio_t as u32 * 800;
    std::fs::write(&wav_path, media::wav_bytes(&samples, n))
        .with_context(|| format!("cannot write {}", wav_path.display()))?;

    if let Some(still) = &cli.still {
        let idx = (shape.frames - 1).min(cli.still_frame) as usize;
        let fb = params.height as usize * params.width as usize * 3;
        media::write_still(
            still,
            &frames[idx * fb..(idx + 1) * fb],
            params.width,
            params.height,
        )?;
        eprintln!("wrote {} (frame {idx})", still.display());
    }
    if cli.audio_only {
        eprintln!(
            "wrote {}.wav ({:.2} s, 32 kHz stereo)",
            out_stem.display(),
            f64::from(n) / f64::from(media::RATE)
        );
        println!("{}.wav", out_stem.display());
        return Ok(());
    }

    media::mux(&out, &wav_path, &frames, params.width, params.height)?;
    eprintln!(
        "wrote {} ({} frames, {}x{}, {:.2} s) and {}.wav",
        out.display(),
        shape.frames,
        params.width,
        params.height,
        f64::from(shape.frames) / f64::from(media::FPS),
        out_stem.display()
    );
    println!("{}", out.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn the_command_is_well_formed() {
        Cli::command().debug_assert();
    }

    #[test]
    fn defaults_match_the_documented_ones() {
        let c = Cli::try_parse_from(["h3"]).unwrap();
        assert_eq!(
            (c.frames, c.steps, c.width, c.height, c.seed),
            (124, 31, 864, 480, 0)
        );
        assert!(matches!(c.attn, Attn::I8) && matches!(c.sampler, Sampler::ResMultistep));
        assert_eq!(c.out, PathBuf::from("h3_out.mp4"));
    }

    #[test]
    fn a_prompt_may_look_like_an_option() {
        let c = Cli::try_parse_from(["h3", "-p", "--help", "ref.jpg", "--no-decode"]).unwrap();
        assert_eq!(c.prompt.as_deref(), Some("--help"));
        assert_eq!(c.files, vec![PathBuf::from("ref.jpg")]);
        assert!(c.no_decode);
    }

    #[test]
    fn unknown_options_and_missing_values_are_rejected() {
        assert!(Cli::try_parse_from(["h3", "--precison", "int8"]).is_err());
        assert!(Cli::try_parse_from(["h3", "-p"]).is_err());
    }

    #[test]
    fn numeric_ranges_are_enforced() {
        assert!(Cli::try_parse_from(["h3", "--steps", "12x"]).is_err());
        assert!(Cli::try_parse_from(["h3", "--width", "0"]).is_err());
        assert!(Cli::try_parse_from(["h3", "--steps", "1"]).is_err());
        assert_eq!(
            Cli::try_parse_from(["h3", "--height", "480"])
                .unwrap()
                .height,
            480
        );
        assert_eq!(
            Cli::try_parse_from(["h3", "--frames", "124"])
                .unwrap()
                .frames,
            124
        );
    }

    #[test]
    fn choices_are_closed() {
        assert!(Cli::try_parse_from(["h3", "--attn", "int7"]).is_err());
        assert!(Cli::try_parse_from(["h3", "--sampler", "heun"]).is_err());
        assert!(matches!(
            Cli::try_parse_from(["h3", "--attn", "f16"]).unwrap().attn,
            Attn::F16
        ));
        assert!(matches!(
            Cli::try_parse_from(["h3", "--sampler", "euler"])
                .unwrap()
                .sampler,
            Sampler::Euler
        ));
        // the documented spelling is the underscore one; the kebab spelling is accepted as an alias
        assert!(matches!(
            Cli::try_parse_from(["h3", "--sampler", "res_multistep"])
                .unwrap()
                .sampler,
            Sampler::ResMultistep
        ));
        assert!(matches!(
            Cli::try_parse_from(["h3", "--sampler", "res-multistep"])
                .unwrap()
                .sampler,
            Sampler::ResMultistep
        ));
    }

    #[test]
    fn extensions_classify_references() {
        assert!(is_image("jpg") && is_image("PNG".to_lowercase().as_str()) && is_image("webp"));
        assert!(is_audio("wav") && is_audio("opus"));
        assert!(!is_image("txt") && !is_audio("mp4"));
        assert_eq!(lower_ext(Path::new("a/b/REF.JPG")), "jpg");
        assert_eq!(lower_ext(Path::new("noext")), "");
    }

    #[test]
    fn output_locations_are_checked_before_the_long_run() {
        let dir = std::env::temp_dir();
        assert!(dir_writable(&dir.join("clip.mp4")));
        assert!(dir_writable(Path::new("clip.mp4")));
        assert!(!dir_writable(&dir.join("missing-dir-for-h3-test/clip.mp4")));
    }
}

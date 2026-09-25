//! `h3`: the MiniMax H3 pipeline from the shell, powered by Loom and HRX.
//!
//! ```text
//! h3 [ref1.jpg ref2.png voice.wav ...] [-p "prompt"] [options] < prompt
//! ```
//!
//! Positional files are references by extension: images are presented as `<Picture i>` and encoded by the
//! video VAE's encoder, audio as `<Audio j>` through the audio VAE's encoder. The prompt is read from stdin
//! unless `-p` is given; `docs/prompting.md` is the format the model expects. ffmpeg decodes the inputs and
//! muxes the output.
//!
//! The CLI owns a Rust `Session`; application adapters call that same API.
mod media;

use h3_hrx::resize;

use h3_hrx::{
    Attention as Attn16, Clip, Config, DenoiseParams, Keyframe, LatentGrid, Noise, Presentation,
    Presented, Reference, Sampler as Sampler16, Session, Tokenizer,
};

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

/// Usage errors follow clap (2); runtime failures exit 1.
const EXIT_USAGE: u8 = 2;

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

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum Residency {
    StageScoped,
    Retain,
    Budgeted,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum Preset {
    Base,
    #[value(name = "turbo-768p-4")]
    TurboFour,
    #[value(name = "turbo-768p-8")]
    TurboEight,
}

impl Preset {
    fn turbo(self) -> Option<h3_hrx::adapter::TurboPreset> {
        match self {
            Self::Base => None,
            Self::TurboFour => Some(h3_hrx::adapter::TurboPreset::Four),
            Self::TurboEight => Some(h3_hrx::adapter::TurboPreset::Eight),
        }
    }
}

#[derive(Parser)]
#[command(
    name = "h3",
    version,
    about = "MiniMax H3 video and audio generation on AMD Strix Halo, powered by Loom and HRX",
    disable_help_subcommand = true
)]
struct Cli {
    /// Reference files, by extension: images become `<Picture i>`, audio becomes `<Audio j>`
    #[arg(value_name = "FILE")]
    files: Vec<PathBuf>,

    /// Generation preset (Turbo is experimental until qualification completes)
    #[arg(long, value_enum, default_value = "base", hide = true)]
    preset: Preset,

    /// The prompt (otherwise read from stdin)
    #[arg(short = 'p', value_name = "TEXT", allow_hyphen_values = true)]
    prompt: Option<String>,

    /// fl2va keyframe: the generated clip starts from this image
    #[arg(long, value_name = "IMG")]
    first_frame: Option<PathBuf>,

    /// Output clip; `<out>.wav` is kept next to it
    #[arg(long, default_value = "h3_out.mp4", value_name = "CLIP")]
    out: PathBuf,

    /// Frame count (base default: 124)
    #[arg(long, value_parser = clap::value_parser!(i32).range(1..=1 << 20))]
    frames: Option<i32>,

    /// Sigma grid points; one fewer model evaluations (base default: 31)
    #[arg(long, value_parser = clap::value_parser!(i32).range(2..=1000))]
    steps: Option<i32>,

    /// Output width (base default: 864)
    #[arg(long, value_parser = clap::value_parser!(i32).range(32..=8192))]
    width: Option<i32>,

    /// Output height (base default: 480)
    #[arg(long, value_parser = clap::value_parser!(i32).range(32..=8192))]
    height: Option<i32>,

    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// The DiT attention's QK^T operands (i8 is the parity path)
    #[arg(long, value_enum, default_value = "i8")]
    attn: Attn,

    /// Sampler (base default: res_multistep)
    #[arg(long, value_enum)]
    sampler: Option<Sampler>,

    /// Use only checkpoints already on disk; never download
    #[arg(long)]
    offline: bool,

    /// Release completed models, retain them, or cache idle units under a budget
    #[arg(long, value_enum, default_value = "stage-scoped")]
    residency: Residency,

    /// Shared native allocation ceiling in MiB (required for budgeted residency)
    #[arg(long, value_parser = clap::value_parser!(u64).range(1..))]
    memory_budget_mib: Option<u64>,

    /// Record modality-specific cache changes while running every block
    #[arg(long, hide = true)]
    cache_observe: bool,

    /// Experimental conditioning/audio/video cache thresholds; requires quality calibration
    #[arg(long, hide = true, num_args = 3, conflicts_with = "cache_observe")]
    cache_thresholds: Vec<f32>,

    #[arg(long, value_name = "FILE")]
    dit: Option<PathBuf>,
    #[arg(long, value_name = "FILE")]
    te: Option<PathBuf>,
    #[arg(long, value_name = "FILE")]
    video_vae: Option<PathBuf>,
    #[arg(long, value_name = "FILE")]
    audio_vae: Option<PathBuf>,

    /// Optional developer source tree; defaults to embedded kernels
    #[arg(long, value_name = "DIR")]
    root: Option<PathBuf>,

    /// Denoise only, decode nothing
    #[arg(long)]
    no_decode: bool,

    /// Voice and sound only: skip the video decoder and write `<out>.wav` (use a 32x32 canvas)
    #[arg(long)]
    audio_only: bool,

    /// Write one frame as an image; any format ffmpeg writes by extension
    #[arg(long, value_name = "IMG")]
    still: Option<PathBuf>,

    #[arg(long, default_value_t = 0, value_parser = clap::value_parser!(i32).range(0..=1 << 20))]
    still_frame: i32,

    /// Write the raw latents as `<prefix>.video.f32` and `<prefix>.audio.f32`
    #[arg(long, value_name = "PREFIX")]
    latents: Option<PathBuf>,

    /// Use the base checkpoint for reference files instead of ref2va
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
///
/// Exclusive creation, and a fresh name if that collides. `File::create` on a predictable name would
/// truncate whatever is already there — and through a symlink, truncate somewhere else entirely — for
/// a probe whose whole purpose is to touch nothing.
fn tempfile_probe(dir: &Path) -> bool {
    for attempt in 0..8 {
        let probe = dir.join(format!(
            ".h3-write-probe-{}-{attempt}-{:x}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        match std::fs::File::options()
            .write(true)
            .create_new(true)
            .open(&probe)
        {
            Ok(_) => {
                let _ = std::fs::remove_file(&probe);
                return true;
            }
            // taken: try another name rather than touching what is there
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(_) => return false,
        }
    }
    false
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

/// Marks the failures that are the caller's mistake rather than the run's, so they exit 2.
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

fn cache_policy(cli: &Cli) -> Result<h3_hrx::CachePolicy> {
    use h3_hrx::{CachePolicy, CacheThresholds};
    let policy = match cli.cache_thresholds.as_slice() {
        [] if cli.cache_observe => CachePolicy::Observe,
        [] => CachePolicy::Off,
        &[conditioning, audio, video] => CachePolicy::Conservative(CacheThresholds {
            conditioning,
            audio,
            video,
        }),
        _ => usage!("pass three cache thresholds: conditioning audio video"),
    };
    if let Err(message) = policy.validate(0.0) {
        usage!("{message}");
    }
    Ok(policy)
}

fn memory_budget_bytes(cli: &Cli) -> Result<Option<usize>> {
    let Some(mib) = cli.memory_budget_mib else {
        if cli.residency == Residency::Budgeted {
            usage!("--residency budgeted requires --memory-budget-mib");
        }
        return Ok(None);
    };
    let Some(bytes) = usize::try_from(mib)
        .ok()
        .and_then(|n| n.checked_mul(1 << 20))
    else {
        usage!("--memory-budget-mib overflows the allocation address space");
    };
    Ok(Some(bytes))
}

fn parameters(cli: &Cli) -> Result<DenoiseParams> {
    memory_budget_bytes(cli)?;
    let cache = cache_policy(cli)?;
    let mut params = DenoiseParams {
        width: cli.width.unwrap_or(864),
        height: cli.height.unwrap_or(480),
        frames: cli.frames.unwrap_or(124),
        steps: cli.steps.unwrap_or(31) as usize,
        seed: cli.seed,
        sampler: match cli.sampler.unwrap_or(Sampler::ResMultistep) {
            Sampler::Euler => Sampler16::Euler,
            Sampler::ResMultistep => Sampler16::ResMultistep,
        },
        ..DenoiseParams::default()
    };
    if let Some(preset) = cli.preset.turbo() {
        for (name, given, expected) in [
            ("width", cli.width, 1344),
            ("height", cli.height, 768),
            ("frames", cli.frames, 124),
            ("steps", cli.steps, preset.evaluations() as i32 + 1),
        ] {
            if given.is_some_and(|v| v != expected) {
                usage!("Turbo requires --{name} {expected}");
            }
        }
        if cli.sampler.is_some_and(|s| s != Sampler::Euler)
            || cli.dit.is_some()
            || !cli.files.is_empty()
            || cli.audio_only
            || cli.attn != Attn::I8
            || cache != h3_hrx::CachePolicy::Off
        {
            usage!("Turbo requires Euler, the default i8 base, cache off, and text or --first-frame conditioning");
        }
        preset.configure(&mut params);
    }
    Ok(params)
}

fn run(cli: Cli) -> Result<()> {
    let params = parameters(&cli)?;
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

    if params.height % 32 != 0 || params.width % 32 != 0 {
        usage!("--width and --height must be multiples of 32");
    }
    if cli.audio_only && cli.still.is_some() {
        usage!("--audio-only decodes no frames; it cannot be combined with --still");
    }
    if cli.no_decode && (cli.still.is_some() || cli.audio_only) {
        usage!("--no-decode decodes nothing; it cannot be combined with --still or --audio-only");
    }

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

    // Reuse the standard Hub cache before downloading missing checkpoints.
    // --offline stops at what is already on disk instead of downloading.
    let resolver = h3_hrx::models::Resolver::new().offline(cli.offline);
    let want_refs = !image_files.is_empty() || !audio_files.is_empty();
    let ref2va = want_refs && !cli.base_weights;
    let resolve = |explicit: &Option<PathBuf>, relative: &str| -> Result<PathBuf> {
        match explicit {
            Some(path) => Ok(path.clone()),
            None => Ok(resolver.find(relative)?),
        }
    };
    let dit = resolve(
        &cli.dit,
        if ref2va {
            h3_hrx::models::DIT_REF2VA
        } else {
            h3_hrx::models::DIT_FL2VA
        },
    )?;
    let te = resolve(&cli.te, h3_hrx::models::TE)?;
    let video_vae = resolve(&cli.video_vae, h3_hrx::models::VIDEO_VAE)?;
    let audio_vae = resolve(&cli.audio_vae, h3_hrx::models::AUDIO_VAE)?;
    if !dit.exists() {
        usage!("{} not found (README, Weights; --dit)", dit.display());
    }

    let shape = Session::shape_for(params.height, params.width, params.frames)
        .ok_or_else(|| UsageError("no such shape".into()))?;

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
        let (tw, th) = resize::fit(w, h, params.width, params.height);
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
    let tok = Tokenizer::new()?; // the vocabulary compiled into the crate (H3_TOKENIZER overrides it)
    let mut presentation = Presentation::new(&tok);
    if let Some(k) = &keyframe {
        presentation.picture(k.w, k.h)?;
    }
    for im in &ref_images {
        presentation.picture(im.w, im.h)?;
    }
    for _ in 0..ref_audio.len() {
        presentation.audio()?;
    }
    let ids = presentation
        .finish(&prompt, &shape)
        .map_err(|e| UsageError(e.to_string()))?;

    eprintln!(
        "{} frames at {}x{}: {}x{}x{} latents, {} audio latents; {} prompt tokens ({} keyframe, {} reference images, {} reference audio){}",
        shape.frames, params.width, params.height, shape.latent_t, shape.lat_h, shape.lat_w, shape.audio_t,
        ids.len(), keyframe.is_some() as i32, ref_images.len(), ref_audio.len(),
        if ref2va { ", ref2va weights" } else { "" }
    );

    // Installed binaries use their packaged kernel sources; --root opts into a working tree's
    // instead. Compiled artifacts always go to the one per-user HRX cache, whoever built them.
    let sources = cli
        .root
        .as_ref()
        .map(|root| root.join("kernels"))
        .unwrap_or_default();
    let loom_library = std::env::var_os("HRX_LOOM_LIBRARY").map(std::path::PathBuf::from);
    let config = Config {
        dit: if cli.preset.turbo().is_some() {
            None
        } else {
            Some(dit.clone())
        },
        te: Some(te),
        video_vae: Some(video_vae),
        audio_vae: Some(audio_vae),
        kernel_sources: sources,
        loom_library,
        attention: match cli.attn {
            Attn::F16 => Attn16::F16,
            Attn::I8 => Attn16::I8,
            Attn::I4 => Attn16::I4,
        },
    };

    let t0 = Instant::now();
    // Safety: the checkpoints are the files this command was pointed at, and it does not write to
    // them. A user who edits a checkpoint mid-run gets what the documentation says they get.
    let residency_manager = memory_budget_bytes(&cli)?
        .map(hrx::residency::ResidencyManager::new)
        .transpose()?;
    let context = hrx::inference::ModelContext::new(hrx::execution::RuntimeOptions {
        memory_budget: residency_manager.as_ref().map(|manager| manager.budget()),
        ..Default::default()
    })?;
    let mut session = unsafe {
        Session::new_in(
            config,
            h3_hrx::SessionOptions {
                residency: match cli.residency {
                    Residency::StageScoped => h3_hrx::ResidencyPolicy::StageScoped,
                    Residency::Retain => h3_hrx::ResidencyPolicy::Retain,
                    Residency::Budgeted => h3_hrx::ResidencyPolicy::Budgeted,
                },
                cache: cache_policy(&cli)?,
                turbo: cli.preset.turbo(),
            },
            &context,
        )
    }?;
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

    if let Some(preset) = cli.preset.turbo() {
        eprintln!(
            "experimental Turbo: {} Euler evaluations, video/audio shifts 6/3",
            preset.evaluations()
        );
    }

    // encoders: latents for the keyframe and the references, encoded before the run so a bad input
    // fails early. Each reference borrows its own latents, which live until the denoise returns.
    let keyframe_latents = match &keyframe {
        Some(k) => Some(
            session
                .encode_video(Clip {
                    pixels: &k.pixels,
                    frames: 1,
                    height: k.h as usize,
                    width: k.w as usize,
                })
                .context("encode keyframe")?
                .0,
        ),
        None => None,
    };
    let mut image_latents = Vec::with_capacity(ref_images.len());
    for im in &ref_images {
        image_latents.push(
            session
                .encode_video(Clip {
                    pixels: &im.pixels,
                    frames: 1,
                    height: im.h as usize,
                    width: im.w as usize,
                })
                .context("encode reference image")?
                .0,
        );
    }
    let mut audio_latents = Vec::with_capacity(ref_audio.len());
    for (samples, n) in &ref_audio {
        audio_latents.push(
            session
                .encode_audio(samples, *n as usize)
                .context("encode reference audio")?,
        );
    }

    let keyframes: Vec<Keyframe<'_>> = keyframe
        .iter()
        .zip(keyframe_latents.iter())
        .map(|(k, z)| Keyframe {
            frame_index: 0,
            latents: z,
            presented: Some(Presented {
                pixels: &k.pixels,
                height: k.h as usize,
                width: k.w as usize,
            }),
            audio: None,
        })
        .collect();
    let mut refs: Vec<Reference<'_>> = ref_images
        .iter()
        .zip(image_latents.iter())
        .map(|(im, z)| Reference::Image {
            latents: z,
            grid: LatentGrid {
                frames: 1,
                height: (im.h / 16) as usize,
                width: (im.w / 16) as usize,
            },
            presented: Some(Presented {
                pixels: &im.pixels,
                height: im.h as usize,
                width: im.w as usize,
            }),
        })
        .collect();
    for (z, t) in &audio_latents {
        refs.push(Reference::Audio {
            latents: z,
            frames: *t,
        });
    }

    let t0 = Instant::now();
    let trace = std::env::var_os("H3_STAGE_TRACE").is_some_and(|v| !v.is_empty() && v != "0");
    let mut show = |step: usize, steps: usize, seconds: f64| {
        if trace {
            eprintln!("  step {step}/{steps}  {seconds:5.1} s");
        } else {
            eprint!("\r  step {step}/{steps}  {seconds:5.1} s");
        }
        let _ = std::io::Write::flush(&mut std::io::stderr());
        false // true would cancel
    };
    let latents = session.denoise(
        &ids,
        &params,
        Noise::default(),
        &refs,
        &keyframes,
        Some(&mut show),
    )?;
    eprintln!();
    let (video, audio) = (latents.video, latents.audio);
    eprintln!("denoised in {:.1} s", t0.elapsed().as_secs_f64());
    if let Some(report) = session.profile_report()? {
        eprintln!("  denoise stages ({report})");
    }

    if let Some(prefix) = &cli.latents {
        for (suffix, data) in [("video.f32", &video), ("audio.f32", &audio)] {
            // appended, not substituted: `with_extension` on `run.seed7` would drop `.seed7` and
            // write `run.video.f32`, so two different prefixes could name one file
            let mut name = prefix.clone().into_os_string();
            name.push(".");
            name.push(suffix);
            let path = PathBuf::from(name);
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
        session.decode_video(&shape, &video, &mut frames)?;
    }
    let mut samples = vec![0.0f32; shape.audio_samples()];
    session.decode_audio(&audio, shape.audio_t as usize, &mut samples)?;
    eprintln!("decoded in {:.1} s", t0.elapsed().as_secs_f64());
    if let Some(report) = session.profile_report()? {
        eprintln!("  decode stages ({report})");
    }
    // the weights are released before muxing, which is where the memory is wanted
    drop(session);

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
        let p = parameters(&c).unwrap();
        assert_eq!(
            (p.frames, p.steps, p.width, p.height, p.seed),
            (124, 31, 864, 480, 0)
        );
        assert!(matches!(c.attn, Attn::I8) && matches!(p.sampler, Sampler16::ResMultistep));
        assert_eq!(c.out, PathBuf::from("h3_out.mp4"));
        assert!(matches!(c.residency, Residency::StageScoped));
        assert!(c.preset.turbo().is_none());
    }

    #[test]
    fn budgeted_residency_requires_an_explicit_representable_ceiling() {
        let parse = |args: &[&str]| Cli::try_parse_from(args).unwrap();
        assert!(memory_budget_bytes(&parse(&["h3", "--residency", "budgeted"])).is_err());
        assert!(Cli::try_parse_from(["h3", "--memory-budget-mib", "0"]).is_err());
        assert!(memory_budget_bytes(&parse(&[
            "h3",
            "--memory-budget-mib",
            "18446744073709551615"
        ]))
        .is_err());
        let cli = parse(&[
            "h3",
            "--residency",
            "budgeted",
            "--memory-budget-mib",
            "81920",
        ]);
        assert_eq!(memory_budget_bytes(&cli).unwrap(), Some(80 << 30));
    }

    #[test]
    fn cache_thresholds_are_complete_positive_and_finite() {
        for values in [
            ["0", "0.1", "0.1"],
            ["0.1", "NaN", "0.1"],
            ["0.1", "0.1", "inf"],
        ] {
            let cli =
                Cli::try_parse_from(["h3", "--cache-thresholds", values[0], values[1], values[2]])
                    .unwrap();
            assert!(parameters(&cli).is_err());
        }
        assert!(Cli::try_parse_from(["h3", "--cache-thresholds", "0.1", "0.1"]).is_err());
        assert!(Cli::try_parse_from([
            "h3",
            "--cache-observe",
            "--cache-thresholds",
            "0.1",
            "0.1",
            "0.1"
        ])
        .is_err());
        let cli = Cli::try_parse_from(["h3", "--cache-thresholds", "0.1", "0.2", "0.3"]).unwrap();
        assert_eq!(
            cache_policy(&cli).unwrap(),
            h3_hrx::CachePolicy::Conservative(h3_hrx::CacheThresholds {
                conditioning: 0.1,
                audio: 0.2,
                video: 0.3
            })
        );
    }

    #[test]
    fn turbo_selects_all_trained_parameters_together() {
        for (name, evaluations) in [("turbo-768p-4", 4), ("turbo-768p-8", 8)] {
            let cli = Cli::try_parse_from(["h3", "--preset", name]).unwrap();
            let p = parameters(&cli).unwrap();
            assert_eq!((p.width, p.height, p.frames), (1344, 768, 124));
            assert_eq!(p.steps, evaluations + 1);
            assert_eq!(p.sampler, Sampler16::Euler);
            assert_eq!((p.video_shift, p.audio_shift), (6.0, 3.0));
        }
    }

    #[test]
    fn turbo_conflicts_fail_before_model_resolution() {
        for extra in [
            vec!["--steps", "4"],
            vec!["--width", "864"],
            vec!["--sampler", "res_multistep"],
            vec!["--dit", "custom.safetensors"],
            vec!["ref.jpg"],
            vec!["--audio-only"],
            vec!["--cache-observe"],
            vec!["--cache-thresholds", "0.1", "0.1", "0.1"],
            vec!["--attn", "i4"],
        ] {
            let mut args = vec!["h3", "--preset", "turbo-768p-4"];
            args.extend(extra);
            let cli = Cli::try_parse_from(args).unwrap();
            assert!(parameters(&cli).is_err());
        }
        let cli = Cli::try_parse_from([
            "h3",
            "--preset",
            "turbo-768p-4",
            "--steps",
            "5",
            "--sampler",
            "euler",
            "--first-frame",
            "alien.png",
        ])
        .unwrap();
        assert!(parameters(&cli).is_ok());
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
            Some(480)
        );
        assert_eq!(
            Cli::try_parse_from(["h3", "--frames", "124"])
                .unwrap()
                .frames,
            Some(124)
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
            Some(Sampler::Euler)
        ));
        // the documented spelling is the underscore one; the kebab spelling is accepted as an alias
        assert!(matches!(
            Cli::try_parse_from(["h3", "--sampler", "res_multistep"])
                .unwrap()
                .sampler,
            Some(Sampler::ResMultistep)
        ));
        assert!(matches!(
            Cli::try_parse_from(["h3", "--sampler", "res-multistep"])
                .unwrap()
                .sampler,
            Some(Sampler::ResMultistep)
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

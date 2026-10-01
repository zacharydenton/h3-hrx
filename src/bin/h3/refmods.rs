//! RefMod CLI policy and media handling. The library owns the file/latent semantics.
use super::{is_image, lower_ext, media, Cli, UsageError};
use anyhow::{bail, Context, Result};
use clap::{Args, Subcommand};
use h3_hrx::{
    refmod::{ApplyOptions, AudioInput, CreateOptions, ImageInput, PreparedRefMod, RefMod},
    Config, Session,
};
use std::path::PathBuf;

#[derive(Subcommand)]
pub enum Command {
    /// Encode images and optional audio into a portable reference file
    Create(Box<Create>),
    /// Inspect current v4/v5 files without opening a GPU or model
    Inspect { file: PathBuf },
}
#[derive(Args)]
pub struct Create {
    /// Image directory (nonrecursive), or image files in the desired order
    #[arg(value_name = "IMAGE_OR_DIR")]
    images: Vec<PathBuf>,
    /// Optional voice/sound reference, bundled with the images
    #[arg(long)]
    audio: Option<PathBuf>,
    #[arg(long)]
    out: PathBuf,
    #[arg(long)]
    name: Option<String>,
    #[arg(long, default_value = "")]
    description: String,
    #[arg(long, default_value = "identity")]
    concept_type: String,
    #[arg(long, default_value_t = 1024)]
    resolution: i32,
    #[arg(long, default_value_t = 8192)]
    max_tokens: usize,
    #[arg(long, default_value_t = 30.0)]
    audio_max_seconds: f32,
    #[arg(long, default_value_t = 5120)]
    audio_max_tokens: usize,
    /// Permit dropping audio latents beyond the audio token budget
    #[arg(long)]
    truncate_audio: bool,
    /// Report inputs and estimated costs without models or output writes
    #[arg(long)]
    dry_run: bool,
    /// Replace an existing destination atomically
    #[arg(long)]
    force: bool,
    #[arg(long)]
    video_vae: Option<PathBuf>,
    #[arg(long)]
    audio_vae: Option<PathBuf>,
    #[arg(long)]
    offline: bool,
    /// Developer kernel source tree
    #[arg(long)]
    root: Option<PathBuf>,
}
fn report(mods: &RefMod) -> Result<()> {
    println!(
        "{} — {} member(s), {} tokens",
        mods.metadata()["name"].as_str().unwrap_or("refmod"),
        mods.members().len(),
        mods.token_count()?
    );
    for (i, m) in mods.members().iter().enumerate() {
        println!(
            "  {}: {} / {} {:?} {:?}, {} tokens",
            i + 1,
            m.metadata()["name"].as_str().unwrap_or("unnamed"),
            m.metadata()["kind"].as_str().unwrap_or("unknown"),
            m.shape(),
            m.dtype(),
            m.token_count()
        );
        println!("    {}", m.metadata());
        if m.metadata().get("refmod_config").is_some() {
            eprintln!("  Saved config is informational; curves and saved-setting application are unsupported.");
        }
    }
    Ok(())
}
pub fn run(command: Command) -> Result<()> {
    match command {
        Command::Inspect { file } => report(&RefMod::load(file)?),
        Command::Create(args) => create(*args),
    }
}
fn image_paths(inputs: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for path in inputs {
        if path.is_dir() {
            let mut found = std::fs::read_dir(path)?
                .map(|e| e.map(|e| e.path()))
                .collect::<std::io::Result<Vec<_>>>()?;
            found.retain(|p| p.is_file() && is_image(&lower_ext(p)));
            found.sort();
            paths.extend(found);
        } else if path.is_file() && is_image(&lower_ext(path)) {
            paths.push(path.clone());
        } else {
            bail!("{}: expected an image or image directory", path.display());
        }
    }
    Ok(paths)
}
fn create(args: Create) -> Result<()> {
    let name = args.name.clone().unwrap_or_else(|| {
        args.out
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned()
    });
    let options = CreateOptions {
        name,
        description: args.description,
        concept_type: args.concept_type,
        resolution: args.resolution,
        max_tokens: args.max_tokens,
        audio_max_seconds: args.audio_max_seconds,
        audio_max_tokens: args.audio_max_tokens,
        truncate_audio: args.truncate_audio,
    };
    options.check().map_err(|e| UsageError(e.to_string()))?;
    if lower_ext(&args.out) != "safetensors" {
        bail!("--out must end in .safetensors");
    }
    if args.out.exists() && !args.force {
        bail!("{} exists; use --force to replace it", args.out.display());
    }
    let paths = image_paths(&args.images)?;
    if paths.is_empty() && args.audio.is_none() {
        bail!("no images or audio found");
    }
    let mut decoded = Vec::new();
    for path in &paths {
        decoded
            .push(media::decode_image(path).with_context(|| format!("decode {}", path.display()))?);
    }
    if let Some((_, w, h)) = decoded.first() {
        let (cw, ch) = options.canvas(*w, *h)?;
        let per_frame = cw as usize / 32 * (ch as usize / 32);
        eprintln!("{} image(s), canvas {cw}x{ch}, {per_frame} tokens/image; at most {} image(s) fit before deduplication",
            paths.len(),options.max_tokens/per_frame);
        for (i, path) in paths.iter().enumerate() {
            eprintln!("  {i}: {}", path.display());
        }
    }
    let audio = args
        .audio
        .as_ref()
        .map(|p| media::decode_audio_limited(p, options.audio_max_seconds))
        .transpose()?;
    if let Some((_, n)) = &audio {
        let tokens = (*n as usize).div_ceil(800) * 2;
        eprintln!(
            "audio: {:.2} seconds, {tokens} tokens before budget truncation",
            *n as f64 / 32000.0
        );
        if tokens > options.audio_max_tokens && !options.truncate_audio {
            bail!("audio exceeds budget; use --truncate-audio or shorten --audio-max-seconds");
        }
    }
    if args.dry_run {
        return Ok(());
    }
    if !super::dir_writable(&args.out) {
        bail!("output directory is missing or not writable");
    }
    let resolver = h3_hrx::models::Resolver::new().offline(args.offline);
    let resolve = |explicit: Option<PathBuf>, relative: &str| -> Result<PathBuf> {
        match explicit {
            Some(p) => Ok(p),
            None => Ok(resolver.find(relative)?),
        }
    };
    let config = Config {
        video_vae: if decoded.is_empty() {
            None
        } else {
            Some(resolve(args.video_vae, h3_hrx::models::VIDEO_VAE)?)
        },
        audio_vae: if audio.is_none() {
            None
        } else {
            Some(resolve(args.audio_vae, h3_hrx::models::AUDIO_VAE)?)
        },
        kernel_sources: args.root.map(|p| p.join("kernels")).unwrap_or_default(),
        loom_library: std::env::var_os("HRX_LOOM_LIBRARY").map(PathBuf::from),
        ..Config::default()
    };
    // Safety: this command only reads the checkpoints and never replaces them.
    let mut session = unsafe { Session::new(config) }?;
    let images = decoded
        .iter()
        .map(|(pixels, w, h)| ImageInput {
            pixels,
            width: *w,
            height: *h,
        })
        .collect::<Vec<_>>();
    let audio = audio.as_ref().map(|(samples, n)| AudioInput {
        samples,
        samples_per_channel: *n as usize,
    });
    let mods = session.create_refmod(&images, audio, &options)?;
    report(&mods)?;
    mods.save(&args.out, args.force)?;
    eprintln!("saved {}", args.out.display());
    Ok(())
}

fn indexed<T: std::str::FromStr>(
    values: &[String],
    len: usize,
    name: &str,
) -> Result<Vec<(usize, T)>> {
    let mut out = Vec::new();
    for value in values {
        let (index, number) = value
            .split_once('=')
            .ok_or_else(|| UsageError(format!("--{name}: use INDEX=VALUE")))?;
        let index = index
            .parse::<usize>()
            .ok()
            .filter(|i| *i > 0 && *i <= len)
            .ok_or_else(|| {
                UsageError(format!("--{name}: index must name a --refmod (1..={len})"))
            })?;
        if out.iter().any(|(i, _)| *i == index - 1) {
            bail!("--{name}: duplicate index {index}");
        }
        out.push((
            index - 1,
            number
                .parse::<T>()
                .map_err(|_| UsageError(format!("--{name}: invalid value {number}")))?,
        ));
    }
    Ok(out)
}
pub fn prepare(cli: &Cli) -> Result<Vec<PreparedRefMod>> {
    let mut options = vec![ApplyOptions::default(); cli.refmods.len()];
    for (i, s) in indexed::<f32>(
        &cli.refmod_visual_strength,
        options.len(),
        "refmod-visual-strength",
    )? {
        options[i].visual_strength = s;
    }
    for (i, s) in indexed::<f32>(
        &cli.refmod_audio_strength,
        options.len(),
        "refmod-audio-strength",
    )? {
        options[i].audio_strength = s;
    }
    for (i, c) in indexed::<usize>(&cli.refmod_copies, options.len(), "refmod-copies")? {
        options[i].copies = c;
    }
    let mut prepared = Vec::new();
    let mut total = 0usize;
    for (path, mut option) in cli.refmods.iter().zip(options) {
        option.max_tokens = cli.refmod_max_total_tokens.map(|n| n.saturating_sub(total));
        let mods = RefMod::load(path)?;
        for m in mods.members() {
            if m.metadata().get("refmod_config").is_some() {
                eprintln!("{}: saved config/curves are not applied", path.display());
            }
            if let Some(description) = m.metadata()["description"]
                .as_str()
                .filter(|s| !s.is_empty())
            {
                eprintln!("{}: {description}", path.display());
            }
        }
        let item = mods
            .prepare(option)
            .with_context(|| format!("prepare {}", path.display()))?;
        total = total
            .checked_add(item.token_count())
            .context("refmod token total overflows")?;
        prepared.push(item);
    }
    if total > 0 {
        eprintln!("refmods: {total} effective reference tokens");
    }
    Ok(prepared)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    fn fixture() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/refmod/combined.safetensors")
    }

    #[test]
    fn combined_slots_respect_modality_controls_copies_and_total_budget() {
        let path = fixture();
        let cli = Cli::try_parse_from([
            "h3",
            "--refmod",
            path.to_str().unwrap(),
            "--refmod",
            path.to_str().unwrap(),
            "--refmod-audio-strength",
            "1=0",
            "--refmod-visual-strength",
            "2=0",
            "--refmod-copies",
            "1=2",
            "--refmod-max-total-tokens",
            "74",
        ])
        .unwrap();
        let prepared = prepare(&cli).unwrap();
        assert_eq!(prepared[0].token_count(), 36);
        assert_eq!(prepared[1].token_count(), 38);
        assert!(matches!(
            prepared[0].references()[0],
            h3_hrx::Reference::Video { .. }
        ));
        assert!(matches!(
            prepared[1].references()[0],
            h3_hrx::Reference::Audio { .. }
        ));
        assert_eq!(prepared[0].references().len(), 2);
        let mut too_small = cli;
        too_small.refmod_max_total_tokens = Some(73);
        assert!(prepare(&too_small).is_err());
    }

    #[test]
    fn disabled_mods_cost_zero_and_active_mods_reject_turbo_before_models() {
        let path = fixture();
        let mut cli = Cli::try_parse_from([
            "h3",
            "--refmod",
            path.to_str().unwrap(),
            "--preset",
            "turbo-768p-4",
            "--refmod-visual-strength",
            "1=0",
            "--refmod-audio-strength",
            "1=0",
        ])
        .unwrap();
        assert_eq!(prepare(&cli).unwrap()[0].token_count(), 0);
        assert!(super::super::parameters(&cli).is_ok());
        cli.refmod_visual_strength.clear();
        let error = super::super::run(cli).unwrap_err();
        assert!(error
            .to_string()
            .contains("Turbo does not support active refmods"));
    }

    #[test]
    fn commands_and_existing_generation_parse() {
        assert!(Cli::try_parse_from([
            "h3",
            "refmod",
            "create",
            "images",
            "--audio",
            "voice.wav",
            "--out",
            "x.safetensors",
            "--dry-run"
        ])
        .is_ok());
        assert!(Cli::try_parse_from(["h3", "refmod", "inspect", "x.safetensors"]).is_ok());
        let cli = Cli::try_parse_from([
            "h3",
            "photo.png",
            "--refmod",
            "x.safetensors",
            "--refmod-audio-strength",
            "1=0",
            "-p",
            "hello",
        ])
        .unwrap();
        assert_eq!(cli.files, vec![PathBuf::from("photo.png")]);
        assert_eq!(
            indexed::<f32>(&cli.refmod_audio_strength, 1, "strength").unwrap(),
            vec![(0, 0.)]
        );
        assert!(indexed::<f32>(&["2=1".into()], 1, "strength").is_err());
        assert!(indexed::<usize>(&["1=2".into(), "1=3".into()], 1, "copies").is_err());
    }
    #[test]
    fn directory_order_is_sorted_and_nonrecursive() {
        let dir = tempfile::tempdir().unwrap();
        for name in ["b.png", "a.jpg", "skip.txt"] {
            std::fs::write(dir.path().join(name), []).unwrap();
        }
        std::fs::create_dir(dir.path().join("nested")).unwrap();
        std::fs::write(dir.path().join("nested/c.png"), []).unwrap();
        let paths = image_paths(&[dir.path().into()]).unwrap();
        assert_eq!(
            paths,
            vec![dir.path().join("a.jpg"), dir.path().join("b.png")]
        );
    }
}

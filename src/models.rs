//! Finding the four checkpoints: a local directory, the shared Hugging Face cache, or the hub.
//!
//! The repository lays its files out the same way ComfyUI does — `diffusion_models/`, `text_encoders/`,
//! `vae/` — so one relative path names a file in either place. A local directory is tried first, then
//! whatever is already in the shared cache, and only then the network. That order means an existing
//! download is reused wherever it lives, and nothing is fetched twice.
use std::path::{Path, PathBuf};

pub const REPO_OWNER: &str = "Comfy-Org";
pub const REPO_NAME: &str = "MiniMax-H3";

pub const DIT_FL2VA: &str = "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors";
pub const DIT_REF2VA: &str = "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors";
pub const TE: &str = "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors";
pub const VIDEO_VAE: &str = "vae/minimax_h3_video_vae_fp16.safetensors";
pub const AUDIO_VAE: &str = "vae/minimax_h3_audio_vae_fp32.safetensors";

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{name} not found: {tried}")]
    NotFound { name: String, tried: String },
    #[error("{name}: {source}")]
    Hub {
        name: String,
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

pub type Result<T> = std::result::Result<T, Error>;

/// Where checkpoints are looked for, in order.
pub struct Resolver {
    dir: Option<PathBuf>,
    revision: Option<String>,
    /// When false, only files already on disk are used and the hub is never contacted.
    download: bool,
}

impl Default for Resolver {
    fn default() -> Self {
        Self::new()
    }
}

impl Resolver {
    /// Where checkpoints are looked for by default: `H3_MODELS`, else `~/comfy-models`.
    pub fn default_root() -> PathBuf {
        std::env::var_os("H3_MODELS")
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                let home = std::env::var_os("HOME").unwrap_or_default();
                Path::new(&home).join("comfy-models")
            })
    }

    /// `H3_MODELS`, else `~/comfy-models` when it exists; downloads allowed.
    pub fn new() -> Self {
        let dir = std::env::var_os("H3_MODELS")
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
            .or_else(|| {
                let home = std::env::var_os("HOME")?;
                let guess = Path::new(&home).join("comfy-models");
                guess.is_dir().then_some(guess)
            });
        Self {
            dir,
            revision: None,
            download: true,
        }
    }

    pub fn with_dir(mut self, dir: Option<PathBuf>) -> Self {
        if dir.is_some() {
            self.dir = dir;
        }
        self
    }

    pub fn revision(mut self, revision: Option<String>) -> Self {
        self.revision = revision;
        self
    }

    /// Use only what is already on disk. A missing file is then an error rather than a download.
    pub fn offline(mut self, offline: bool) -> Self {
        self.download = !offline;
        self
    }

    /// The path of one checkpoint, fetching it into the shared cache if it is not already somewhere.
    pub fn find(&self, relative: &str) -> Result<PathBuf> {
        if let Some(path) = self.local(relative) {
            return Ok(path);
        }
        // Already in the shared cache? This never touches the network.
        if let Ok(path) = self.hub(relative, true) {
            return Ok(path);
        }
        if !self.download {
            return Err(self.missing(relative));
        }
        self.hub(relative, false)
    }

    /// As [`Resolver::find`], but a file that is simply absent is not an error: the ref2va checkpoint
    /// is optional, and asking for it must not start a 21 GB download.
    pub fn find_local(&self, relative: &str) -> Option<PathBuf> {
        self.local(relative)
            .or_else(|| self.hub(relative, true).ok())
    }

    fn local(&self, relative: &str) -> Option<PathBuf> {
        let path = self.dir.as_ref()?.join(relative);
        path.is_file().then_some(path)
    }

    fn hub(&self, relative: &str, cached_only: bool) -> Result<PathBuf> {
        let client = hf_hub::HFClientSync::new().map_err(|e| Error::Hub {
            name: relative.into(),
            source: Box::new(e),
        })?;
        client
            .model(REPO_OWNER, REPO_NAME)
            .download_file()
            .filename(relative)
            .maybe_revision(self.revision.clone())
            .local_files_only(cached_only)
            .send()
            .map_err(|e| Error::Hub {
                name: relative.into(),
                source: Box::new(e),
            })
    }

    fn missing(&self, relative: &str) -> Error {
        let mut tried = String::new();
        if let Some(dir) = &self.dir {
            tried.push_str(&format!("not under {}", dir.display()));
        } else {
            tried.push_str("no models directory (--models or H3_MODELS)");
        }
        tried.push_str(&format!(
            ", and not in the Hugging Face cache for {REPO_OWNER}/{REPO_NAME}"
        ));
        Error::NotFound {
            name: relative.into(),
            tried,
        }
    }
}

/// The four checkpoints a session needs.
#[derive(Debug, Clone)]
pub struct Checkpoints {
    pub dit: PathBuf,
    pub te: PathBuf,
    pub video_vae: PathBuf,
    pub audio_vae: PathBuf,
}

impl Checkpoints {
    /// Resolves all four, with the caller naming which DiT it wants. Choosing between the base and the
    /// reference-conditioned checkpoint is policy — the command line errors rather than silently
    /// substituting one for the other — so it is not decided here.
    pub fn resolve(resolver: &Resolver, dit: &str) -> Result<Self> {
        Ok(Self {
            dit: resolver.find(dit)?,
            te: resolver.find(TE)?,
            video_vae: resolver.find(VIDEO_VAE)?,
            audio_vae: resolver.find(AUDIO_VAE)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_local_directory_wins_and_needs_no_network() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(VIDEO_VAE);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"not really a checkpoint").unwrap();
        let r = Resolver::new()
            .with_dir(Some(dir.path().to_path_buf()))
            .offline(true);
        assert_eq!(r.find(VIDEO_VAE).unwrap(), path);
    }

    #[test]
    fn offline_reports_where_it_looked() {
        let dir = tempfile::tempdir().unwrap();
        let r = Resolver::new()
            .with_dir(Some(dir.path().to_path_buf()))
            .offline(true);
        // A name the repository does not have, so the result cannot depend on what this machine
        // happens to have cached.
        const ABSENT: &str = "vae/not-a-real-checkpoint.safetensors";
        let message = r.find(ABSENT).unwrap_err().to_string();
        assert!(message.contains(ABSENT), "{message}");
        assert!(message.contains(dir.path().to_str().unwrap()), "{message}");
        assert!(message.contains("Hugging Face cache"), "{message}");
    }

    #[test]
    fn an_optional_checkpoint_does_not_start_a_download() {
        let dir = tempfile::tempdir().unwrap();
        // download allowed, but find_local must still not reach for the network
        let r = Resolver::new().with_dir(Some(dir.path().to_path_buf()));
        assert!(r
            .find_local("diffusion_models/definitely-not-a-real-file.safetensors")
            .is_none());
    }

    #[test]
    fn the_repository_paths_match_what_the_readme_documents() {
        // These are the --include arguments in the README's `hf download` line, and the same relative
        // paths ComfyUI uses, which is why one string serves both.
        for path in [DIT_FL2VA, DIT_REF2VA, TE, VIDEO_VAE, AUDIO_VAE] {
            assert!(path.ends_with(".safetensors"), "{path}");
            let dir = path.split('/').next().unwrap();
            assert!(
                matches!(dir, "diffusion_models" | "text_encoders" | "vae"),
                "{path}"
            );
        }
    }
}

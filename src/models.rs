//! Resolve checkpoint files through the standard Hugging Face Hub cache.
//!
//! Repository-relative filenames are passed directly to the Hub client, which owns
//! snapshot paths, cache lookup, and downloads. No separate model directory is used.
use hrx::artifacts::hf::{HubFile, Repository, Resolver as HubResolver};
use std::path::PathBuf;

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

/// Resolve Comfy-Org checkpoints through the shared Hugging Face Hub cache.
pub struct Resolver {
    owner: String,
    repository: String,
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
    /// Resolve through the standard Hugging Face cache, downloading missing files on demand.
    pub fn new() -> Self {
        Self {
            owner: REPO_OWNER.into(),
            repository: REPO_NAME.into(),
            revision: None,
            download: true,
        }
    }

    pub fn revision(mut self, revision: Option<String>) -> Self {
        self.revision = revision;
        self
    }

    /// Select another Hub model repository, using the same standard cache and offline policy.
    pub fn repository(mut self, owner: &str, repository: &str) -> Self {
        self.owner = owner.into();
        self.repository = repository.into();
        self
    }

    /// Use only what is already on disk. A missing file is then an error rather than a download.
    pub fn offline(mut self, offline: bool) -> Self {
        self.download = !offline;
        self
    }

    /// The path of one checkpoint, fetching it into the shared cache if it is not already cached.
    pub fn find(&self, relative: &str) -> Result<PathBuf> {
        let resolver = self.hub();
        let file = HubFile::new(relative);
        if let Ok(Some(path)) = resolver.local(&file) {
            return Ok(path);
        }
        if !self.download {
            return Err(self.missing(relative));
        }
        resolver.resolve(&file).map_err(|source| Error::Hub {
            name: relative.into(),
            source: Box::new(source),
        })
    }

    /// As [`Resolver::find`], but a file that is simply absent is not an error: the ref2va checkpoint
    /// is optional, and asking for it must not start a 21 GB download.
    pub fn find_local(&self, relative: &str) -> Option<PathBuf> {
        self.hub().local(&HubFile::new(relative)).ok().flatten()
    }

    fn hub(&self) -> HubResolver {
        let mut repository = Repository::new(&self.owner, &self.repository);
        if let Some(revision) = &self.revision {
            repository = repository.at(revision);
        }
        HubResolver::new(repository)
    }

    fn missing(&self, relative: &str) -> Error {
        let tried = format!(
            "not in the Hugging Face cache for {}/{}",
            self.owner, self.repository
        );
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
    fn offline_reports_where_it_looked() {
        let r = Resolver::new().offline(true);
        // A name the repository does not have, so the result cannot depend on what this machine
        // happens to have cached.
        const ABSENT: &str = "vae/not-a-real-checkpoint.safetensors";
        let message = r.find(ABSENT).unwrap_err().to_string();
        assert!(message.contains(ABSENT), "{message}");
        assert!(message.contains("Hugging Face cache"), "{message}");
    }

    #[test]
    fn an_optional_checkpoint_does_not_start_a_download() {
        // Download is allowed, but find_local must still not reach for the network.
        let r = Resolver::new();
        assert!(r
            .find_local("diffusion_models/definitely-not-a-real-file.safetensors")
            .is_none());
    }
}

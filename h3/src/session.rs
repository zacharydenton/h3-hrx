//! A session: the four checkpoints, the compiler and the device, opened lazily and reused.
//!
//! This is the Rust API — slices, borrows, `Result` — that `capi` wraps and that `cli/` uses directly.
//! Checkpoints open on first use because most callers want one or two of them: decoding a clip needs
//! the video VAE alone, and opening the 32B text encoder to do it would cost two minutes for nothing.
use crate::avae::AudioVae;
use crate::compile::Compiler;
use crate::dispatch::Profile;
use crate::dit::{DenoiseParams, Dit, KeyframeInput, Latents, Noise, RefInput};
use crate::error::{invalid, Result};
use crate::layout::{shape_for, Shape};
use crate::te::TextEncoder;
use crate::vvae::{Clip, VideoVae};

/// Where the four checkpoints live, and how kernels are built.
pub struct Config {
    pub dit: Option<std::path::PathBuf>,
    pub te: Option<std::path::PathBuf>,
    pub video_vae: Option<std::path::PathBuf>,
    pub audio_vae: Option<std::path::PathBuf>,
    pub kernel_sources: std::path::PathBuf,
    pub cache_dir: std::path::PathBuf,
    pub loom_compile: String,
    /// the DiT attention's QK operands: 16, 8 or 4
    pub attn_qk_bits: usize,
}

pub struct Session {
    gpu: hrx::Gpu,
    compiler: Compiler,
    config: Config,
    prof: Profile,
    dit: Option<Dit>,
    te: Option<TextEncoder>,
    vvae: Option<VideoVae>,
    avae: Option<AudioVae>,
}

impl Session {
    pub fn new(config: Config) -> Result<Self> {
        if !matches!(config.attn_qk_bits, 4 | 8 | 16) {
            return invalid("attn_qk_bits must be 4, 8 or 16");
        }
        let compiler = Compiler::new(
            config.loom_compile.clone(),
            config.kernel_sources.clone(),
            config.cache_dir.clone(),
        );
        Ok(Self {
            gpu: hrx::Gpu::open()?,
            compiler,
            config,
            prof: Profile::from_env(),
            dit: None,
            te: None,
            vvae: None,
            avae: None,
        })
    }

    /// The shapes a request produces, or `None` when it is not one this model serves.
    pub fn shape_for(height: i32, width: i32, frames: i32) -> Option<Shape> {
        shape_for(height, width, frames)
    }

    fn dit(&mut self) -> Result<&mut Dit> {
        if self.dit.is_none() {
            let Some(path) = &self.config.dit else {
                return invalid("no DiT checkpoint was configured");
            };
            self.dit = Some(Dit::open(&self.gpu, path)?);
        }
        Ok(self.dit.as_mut().expect("opened above"))
    }

    fn te(&mut self) -> Result<&mut TextEncoder> {
        if self.te.is_none() {
            let Some(path) = &self.config.te else {
                return invalid("no text encoder checkpoint was configured");
            };
            self.te = Some(TextEncoder::open(&self.gpu, path)?);
        }
        Ok(self.te.as_mut().expect("opened above"))
    }

    fn vvae(&mut self) -> Result<&mut VideoVae> {
        if self.vvae.is_none() {
            let Some(path) = &self.config.video_vae else {
                return invalid("no video VAE checkpoint was configured");
            };
            self.vvae = Some(VideoVae::open(&self.gpu, path)?);
        }
        Ok(self.vvae.as_mut().expect("opened above"))
    }

    fn avae(&mut self) -> Result<&mut AudioVae> {
        if self.avae.is_none() {
            let Some(path) = &self.config.audio_vae else {
                return invalid("no audio VAE checkpoint was configured");
            };
            self.avae = Some(AudioVae::open(&self.gpu, path)?);
        }
        Ok(self.avae.as_mut().expect("opened above"))
    }

    /// Both checkpoints the prompt path needs, borrowed at once.
    fn prompt_pair(
        &mut self,
    ) -> Result<(
        &hrx::Gpu,
        &Compiler,
        &mut Profile,
        &mut Dit,
        &mut TextEncoder,
    )> {
        self.dit()?;
        self.te()?;
        let Self {
            gpu,
            compiler,
            prof,
            dit,
            te,
            ..
        } = self;
        Ok((
            gpu,
            compiler,
            prof,
            dit.as_mut().expect("opened"),
            te.as_mut().expect("opened"),
        ))
    }

    /// The refined text rows the blocks see, `[n][5376]`.
    pub fn text_in(&mut self, ids: &[i32], out: &mut [f32]) -> Result<()> {
        let (gpu, c, prof, dit, te) = self.prompt_pair()?;
        dit.text_in(gpu, c, prof, te, ids, &[])?;
        dit.read_rows(gpu, ids.len(), out)
    }

    /// The whole denoising run.
    #[allow(clippy::too_many_arguments)]
    pub fn denoise(
        &mut self,
        ids: &[i32],
        p: &DenoiseParams,
        noise: Noise<'_>,
        refs: &[RefInput<'_>],
        kfs: &[KeyframeInput<'_>],
        progress: Option<&mut dyn FnMut(usize, usize, f64) -> bool>,
    ) -> Result<Latents> {
        let (gpu, c, prof, dit, te) = self.prompt_pair()?;
        dit.denoise(gpu, c, prof, te, ids, p, noise, refs, kfs, progress)
    }

    /// The vision tower over one image.
    pub fn vision_embed(
        &mut self,
        pixels: &[f32],
        height: usize,
        width: usize,
    ) -> Result<crate::vision::Embedding> {
        self.te()?;
        let Self {
            gpu,
            compiler,
            prof,
            te,
            ..
        } = self;
        let te = te.as_ref().expect("opened");
        crate::vision::embed(gpu, compiler, prof, te.weights(), pixels, height, width)
    }

    pub fn decode_video(&mut self, shape: &Shape, latents: &[f32], out: &mut [u8]) -> Result<()> {
        self.vvae()?;
        let Self {
            gpu,
            compiler,
            prof,
            vvae,
            ..
        } = self;
        vvae.as_mut()
            .expect("opened")
            .decode_video(gpu, compiler, prof, shape, latents, out)
    }

    pub fn encode_video(&mut self, clip: Clip<'_>) -> Result<(Vec<f32>, usize)> {
        self.vvae()?;
        let Self {
            gpu,
            compiler,
            prof,
            vvae,
            ..
        } = self;
        vvae.as_mut()
            .expect("opened")
            .encode_video(gpu, compiler, prof, clip)
    }

    pub fn decode_audio(
        &mut self,
        latents: &[f32],
        audio_t: usize,
        samples: &mut [f32],
    ) -> Result<()> {
        self.avae()?;
        let Self {
            gpu,
            compiler,
            prof,
            avae,
            ..
        } = self;
        avae.as_mut()
            .expect("opened")
            .decode(gpu, compiler, prof, latents, audio_t, samples)
    }

    pub fn encode_audio(&mut self, samples: &[f32], n: usize) -> Result<(Vec<f32>, usize)> {
        self.avae()?;
        let Self {
            gpu,
            compiler,
            prof,
            avae,
            ..
        } = self;
        avae.as_mut()
            .expect("opened")
            .encode(gpu, compiler, prof, samples, n)
    }
}

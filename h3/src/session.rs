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

    /// The stage timings gathered since the last call, and a reset.
    ///
    /// Empty unless `H3_PROFILE` is set, which is what turns on the synchronise around each launch.
    pub fn profile_report(&mut self) -> Option<String> {
        if !self.prof.on {
            return None;
        }
        let report = self.prof.report(0.01);
        self.prof.take();
        (!report.is_empty()).then_some(report)
    }

    /// The attention width this session was created with. The stack is built for it, so it is not a
    /// per-run parameter — and a run that quietly ignored it would make a precision comparison
    /// meaningless rather than wrong in any visible way.
    pub fn attn_qk_bits(&self) -> usize {
        self.config.attn_qk_bits
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
        // checked before a checkpoint is opened: a bad request should cost nothing
        if ids.is_empty() {
            return invalid("ids must hold at least one token");
        }
        if out.len() < ids.len() * crate::model::HID {
            return invalid(format!(
                "text_in needs {} floats for {} tokens, {} given",
                ids.len() * crate::model::HID,
                ids.len(),
                out.len()
            ));
        }
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
        // the attention width is the session's, set once at creation: the stack is built for it
        let qk_bits = self.config.attn_qk_bits;
        let (gpu, c, prof, dit, te) = self.prompt_pair()?;
        dit.denoise(
            gpu, c, prof, te, ids, p, qk_bits, noise, refs, kfs, progress,
        )
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dit::{KeyframeInput, RefInput};
    use crate::vvae::Clip;

    fn config(bits: usize) -> Config {
        Config {
            dit: None,
            te: None,
            video_vae: None,
            audio_vae: None,
            kernel_sources: "kernels".into(),
            cache_dir: "build/kernel_cache".into(),
            loom_compile: "loom-compile".into(),
            attn_qk_bits: bits,
        }
    }

    #[test]
    fn a_clip_must_match_the_buffer_it_names() {
        let pixels = vec![0.0f32; 2 * 32 * 32 * 3];
        let ok = Clip {
            pixels: &pixels,
            frames: 2,
            height: 32,
            width: 32,
        };
        assert!(ok.check().is_ok());
        // no frames at all: this reached a temporal underflow inside the encoder
        assert!(Clip { frames: 0, ..ok }.check().is_err());
        assert!(Clip { height: 0, ..ok }.check().is_err());
        // a shape the buffer cannot cover
        assert!(Clip { frames: 3, ..ok }.check().is_err());
        assert!(Clip {
            height: 64,
            width: 64,
            ..ok
        }
        .check()
        .is_err());
        // and one that is not a whole number of patches
        assert!(Clip { height: 40, ..ok }.check().is_err());
    }

    #[test]
    fn a_reference_must_match_the_buffers_it_names() {
        let z = vec![0.0f32; crate::model::LATENT_CH * 4 * 4];
        let image = RefInput {
            kind: 0,
            video_latent: Some(&z),
            latent_t: 1,
            lat_h: 4,
            lat_w: 4,
            audio_latent: None,
            audio_t: 0,
            pixels: None,
            height: 0,
            width: 0,
        };
        assert!(image.check(0).is_ok());
        // latents that do not cover the grid claimed
        assert!(RefInput { lat_h: 8, ..image }.check(0).is_err());
        // a visual reference with no latents at all
        assert!(RefInput {
            video_latent: None,
            ..image
        }
        .check(0)
        .is_err());
        // an audio reference needs audio
        assert!(RefInput {
            kind: 1,
            video_latent: None,
            ..image
        }
        .check(0)
        .is_err());
        // and an unknown kind is refused rather than silently treated as visual
        assert!(RefInput { kind: 7, ..image }.check(0).is_err());
    }

    #[test]
    fn a_keyframe_must_sit_on_the_generations_grid() {
        let z = vec![0.0f32; crate::model::LATENT_CH * 16 * 16];
        let kf = KeyframeInput {
            frame_index: 0,
            video_latent: &z,
            audio_latent: None,
            audio_t: 0,
            pixels: None,
            height: 0,
            width: 0,
        };
        assert!(kf.check(0, 16, 16).is_ok());
        assert!(
            kf.check(0, 32, 32).is_err(),
            "a larger grid needs more latents"
        );
        assert!(kf.check(0, 0, 16).is_err(), "an empty grid is not a grid");
    }

    #[test]
    fn only_the_three_attention_widths_are_accepted() {
        for bad in [0, 1, 2, 7, 9, 32] {
            let e = Session::new(config(bad));
            assert!(
                matches!(e, Err(crate::error::Error::Invalid(_))),
                "{bad} bits should be refused"
            );
        }
    }
}

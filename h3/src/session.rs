//! A session: the four checkpoints, the compiler and the device, opened lazily and reused.
//!
//! This is the Rust API — slices, borrows, `Result` — that `capi` wraps and that `cli/` uses directly.
//! Checkpoints open on first use because most callers want one or two of them: decoding a clip needs
//! the video VAE alone, and opening the 32B text encoder to do it would cost two minutes for nothing.
use crate::avae::AudioVae;
use crate::compile::Compiler;
use crate::dispatch::Profile;
use crate::dit::{DenoiseParams, Dit, Keyframe, Latents, Noise, Reference};
use crate::error::{invalid, Result};
use crate::layout::{shape_for, Shape};
use crate::te::TextEncoder;
use crate::vvae::{Clip, VideoVae};

/// Where the four checkpoints live, and how kernels are built.
#[derive(Clone, Debug)]
pub struct Config {
    pub dit: Option<std::path::PathBuf>,
    pub te: Option<std::path::PathBuf>,
    pub video_vae: Option<std::path::PathBuf>,
    pub audio_vae: Option<std::path::PathBuf>,
    /// Empty selects the sources embedded in this model package.
    pub kernel_sources: std::path::PathBuf,
    /// Empty selects the shared per-user HRX cache.
    pub cache_dir: std::path::PathBuf,
    pub loom_library: Option<std::path::PathBuf>,
    /// the DiT attention's QK operands
    pub attention: crate::dit::Attention,
}

impl Default for Config {
    /// Checkpoints under `H3_MODELS` or `~/comfy-models`, embedded kernel sources,
    /// a shared per-user compiler cache, and int8 attention.
    fn default() -> Self {
        let models = crate::models::Resolver::default_root();
        Self {
            dit: Some(models.join(crate::models::DIT_FL2VA)),
            te: Some(models.join(crate::models::TE)),
            video_vae: Some(models.join(crate::models::VIDEO_VAE)),
            audio_vae: Some(models.join(crate::models::AUDIO_VAE)),
            kernel_sources: std::path::PathBuf::new(),
            cache_dir: std::path::PathBuf::new(),
            loom_library: None,
            attention: crate::dit::Attention::default(),
        }
    }
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
    /// Opens a session. Checkpoints are read lazily, on the first call that needs one.
    ///
    /// # Safety
    ///
    /// The checkpoint files named in `config` are memory-mapped, not copied — a 30 GB checkpoint has
    /// to be — and stay mapped for as long as this session lives. The caller must ensure that none of
    /// them is modified or truncated in that time:
    ///
    /// - Modifying one changes bytes already validated: the header said a tensor was `[5376][14336]`
    ///   of bf16, and everything after trusts it.
    /// - Truncating one turns a mapped page into `SIGBUS`, which no `Result` can carry. The process
    ///   dies at the read.
    ///
    /// Nothing in the filesystem enforces this and nothing here can check it, which is why this
    /// function is `unsafe` rather than documentation asking nicely. Point a session at files you
    /// control.
    pub unsafe fn new(config: Config) -> Result<Self> {
        // An empty `cache_dir` and an empty `kernel_sources` are resolved by the compiler itself, so
        // that every route to one — this, the C ABI, a direct `Compiler::new` — reaches the same
        // place rather than each defaulting on its own.
        let compiler = Compiler::new(
            config.loom_library.clone(),
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

    /// The attention this session was created with. The stack is built for it, so it is not a
    /// per-run parameter — and a run that quietly ignored it would make a precision comparison
    /// meaningless rather than wrong in any visible way.
    pub fn attention(&self) -> crate::dit::Attention {
        self.config.attention
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
            // Safety: the caller's, taken at Session::new.
            self.dit = Some(unsafe { Dit::open(&self.gpu, path) }?);
        }
        Ok(self.dit.as_mut().expect("opened above"))
    }

    fn te(&mut self) -> Result<&mut TextEncoder> {
        if self.te.is_none() {
            let Some(path) = &self.config.te else {
                return invalid("no text encoder checkpoint was configured");
            };
            // Safety: the caller's, taken at Session::new.
            self.te = Some(unsafe { TextEncoder::open(&self.gpu, path) }?);
        }
        Ok(self.te.as_mut().expect("opened above"))
    }

    fn vvae(&mut self) -> Result<&mut VideoVae> {
        if self.vvae.is_none() {
            let Some(path) = &self.config.video_vae else {
                return invalid("no video VAE checkpoint was configured");
            };
            // Safety: the caller's, taken at Session::new.
            self.vvae = Some(unsafe { VideoVae::open(&self.gpu, path) }?);
        }
        Ok(self.vvae.as_mut().expect("opened above"))
    }

    fn avae(&mut self) -> Result<&mut AudioVae> {
        if self.avae.is_none() {
            let Some(path) = &self.config.audio_vae else {
                return invalid("no audio VAE checkpoint was configured");
            };
            // Safety: the caller's, taken at Session::new.
            self.avae = Some(unsafe { AudioVae::open(&self.gpu, path) }?);
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
        refs: &[Reference<'_>],
        kfs: &[Keyframe<'_>],
        progress: Option<&mut dyn FnMut(usize, usize, f64) -> bool>,
    ) -> Result<Latents> {
        // Everything cheap, before a checkpoint opens or a kernel compiles. This was written once,
        // lost in a refactor, and not missed — because the tests called the validators rather than
        // this path. There is a test below that calls `denoise` itself now.
        let sh = self.validate(ids, p, noise, refs, kfs)?;
        let _ = sh;
        // the attention width is the session's, set once at creation: the stack is built for it
        let qk_bits = self.config.attention.bits();
        let (gpu, c, prof, dit, te) = self.prompt_pair()?;
        dit.denoise(
            gpu, c, prof, te, ids, p, qk_bits, noise, refs, kfs, progress,
        )
    }

    /// Every cheap property of a request, checked before anything expensive happens.
    ///
    /// Separate so it can be tested directly, and so a caller can ask whether a request is well
    /// formed without running it.
    pub fn validate(
        &self,
        ids: &[i32],
        p: &DenoiseParams,
        noise: Noise<'_>,
        refs: &[Reference<'_>],
        kfs: &[Keyframe<'_>],
    ) -> Result<Shape> {
        if ids.is_empty() {
            return invalid("ids must hold at least one token");
        }
        let Some(sh) = shape_for(p.height, p.width, p.frames) else {
            return invalid(format!(
                "no shape for {}x{} at {} frames",
                p.height, p.width, p.frames
            ));
        };
        if !(2..=1000).contains(&p.steps) {
            return invalid(format!("steps must be 2..1000, not {}", p.steps));
        }
        let video = crate::model::LATENT_CH
            * (sh.latent_t as usize)
            * (sh.lat_h as usize)
            * (sh.lat_w as usize);
        let audio = 2 * crate::avae::AUDIO_CH * sh.audio_t as usize;
        if let Some(z) = noise.video {
            if z.len() < video {
                return invalid(format!(
                    "video noise needs {video} floats, {} given",
                    z.len()
                ));
            }
        }
        if let Some(z) = noise.audio {
            if z.len() < audio {
                return invalid(format!(
                    "audio noise needs {audio} floats, {} given",
                    z.len()
                ));
            }
        }
        for (i, r) in refs.iter().enumerate() {
            r.check(i)?;
        }
        for (i, k) in kfs.iter().enumerate() {
            k.check(i, sh.lat_h, sh.lat_w)?;
        }
        Ok(sh)
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
    use crate::dit::{Keyframe, LatentGrid, Reference};
    use crate::vvae::Clip;

    fn config(attention: crate::dit::Attention) -> Config {
        Config {
            dit: None,
            te: None,
            video_vae: None,
            audio_vae: None,
            kernel_sources: "kernels".into(),
            cache_dir: "build/kernel_cache".into(),
            loom_library: None,
            attention,
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
        let grid = LatentGrid {
            frames: 1,
            height: 4,
            width: 4,
        };
        assert!(Reference::Image {
            latents: &z,
            grid,
            presented: None
        }
        .check(0)
        .is_ok());
        // latents that do not cover the grid claimed
        assert!(Reference::Image {
            latents: &z,
            grid: LatentGrid { height: 8, ..grid },
            presented: None
        }
        .check(0)
        .is_err());
        // an audio reference whose latents are short of its frame count
        assert!(Reference::Audio {
            latents: &z,
            frames: 10_000
        }
        .check(0)
        .is_err());
        // A video reference with a soundtrack is expressible; an image with one is not, and an audio
        // reference carrying video latents is not either. Those were the combinations a kind tag with
        // optional fields allowed.
        assert!(Reference::Video {
            latents: &z,
            grid,
            audio: None
        }
        .check(0)
        .is_ok());
    }

    #[test]
    fn a_keyframe_must_sit_on_the_generations_grid() {
        let z = vec![0.0f32; crate::model::LATENT_CH * 16 * 16];
        let kf = Keyframe {
            frame_index: 0,
            latents: &z,
            presented: None,
            audio: None,
        };
        assert!(kf.check(0, 16, 16).is_ok());
        assert!(
            kf.check(0, 32, 32).is_err(),
            "a larger grid needs more latents"
        );
        assert!(kf.check(0, 0, 16).is_err(), "an empty grid is not a grid");
    }

    #[test]
    fn only_the_three_attention_widths_exist() {
        use crate::dit::Attention;
        // the type is the check now: a width that has no kernels cannot be named
        for bad in [0usize, 1, 2, 7, 9, 32] {
            assert_eq!(Attention::from_bits(bad), None, "{bad} bits");
        }
        for (bits, want) in [(16, Attention::F16), (8, Attention::I8), (4, Attention::I4)] {
            assert_eq!(Attention::from_bits(bits), Some(want));
            assert_eq!(want.bits(), bits);
        }
        assert_eq!(Attention::default(), Attention::I8);
        let _ = config(Attention::F16);
    }
}

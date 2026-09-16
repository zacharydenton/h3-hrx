//! A session: the four checkpoints, the compiler and the device, opened lazily and reused.
//!
//! The CLI and application adapters use this API directly.
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

/// How long a session keeps completed models on the device.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ResidencyPolicy {
    /// Keep weights for subsequent requests.
    #[default]
    Retain,
    /// Release completed stages before loading the next model.
    StageScoped,
}

/// Optional session behavior. Existing callers retain weights by default.
#[derive(Clone, Copy, Debug, Default)]
pub struct SessionOptions {
    pub residency: ResidencyPolicy,
    pub cache: crate::cache::CachePolicy,
    /// Experimental native Turbo path; requires its matching parameters and qualification.
    pub turbo: Option<crate::adapter::TurboPreset>,
}

/// Where the four checkpoints live, and how kernels are built.
#[derive(Clone, Debug)]
pub struct Config {
    /// Explicit checkpoint paths override automatic resolution through the Hugging Face cache.
    /// `None` resolves the corresponding checkpoint on first use.
    pub dit: Option<std::path::PathBuf>,
    pub te: Option<std::path::PathBuf>,
    pub video_vae: Option<std::path::PathBuf>,
    pub audio_vae: Option<std::path::PathBuf>,
    /// Empty selects the sources embedded in this model package.
    pub kernel_sources: std::path::PathBuf,
    pub loom_library: Option<std::path::PathBuf>,
    /// the DiT attention's QK operands
    pub attention: crate::dit::Attention,
}

impl Default for Config {
    /// Checkpoints resolved on demand through the Hugging Face cache, embedded kernel sources,
    /// a shared per-user compiler cache, and int8 attention.
    fn default() -> Self {
        Self {
            dit: None,
            te: None,
            video_vae: None,
            audio_vae: None,
            kernel_sources: std::path::PathBuf::new(),
            loom_library: None,
            attention: crate::dit::Attention::default(),
        }
    }
}

/// Resolve only the checkpoint needed by a stage; constructing a config does no I/O.
fn checkpoint_path(
    explicit: &Option<std::path::PathBuf>,
    relative: &str,
) -> Result<std::path::PathBuf> {
    match explicit {
        Some(path) => Ok(path.clone()),
        None => Ok(crate::models::Resolver::new().find(relative)?),
    }
}

/// One model, held open: the checkpoints mapped, the weights uploaded as stages ask for them, and
/// the kernels compiled for the shapes seen so far. Reuse a session across requests rather than
/// opening one per request — that is what keeps the weights resident and the kernels compiled.
///
/// Every method takes `&mut self`, so the borrow checker serialises the calls; nothing here needs a
/// lock of its own.
pub struct Session {
    stream: hrx::Stream,
    compiler: Compiler,
    config: Config,
    options: SessionOptions,
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
    /// The checkpoint files named in `config` or resolved from the cache are memory-mapped, not copied — a 30 GB checkpoint has
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
        // Safety: forwarded unchanged from this constructor's contract.
        unsafe { Self::new_with_options(config, SessionOptions::default()) }
    }

    /// Opens a session with explicit model residency behavior.
    ///
    /// # Safety
    /// The checkpoint immutability requirements of [`Session::new`] apply for the
    /// entire session, including files temporarily released and reopened later.
    pub unsafe fn new_with_options(config: Config, options: SessionOptions) -> Result<Self> {
        // An empty `kernel_sources` selects the sources embedded in this package. Compiled
        // artifacts go wherever HRX keeps them, which is one cache per user for every consumer.
        let compiler = Compiler::new(config.loom_library.clone(), config.kernel_sources.clone());
        Ok(Self {
            stream: hrx::Stream::open()?,
            compiler,
            config,
            options,
            prof: Profile::from_env(),
            dit: None,
            te: None,
            vvae: None,
            avae: None,
        })
    }

    /// The lifetime policy selected when this session was opened.
    pub fn residency(&self) -> ResidencyPolicy {
        self.options.residency
    }

    /// Fence before dropping model owners and the graph executions they contain.
    fn release_completed(&mut self, encoder: bool, dit: bool, vaes: bool) -> Result<()> {
        if self.options.residency == ResidencyPolicy::Retain {
            return Ok(());
        }
        self.stream.synchronize()?;
        if encoder {
            self.te = None;
        }
        if dit {
            self.dit = None;
        }
        if vaes {
            self.vvae = None;
            self.avae = None;
        }
        Ok(())
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
            let path = checkpoint_path(&self.config.dit, crate::models::DIT_FL2VA)?;
            crate::trace::checkpoint("dit", &path);
            // Safety: the caller's, taken at Session::new.
            let adapter = if let Some(preset) = self.options.turbo {
                let adapter_path = preset.resolve(false)?;
                crate::trace::checkpoint("adapter", &adapter_path);
                // Safety: session constructor also requires adapter cache files to remain immutable.
                Some(unsafe { crate::adapter::Adapter::open(&adapter_path) }?)
            } else {
                None
            };
            let mut dit = unsafe { Dit::open(&mut self.stream, &path) }?;
            if let Some(adapter) = adapter {
                dit.set_adapter(adapter);
            }
            self.dit = Some(dit);
        }
        Ok(self.dit.as_mut().expect("opened above"))
    }

    fn te(&mut self) -> Result<&mut TextEncoder> {
        if self.te.is_none() {
            let path = checkpoint_path(&self.config.te, crate::models::TE)?;
            crate::trace::checkpoint("te", &path);
            // Safety: the caller's, taken at Session::new.
            self.te = Some(unsafe { TextEncoder::open(&mut self.stream, &path) }?);
        }
        Ok(self.te.as_mut().expect("opened above"))
    }

    fn vvae(&mut self) -> Result<&mut VideoVae> {
        if self.vvae.is_none() {
            let path = checkpoint_path(&self.config.video_vae, crate::models::VIDEO_VAE)?;
            crate::trace::checkpoint("video_vae", &path);
            // Safety: the caller's, taken at Session::new.
            self.vvae = Some(unsafe { VideoVae::open(&mut self.stream, &path) }?);
        }
        Ok(self.vvae.as_mut().expect("opened above"))
    }

    fn avae(&mut self) -> Result<&mut AudioVae> {
        if self.avae.is_none() {
            let path = checkpoint_path(&self.config.audio_vae, crate::models::AUDIO_VAE)?;
            crate::trace::checkpoint("audio_vae", &path);
            // Safety: the caller's, taken at Session::new.
            self.avae = Some(unsafe { AudioVae::open(&mut self.stream, &path) }?);
        }
        Ok(self.avae.as_mut().expect("opened above"))
    }

    /// Both checkpoints the prompt path needs, borrowed at once.
    fn prompt_pair(
        &mut self,
    ) -> Result<(
        &mut hrx::Stream,
        &Compiler,
        &mut Profile,
        &mut Dit,
        &mut TextEncoder,
    )> {
        self.dit()?;
        self.te()?;
        let Self {
            stream,
            compiler,
            prof,
            dit,
            te,
            ..
        } = self;
        Ok((
            stream,
            compiler,
            prof,
            dit.as_mut().expect("opened"),
            te.as_mut().expect("opened"),
        ))
    }

    /// The refined text rows the blocks see, `[n][5376]`.
    ///
    /// `out` must hold at least that many floats; a longer one — a pooled buffer, say — is accepted
    /// and only the rows the request needs are written. Inputs are read from the caller's memory
    /// rather than copied, so `ids` and `out` must not overlap.
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
        let (stream, c, prof, dit, te) = self.prompt_pair()?;
        dit.text_in(stream, c, prof, te, ids, &[])?;
        dit.read_rows(stream, ids.len(), out)
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
        let cache_policy = self.options.cache;
        self.release_completed(false, false, true)?;
        crate::trace::event("conditioning_start", String::new);
        let result = (|| {
            let (stream, c, prof, dit, te) = self.prompt_pair()?;
            let prepared = dit.prepare_denoise(stream, c, prof, te, ids, p, refs, kfs)?;
            crate::trace::event("conditioning_ready", String::new);
            self.release_completed(true, false, false)?;
            crate::trace::event("encoder_released", || {
                format!(",\"released\":{}", self.te.is_none())
            });
            self.dit
                .as_mut()
                .expect("conditioning prepared")
                .sample_prepared(
                    &mut self.stream,
                    &self.compiler,
                    &mut self.prof,
                    prepared,
                    p,
                    qk_bits,
                    noise,
                    refs,
                    kfs,
                    progress,
                    cache_policy,
                )
        })();
        // Cleanup also runs after cancellation or a failed preparation. A failed fence
        // retains owners so in-flight work cannot observe freed allocations.
        self.release_completed(true, true, false)?;
        crate::trace::event("denoise_finished", || {
            format!(
                ",\"success\":{},\"dit_released\":{}",
                result.is_ok(),
                self.dit.is_none()
            )
        });
        result
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
        self.options
            .cache
            .validate(p.cache_threshold)
            .map_err(|e| crate::Error::Invalid(e.into()))?;
        if let Some(preset) = self.options.turbo {
            if (p.width, p.height, p.frames) != (1344, 768, 124)
                || !refs.is_empty()
                || kfs.len() > 1
                || kfs.iter().any(|k| k.frame_index != 0 || k.audio.is_some())
            {
                return invalid(
                    "768p Turbo supports 1344x768, 124 frames, and at most one first-frame image",
                );
            }
            if p.steps != preset.evaluations() + 1
                || p.sampler != crate::Sampler::Euler
                || p.video_shift != 6.0
                || p.audio_shift != 3.0
            {
                return invalid("Turbo requires its trained evaluation count, Euler, and video/audio shifts 6/3");
            }
            if self.config.dit.is_some()
                || self.config.attention != crate::Attention::I8
                || p.cache_threshold > 0.0
                || self.options.cache != crate::CachePolicy::Off
            {
                return invalid(
                    "Turbo requires the default quantized base, i8 attention, and cache off",
                );
            }
        }
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
            stream,
            compiler,
            prof,
            te,
            ..
        } = self;
        let te = te.as_ref().expect("opened");
        crate::vision::embed(stream, compiler, prof, te.weights(), pixels, height, width)
    }

    /// Model-space latents to `[frames][height][width][3]` RGB8.
    ///
    /// Arguments are checked before anything is written, so an [`crate::Error::Invalid`] leaves `out` as it
    /// was. A failure raised once the decode is under way — a device error, or a cancelled run —
    /// can leave it *partly* written, because the decoder commits each temporal chunk as it
    /// finishes rather than staging a whole clip. Read `out` only after this returns `Ok`.
    ///
    /// `latents` and `out` must not overlap.
    pub fn decode_video(&mut self, shape: &Shape, latents: &[f32], out: &mut [u8]) -> Result<()> {
        self.release_completed(true, true, false)?;
        crate::trace::event("video_decode_start", String::new);
        self.vvae()?;
        let Self {
            stream,
            compiler,
            prof,
            vvae,
            ..
        } = self;
        let result = vvae
            .as_mut()
            .expect("opened")
            .decode_video(stream, compiler, prof, shape, latents, out);
        self.release_completed(false, false, true)?;
        crate::trace::event("video_decode_finished", || {
            format!(",\"success\":{}", result.is_ok())
        });
        result
    }

    pub fn encode_video(&mut self, clip: Clip<'_>) -> Result<(Vec<f32>, usize)> {
        self.vvae()?;
        let Self {
            stream,
            compiler,
            prof,
            vvae,
            ..
        } = self;
        vvae.as_mut()
            .expect("opened")
            .encode_video(stream, compiler, prof, clip)
    }

    pub fn decode_audio(
        &mut self,
        latents: &[f32],
        audio_t: usize,
        samples: &mut [f32],
    ) -> Result<()> {
        self.release_completed(true, true, false)?;
        crate::trace::event("audio_decode_start", String::new);
        self.avae()?;
        let Self {
            stream,
            compiler,
            prof,
            avae,
            ..
        } = self;
        let result = avae
            .as_mut()
            .expect("opened")
            .decode(stream, compiler, prof, latents, audio_t, samples);
        self.release_completed(false, false, true)?;
        crate::trace::event("audio_decode_finished", || {
            format!(",\"success\":{}", result.is_ok())
        });
        result
    }

    pub fn encode_audio(&mut self, samples: &[f32], n: usize) -> Result<(Vec<f32>, usize)> {
        self.avae()?;
        let Self {
            stream,
            compiler,
            prof,
            avae,
            ..
        } = self;
        avae.as_mut()
            .expect("opened")
            .encode(stream, compiler, prof, samples, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn existing_sessions_retain_models_by_default() {
        assert_eq!(SessionOptions::default().residency, ResidencyPolicy::Retain);
    }

    #[test]
    #[ignore = "requires an idle gfx1151 and the DiT/text encoder checkpoints"]
    fn stage_scoped_denoise_releases_models_and_recovers_after_cancellation() {
        let ids = crate::Tokenizer::new()
            .unwrap()
            .encode("a red fox in snow")
            .unwrap();
        let p = DenoiseParams {
            height: 256,
            width: 256,
            frames: 5,
            steps: 2,
            sampler: crate::Sampler::Euler,
            seed: 7,
            ..DenoiseParams::default()
        };
        // Safety: the fixture never modifies checkpoint files.
        let mut session = unsafe { Session::new(Config::default()) }.unwrap();
        let expected = session
            .denoise(&ids, &p, Noise::default(), &[], &[], None)
            .unwrap();
        assert!(session.te.is_some() && session.dit.is_some());
        session.options.residency = ResidencyPolicy::StageScoped;
        session.release_completed(true, true, true).unwrap();
        for cancel in [true, false] {
            let result = session.denoise(
                &ids,
                &p,
                Noise::default(),
                &[],
                &[],
                Some(&mut |_, _, _| cancel),
            );
            assert!(session.te.is_none() && session.dit.is_none());
            assert!(session.vvae.is_none() && session.avae.is_none());
            if cancel {
                assert!(matches!(result, Err(crate::Error::Cancelled)));
            } else {
                let actual = result.unwrap();
                assert_eq!(
                    crate::vvae::as_bytes(&actual.video),
                    crate::vvae::as_bytes(&expected.video)
                );
                assert_eq!(
                    crate::vvae::as_bytes(&actual.audio),
                    crate::vvae::as_bytes(&expected.audio)
                );
            }
        }
    }

    #[test]
    #[ignore = "requires an idle gfx1151 and the DiT/text encoder checkpoints"]
    fn observing_and_forcing_full_cache_evaluations_preserve_the_trajectory() {
        use crate::{CachePolicy, CacheThresholds};
        let ids = crate::Tokenizer::new()
            .unwrap()
            .encode("a fox running through snow")
            .unwrap();
        let p = DenoiseParams {
            height: 256,
            width: 256,
            frames: 22,
            steps: 6,
            seed: 7,
            ..DenoiseParams::default()
        };
        // Safety: the fixture never modifies checkpoint files.
        let mut session = unsafe { Session::new(Config::default()) }.unwrap();
        let expected = session
            .denoise(&ids, &p, Noise::default(), &[], &[], None)
            .unwrap();
        // Five evaluations include one eligible interior skip. A very small threshold
        // forces its suffix to run, exercising the split graph and metric reductions.
        for policy in [
            CachePolicy::Observe,
            CachePolicy::Conservative(CacheThresholds {
                conditioning: f32::MIN_POSITIVE,
                audio: f32::MIN_POSITIVE,
                video: f32::MIN_POSITIVE,
            }),
        ] {
            session.options.cache = policy;
            let actual = session
                .denoise(&ids, &p, Noise::default(), &[], &[], None)
                .unwrap();
            assert_eq!(
                crate::vvae::as_bytes(&actual.video),
                crate::vvae::as_bytes(&expected.video)
            );
            assert_eq!(
                crate::vvae::as_bytes(&actual.audio),
                crate::vvae::as_bytes(&expected.audio)
            );
        }
        // The legacy scalar maps to the hardened policy. Its old digest intentionally
        // no longer applies: evaluation one and consecutive suffix skips are forbidden.
        session.options.cache = CachePolicy::Off;
        let cached = session
            .denoise(
                &ids,
                &DenoiseParams {
                    cache_threshold: f32::MAX,
                    ..p
                },
                Noise::default(),
                &[],
                &[],
                None,
            )
            .unwrap();
        assert!(cached
            .video
            .iter()
            .chain(&cached.audio)
            .all(|v| v.is_finite()));
        assert_ne!(
            crate::vvae::as_bytes(&cached.video),
            crate::vvae::as_bytes(&expected.video)
        );
    }

    #[test]
    fn default_checkpoints_reuse_the_standard_hub_cache() {
        const CHILD: &str = "H3_TEST_CACHED_CHECKPOINT";
        if let Some(expected) = std::env::var_os(CHILD) {
            let config = Config::default();
            let expected = std::path::PathBuf::from(expected);
            assert_eq!(
                checkpoint_path(&config.video_vae, crate::models::VIDEO_VAE).unwrap(),
                expected
            );
            // A caller's explicit file must still win over a cached checkpoint.
            let explicit = expected.with_file_name("explicit.safetensors");
            assert_eq!(
                checkpoint_path(&Some(explicit.clone()), crate::models::VIDEO_VAE).unwrap(),
                explicit
            );
            return;
        }

        // Isolate environment changes in subprocesses so parallel tests cannot race on them.
        let dir = tempfile::tempdir().unwrap();
        let cache = dir.path().join("huggingface/hub");
        let repo = cache.join("models--Comfy-Org--MiniMax-H3");
        let revision = "0123456789abcdef0123456789abcdef01234567";
        let checkpoint = repo
            .join("snapshots")
            .join(revision)
            .join(crate::models::VIDEO_VAE);
        std::fs::create_dir_all(checkpoint.parent().unwrap()).unwrap();
        std::fs::create_dir_all(repo.join("refs")).unwrap();
        std::fs::write(repo.join("refs/main"), revision).unwrap();
        std::fs::write(&checkpoint, b"cached checkpoint fixture").unwrap();

        // The retired model-directory override must not shadow the Hub cache.
        let legacy = dir.path().join("legacy-models");
        let legacy_checkpoint = legacy.join(crate::models::VIDEO_VAE);
        std::fs::create_dir_all(legacy_checkpoint.parent().unwrap()).unwrap();
        std::fs::write(legacy_checkpoint, b"legacy checkpoint fixture").unwrap();

        for (variable, value) in [
            ("HF_HUB_CACHE", cache.clone()),
            ("HF_HOME", dir.path().join("huggingface")),
            ("XDG_CACHE_HOME", dir.path().to_path_buf()),
        ] {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "session::tests::default_checkpoints_reuse_the_standard_hub_cache",
                    "--nocapture",
                ])
                .env("H3_MODELS", &legacy)
                .env_remove("HF_HUB_CACHE")
                .env_remove("HUGGINGFACE_HUB_CACHE")
                .env_remove("HF_HOME")
                .env_remove("XDG_CACHE_HOME")
                .env("HF_ENDPOINT", "http://127.0.0.1:1")
                .env(variable, value)
                .env(CHILD, &checkpoint)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{variable}: {}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }
    use crate::dit::{Keyframe, LatentGrid, Reference};
    use crate::vvae::Clip;

    fn config(attention: crate::dit::Attention) -> Config {
        Config {
            dit: None,
            te: None,
            video_vae: None,
            audio_vae: None,
            kernel_sources: "kernels".into(),
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

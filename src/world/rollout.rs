//! Observation-conditioned rollout. This is an application layer over the released
//! fixed-horizon model, not a recurrent model or a persistent 3D representation.
use super::{ActionSchedule, WorldRequest, UPSTREAM_REVISION};
use crate::{error::invalid, DenoiseParams, Error, Result, Sampler, Session, Shape, Tokenizer};
use std::io::{Read, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"H3WORLD1";
const MAX_HEADER: usize = 64 * 1024;
// Bounds host output independently of the native allocation ceiling.
const MAX_VIDEO_BYTES: usize = 512 * 1024 * 1024;
const MAX_OBSERVATION_BYTES: usize = 64 * 1024 * 1024;

/// Portable, lossless current observation and rollout cursor. Clone to branch.
/// No GPU buffers or past clips are retained. Use the same model configuration
/// when resuming; the state cannot recover hidden objects or motion history.
#[derive(Clone, Debug)]
pub struct WorldState {
    scene: String,
    parameters: DenoiseParams,
    observation: Vec<u8>,
    segments: u64,
    frame: u64,
    adapter_sha256: Option<String>,
}

/// A complete decoded segment. Frame zero shares the prior observation's time;
/// append `video[frame_bytes..]` when assembling a trajectory without duplicate
/// boundary times. Audio is planar stereo at 32 kHz and starts afresh per
/// segment; the released model does not condition on prior audio.
pub struct WorldSegment {
    pub video: Vec<u8>,
    pub audio: Vec<f32>,
    pub shape: Shape,
    pub index: u64,
    pub start_frame: u64,
    pub seed: u64,
    pub action_sentences: Vec<String>,
}

fn shape(p: &DenoiseParams) -> Result<Shape> {
    let sh = Session::shape_for(p.height, p.width, p.frames)
        .ok_or_else(|| Error::Invalid("invalid world canvas or duration".into()))?;
    if p.frames != sh.frames
        || p.sampler != Sampler::Euler
        || p.cache_threshold != 0.0
        || !(2..=1000).contains(&p.steps)
        || !p.video_shift.is_finite()
        || p.video_shift <= 0.0
        || !p.audio_shift.is_finite()
        || p.audio_shift <= 0.0
    {
        return invalid(
            "world state requires frames = 17k+5, Euler, valid shifts and 2..1000 grid points",
        );
    }
    if sh.video_bytes() > MAX_VIDEO_BYTES
        || p.width as usize * p.height as usize * 3 > MAX_OBSERVATION_BYTES
    {
        return invalid(
            "world rollout exceeds the 512 MiB decoded segment or 64 MiB observation limit",
        );
    }
    let rows = (sh.latent_t as usize + 1) * (sh.lat_h as usize / 2) * (sh.lat_w as usize / 2)
        + sh.audio_t as usize * 2;
    if rows >= 65536 {
        return invalid("world sequence exceeds native attention capacity");
    }
    Ok(sh)
}

/// Check model geometry and bounded host output before allocating a canvas.
pub fn validate_rollout_parameters(p: &DenoiseParams) -> Result<()> {
    shape(p).map(|_| ())
}

impl WorldState {
    /// `observation` is RGB8 at exactly the requested canvas size. Resize the
    /// initial image with `resize::world_first_frame` before constructing it.
    pub fn new(
        scene: impl Into<String>,
        parameters: DenoiseParams,
        observation: Vec<u8>,
    ) -> Result<Self> {
        shape(&parameters)?;
        let scene = scene.into();
        if scene.trim().is_empty() || scene.len() > 32768 {
            return invalid("world scene must be nonempty and at most 32768 bytes");
        }
        if observation.len() != parameters.width as usize * parameters.height as usize * 3 {
            return invalid("world observation must be exactly one RGB8 frame at the canvas size");
        }
        Ok(Self {
            scene,
            parameters,
            observation,
            segments: 0,
            frame: 0,
            adapter_sha256: None,
        })
    }

    pub fn scene(&self) -> &str {
        &self.scene
    }
    /// Parameters for the *next* segment, including its deterministic seed.
    pub fn parameters(&self) -> DenoiseParams {
        self.parameters
    }
    pub fn observation(&self) -> &[u8] {
        &self.observation
    }
    pub fn segments(&self) -> u64 {
        self.segments
    }
    /// Current observation's frame index in the rollout, at 24 fps.
    pub fn frame(&self) -> u64 {
        self.frame
    }

    /// Bind an adapter on first use and reject a different one on resume.
    /// Applications remain responsible for keeping the base checkpoints fixed.
    pub fn bind_adapter(&mut self, sha256: &str) -> Result<()> {
        if sha256.len() != 64
            || !sha256
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        {
            return invalid("adapter identity must be a lowercase SHA-256 digest");
        }
        if self
            .adapter_sha256
            .as_deref()
            .is_some_and(|old| old != sha256)
        {
            return invalid("world state was created with a different adapter");
        }
        self.adapter_sha256 = Some(sha256.into());
        Ok(())
    }

    /// Atomically replace a checkpoint only after its header and pixels have
    /// been written and synced. The old file survives a failed write.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let p = self.parameters;
        let header = serde_json::to_vec(&serde_json::json!({
            "version": 1, "upstream_revision": UPSTREAM_REVISION,
            "scene": self.scene, "width": p.width, "height": p.height,
            "frames": p.frames, "steps": p.steps, "next_seed": p.seed,
            "video_shift": p.video_shift, "audio_shift": p.audio_shift,
            "segments": self.segments, "frame": self.frame,
            "adapter_sha256": self.adapter_sha256
        }))
        .map_err(io_error)?;
        if header.len() > MAX_HEADER {
            return invalid("world state header too large");
        }
        let mut file = tempfile::NamedTempFile::new_in(parent).map_err(io_error)?;
        file.write_all(MAGIC).map_err(io_error)?;
        file.write_all(&(header.len() as u32).to_le_bytes())
            .map_err(io_error)?;
        file.write_all(&header).map_err(io_error)?;
        file.write_all(&self.observation).map_err(io_error)?;
        file.as_file().sync_all().map_err(io_error)?;
        file.persist(path).map_err(io_error)?;
        Ok(())
    }

    /// Read bounded metadata before allocating pixels. Unknown versions,
    /// incompatible method revisions, truncated and trailing data are errors.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let mut file = std::fs::File::open(path).map_err(io_error)?;
        let mut magic = [0; 8];
        file.read_exact(&mut magic).map_err(io_error)?;
        if &magic != MAGIC {
            return invalid("not an H3 world state");
        }
        let mut len = [0; 4];
        file.read_exact(&mut len).map_err(io_error)?;
        let len = u32::from_le_bytes(len) as usize;
        if len > MAX_HEADER {
            return invalid("world state header too large");
        }
        let mut bytes = vec![0; len];
        file.read_exact(&mut bytes).map_err(io_error)?;
        let h: serde_json::Value = serde_json::from_slice(&bytes).map_err(io_error)?;
        if h["version"] != 1 || h["upstream_revision"] != UPSTREAM_REVISION {
            return invalid("unsupported world state version or method revision");
        }
        let number = |name: &str| {
            h[name]
                .as_u64()
                .ok_or_else(|| Error::Invalid(format!("invalid state {name}")))
        };
        let integer = |name: &str| i32::try_from(number(name)?).map_err(io_error);
        let float = |name: &str| {
            h[name]
                .as_f64()
                .ok_or_else(|| Error::Invalid(format!("invalid state {name}")))
        };
        let p = DenoiseParams {
            width: integer("width")?,
            height: integer("height")?,
            frames: integer("frames")?,
            steps: usize::try_from(number("steps")?).map_err(io_error)?,
            seed: number("next_seed")?,
            video_shift: float("video_shift")?,
            audio_shift: float("audio_shift")?,
            sampler: Sampler::Euler,
            cache_threshold: 0.0,
        };
        shape(&p)?;
        let length = p.width as usize * p.height as usize * 3;
        if file.metadata().map_err(io_error)?.len() != (12 + len + length) as u64 {
            return invalid("world state pixel length mismatch");
        }
        let mut pixels = vec![0; length];
        file.read_exact(&mut pixels).map_err(io_error)?;
        let mut state = Self::new(
            h["scene"]
                .as_str()
                .ok_or_else(|| Error::Invalid("missing scene".into()))?,
            p,
            pixels,
        )?;
        state.segments = number("segments")?;
        state.frame = number("frame")?;
        if state.segments.checked_mul((p.frames - 1) as u64) != Some(state.frame) {
            return invalid("inconsistent world state timeline");
        }
        if !h["adapter_sha256"].is_null() {
            state.bind_adapter(
                h["adapter_sha256"]
                    .as_str()
                    .ok_or_else(|| Error::Invalid("invalid adapter identity".into()))?,
            )?;
        }
        Ok(state)
    }

    fn advance_with(
        &mut self,
        actions: &ActionSchedule,
        generate: impl FnOnce(&Self, &Shape) -> Result<(Vec<u8>, Vec<f32>)>,
    ) -> Result<WorldSegment> {
        let sh = shape(&self.parameters)?;
        let sentences = actions.sentences(sh.frames as usize)?;
        let next_segment = self
            .segments
            .checked_add(1)
            .ok_or_else(|| Error::Invalid("world segment counter exhausted".into()))?;
        let next_frame = self
            .frame
            .checked_add((sh.frames - 1) as u64)
            .ok_or_else(|| Error::Invalid("world frame counter exhausted".into()))?;
        let (video, audio) = generate(self, &sh)?;
        if video.len() != sh.video_bytes()
            || audio.len() != sh.audio_samples()
            || audio.iter().any(|v| !v.is_finite())
        {
            return invalid("world generator returned invalid decoded output");
        }
        let frame_bytes = self.observation.len();
        let result = WorldSegment {
            video,
            audio,
            shape: sh,
            index: self.segments,
            start_frame: self.frame,
            seed: self.parameters.seed,
            action_sentences: sentences,
        };
        // Commit last: failures and cancellation above leave the observation,
        // cursor and seed unchanged, so the caller can retry the same step.
        self.observation
            .copy_from_slice(&result.video[result.video.len() - frame_bytes..]);
        self.segments = next_segment;
        self.frame = next_frame;
        self.parameters.seed = self.parameters.seed.wrapping_add(1);
        Ok(result)
    }
}

fn io_error(e: impl std::fmt::Display) -> Error {
    Error::Other(e.to_string())
}

impl Session {
    /// Generate, decode, and advance a world from its latest observation.
    /// Errors/cancellation leave `state` unchanged. Uses the ordinary session
    /// residency and allocation budget, and retains no trajectory on the GPU.
    pub fn step_world(
        &mut self,
        state: &mut WorldState,
        tokenizer: &Tokenizer,
        actions: &ActionSchedule,
        progress: Option<&mut dyn FnMut(usize, usize, f64) -> bool>,
    ) -> Result<WorldSegment> {
        let digest = self.world_adapter_identity()?;
        if state
            .adapter_sha256
            .as_deref()
            .is_some_and(|old| old != digest)
        {
            return invalid("world state was created with a different adapter");
        }
        let segment = state.advance_with(actions, |state, sh| {
            use crate::media_context::{Frame, Media, MediaEntry};
            let p = state.parameters;
            let request = WorldRequest::new(tokenizer, actions, sh.frames as usize)?;
            let frame = Frame {
                pixels: state
                    .observation
                    .iter()
                    .map(|b| *b as f32 / 255.0)
                    .collect::<Vec<_>>()
                    .into(),
                width: p.width as usize,
                height: p.height as usize,
            };
            let entries = [MediaEntry {
                media: Media::Picture(frame.clone()),
                role: "first_frame".into(),
                metadata: serde_json::Value::Null,
            }];
            let mut prompt_shape = *sh;
            prompt_shape.text_rows_max -= request.token_count() as i32;
            let presentation =
                crate::PreparedPresentation::new(tokenizer, &entries, &state.scene, &prompt_shape)?;
            let rows = presentation.ids().len()
                + request.token_count()
                + (sh.latent_t as usize + 1) * (sh.lat_h as usize / 2) * (sh.lat_w as usize / 2)
                + sh.audio_t as usize * 2;
            if rows > 65536 {
                return invalid("world sequence exceeds native attention capacity");
            }
            let key = self
                .encode_video(crate::Clip {
                    pixels: &frame.pixels,
                    frames: 1,
                    width: frame.width,
                    height: frame.height,
                })?
                .0;
            let latents = self.denoise_world(
                &presentation,
                &request,
                &p,
                crate::Noise::default(),
                &crate::Keyframe {
                    frame_index: 0,
                    latents: &key,
                    audio: None,
                    presented: None,
                },
                progress,
            )?;
            let mut video = vec![0; sh.video_bytes()];
            self.decode_video(sh, &latents.video, &mut video)?;
            let mut audio = vec![0.; sh.audio_samples()];
            self.decode_audio(&latents.audio, sh.audio_t as usize, &mut audio)?;
            Ok((video, audio))
        })?;
        state.adapter_sha256 = Some(digest);
        Ok(segment)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn initial() -> WorldState {
        WorldState::new(
            "A man in a garage.",
            DenoiseParams {
                width: 64,
                height: 64,
                frames: 5,
                steps: 2,
                seed: 7,
                ..WorldRequest::parameters()
            },
            vec![13; 64 * 64 * 3],
        )
        .unwrap()
    }
    #[test]
    fn rollout_resume_branch_and_failed_step_are_transactional() {
        let mut state = initial();
        let schedule = ActionSchedule::preset("forward", 5).unwrap();
        let out = state
            .advance_with(&schedule, |s, sh| {
                assert_eq!(s.observation(), vec![13; 64 * 64 * 3]);
                let mut frames = vec![21; sh.video_bytes()];
                frames[sh.video_bytes() - 64 * 64 * 3..].fill(37);
                Ok((frames, vec![0.; sh.audio_samples()]))
            })
            .unwrap();
        assert_eq!((out.seed, out.index, out.start_frame), (7, 0, 0));
        assert_eq!(
            (state.parameters().seed, state.segments(), state.frame()),
            (8, 1, 4)
        );
        assert_eq!(state.observation(), vec![37; 64 * 64 * 3]);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("state");
        state.bind_adapter(&"a".repeat(64)).unwrap();
        state.save(&path).unwrap();
        let mut resumed = WorldState::load(&path).unwrap();
        assert_eq!(format!("{state:?}"), format!("{resumed:?}"));
        let branch = resumed.clone();
        assert!(resumed
            .advance_with(&schedule, |_, _| Err(Error::Cancelled))
            .is_err());
        assert!(resumed
            .advance_with(&schedule, |_, _| Ok((vec![], vec![])))
            .is_err());
        assert_eq!(format!("{branch:?}"), format!("{resumed:?}"));
        assert!(resumed.bind_adapter(&"b".repeat(64)).is_err());
        resumed
            .advance_with(&schedule, |s, sh| {
                assert_eq!(s.observation(), vec![37; 64 * 64 * 3]);
                assert_eq!(s.parameters().seed, 8);
                Ok((vec![42; sh.video_bytes()], vec![0.; sh.audio_samples()]))
            })
            .unwrap();
        resumed.save(&path).unwrap();
        assert_eq!(WorldState::load(&path).unwrap().frame(), 8);
        assert_eq!(branch.observation(), vec![37; 64 * 64 * 3]);
    }
    #[test]
    fn state_rejects_corruption_and_unbounded_allocations() {
        let state = initial();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("state");
        state.save(&path).unwrap();
        let good = std::fs::read(&path).unwrap();
        for len in [0, 8, 11, good.len() - 1] {
            std::fs::write(&path, &good[..len]).unwrap();
            assert!(WorldState::load(&path).is_err());
        }
        let mut corrupt = good.clone();
        corrupt[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        std::fs::write(&path, &corrupt).unwrap();
        assert!(WorldState::load(&path).is_err());
        let mut extra = good;
        extra.push(0);
        std::fs::write(&path, extra).unwrap();
        assert!(WorldState::load(&path).is_err());
        let mut p = state.parameters();
        p.frames = 6;
        assert!(WorldState::new("scene", p, state.observation.clone()).is_err());
        p.frames = 124;
        p.width = 8192;
        p.height = 8192;
        assert!(WorldState::new("scene", p, vec![]).is_err());
    }
}

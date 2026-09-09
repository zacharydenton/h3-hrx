//! The DiT side of the pipeline: the packed sequence buffer, the token refiner, and the pieces that
//! write into the sequence before the blocks run.
//!
//! The sequence buffer is one allocation shared by everything — text rows, keyframes, references,
//! audio and video all live in it at offsets the layout decides — so it is sized once for the longest
//! sequence a session will see and reused. That is also why `text_in` lands here rather than in the
//! text encoder: the encoder produces 5120-wide rows, and it is the DiT that projects them to 5376 and
//! refines them in place at the front of its own sequence.
use crate::compile::{Cfg, Compiler};
use crate::dispatch::{axpy, checked, Classes, Matmul16, NormMod, Profile};
use crate::error::{invalid, Result};
use crate::model::*;
use crate::rope::VisionSpan;
use crate::stack::{Constants, Stack, StackDims};
use crate::te::{Span, TextEncoder};
use crate::weights::Weights;

/// The refiner: two H3-shaped blocks on bf16 rows with no rope, then a modulated norm.
fn refiner_dims() -> StackDims {
    StackDims {
        hidden: HID,
        heads: HEADS,
        kv_heads: HEADS,
        head_dim: HEAD_DIM,
        ffn: FFN,
        rope_dim: ROPE_DIM,
        classes: 1,
        wbits: 16,
        eps: 1e-5,
        bias: false,
        gate_first: true,
        causal: false,
        attn_i4: false,
        attn_qk_bits: 16,
        bf16: true,
    }
}

/// How many rows a sequence of `seq` gets: rounded to 256, plus the 32 the attention kernels read
/// past the end of a sequence.
pub fn seq_capacity(seq: usize) -> usize {
    seq.div_ceil(256) * 256 + 32
}

/// The sequence buffers, sized for the longest sequence asked for so far.
///
/// Several of these are written only by the denoise loop; they are allocated together because they are
/// all functions of the same capacity and reallocating them apart would be a way to get them out of
/// step.
struct Seq {
    capacity: usize,
    x: hrx::Buffer,
    /// the per-row timestep class the modulated kernels index with
    cls: Classes,
    /// all zeros: what the single-class GEMMs index their gate table with
    cls0: Classes,
    tcls: Classes,
    cos: hrx::Buffer,
    sin: hrx::Buffer,
    /// the embedders' f32 input rows, `[rows][96]` at the widest
    in32: hrx::Buffer,
    /// the final layer's f32 output rows, `[rows][128]`
    out32: hrx::Buffer,
}

struct Refiner {
    stack: Stack,
    tokens: usize,
    cos: hrx::Buffer,
    sin: hrx::Buffer,
    norm: NormMod,
}

pub struct Dit {
    weights: Weights,
    constants: Constants,
    seq: Option<Seq>,
    refiner: Option<Refiner>,
    cond: Option<Conditioning>,
    blocks: Option<Blocks>,
    cache: Option<CacheBuffers>,
    euler: Option<crate::sampler::device::Euler>,
}

impl Dit {
    /// # Safety
    ///
    /// Maps the checkpoint; see [`crate::Session::new`].
    pub unsafe fn open(
        stream: &mut hrx::Stream,
        path: impl AsRef<std::path::Path>,
    ) -> Result<Self> {
        Ok(Self {
            weights: unsafe { Weights::open(path, crate::plan::dit::plan) }?,
            constants: Constants::new(stream)?,
            seq: None,
            refiner: None,
            cond: None,
            blocks: None,
            cache: None,
            euler: None,
        })
    }

    pub fn weights(&self) -> &Weights {
        &self.weights
    }

    /// Grows the sequence buffers if this length does not fit. The capacity is rounded to 256 rows
    /// plus 32, which is the slack the attention kernels read past the end of a sequence.
    pub fn ensure_seq(&mut self, stream: &mut hrx::Stream, seq: usize) -> Result<()> {
        if self.seq.as_ref().is_some_and(|s| seq <= s.capacity) {
            return Ok(());
        }
        self.seq = None;
        let t = seq_capacity(seq);
        let mut zeroed = |bytes: usize| -> Result<hrx::Buffer> {
            let b = stream.allocate(bytes)?;
            stream.fill(b.slice(0, bytes), 0)?;
            Ok(b)
        };
        self.seq = Some(Seq {
            capacity: t,
            x: zeroed(t * HID * 4)?,
            cls: Classes::zeroed(stream, t)?,
            cls0: Classes::zeroed(stream, t)?,
            tcls: Classes::zeroed(stream, t)?,
            cos: stream.allocate(t * ROPE_HALF * 4)?,
            sin: stream.allocate(t * ROPE_HALF * 4)?,
            in32: stream.allocate(t * VIDEO_PATCH * 4)?,
            out32: stream.allocate(t * FINAL_N * 4)?,
        });
        Ok(())
    }

    fn ensure_refiner(&mut self, stream: &mut hrx::Stream, c: &Compiler, n: usize) -> Result<()> {
        if self.refiner.as_ref().is_some_and(|r| r.tokens == n) {
            return Ok(());
        }
        self.refiner = None;
        let stack = Stack::new(
            c,
            stream,
            refiner_dims(),
            n,
            REFINER_BLOCKS,
            &self.weights,
            |i| format!("h3.refiner.{i}."),
            true,
            self.constants.ones.clone(),
            "refiner",
        )?;
        // no rope: the identity rotation at every row
        let cos = stream.allocate(n * ROPE_HALF * 4)?;
        let sin = stream.allocate(n * ROPE_HALF * 4)?;
        stream.upload(
            cos.binding(),
            crate::vvae::as_bytes(&vec![1.0f32; n * ROPE_HALF]),
        )?;
        stream.fill(sin.slice(0, n * ROPE_HALF * 4), 0)?;
        self.refiner = Some(Refiner {
            stack,
            tokens: n,
            cos,
            sin,
            norm: NormMod::build(c, stream, HID, 1e-5, 1)?,
        });
        Ok(())
    }

    /// The prompt into the front of the sequence: the encoder, `condition_proj` to the DiT's width,
    /// then the refiner and its final norm, all landing in `x` rows `[0, n)`.
    pub fn text_in(
        &mut self,
        stream: &mut hrx::Stream,
        c: &Compiler,
        prof: &mut Profile,
        te: &mut TextEncoder,
        ids: &[i32],
        spans: &[Span<'_>],
    ) -> Result<()> {
        let n = ids.len();
        self.ensure_seq(stream, n)?;
        let hidden = te.encode(stream, c, prof, ids, spans)?;

        let seq = self.seq.as_ref().expect("sized above");
        let held_1 = self.weights.at(stream, "h3.cond.w", HID * TEXT_DIM * 2)?;
        let held_2 = self.weights.at(stream, "h3.cond.b", HID * 4)?;
        Matmul16::build(c, stream, "bias", TEXT_DIM, HID)?.run(
            stream,
            Some(prof),
            "condition proj",
            n,
            hidden,
            held_1.binding(),
            held_2.binding(),
            seq.x.binding(),
            None,
        )?;

        self.ensure_refiner(stream, c, n)?;
        let r = self.refiner.as_mut().expect("built above");
        let seq = self.seq.as_ref().expect("sized above");
        let cond = self.constants.identity();
        let cond_fn = |_: usize| crate::stack::LayerCond { ..cond };
        r.stack.forward(
            stream,
            prof,
            seq.x.binding(),
            seq.cls0.slice(0, n),
            r.cos.binding(),
            r.sin.binding(),
            &cond_fn,
            0,
            None,
        )?;
        // the refiner modulates with a single class, so its table is the two zero rows
        let held_1 = self.weights.at(stream, "h3.refiner.final_norm", HID * 4)?;
        r.norm.run(
            stream,
            Some(prof),
            "refiner final norm",
            n,
            seq.x.binding(),
            held_1.binding(),
            self.constants.zeros.binding(),
            seq.cls0.slice(0, n),
        )?;
        Ok(())
    }

    /// The sequence's first `rows` rows, `[rows][5376]` f32 — what `h3_text_in` hands back.
    pub fn read_rows(&self, stream: &mut hrx::Stream, rows: usize, out: &mut [f32]) -> Result<()> {
        let seq = self.seq.as_ref().expect("a sequence has been sized");
        stream.synchronize()?;
        stream.read(
            seq.x.slice(0, rows * HID * 4),
            crate::vvae::as_bytes_mut(out),
        )?;
        Ok(())
    }
}

/// The QK operands' width. The stack is built for it, so it belongs to the session, not a run.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Attention {
    /// f16 operands: the widest, and what the reference comparison uses
    F16,
    /// int8: the parity path, and the default
    #[default]
    I8,
    /// int4: least operand traffic, and it ghosts conditioned clips
    I4,
}

impl Attention {
    pub fn bits(self) -> usize {
        match self {
            Self::F16 => 16,
            Self::I8 => 8,
            Self::I4 => 4,
        }
    }

    /// The three the kernels exist for; anything else is a caller's mistake.
    pub fn from_bits(bits: usize) -> Option<Self> {
        match bits {
            16 => Some(Self::F16),
            8 => Some(Self::I8),
            4 => Some(Self::I4),
            _ => None,
        }
    }
}

/// How a step advances.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Sampler {
    /// Euler on each stream's own schedule, as diffusers does it
    Euler,
    /// ComfyUI's res_multistep over the pack on the video sigma grid, which the stock workflows use
    #[default]
    ResMultistep,
}

/// What a run asks for. The shifts and the sampler are the schedule's, the threshold the step cache's.
#[derive(Clone, Copy, Debug)]
pub struct DenoiseParams {
    pub height: i32,
    pub width: i32,
    pub frames: i32,
    pub steps: usize,
    pub seed: u64,
    pub sampler: Sampler,
    pub video_shift: f64,
    pub audio_shift: f64,
    pub cache_threshold: f32,
}

impl Default for DenoiseParams {
    fn default() -> Self {
        Self {
            height: 480,
            width: 864,
            frames: 90,
            steps: 30,
            seed: 0,
            sampler: Sampler::ResMultistep,
            video_shift: 12.0,
            audio_shift: 3.0,
            cache_threshold: 0.0,
        }
    }
}

/// Explicit noise, so a run can be reproduced without depending on the generator.
#[derive(Clone, Copy, Default)]
pub struct Noise<'a> {
    pub video: Option<&'a [f32]>,
    pub audio: Option<&'a [f32]>,
}

/// The small host interpolation curve and prepared device conditioning tables.
struct Conditioning {
    curve: Vec<f32>,
    inv_freq: Vec<f32>,
    adaln: crate::conditioning::device::Projection,
    final_adaln: crate::conditioning::device::Projection,
    mods: hrx::Buffer,
    final_table: hrx::Buffer,
}

/// The 50-block stack and the buffers that depend on the sequence length.
struct Blocks {
    stack: Stack,
    tokens: usize,
    qk_bits: usize,
    final_norm: NormMod,
    final_norm_scale: std::sync::Arc<hrx::Buffer>,
    audio_in: Projection,
    video_in: Projection,
    final_out: Projection,
    /// the text rows, kept so they can be restored each step
    text_copy: hrx::Buffer,
}

/// A projection and its resident operands, prepared outside the denoise loop.
struct Projection {
    op: crate::dispatch::MatmulF32,
    weight: std::sync::Arc<hrx::Buffer>,
    bias: std::sync::Arc<hrx::Buffer>,
}

impl Projection {
    fn build(
        c: &Compiler,
        stream: &mut hrx::Stream,
        weights: &Weights,
        name: &str,
        k: usize,
        n: usize,
    ) -> Result<Self> {
        Ok(Self {
            op: crate::dispatch::MatmulF32::build(c, stream, k, n)?,
            weight: weights.at(stream, &format!("{name}.w"), n * k * 4)?,
            bias: weights.at(stream, &format!("{name}.b"), n * 4)?,
        })
    }

    fn run(
        &self,
        stream: &mut hrx::Stream,
        prof: &mut Profile,
        stage: &str,
        rows: usize,
        input: hrx::View<'_>,
        output: hrx::View<'_>,
    ) -> Result<()> {
        Ok(self.op.run(
            stream,
            Some(prof),
            stage,
            rows,
            input,
            self.weight.binding(),
            self.bias.binding(),
            output,
        )?)
    }
}

/// The step cache's device side.
struct CacheBuffers {
    n: usize,
    xb0: hrx::Buffer,
    prev: hrx::Buffer,
    resid: hrx::Buffer,
    partials: hrx::Buffer,
    metric: std::sync::Arc<hrx::Kernel>,
}

/// A latent grid: the extent of a reference's own latents.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct LatentGrid {
    pub frames: usize,
    pub height: usize,
    pub width: usize,
}

impl LatentGrid {
    /// The latents this grid holds, or `None` if the extents overflow.
    ///
    /// Checked, because a validator that computes its requirement by wrapping accepts whatever it
    /// wrapped to: a height of 2^63 becomes a small number of floats and any buffer passes.
    pub fn elements(&self) -> Option<usize> {
        [LATENT_CH, self.frames, self.height, self.width]
            .into_iter()
            .try_fold(1usize, |a, b| a.checked_mul(b))
    }
}

/// An image as the text encoder is shown it, alongside a visual reference's latents.
#[derive(Clone, Copy, Debug)]
pub struct Presented<'a> {
    pub pixels: &'a [f32],
    pub height: usize,
    pub width: usize,
}

/// A reference the pack conditions on, in presentation order.
///
/// An enum rather than a kind tag with optional fields: an audio reference has no latent grid and an
/// image has no soundtrack, and the combinations that used to be constructible — a kind of 1 carrying
/// video latents, a visual kind carrying none — cannot be written down here at all.
#[derive(Clone, Copy, Debug)]
pub enum Reference<'a> {
    /// One latent frame, optionally presented to the text encoder as pixels
    Image {
        latents: &'a [f32],
        grid: LatentGrid,
        presented: Option<Presented<'a>>,
    },
    /// `[2][32][frames]`
    Audio { latents: &'a [f32], frames: usize },
    /// A clip, with its own soundtrack if it has one
    Video {
        latents: &'a [f32],
        grid: LatentGrid,
        audio: Option<(&'a [f32], usize)>,
    },
}

impl Reference<'_> {
    /// The tag the packed layout orders by: 0 image, 1 audio, 2 video.
    pub(crate) fn kind(&self) -> i32 {
        match self {
            Self::Image { .. } => 0,
            Self::Audio { .. } => 1,
            Self::Video { .. } => 2,
        }
    }

    pub(crate) fn grid(&self) -> LatentGrid {
        match self {
            Self::Image { grid, .. } => LatentGrid { frames: 1, ..*grid },
            Self::Video { grid, .. } => *grid,
            Self::Audio { .. } => LatentGrid {
                frames: 0,
                height: 0,
                width: 0,
            },
        }
    }

    pub(crate) fn video_latents(&self) -> Option<&[f32]> {
        match self {
            Self::Image { latents, .. } | Self::Video { latents, .. } => Some(latents),
            Self::Audio { .. } => None,
        }
    }

    pub(crate) fn audio_latents(&self) -> Option<(&[f32], usize)> {
        match self {
            Self::Audio { latents, frames } => Some((latents, *frames)),
            Self::Video { audio, .. } => *audio,
            Self::Image { .. } => None,
        }
    }

    pub(crate) fn presented(&self) -> Option<Presented<'_>> {
        match self {
            Self::Image { presented, .. } => *presented,
            _ => None,
        }
    }

    /// The buffers against the extents they claim.
    pub fn check(&self, i: usize) -> crate::error::Result<()> {
        let at = |what: String| crate::error::Error::Invalid(format!("reference {i}: {what}"));
        if let Some(z) = self.video_latents() {
            let g = self.grid();
            // the latents are packed as 2x2 patches, four spatial neighbours to a row, so an odd
            // extent has a half patch at its edge and the packing indexes past the rows it sized
            if !g.height.is_multiple_of(2) || !g.width.is_multiple_of(2) {
                return Err(at(format!(
                    "a {}x{} latent grid is not whole 2x2 patches",
                    g.height, g.width
                )));
            }
            let Some(need) = g.elements() else {
                return Err(at(format!("latents for {g:?} overflow a usize")));
            };
            if need == 0 || z.len() < need {
                return Err(at(format!(
                    "latents for {g:?} need {need} floats, {} given",
                    z.len()
                )));
            }
        }
        if let Some((z, frames)) = self.audio_latents() {
            let Some(need) = frames.checked_mul(2 * crate::avae::AUDIO_CH) else {
                return Err(at(format!("{frames} audio frames overflow a usize")));
            };
            if need == 0 || z.len() < need {
                return Err(at(format!(
                    "audio latents for {frames} frames need {need} floats, {} given",
                    z.len()
                )));
            }
        }
        if let Some(p) = self.presented() {
            let Some(need) = p.height.checked_mul(p.width).and_then(|n| n.checked_mul(3)) else {
                return Err(at(format!(
                    "pixels of {}x{} overflow a usize",
                    p.height, p.width
                )));
            };
            if need == 0 || p.pixels.len() < need {
                return Err(at(format!(
                    "pixels of {}x{} need {need} floats, {} given",
                    p.height,
                    p.width,
                    p.pixels.len()
                )));
            }
        }
        Ok(())
    }
}

/// A keyframe: one latent frame pinned at a frame index, on the generation's own latent grid.
#[derive(Clone, Copy, Debug)]
pub struct Keyframe<'a> {
    /// 0 for the first frame, or `frames - 1` after snapping for the last
    pub frame_index: i32,
    /// `[24][1][lat_h][lat_w]`, on the grid the run itself uses
    pub latents: &'a [f32],
    /// the same frame as pixels, for the encoder's presentation
    pub presented: Option<Presented<'a>>,
    /// optional, and never denoised
    pub audio: Option<(&'a [f32], usize)>,
}

impl Keyframe<'_> {
    /// The buffers against the generation's grid, which is what a keyframe sits on.
    pub fn check(&self, i: usize, lat_h: i32, lat_w: i32) -> crate::error::Result<()> {
        let at = |what: String| crate::error::Error::Invalid(format!("keyframe {i}: {what}"));
        let need = LATENT_CH * (lat_h.max(0) as usize) * (lat_w.max(0) as usize);
        if need == 0 || self.latents.len() < need {
            return Err(at(format!(
                "latents on the {lat_h}x{lat_w} grid need {need} floats, {} given",
                self.latents.len()
            )));
        }
        if let Some((z, frames)) = self.audio {
            let Some(want) = frames.checked_mul(2 * crate::avae::AUDIO_CH) else {
                return Err(at(format!("{frames} audio frames overflow a usize")));
            };
            if want == 0 || z.len() < want {
                return Err(at(format!(
                    "audio latents for {frames} frames need {want} floats, {} given",
                    z.len()
                )));
            }
        }
        if let Some(p) = self.presented {
            let Some(want) = p.height.checked_mul(p.width).and_then(|n| n.checked_mul(3)) else {
                return Err(at(format!(
                    "pixels of {}x{} overflow a usize",
                    p.height, p.width
                )));
            };
            if want == 0 || p.pixels.len() < want {
                return Err(at(format!(
                    "pixels of {}x{} need {want} floats, {} given",
                    p.height,
                    p.width,
                    p.pixels.len()
                )));
            }
        }
        Ok(())
    }
}

/// What a run produces.
pub struct Latents {
    /// `[24][latent_t][lat_h][lat_w]`
    pub video: Vec<f32>,
    /// `[2][32][audio_t]`
    pub audio: Vec<f32>,
}

impl Dit {
    /// Upload conditioning weights and prepare both table projections once.
    fn ensure_conditioning(&mut self, stream: &mut hrx::Stream, c: &Compiler) -> Result<()> {
        if self.cond.is_some() {
            return Ok(());
        }
        let w = &self.weights;
        let mut adaln_w = Vec::with_capacity(BLOCKS);
        let mut adaln_b = Vec::with_capacity(BLOCKS);
        for i in 0..BLOCKS {
            adaln_w.push(w.host_f32(&format!("h3.blocks.{i}.adaln.w"), MODALITIES * 6 * HID * 8)?);
            adaln_b.push(w.host_f32(&format!("h3.blocks.{i}.adaln.b"), MODALITIES * 6 * HID)?);
        }
        self.cond = Some(Conditioning {
            curve: w.host_f32("h3.adaln_t_table", 1025 * 8)?,
            inv_freq: w.host_f32("h3.rope_inv_freq", 16)?,
            adaln: crate::conditioning::device::Projection::blocks(stream, c, &adaln_w, &adaln_b)?,
            final_adaln: crate::conditioning::device::Projection::final_layer(
                stream,
                c,
                w.host_f32("h3.final.adaln.w", 2 * HID * 8)?,
                w.host_f32("h3.final.adaln.b", 2 * HID)?,
            )?,
            mods: stream.allocate(BLOCKS * MODS_ROWS * HID * 4)?,
            final_table: stream.allocate(4 * HID * 4)?,
        });
        Ok(())
    }

    /// The 50 blocks and the final layer's norm, for one sequence length.
    fn ensure_blocks(
        &mut self,
        stream: &mut hrx::Stream,
        c: &Compiler,
        tokens: usize,
        qk_bits: usize,
    ) -> Result<()> {
        if self
            .blocks
            .as_ref()
            .is_some_and(|b| b.tokens == tokens && b.qk_bits == qk_bits)
        {
            return Ok(());
        }
        self.blocks = None;
        let mut d = StackDims {
            hidden: HID,
            heads: HEADS,
            kv_heads: HEADS,
            head_dim: HEAD_DIM,
            ffn: FFN,
            rope_dim: ROPE_DIM,
            classes: CLASSES,
            wbits: 8,
            eps: 1e-5,
            bias: false,
            gate_first: true,
            causal: false,
            attn_i4: qk_bits == 4,
            attn_qk_bits: qk_bits,
            bf16: false,
        };
        // H3_ATTN_QK=f16|i8|i4 overrides the configured width
        if let Some(v) = crate::stack::env_once("H3_ATTN_QK") {
            let bits = match v {
                "f16" => 16,
                "i8" => 8,
                _ => 4,
            };
            d.attn_qk_bits = bits;
            d.attn_i4 = bits == 4;
        }
        let qk = d.attn_qk_bits;
        let stack = Stack::new(
            c,
            stream,
            d,
            tokens,
            BLOCKS,
            &self.weights,
            |i| format!("blocks.{i}."),
            true,
            self.constants.ones.clone(),
            "dit",
        )?;
        self.blocks = Some(Blocks {
            stack,
            tokens,
            qk_bits: qk,
            // the final head has only the video and audio timestep classes
            final_norm: NormMod::build(c, stream, HID, 1e-5, 2)?,
            final_norm_scale: self.weights.at(stream, "h3.final.norm", HID * 4)?,
            audio_in: Projection::build(c, stream, &self.weights, "h3.audio_in", AUDIO_CH, HID)?,
            video_in: Projection::build(c, stream, &self.weights, "h3.video_in", VIDEO_PATCH, HID)?,
            final_out: Projection::build(c, stream, &self.weights, "h3.final.out", HID, FINAL_N)?,
            text_copy: stream.allocate(seq_capacity(tokens) * HID * 4)?,
        });
        Ok(())
    }

    /// The two modulation tables for one step's four timestep embeddings.
    fn build_mods(
        &self,
        stream: &mut hrx::Stream,
        te: (&[f32; 8], &[f32; 8], &[f32; 8], &[f32; 8]),
    ) -> Result<()> {
        let cond = self.cond.as_ref().expect("conditioning is ready");
        let (tv, ta, tcv, tca) = te;
        cond.adaln
            .run(stream, &[*tv, *ta, *tcv, *tca], &cond.mods)?;
        cond.final_adaln
            .run(stream, &[*tv, *ta], &cond.final_table)?;
        Ok(())
    }

    /// `x` rows `[row0, row0 + rows)` from the f32 patch projection the checkpoint stores.
    #[allow(clippy::too_many_arguments)]
    fn embed_f32(
        &self,
        stream: &mut hrx::Stream,
        prof: &mut Profile,
        stage: &str,
        row0: usize,
        rows: usize,
        audio: bool,
        input: Option<hrx::View<'_>>,
    ) -> Result<()> {
        let seq = self.seq.as_ref().expect("a sequence has been sized");
        let blocks = self.blocks.as_ref().expect("projections prepared");
        let projection = if audio {
            &blocks.audio_in
        } else {
            &blocks.video_in
        };
        projection.run(
            stream,
            prof,
            stage,
            rows,
            input.unwrap_or_else(|| seq.in32.binding()),
            seq.x.slice(row0 * HID * 4, rows * HID * 4),
        )
    }

    /// The whole denoising run: prompt in, model-space latents out.
    #[allow(clippy::too_many_arguments)]
    pub fn denoise(
        &mut self,
        stream: &mut hrx::Stream,
        c: &Compiler,
        prof: &mut Profile,
        te: &mut TextEncoder,
        ids: &[i32],
        p: &DenoiseParams,
        // the QK operands' width, from the session's configuration: 16, 8 or 4
        qk_bits: usize,
        noise: Noise<'_>,
        refs: &[Reference<'_>],
        kfs: &[Keyframe<'_>],
        mut progress: Option<&mut dyn FnMut(usize, usize, f64) -> bool>,
    ) -> Result<Latents> {
        let sh = crate::layout::shape_for(p.height, p.width, p.frames)
            .ok_or_else(|| crate::error::Error::Invalid("no such shape".into()))?;
        if !(2..=1000).contains(&p.steps) {
            return invalid("steps must be 2..1000");
        }
        self.ensure_conditioning(stream, c)?;

        let n = ids.len();
        let lay_refs: Vec<crate::layout::Ref> = refs
            .iter()
            .map(|r| {
                let g = r.grid();
                crate::layout::Ref {
                    kind: r.kind(),
                    latent_t: g.frames as i32,
                    lat_h: g.height as i32,
                    lat_w: g.width as i32,
                    audio_t: r.audio_latents().map_or(0, |(_, n)| n as i32),
                    has_audio: r.audio_latents().is_some(),
                }
            })
            .collect();
        let lay_kfs: Vec<crate::layout::Keyframe> = kfs
            .iter()
            .map(|k| crate::layout::Keyframe {
                frame_index: k.frame_index,
                audio_t: k.audio.map_or(0, |(_, n)| n as i32),
                has_audio: k.audio.is_some(),
            })
            .collect();
        let mut lay = crate::layout::Layout::new(
            n,
            sh.latent_t as usize,
            sh.lat_h as usize,
            sh.lat_w as usize,
            sh.audio_t as usize,
            &lay_refs,
            &lay_kfs,
        )
        .map_err(crate::error::Error::Invalid)?;

        let (l, rr) = (n, lay.ref_rows);
        let lr = l + rr;
        let (na, nv, s) = (lay.audio_rows, lay.video_rows, lay.seq_len);
        self.ensure_seq(stream, s)?;

        // the images this prompt presents, through the tower, in the order of the placeholder runs
        let embeddings = self.vision_for(stream, c, prof, te, ids, refs, kfs)?;
        let spans: Vec<crate::te::Span<'_>> = embeddings
            .iter()
            .map(|(at, e)| crate::te::Span {
                at: *at,
                merged: &e.merged,
                deepstack: &e.deepstack,
            })
            .collect();
        for sp in &spans {
            lay.mark_vision(sp.at.start, sp.at.count)
                .map_err(crate::error::Error::Invalid)?;
        }
        self.text_in(stream, c, prof, te, ids, &spans)?;

        // the packed layout's per-row tables
        {
            let (mut cos, mut sin) = (vec![0.0f32; s * ROPE_HALF], vec![0.0f32; s * ROPE_HALF]);
            let inv = &self.cond.as_ref().expect("ready").inv_freq;
            crate::rope::dit(&lay.pos, inv, &mut cos, &mut sin);
            let seq = self.seq.as_mut().expect("sized above");
            seq.cls.write(stream, &lay.adaln_rows, CLASSES)?;
            // The final head runs on the generated rows alone, and has only the video and audio
            // timestep classes. The rows before them carry conditioning classes its two-row table
            // has no place for — 2 for a reference's video, 3 for its audio — so only the generated
            // span goes up, and the head binds it from row zero.
            seq.tcls.write(stream, &lay.tclass[lr..lr + na + nv], 2)?;
            crate::dispatch::upload_at(stream, &seq.cos, 0, crate::vvae::as_bytes(&cos))?;
            crate::dispatch::upload_at(stream, &seq.sin, 0, crate::vvae::as_bytes(&sin))?;
        }
        self.ensure_blocks(stream, c, s, qk_bits)?;

        // the latents, as rows
        let generated = na + nv;
        let (t_len, h, w, a) = (
            sh.latent_t as usize,
            sh.lat_h as usize,
            sh.lat_w as usize,
            sh.audio_t as usize,
        );
        let res = p.sampler == Sampler::ResMultistep;
        let (shift_v, shift_a) = (
            if p.video_shift > 0.0 {
                p.video_shift
            } else {
                12.0
            },
            if p.audio_shift > 0.0 {
                p.audio_shift
            } else {
                3.0
            },
        );
        let ascale = shift_v / shift_a;

        let mut rng = crate::noise::Noise::new(p.seed);
        let mut vrows = vec![0.0f32; nv * VIDEO_PATCH];
        let mut arows = vec![0.0f32; na * AUDIO_CH];
        match noise.video {
            Some(z) => crate::sampler::tensor_to_rows(z, &mut vrows, t_len, h, w),
            None => {
                let mut lat = vec![0.0f32; LATENT_CH * t_len * h * w];
                rng.fill(&mut lat);
                crate::sampler::tensor_to_rows(&lat, &mut vrows, t_len, h, w);
            }
        }
        match noise.audio {
            Some(z) => crate::sampler::audio_tensor_to_rows(z, &mut arows, a),
            None => rng.fill(&mut arows),
        }
        // carry = sigma_a / sigma_v is one at sigma_v = 1, so the carried variable starts as the noise
        let mut yrows = if res { arows.clone() } else { Vec::new() };
        // Multistep reuses all host workspaces; Euler keeps its evolving latents on GPU.
        let host_v = if res { vrows.len() } else { 0 };
        let host_a = if res { arows.len() } else { 0 };
        let (mut old_v, mut old_a) = (vec![0.0; host_v], vec![0.0; host_a]);
        let (mut den_v, mut den_a) = (vec![0.0; host_v], vec![0.0; host_a]);
        let (mut vout, mut aout) = (vec![0.0; host_v], vec![0.0; host_a]);
        if !res {
            if !self.euler.as_ref().is_some_and(|e| e.matches(na, nv)) {
                self.euler = None;
                self.euler = Some(crate::sampler::device::Euler::new(stream, c, na, nv)?);
            }
            self.euler
                .as_ref()
                .expect("Euler prepared")
                .upload(stream, &arows, &vrows)?;
        }

        let sv = crate::layout::Schedule::new(p.steps, shift_v);
        let sa = crate::layout::Schedule::new(p.steps, shift_a);
        if sv.timesteps.len() != sa.timesteps.len() {
            return crate::error::other("the two schedules differ in length");
        }
        let mut cache = crate::cache::StepCache::new(p.cache_threshold);
        if cache.is_some() {
            self.ensure_cache(stream, c, s * HID)?;
        }

        let mut in_rows = na.max(nv);
        for sg in &lay.ref_segs {
            in_rows = in_rows.max(sg.rows);
        }
        let started = std::time::Instant::now();
        let mut in32 = vec![0.0f32; in_rows * VIDEO_PATCH];
        let mut out32 = vec![0.0f32; if res { generated * FINAL_N } else { 0 }];
        // Retain the curve independently across the mutable block execution below.
        let curve = self.cond.as_ref().expect("ready").curve.clone();

        for step in 0..sv.timesteps.len() {
            // the four timestep embeddings: video, audio, and the two conditioning classes, which
            // sit at least at the augmentation timestep so a reference never reads as fully denoised
            let tv = crate::conditioning::temb(&curve, sv.timesteps[step]);
            let ta = crate::conditioning::temb(&curve, sa.timesteps[step]);
            let tcv = crate::conditioning::temb(&curve, sv.timesteps[step].max(VISUAL_COND_AUG));
            let tca = crate::conditioning::temb(&curve, sa.timesteps[step].max(1.0));
            self.build_mods(stream, (&tv, &ta, &tcv, &tca))?;

            // audio rows then video rows, each through its f32 patch projection
            if res {
                let carry = sa.sigmas[step] / sv.sigmas[step];
                for (x, y) in arows.iter_mut().zip(&yrows) {
                    *x = y * carry;
                }
            }
            if res {
                let seq = self.seq.as_ref().expect("sized above");
                crate::dispatch::upload_at(stream, &seq.in32, 0, crate::vvae::as_bytes(&arows))?;
            }
            let input = self
                .euler
                .as_ref()
                .filter(|_| !res)
                .map(|e| e.audio.binding());
            self.embed_f32(stream, prof, "audio in", lr, na, true, input)?;
            if res {
                let seq = self.seq.as_ref().expect("sized above");
                crate::dispatch::upload_at(stream, &seq.in32, 0, crate::vvae::as_bytes(&vrows))?;
            }
            let input = self
                .euler
                .as_ref()
                .filter(|_| !res)
                .map(|e| e.video.binding());
            self.embed_f32(stream, prof, "video in", lr + na, nv, false, input)?;

            if step == 0 {
                self.inject_references(stream, prof, &lay, refs, kfs, p.seed, &mut in32)?;
                let seq = self.seq.as_ref().expect("sized above");
                let b = self.blocks.as_ref().expect("built above");
                stream.copy(
                    b.text_copy.slice(0, lr * HID * 4),
                    seq.x.slice(0, lr * HID * 4),
                )?;
            } else {
                let seq = self.seq.as_ref().expect("sized above");
                let b = self.blocks.as_ref().expect("built above");
                // the blocks update x in place, so the text and reference rows are restored each step
                stream.copy(
                    seq.x.slice(0, lr * HID * 4),
                    b.text_copy.slice(0, lr * HID * 4),
                )?;
            }

            self.run_blocks(stream, c, prof, s, step, cache.as_mut())?;

            // the final layer: the modulated norm on the generated rows, then the two f32 heads
            {
                let seq = self.seq.as_ref().expect("sized above");
                let b = self.blocks.as_ref().expect("built above");
                let cond = self.cond.as_ref().expect("ready");
                // two classes here, the video and audio timesteps, so four table rows
                b.final_norm.run(
                    stream,
                    Some(prof),
                    "final norm",
                    generated,
                    seq.x.slice(lr * HID * 4, generated * HID * 4),
                    b.final_norm_scale.binding(),
                    cond.final_table.binding(),
                    seq.tcls.slice(0, generated),
                )?;
            }
            {
                let seq = self.seq.as_ref().expect("sized above");
                let b = self.blocks.as_ref().expect("built above");
                b.final_out.run(
                    stream,
                    prof,
                    "final out",
                    generated,
                    seq.x.slice(lr * HID * 4, generated * HID * 4),
                    seq.out32.binding(),
                )?;
                if res {
                    stream.read(
                        seq.out32.slice(0, generated * FINAL_N * 4),
                        crate::vvae::as_bytes_mut(&mut out32),
                    )?;
                } else {
                    self.euler.as_ref().expect("Euler prepared").step(
                        stream,
                        seq.out32.binding(),
                        (sa.sigmas[step], sa.sigmas[step + 1] / sa.sigmas[step]),
                        (sv.sigmas[step], sv.sigmas[step + 1] / sv.sigmas[step]),
                    )?;
                }
            }

            if res {
                for (i, value) in vout.iter_mut().enumerate() {
                    *value = out32[(na + i / VIDEO_PATCH) * FINAL_N + i % VIDEO_PATCH];
                }
                for (i, value) in aout.iter_mut().enumerate() {
                    *value = out32[(i / AUDIO_CH) * FINAL_N + VIDEO_PATCH + i % AUDIO_CH];
                }
                let (sg_v, sg_a) = (sv.sigmas[step], sa.sigmas[step]);
                crate::sampler::denoised_video_into(&vrows, &vout, sg_v, &mut den_v);
                crate::sampler::denoised_audio_into(
                    &yrows, &arows, &aout, sg_v, sg_a, ascale, &mut den_a,
                );
                let previous_v = (step > 0).then_some(old_v.as_slice());
                let previous_a = (step > 0).then_some(old_a.as_slice());
                crate::sampler::advance(&mut vrows, &den_v, previous_v, &sv.sigmas, step);
                crate::sampler::advance(&mut yrows, &den_a, previous_a, &sv.sigmas, step);
                std::mem::swap(&mut old_v, &mut den_v);
                std::mem::swap(&mut old_a, &mut den_a);
            }
            if let Some(cb) = progress.as_deref_mut() {
                // Progress reports completed steps, even when Euler stays on the GPU.
                if !res {
                    stream.synchronize()?;
                }
                if cb(
                    step + 1,
                    sv.timesteps.len(),
                    started.elapsed().as_secs_f64(),
                ) {
                    return Err(crate::error::Error::Cancelled);
                }
            }
        }

        if let Some(c) = cache.as_ref() {
            if crate::stack::env_once("H3_CACHE_TRACE").is_some() {
                eprintln!(
                    "  step cache: {} of {} evaluations skipped",
                    c.skipped(),
                    sv.timesteps.len()
                );
            }
        }

        if !res {
            self.euler
                .as_ref()
                .expect("Euler prepared")
                .download(stream, &mut arows, &mut vrows)?;
        }
        let mut video = vec![0.0f32; LATENT_CH * t_len * h * w];
        let mut audio = vec![0.0f32; 2 * AUDIO_CH * a];
        crate::sampler::rows_to_tensor(&vrows, &mut video, t_len, h, w);
        crate::sampler::audio_rows_to_tensor(
            if res { &yrows } else { &arows },
            &mut audio,
            a,
            res.then_some(ascale),
        );
        Ok(Latents { video, audio })
    }
}

impl Dit {
    /// The tower's output for every image this prompt presents, paired with the rows it fills.
    ///
    /// A run of negative ids is a placeholder for an image; the images are matched to those runs in
    /// presentation order — keyframes first, then image references — and each must be exactly the size
    /// its run reserved.
    #[allow(clippy::too_many_arguments)]
    fn vision_for(
        &self,
        stream: &mut hrx::Stream,
        c: &Compiler,
        prof: &mut Profile,
        te: &TextEncoder,
        ids: &[i32],
        refs: &[Reference<'_>],
        kfs: &[Keyframe<'_>],
    ) -> Result<Vec<(VisionSpan, crate::vision::Embedding)>> {
        let mut runs: Vec<(usize, usize)> = Vec::new();
        for (i, id) in ids.iter().enumerate() {
            if *id >= 0 {
                continue;
            }
            match runs.last_mut() {
                Some(r) if r.0 + r.1 == i => r.1 += 1,
                _ => runs.push((i, 1)),
            }
        }
        let mut pics: Vec<(&[f32], usize, usize)> = Vec::new();
        for kf in kfs {
            if let Some(p) = kf.presented {
                pics.push((p.pixels, p.height, p.width));
            }
        }
        for rf in refs {
            if let Some(p) = rf.presented() {
                pics.push((p.pixels, p.height, p.width));
            }
        }
        if pics.len() > runs.len() {
            return invalid("more images than placeholder runs in the ids");
        }
        if pics.len() < runs.len() {
            return invalid("placeholder runs in the ids without an image reference with pixels");
        }
        let mut out = Vec::with_capacity(pics.len());
        for (ri, (px, ph, pw)) in pics.into_iter().enumerate() {
            let (mh, mw) = (ph / 32, pw / 32);
            if runs[ri].1 != mh * mw {
                return invalid(format!(
                    "placeholder run {ri} has {} ids, the image needs {}",
                    runs[ri].1,
                    mh * mw
                ));
            }
            let e = crate::vision::embed(stream, c, prof, te.weights(), px, ph, pw)?;
            out.push((
                VisionSpan {
                    start: runs[ri].0,
                    count: runs[ri].1,
                    merged_h: mh,
                    merged_w: mw,
                },
                e,
            ));
        }
        Ok(out)
    }

    /// The reference rows, written once at the first step and restored from the copy thereafter.
    ///
    /// Visual references are mixed with seeded noise at 0.999 — ComfyUI's condition augmentation —
    /// from a generator seeded per segment, so a reference's augmentation does not depend on how many
    /// latents preceded it.
    #[allow(clippy::too_many_arguments)]
    fn inject_references(
        &self,
        stream: &mut hrx::Stream,
        prof: &mut Profile,
        lay: &crate::layout::Layout,
        refs: &[Reference<'_>],
        kfs: &[Keyframe<'_>],
        seed: u64,
        in32: &mut [f32],
    ) -> Result<()> {
        for sg in &lay.ref_segs {
            // a keyframe presents as a one-frame image reference
            let (video_latent, audio_latent) = if sg.kind == 3 {
                let kf = &kfs[sg.index];
                (Some(kf.latents), kf.audio.map(|(z, _)| z))
            } else {
                let rf = &refs[sg.index];
                (rf.video_latents(), rf.audio_latents().map(|(z, _)| z))
            };
            if sg.audio {
                let Some(al) = audio_latent else {
                    return invalid("an audio reference segment without audio latents");
                };
                let at = sg.audio_t as usize;
                for ch in 0..2 {
                    for t in 0..at {
                        for k in 0..AUDIO_CH {
                            in32[(ch * at + t) * AUDIO_CH + k] = al[(ch * AUDIO_CH + k) * at + t];
                        }
                    }
                }
                let seq = self.seq.as_ref().expect("sized above");
                crate::dispatch::upload_at(
                    stream,
                    &seq.in32,
                    0,
                    crate::vvae::as_bytes(&in32[..sg.rows * AUDIO_CH]),
                )?;
                self.embed_f32(stream, prof, "ref audio in", sg.row0, sg.rows, true, None)?;
            } else {
                let Some(vl) = video_latent else {
                    return invalid("a visual reference segment without video latents");
                };
                crate::noise::augment_ref(
                    vl,
                    &mut in32[..sg.rows * VIDEO_PATCH],
                    sg.latent_t as usize,
                    sg.lat_h as usize,
                    sg.lat_w as usize,
                    seed,
                );
                let seq = self.seq.as_ref().expect("sized above");
                crate::dispatch::upload_at(
                    stream,
                    &seq.in32,
                    0,
                    crate::vvae::as_bytes(&in32[..sg.rows * VIDEO_PATCH]),
                )?;
                self.embed_f32(stream, prof, "ref video in", sg.row0, sg.rows, false, None)?;
            }
        }
        Ok(())
    }

    fn ensure_cache(&mut self, stream: &mut hrx::Stream, c: &Compiler, n: usize) -> Result<()> {
        if self.cache.as_ref().is_some_and(|b| b.n >= n) {
            return Ok(());
        }
        self.cache = None;
        self.cache = Some(CacheBuffers {
            n,
            xb0: stream.allocate(n * 4)?,
            prev: stream.allocate(n * 4)?,
            resid: stream.allocate(n * 4)?,
            partials: stream.allocate(crate::cache::groups(n) * 8)?,
            metric: c.get(stream, "absdiff_sum_f32", "h3_absdiff_sum_f32", &Cfg::new())?,
        });
        Ok(())
    }

    /// The 50 blocks, either straight through or with the first-block cache deciding.
    fn run_blocks(
        &mut self,
        stream: &mut hrx::Stream,
        c: &Compiler,
        prof: &mut Profile,
        s: usize,
        step: usize,
        cache: Option<&mut crate::cache::StepCache>,
    ) -> Result<()> {
        // The table is one field and the buffers another, so they are taken apart before the views
        // are made: a view borrows its allocation, and borrowing all of `self` to build them would
        // conflict with the mutable borrow the stack needs.
        let Dit {
            seq,
            blocks,
            cache: cbuf,
            cond,
            ..
        } = self;
        let mods = &cond.as_ref().expect("conditioning is ready").mods;
        let conds: Vec<crate::stack::LayerCond> =
            (0..BLOCKS).map(|i| layer_cond(mods, i)).collect();
        let cond_fn = |i: usize| conds[i];
        let seq = seq.as_ref().expect("sized above");
        let b = blocks.as_mut().expect("built above");
        let (x, cls, cos, sin) = (
            seq.x.binding(),
            seq.cls.slice(0, s),
            seq.cos.binding(),
            seq.sin.binding(),
        );
        let Some(cache) = cache else {
            b.stack
                .forward(stream, prof, x, cls, cos, sin, &cond_fn, 0, None)?;
            return Ok(());
        };

        let cb = cbuf.as_ref().expect("cache buffers");
        let n = s * HID;
        b.stack
            .forward(stream, prof, x, cls, cos, sin, &cond_fn, 0, Some(1))?;
        let (mut d, mut m) = (0.0f64, 0.0f64);
        if step > 0 {
            let groups = crate::cache::groups(n);
            checked(
                stream,
                &cb.metric,
                Some(prof),
                "cache metric",
                [groups as u32, 1, 1],
                [THREADS, 1, 1],
                &[n as u32],
                &[x, cb.prev.binding(), cb.partials.binding()],
                &[n * 4, n * 4, groups * 8],
            )?;
            let mut ps = vec![0.0f32; groups * 2];
            stream.synchronize()?;
            stream.read(
                cb.partials.slice(0, groups * 8),
                crate::vvae::as_bytes_mut(&mut ps),
            )?;
            for i in 0..groups {
                d += f64::from(ps[2 * i]);
                m += f64::from(ps[2 * i + 1]);
            }
        }
        let change = cache.consider(step, d, m);
        // H3_CACHE_TRACE: the decision per step, as the C printed it. Without it the threshold is
        // impossible to choose — the useful range is narrow and depends on the prompt.
        if crate::stack::env_once("H3_CACHE_TRACE").is_some() {
            eprintln!(
                "  step {}: block-0 change {:.4}, accumulated {:.4} -> {}",
                step + 1,
                change.relative,
                change.accumulated,
                if change.skip { "cached" } else { "full" }
            );
        }
        stream.copy(cb.prev.slice(0, n * 4), seq.x.slice(0, n * 4))?;
        if change.skip {
            axpy(
                c,
                stream,
                Some(prof),
                "cache add",
                1.0,
                1.0,
                n,
                cb.resid.binding(),
                x,
            )?;
        } else {
            stream.copy(cb.xb0.slice(0, n * 4), seq.x.slice(0, n * 4))?;
            b.stack
                .forward(stream, prof, x, cls, cos, sin, &cond_fn, 1, None)?;
            // the residual of blocks 1..49, which a skipped step adds instead of running them
            stream.copy(cb.resid.slice(0, n * 4), seq.x.slice(0, n * 4))?;
            axpy(
                c,
                stream,
                Some(prof),
                "cache resid",
                -1.0,
                1.0,
                n,
                cb.xb0.binding(),
                cb.resid.binding(),
            )?;
            cache.recorded();
        }
        Ok(())
    }
}

/// One layer's slice of the modulation table: rows [0, 2C) are (scale, shift) per class for the
/// attention, [2C, 3C) its gate, [3C, 5C) and [5C, 6C) the same for the MLP.
fn layer_cond(mods: &hrx::Buffer, i: usize) -> crate::stack::LayerCond<'_> {
    let row = |r: usize| mods.slice((i * MODS_ROWS + r) * HID * 4, (MODS_ROWS - r) * HID * 4);
    crate::stack::LayerCond {
        table_msa: row(0),
        gate_msa: row(2 * CLASSES),
        table_mlp: row(3 * CLASSES),
        gate_mlp: row(5 * CLASSES),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_reference_grid_is_whole_patches() {
        let grid = |height, width| LatentGrid {
            frames: 1,
            height,
            width,
        };
        let lat = vec![0.0f32; LATENT_CH * 8 * 8];
        let of = |g| Reference::Image {
            latents: &lat,
            grid: g,
            presented: None,
        };
        of(grid(4, 4)).check(0).expect("an even grid");
        // 3x2 sizes its rows as (3/2)*(2/2) = 1 and then packs a second row of patches into it
        assert!(of(grid(3, 2)).check(0).is_err(), "an odd height");
        assert!(of(grid(4, 3)).check(0).is_err(), "an odd width");
    }

    #[test]
    fn the_refiner_has_no_rope_and_one_class() {
        let d = refiner_dims();
        assert!(d.bf16, "its rows are bf16 as stored");
        assert_eq!(d.classes, 1);
        assert_eq!(d.kv_heads, d.heads, "no grouped-query attention here");
        assert!(!d.causal);
    }

    #[test]
    fn the_sequence_capacity_leaves_the_attention_its_slack() {
        assert_eq!(seq_capacity(1), 288);
        assert_eq!(seq_capacity(256), 288);
        assert_eq!(seq_capacity(257), 544);
        assert_eq!(seq_capacity(30_000), 30_240);
        // it is monotonic and always leaves room, which is what makes "does it fit" a valid test
        let mut prev = 0;
        for n in (1..4000).step_by(7) {
            let cap = seq_capacity(n);
            assert!(cap >= n + 32, "{n} fits in {cap} with slack");
            assert!(cap >= prev);
            prev = cap;
        }
    }
}

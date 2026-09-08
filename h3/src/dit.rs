//! The DiT side of the pipeline: the packed sequence buffer, the token refiner, and the pieces that
//! write into the sequence before the blocks run.
//!
//! The sequence buffer is one allocation shared by everything — text rows, keyframes, references,
//! audio and video all live in it at offsets the layout decides — so it is sized once for the longest
//! sequence a session will see and reused. That is also why `text_in` lands here rather than in the
//! text encoder: the encoder produces 5120-wide rows, and it is the DiT that projects them to 5376 and
//! refines them in place at the front of its own sequence.
use crate::compile::{Cfg, Compiler};
use crate::dispatch::{launch, Matmul16, Profile};
use crate::error::Result;
use crate::model::*;
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
#[allow(dead_code)]
struct Seq {
    capacity: usize,
    x: hrx::Buffer,
    /// the per-row timestep class the modulated kernels index with
    cls: hrx::Buffer,
    /// all zeros: what the single-class GEMMs index their gate table with
    cls0: hrx::Buffer,
    tcls: hrx::Buffer,
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
    norm: std::sync::Arc<hrx::Kernel>,
}

pub struct Dit {
    weights: Weights,
    constants: Constants,
    seq: Option<Seq>,
    refiner: Option<Refiner>,
}

impl Dit {
    pub fn open(gpu: &hrx::Gpu, path: impl AsRef<std::path::Path>) -> Result<Self> {
        Ok(Self {
            weights: Weights::open(path, crate::plan::dit::plan)?,
            constants: Constants::new(gpu)?,
            seq: None,
            refiner: None,
        })
    }

    pub fn weights(&self) -> &Weights {
        &self.weights
    }

    /// Grows the sequence buffers if this length does not fit. The capacity is rounded to 256 rows
    /// plus 32, which is the slack the attention kernels read past the end of a sequence.
    pub fn ensure_seq(&mut self, gpu: &hrx::Gpu, seq: usize) -> Result<()> {
        if self.seq.as_ref().is_some_and(|s| seq <= s.capacity) {
            return Ok(());
        }
        self.seq = None;
        let t = seq_capacity(seq);
        let zeroed = |bytes: usize| -> Result<hrx::Buffer> {
            let b = gpu.alloc(bytes)?;
            gpu.memset(&b, 0, bytes)?;
            Ok(b)
        };
        self.seq = Some(Seq {
            capacity: t,
            x: zeroed(t * HID * 4)?,
            cls: zeroed(t * 4)?,
            cls0: zeroed(t * 4)?,
            tcls: zeroed(t * 4)?,
            cos: gpu.alloc(t * ROPE_HALF * 4)?,
            sin: gpu.alloc(t * ROPE_HALF * 4)?,
            in32: gpu.alloc(t * VIDEO_PATCH * 4)?,
            out32: gpu.alloc(t * FINAL_N * 4)?,
        });
        Ok(())
    }

    fn ensure_refiner(&mut self, gpu: &hrx::Gpu, c: &Compiler, n: usize) -> Result<()> {
        if self.refiner.as_ref().is_some_and(|r| r.tokens == n) {
            return Ok(());
        }
        self.refiner = None;
        let stack = Stack::new(
            c,
            gpu,
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
        let cos = gpu.alloc(n * ROPE_HALF * 4)?;
        let sin = gpu.alloc(n * ROPE_HALF * 4)?;
        gpu.h2d(&cos, crate::vvae::as_bytes(&vec![1.0f32; n * ROPE_HALF]))?;
        gpu.memset(&sin, 0, n * ROPE_HALF * 4)?;
        let ns = "h3.norm_mod_f32.";
        let cfg: Cfg = vec![
            (format!("{ns}width"), HID.to_string()),
            (
                format!("{ns}lanes"),
                lanes_for(HID)
                    .expect("a lane count for the hidden width")
                    .to_string(),
            ),
            (format!("{ns}eps"), crate::compile::num(1e-5)),
            (format!("{ns}classes"), "1".into()),
        ];
        self.refiner = Some(Refiner {
            stack,
            tokens: n,
            cos,
            sin,
            norm: c.get(gpu, "norm_mod_f32", "h3_norm_mod_f32", &cfg)?,
        });
        Ok(())
    }

    /// The prompt into the front of the sequence: the encoder, `condition_proj` to the DiT's width,
    /// then the refiner and its final norm, all landing in `x` rows `[0, n)`.
    pub fn text_in(
        &mut self,
        gpu: &hrx::Gpu,
        c: &Compiler,
        prof: &mut Profile,
        te: &mut TextEncoder,
        ids: &[i32],
        spans: &[Span<'_>],
    ) -> Result<()> {
        let n = ids.len();
        self.ensure_seq(gpu, n)?;
        let hidden = te.encode(gpu, c, prof, ids, spans)?;

        let seq = self.seq.as_ref().expect("sized above");
        Matmul16::build(c, gpu, "bias", TEXT_DIM, HID)?.run(
            gpu,
            Some(prof),
            "condition proj",
            n,
            hidden,
            self.weights
                .at(gpu, "h3.cond.w", HID * TEXT_DIM * 2)?
                .binding(),
            self.weights.at(gpu, "h3.cond.b", HID * 4)?.binding(),
            seq.x.binding(),
            None,
        )?;

        self.ensure_refiner(gpu, c, n)?;
        let r = self.refiner.as_mut().expect("built above");
        let seq = self.seq.as_ref().expect("sized above");
        let cond = self.constants.identity();
        let cond_fn = |_: usize| crate::stack::LayerCond { ..cond };
        r.stack.forward(
            gpu,
            prof,
            seq.x.binding(),
            seq.cls0.binding(),
            r.cos.binding(),
            r.sin.binding(),
            &cond_fn,
            0,
            None,
        )?;
        launch(
            gpu,
            &r.norm,
            Some(prof),
            "refiner final norm",
            [n as u32, 1, 1],
            [lanes_for(HID).expect("a lane count") as u32, 1, 1],
            &[n as u32],
            &[
                seq.x.binding(),
                self.weights
                    .at(gpu, "h3.refiner.final_norm", HID * 4)?
                    .binding(),
                self.constants.zeros.binding(),
                seq.cls0.binding(),
            ],
        )?;
        Ok(())
    }

    /// The sequence's first `rows` rows, `[rows][5376]` f32 — what `h3pipe_text_in` hands back.
    pub fn read_rows(&self, gpu: &hrx::Gpu, rows: usize, out: &mut [f32]) -> Result<()> {
        let seq = self.seq.as_ref().expect("a sequence has been sized");
        gpu.sync()?;
        gpu.d2h_ref(
            seq.x.slice(0, rows * HID * 4),
            crate::vvae::as_bytes_mut(out),
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

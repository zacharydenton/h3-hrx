//! The text encoder: Qwen3-VL's 50 causal layers over the prompt's token embeddings, with the vision
//! tower's DeepStack features folded in at the image rows.
//!
//! Two things make it more than a `Stack` with different dimensions. Its rotary tables depend on where
//! the images sit in the prompt — Qwen2-VL's mrope gives an image span its grid rather than a running
//! index, and offsets everything after it — so the stack and its tables are rebuilt when the spans
//! move, not only when the length changes. And the DeepStack features are added *between* layers 0, 1
//! and 2, which is why the forward pass is split rather than run in one call.
use crate::compile::Compiler;
use crate::dispatch::{axpy, Profile};
use crate::error::{invalid, Result};
use crate::model::*;
use crate::rope::{self, VisionSpan};
use crate::stack::{Constants, Stack, StackDims};
use crate::weights::Weights;

/// One image's contribution to the prompt: where its rows sit, and what the tower produced for them.
pub struct Span<'a> {
    pub at: VisionSpan,
    /// `[count][5120]`, the tower's merged embedding, which replaces the token embeddings
    pub merged: &'a [f32],
    /// `[3][count][5120]`, added after layers 0, 1 and 2
    pub deepstack: &'a [f32],
}

/// The longest image span the DeepStack staging buffer takes.
const MAX_SPAN: usize = 4096;

fn dims() -> StackDims {
    StackDims {
        hidden: TE_HID,
        heads: TE_HEADS,
        kv_heads: TE_KV,
        head_dim: HEAD_DIM,
        ffn: TE_FFN,
        rope_dim: 128,
        classes: 1,
        wbits: 8,
        eps: 1e-6,
        bias: false,
        gate_first: true,
        causal: true,
        attn_i4: false,
        attn_qk_bits: 16,
        bf16: false,
    }
}

struct Built {
    stack: Stack,
    /// what the tables were built for: the token count and where the image spans sat
    signature: (usize, Vec<(usize, usize, usize, usize)>),
    x: hrx::Buffer,
    cos: hrx::Buffer,
    sin: hrx::Buffer,
    cls: crate::dispatch::Classes,
    ds: hrx::Buffer,
}

pub struct TextEncoder {
    weights: Weights,
    constants: Constants,
    built: Option<Built>,
}

impl TextEncoder {
    /// # Safety
    ///
    /// Maps the checkpoint; see [`crate::Session::new`].
    pub unsafe fn open(
        stream: &mut hrx::Stream,
        path: impl AsRef<std::path::Path>,
    ) -> Result<Self> {
        Ok(Self {
            weights: unsafe { Weights::open(path, crate::plan::te::plan) }?,
            constants: Constants::new(stream)?,
            built: None,
        })
    }

    /// The checkpoint, which also holds the vision tower's weights.
    pub fn weights(&self) -> &Weights {
        &self.weights
    }

    /// The prompt's embedding rows, `[n][5120]` f32.
    ///
    /// The table is bf16 in the file and widened here. A row covered by an image span keeps the
    /// tower's embedding instead, and its token id is allowed to be negative — that is how a caller
    /// says "there is no token here", and it is the only case where a negative id is not an error.
    fn embed(&self, ids: &[i32], spans: &[Span<'_>]) -> Result<Vec<f32>> {
        let n = ids.len();
        let mut emb = vec![0.0f32; n * TEXT_DIM];
        for sp in spans {
            let at = sp.at.start * TEXT_DIM;
            emb[at..at + sp.at.count * TEXT_DIM].copy_from_slice(sp.merged);
        }
        // the table stays mapped and only the rows this prompt names are widened
        let entry = self.weights.file().at_checked(
            "model.embed_tokens.weight",
            hrx::artifacts::safetensors::DType::BF16,
            &[-1, TEXT_DIM as i64],
        )?;
        let table = self.weights.file().bytes(entry);
        let rows = entry.shape[0];
        for (i, id) in ids.iter().enumerate() {
            let covered = spans
                .iter()
                .any(|sp| i >= sp.at.start && i < sp.at.start + sp.at.count);
            if *id < 0 && covered {
                continue;
            }
            if *id < 0 || *id as usize >= rows {
                return invalid(format!("token id out of range: {id}"));
            }
            let src = *id as usize * TEXT_DIM * 2;
            for j in 0..TEXT_DIM {
                let bits = u16::from_le_bytes([table[src + 2 * j], table[src + 2 * j + 1]]);
                emb[i * TEXT_DIM + j] = half::bf16::from_bits(bits).to_f32();
            }
        }
        Ok(emb)
    }

    fn ensure(
        &mut self,
        stream: &mut hrx::Stream,
        c: &Compiler,
        n: usize,
        spans: &[Span<'_>],
    ) -> Result<()> {
        let signature: Vec<(usize, usize, usize, usize)> = spans
            .iter()
            .map(|s| (s.at.start, s.at.count, s.at.merged_h, s.at.merged_w))
            .collect();
        if self
            .built
            .as_ref()
            .is_some_and(|b| b.signature == (n, signature.clone()))
        {
            return Ok(());
        }
        self.built = None;
        let stack = Stack::new(
            c,
            stream,
            dims(),
            n,
            TE_LAYERS,
            &self.weights,
            |i| format!("blocks.{i}."),
            true,
            self.constants.ones.clone(),
            "te",
        )?;
        let t = stack.capacity();
        let x = stream.allocate_zeroed(t * TE_HID * 4)?;
        let cls = crate::dispatch::Classes::zeroed(stream, t)?;

        let positions = rope::mrope_positions(n, &signature_spans(spans));
        let (mut cos_h, mut sin_h) = (
            vec![0.0f32; n * TE_ROPE_HALF],
            vec![0.0f32; n * TE_ROPE_HALF],
        );
        rope::te(&positions, &mut cos_h, &mut sin_h);
        let cos = stream.allocate(t * TE_ROPE_HALF * 4)?;
        let sin = stream.allocate(t * TE_ROPE_HALF * 4)?;
        stream.upload_at(&cos, 0, crate::vvae::as_bytes(&cos_h))?;
        stream.upload_at(&sin, 0, crate::vvae::as_bytes(&sin_h))?;

        self.built = Some(Built {
            stack,
            signature: (n, signature),
            x,
            cos,
            sin,
            cls,
            ds: stream.allocate(MAX_SPAN * TEXT_DIM * 4)?,
        });
        Ok(())
    }

    /// Runs the encoder, leaving `[n][5120]` f32 on the device.
    pub fn encode(
        &mut self,
        stream: &mut hrx::Stream,
        c: &Compiler,
        prof: &mut Profile,
        ids: &[i32],
        spans: &[Span<'_>],
    ) -> Result<hrx::View<'_>> {
        for sp in spans {
            if sp.at.count > MAX_SPAN {
                return invalid(format!("vision span of {} rows is too long", sp.at.count));
            }
        }
        let n = ids.len();
        let emb = self.embed(ids, spans)?;
        self.ensure(stream, c, n, spans)?;
        let b = self.built.as_mut().expect("built above");
        stream.upload_at(&b.x, 0, crate::vvae::as_bytes(&emb))?;

        let cond = self.constants.identity();
        let cond_fn = |_: usize| crate::stack::LayerCond { ..cond };
        let (x, cls, cos, sin) = (b.x.binding(), b.cls.all(), b.cos.binding(), b.sin.binding());
        if spans.is_empty() {
            b.stack
                .forward(stream, prof, x, cls, cos, sin, &cond_fn, 0, None)?;
            return Ok(x);
        }
        // DeepStack: the tower's features from blocks 8, 16 and 24 land at the image rows after each of
        // the first three layers, so the pass is run one layer at a time until they are in.
        for layer in 0..3 {
            b.stack.forward(
                stream,
                prof,
                x,
                cls,
                cos,
                sin,
                &cond_fn,
                layer,
                Some(layer + 1),
            )?;
            for sp in spans {
                let take = sp.at.count * TEXT_DIM;
                let from = layer * take;
                stream.upload_at(
                    &b.ds,
                    0,
                    crate::vvae::as_bytes(&sp.deepstack[from..from + take]),
                )?;
                axpy(
                    c,
                    stream,
                    Some(prof),
                    "deepstack",
                    1.0,
                    1.0,
                    take,
                    b.ds.binding(),
                    b.x.slice(sp.at.start * TEXT_DIM * 4, take * 4),
                )?;
            }
        }
        b.stack
            .forward(stream, prof, x, cls, cos, sin, &cond_fn, 3, None)?;
        Ok(x)
    }
}

fn signature_spans(spans: &[Span<'_>]) -> Vec<VisionSpan> {
    spans.iter().map(|s| s.at).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_staging_buffer_holds_the_longest_span_allowed() {
        // A 4096-row span is a 1024x1024 image, four times the largest canvas; the check exists
        // because the buffer is a fixed allocation and a longer span would overrun it.
        // the patch grid is the image over 16, which is the latent grid, and four patches merge
        let widest = crate::layout::shape_for(768, 1344, 5).expect("the largest canvas");
        let merged = widest.lat_h as usize * widest.lat_w as usize / 4;
        assert_eq!(merged, 1008);
        assert!(merged < MAX_SPAN, "{merged} merged rows must fit");
    }

    #[test]
    fn the_encoder_is_causal_and_the_refiner_is_not() {
        assert!(dims().causal);
        assert_eq!(dims().kv_heads, TE_KV, "grouped-query attention");
        assert!(dims().kv_heads < dims().heads);
        assert!(!dims().bias, "the encoder's projections carry no bias");
    }
}

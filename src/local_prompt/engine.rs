//! Resident shared-prefix Qwen execution. KV state belongs to one completion only.
use super::{tail, LocalPromptConfig};
use crate::{
    compile::Compiler,
    dispatch::{axpy, Classes, Matmul16, Profile},
    model::*,
    rope::{self, VisionSpan},
    stack::{Constants, KvCache, Stack},
    te::{Span, TextEncoder},
    weights::Weights,
    Result,
};
use std::sync::Arc;

struct Run {
    prefix: Stack,
    tail: Stack,
    x: hrx::Buffer,
    cos: hrx::Buffer,
    sin: hrx::Buffer,
    classes: Classes,
}
impl Run {
    fn new(
        stream: &mut hrx::Stream,
        c: &Compiler,
        te: &TextEncoder,
        tail: &Weights,
        constants: &Constants,
        n: usize,
        cache: &[Arc<KvCache>; 2],
    ) -> Result<Self> {
        let mut d = crate::te::dims();
        let mut prefix = Stack::new(
            c,
            stream,
            d.clone(),
            n,
            TE_LAYERS,
            te.weights(),
            |i| format!("blocks.{i}."),
            true,
            constants.ones.clone(),
            "local-prefix",
        )?;
        d.wbits = 16;
        d.bf16 = true;
        let remaining_tail = tail.device_bytes().saturating_sub(tail.uploaded_bytes());
        if super::available_memory()? < remaining_tail.saturating_add(8usize << 30) {
            return crate::error::invalid(
                "insufficient RAM for the local Qwen continuation and 8 GiB system reserve",
            );
        }
        let mut end = Stack::new(
            c,
            stream,
            d,
            n,
            tail::LAYERS,
            tail,
            |i| format!("blocks.{i}."),
            true,
            constants.ones.clone(),
            "local-tail",
        )?;
        prefix.use_kv_cache(c, stream, cache[0].clone())?;
        end.use_kv_cache(c, stream, cache[1].clone())?;
        let capacity = prefix.capacity().max(end.capacity());
        Ok(Self {
            prefix,
            tail: end,
            x: stream.allocate_zeroed(capacity * TE_HID * 4)?,
            cos: stream.allocate_zeroed(capacity * TE_ROPE_HALF * 4)?,
            sin: stream.allocate_zeroed(capacity * TE_ROPE_HALF * 4)?,
            classes: Classes::zeroed(stream, capacity)?,
        })
    }
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &mut self,
        stream: &mut hrx::Stream,
        c: &Compiler,
        prof: &mut Profile,
        te: &TextEncoder,
        constants: &Constants,
        ids: &[i32],
        spans: &[Span<'_>],
        positions: &[f64],
        offset: usize,
        config: &LocalPromptConfig,
    ) -> Result<()> {
        self.prefix.cache_offset(offset)?;
        self.tail.cache_offset(offset)?;
        let mut cos = vec![0.; ids.len() * TE_ROPE_HALF];
        let mut sin = cos.clone();
        rope::te(positions, &mut cos, &mut sin);
        stream.upload_at(&self.cos, 0, bytemuck::cast_slice(&cos))?;
        stream.upload_at(&self.sin, 0, bytemuck::cast_slice(&sin))?;
        stream.upload_at(&self.x, 0, bytemuck::cast_slice(&te.embed(ids, spans)?))?;
        let cond = constants.identity();
        let cond_fn = |_| crate::stack::LayerCond { ..cond };
        for layer in 0..TE_LAYERS {
            if config.cancelled() {
                return Err(crate::Error::Cancelled);
            }
            self.prefix.forward(
                stream,
                prof,
                self.x.binding(),
                self.classes.all(),
                self.cos.binding(),
                self.sin.binding(),
                &cond_fn,
                layer,
                Some(layer + 1),
            )?;
            for sp in spans.iter().filter(|_| layer < 3) {
                let count = sp.at.count * TE_HID;
                let ds = stream.allocate(count * 4)?;
                stream.upload(
                    ds.binding(),
                    bytemuck::cast_slice(&sp.deepstack[layer * count..(layer + 1) * count]),
                )?;
                axpy(
                    c,
                    stream,
                    Some(prof),
                    "local deepstack",
                    1.,
                    1.,
                    count,
                    ds.binding(),
                    self.x.slice(sp.at.start * TE_HID * 4, count * 4),
                )?;
            }
        }
        for layer in 0..tail::LAYERS {
            if config.cancelled() {
                return Err(crate::Error::Cancelled);
            }
            self.tail.forward(
                stream,
                prof,
                self.x.binding(),
                self.classes.all(),
                self.cos.binding(),
                self.sin.binding(),
                &cond_fn,
                layer,
                Some(layer + 1),
            )?;
        }
        Ok(())
    }
}

pub(crate) struct Engine {
    pub stats: Vec<serde_json::Value>,
    weights: Weights,
    constants: Constants,
    norm: Vec<f32>,
    head: Arc<hrx::Buffer>,
    projection: Matmul16,
    bias: hrx::Buffer,
    logits: hrx::Buffer,
    normalized: hrx::Buffer,
}
impl Engine {
    pub fn new(stream: &mut hrx::Stream, c: &Compiler, weights: Weights) -> Result<Self> {
        let norm = weights.host_f32("final_norm", TE_HID)?;
        let head = weights.at(stream, "lm_head", tail::VOCAB * TE_HID * 2)?;
        Ok(Self {
            stats: Vec::new(),
            weights,
            constants: Constants::new(stream)?,
            norm,
            head,
            projection: Matmul16::build(c, stream, "bias", TE_HID, tail::VOCAB)?,
            bias: stream.allocate_zeroed(tail::VOCAB * 4)?,
            logits: stream.allocate(tail::VOCAB * 4)?,
            normalized: stream.allocate(TE_HID * 4)?,
        })
    }
    fn next(
        &self,
        stream: &mut hrx::Stream,
        prof: &mut Profile,
        run: &Run,
        last: usize,
    ) -> Result<u32> {
        let mut x = vec![0f32; TE_HID];
        x.copy_from_slice(bytemuck::cast_slice(
            &stream
                .read(run.x.slice(last * TE_HID * 4, TE_HID * 4))?
                .wait(stream)?,
        ));
        let inv = (x.iter().map(|x| x * x).sum::<f32>() / TE_HID as f32 + 1e-6)
            .sqrt()
            .recip();
        for (x, w) in x.iter_mut().zip(&self.norm) {
            *x *= inv * w;
        }
        stream.upload(self.normalized.binding(), bytemuck::cast_slice(&x))?;
        self.projection.run(
            stream,
            Some(prof),
            "local lm head",
            1,
            self.normalized.binding(),
            self.head.binding(),
            self.bias.binding(),
            self.logits.binding(),
            None,
        )?;
        let mut logits = vec![0f32; tail::VOCAB];
        logits.copy_from_slice(bytemuck::cast_slice(
            &stream.read(self.logits.binding())?.wait(stream)?,
        ));
        if logits.iter().any(|x| !x.is_finite()) {
            return crate::error::invalid("non-finite local prompt logits");
        }
        Ok(logits
            .iter()
            .enumerate()
            .max_by(|(ia, a), (ib, b)| a.total_cmp(b).then_with(|| ib.cmp(ia)))
            .expect("vocabulary")
            .0 as u32)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn complete(
        &mut self,
        stream: &mut hrx::Stream,
        c: &Compiler,
        prof: &mut Profile,
        te: &TextEncoder,
        ids: &[i32],
        spans: &[Span<'_>],
        config: &LocalPromptConfig,
    ) -> Result<String> {
        if ids.is_empty()
            || ids
                .len()
                .checked_add(config.max_tokens)
                .is_none_or(|n| n > config.context_tokens)
        {
            return crate::error::invalid("local prompt plus output budget exceeds context");
        }
        let started = std::time::Instant::now();
        crate::trace::event("local_prefill_start", || {
            format!(",\"prompt_tokens\":{}", ids.len())
        });
        let cache = [
            KvCache::new(stream, TE_LAYERS, config.context_tokens, TE_KV * HEAD_DIM)?,
            KvCache::new(
                stream,
                tail::LAYERS,
                config.context_tokens,
                TE_KV * HEAD_DIM,
            )?,
        ];
        let positions = rope::mrope_positions(
            ids.len(),
            &spans.iter().map(|s| s.at).collect::<Vec<VisionSpan>>(),
        );
        let next_position = positions.iter().copied().fold(0f64, f64::max) + 1.;
        let mut prefill = Run::new(
            stream,
            c,
            te,
            &self.weights,
            &self.constants,
            ids.len(),
            &cache,
        )?;
        prefill.forward(
            stream,
            c,
            prof,
            te,
            &self.constants,
            ids,
            spans,
            &positions,
            0,
            config,
        )?;
        let mut token = self.next(stream, prof, &prefill, ids.len() - 1)?;
        stream.synchronize()?;
        drop(prefill);
        let mut decode = Run::new(stream, c, te, &self.weights, &self.constants, 1, &cache)?;
        let prefill_seconds = started.elapsed().as_secs_f64();
        crate::trace::event("local_prefill_ready", || {
            format!(",\"seconds\":{prefill_seconds}")
        });
        let decode_started = std::time::Instant::now();
        let mut output = Vec::new();
        let tok = crate::Tokenizer::checked_local()?;
        for i in 0..config.max_tokens {
            if config.cancelled() {
                return Err(crate::Error::Cancelled);
            }
            if super::available_memory()? < 8usize << 30 {
                return crate::error::invalid(
                    "local prompting stopped: available RAM fell below the 8 GiB system reserve",
                );
            }
            if token == 151645 || token == 151643 {
                let text = tok.decode(&output)?;
                if text.trim().is_empty() {
                    return crate::error::invalid("local prompt completion was empty");
                }
                self.stats.push(serde_json::json!({"prompt_tokens":ids.len(),"generated_tokens":output.len(),"prefill_seconds":prefill_seconds,"decode_seconds":decode_started.elapsed().as_secs_f64()}));
                return Ok(text.trim().to_owned());
            }
            output.push(token);
            if output.len().is_multiple_of(32) {
                crate::trace::event("local_decode_progress", || {
                    format!(",\"generated_tokens\":{}", output.len())
                });
            }
            if i + 1 == config.max_tokens {
                break;
            }
            let position = next_position + i as f64;
            decode.forward(
                stream,
                c,
                prof,
                te,
                &self.constants,
                &[token as i32],
                &[],
                &[position; 3],
                ids.len() + i,
                config,
            )?;
            token = self.next(stream, prof, &decode, 0)?;
        }
        #[cfg(test)]
        eprintln!("unfinished local completion: {}", tok.decode(&output)?);
        crate::error::invalid("local prompt completion reached its output limit before EOS")
    }
}

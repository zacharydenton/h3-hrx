//! Session-owned local completion: the encoder stays shared; generation-only state is disposable.
use super::*;
use crate::{
    local_prompt::{self, LocalPromptConfig},
    media_context::PreparedPresentation,
    te::Span,
};
use serde_json::Value;
impl Session {
    pub(crate) fn local_resolve_refmod_vaes(
        &mut self,
        offline: bool,
        visual: bool,
        audio: bool,
    ) -> Result<()> {
        self.scheduled(|state| {
            let resolver = crate::models::Resolver::new().offline(offline);
            if visual && state.config.video_vae.is_none() {
                state.config.video_vae = Some(resolver.find(crate::models::VIDEO_VAE)?);
            }
            if audio && state.config.audio_vae.is_none() {
                state.config.audio_vae = Some(resolver.find(crate::models::AUDIO_VAE)?);
            }
            Ok(())
        })
    }
    pub(crate) fn local_complete(
        &mut self,
        config: &LocalPromptConfig,
        messages: &[Value],
        presentation: Option<&PreparedPresentation>,
    ) -> Result<String> {
        if config.cancelled() {
            return Err(crate::Error::Cancelled);
        }
        let (ids, offset) = local_prompt::chat_ids(messages, presentation)?;
        if ids
            .len()
            .checked_add(config.max_tokens)
            .is_none_or(|n| n > config.context_tokens)
        {
            return invalid("local prompt plus output budget exceeds context");
        }
        let manager = self.context.runtime().memory_budget()
            .and_then(|b| b.manager())
            .ok_or_else(|| crate::Error::Invalid(
                "local prompting requires a live ResidencyManager-backed session allocation budget".into()
            ))?;
        let budget = manager.statistics().budget_bytes;
        if budget > config.memory_budget_bytes {
            return invalid(
                "session allocation ceiling exceeds the configured local prompt ceiling",
            );
        }
        self.scheduled(|state| {
            if state.local.is_none() {
                state.release_completed(false, true, true)?;
                if state.config.te.is_none() {
                    state.config.te = Some(crate::models::Resolver::new()
                        .offline(config.offline).find(crate::models::TE)?);
                }
                state.te()?;
                let paths = crate::local_prompt::tail::paths(config.model_dir.as_deref(), config.offline)?;
                // Safety: LocalPromptGenerator extends the checkpoint immutability contract.
                let tail = unsafe { crate::local_prompt::tail::open(&paths) }?;
                let prefix = state.te.as_ref().expect("opened").weights();
                let weights = prefix.device_bytes() + tail.device_bytes();
                let kv = (config.context_tokens.div_ceil(16) * 16 + 16) * 64 * 2 * 8 * 128 * 2;
                let scratch = 4usize << 30;
                let required = weights.checked_add(kv).and_then(|n| n.checked_add(scratch))
                    .ok_or_else(|| crate::Error::Invalid("local memory estimate overflow".into()))?;
                // Shared weights already occupy RAM and the allocation budget. Count only
                // additional bytes when checking available headroom on a reused session.
                let additional = required.saturating_sub(prefix.uploaded_bytes());
                let available = crate::local_prompt::available_memory()?;
                let budget_required = manager.statistics().reserved_bytes.saturating_add(additional);
                if budget_required > budget || additional.saturating_add(8usize << 30) > available {
                    return invalid(format!(
                        "resident local Qwen needs approximately {:.2} GiB additional RAM plus 8 GiB system headroom; allocation estimate {:.2} GiB, budget {:.2} GiB, available {:.2} GiB",
                        additional as f64 / (1u64 << 30) as f64,
                        budget_required as f64 / (1u64 << 30) as f64,
                        budget as f64 / (1u64 << 30) as f64,
                        available as f64 / (1u64 << 30) as f64,
                    ));
                }
                state.local = Some(crate::local_prompt::Engine::new(&mut state.stream, &state.compiler, tail)?);
            }
            let te = state.te.as_ref().expect("local encoder");
            let mut evidence = Vec::new();
            if let Some(p) = presentation {
                let mut runs = Vec::<(usize, usize)>::new();
                for (i, id) in p.ids().iter().enumerate() {
                    if *id < 0 {
                        match runs.last_mut() {
                            Some((start, count)) if *start + *count == i => *count += 1,
                            _ => runs.push((i, 1)),
                        }
                    }
                }
                if runs.len() != p.blocks().len() {
                    return invalid("local vision spans do not match blocks");
                }
                for ((start, count), block) in runs.into_iter().zip(p.blocks()) {
                    let (h, w) = (block.first.height, block.first.width);
                    let embedding = te.vision_pair(
                        &mut state.stream, &state.compiler, &mut state.prof,
                        &block.first.pixels, &block.second.pixels, h, w,
                    )?;
                    evidence.push((crate::rope::VisionSpan {
                        start: start + offset, count, merged_h: h / 32, merged_w: w / 32,
                    }, embedding));
                }
            }
            let spans: Vec<_> = evidence.iter().map(|(at, e)| Span {
                at: *at, merged: &e.merged, deepstack: &e.deepstack,
            }).collect();
            state.local.as_mut().expect("loaded").complete(
                &mut state.stream, &state.compiler, &mut state.prof, te, &ids, &spans, config,
            )
        })
    }
    pub(crate) fn local_prompt_record(&mut self) -> Result<Value> {
        self.scheduled(|state| Ok(serde_json::json!({
            "prefix_checkpoint":state.te.as_ref().map(|te|te.weights().file().path().display().to_string()),
            "tail_revision":crate::local_prompt::tail::REVISION,
            "completions":state.local.as_ref().map(|g| &g.stats),
            "precision":"INT8 ConvRot prefix weights; BF16 continuation weights; F32 residuals; F16 projection outputs, attention and KV",
        })))
    }
    pub(crate) fn finish_local_prompt(&mut self, keep_encoder: bool) -> Result<()> {
        self.scheduled(|state| {
            state.stream.synchronize()?;
            state.local = None;
            if !keep_encoder {
                state.te = None;
            }
            Ok(())
        })
    }
}

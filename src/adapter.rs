//! Pinned Turbo checkpoint descriptions and strict CPU validation.
//! This module inspects adapters; execution is not enabled by inspecting a checkpoint.
use crate::checkpoint::Checkpoint;
use crate::error::{invalid, Result};
use hrx::artifacts::safetensors::{DType as Dtype, Entry};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// The two 1344×768 FL2VA/T2VA Turbo adapters selected for integration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TurboPreset {
    Four,
    Eight,
}

impl TurboPreset {
    /// Select the complete trained configuration; it must accompany this session's adapter.
    pub fn configure(self, params: &mut crate::DenoiseParams) {
        params.width = 1344;
        params.height = 768;
        params.frames = 124;
        params.steps = self.evaluations() + 1;
        params.sampler = crate::Sampler::Euler;
        params.video_shift = 6.0;
        params.audio_shift = 3.0;
        params.cache_threshold = 0.0;
    }
    pub fn evaluations(self) -> usize {
        match self {
            Self::Four => 4,
            Self::Eight => 8,
        }
    }

    /// Resolve the exact upstream version, without making a separate model directory.
    pub fn resolve(self, offline: bool) -> Result<PathBuf> {
        let (owner, repo, revision, filename) = match self {
            Self::Four => (
                "Comfy-Org",
                "MiniMax-H3",
                "a98869194787969724c7425d95d0ed73ce9202af",
                "loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
            ),
            Self::Eight => (
                "lightx2v",
                "Minimax-h3-Turbo",
                "3ec17a324ced54151364f24f8b5fb6bf7e26414f",
                "minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors",
            ),
        };
        Ok(crate::models::Resolver::new()
            .repository(owner, repo)
            .revision(Some(revision.into()))
            .offline(offline)
            .find(filename)?)
    }

    /// Trained video/audio sigma grids, including the terminal zero.
    pub fn schedules(self) -> (crate::layout::Schedule, crate::layout::Schedule) {
        (
            crate::layout::Schedule::new(self.evaluations() + 1, 6.0),
            crate::layout::Schedule::new(self.evaluations() + 1, 3.0),
        )
    }
}

/// A projection in the checkpoint's original, unrotated weight basis.
#[derive(Clone, Debug)]
pub struct LinearAdapter {
    pub prefix: String,
    pub rank: usize,
    pub input: usize,
    pub output: usize,
    /// Multiplier applied to B(Ax); alpha / rank, not the raw alpha.
    pub scale: f32,
    /// The base stack interleaves gate/up in 16-row runs; adapters store the halves contiguously.
    pub gate_up: bool,
}

/// A mapped, fully accounted-for adapter checkpoint.
pub struct Adapter {
    file: Checkpoint,
    pub projections: Vec<LinearAdapter>,
}

fn specifications() -> Vec<LinearAdapter> {
    let mut out = Vec::new();
    for (stack, count) in [("blocks", 50), ("token_refiner.blocks", 2)] {
        for layer in 0..count {
            for (op, rank, input, output, gate_up) in [
                ("attn.qkv_proj", 384, 5376, 21504, false),
                ("attn.out_proj", 128, 7168, 5376, false),
                ("mlp.fc1", 128, 5376, 28672, true),
                ("mlp.fc2", 128, 14336, 5376, false),
            ] {
                out.push(LinearAdapter {
                    prefix: format!("diffusion_model.{stack}.{layer}.{op}"),
                    rank,
                    input,
                    output,
                    scale: 0.0,
                    gate_up,
                });
            }
        }
    }
    out
}

fn validate_header(entries: &BTreeMap<String, Entry>) -> Result<Vec<LinearAdapter>> {
    let specs = specifications();
    let mut expected = BTreeSet::new();
    for spec in &specs {
        for (suffix, dtype, shape) in [
            ("lora_A.weight", Dtype::BF16, vec![spec.rank, spec.input]),
            ("lora_B.weight", Dtype::BF16, vec![spec.output, spec.rank]),
            ("alpha", Dtype::F32, vec![]),
        ] {
            let name = format!("{}.{}", spec.prefix, suffix);
            let Some(entry) = entries.get(&name) else {
                return invalid(format!("adapter missing {name}"));
            };
            if entry.dtype != dtype || entry.shape != shape {
                return invalid(format!(
                    "adapter {name}: expected {dtype:?} {shape:?}, got {:?} {:?}",
                    entry.dtype, entry.shape
                ));
            }
            expected.insert(name);
        }
    }
    if let Some(name) = entries.keys().find(|key| !expected.contains(*key)) {
        return invalid(format!(
            "unsupported adapter tensor {name}; refusing a partial adapter"
        ));
    }
    Ok(specs)
}

impl Adapter {
    /// Validate all 624 tensors, including both refiner blocks, before device allocation.
    ///
    /// # Safety
    /// The file must remain immutable and untruncated while this object is alive.
    pub unsafe fn open(path: &Path) -> Result<Self> {
        // Safety: forwarded from this method's contract.
        let file = unsafe { Checkpoint::open(path) }?;
        let mut projections = validate_header(file.entries())?;
        for spec in &mut projections {
            let e = file.at(&format!("{}.alpha", spec.prefix))?;
            let alpha = f32::from_le_bytes(file.bytes(e).try_into().expect("validated scalar"));
            if !alpha.is_finite() || alpha <= 0.0 {
                return invalid(format!(
                    "{}: adapter alpha must be finite and positive",
                    spec.prefix
                ));
            }
            spec.scale = alpha / spec.rank as f32;
        }
        Ok(Self { file, projections })
    }

    pub fn checkpoint(&self) -> &Checkpoint {
        &self.file
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header() -> BTreeMap<String, Entry> {
        let mut entries = BTreeMap::new();
        for s in specifications() {
            for (suffix, dtype, shape) in [
                ("lora_A.weight", Dtype::BF16, vec![s.rank, s.input]),
                ("lora_B.weight", Dtype::BF16, vec![s.output, s.rank]),
                ("alpha", Dtype::F32, vec![]),
            ] {
                entries.insert(
                    format!("{}.{}", s.prefix, suffix),
                    Entry {
                        dtype,
                        shape,
                        offset: 0,
                        bytes: 0,
                    },
                );
            }
        }
        entries
    }

    #[test]
    fn adapter_cannot_silently_omit_refiner_or_ignore_unknown_tensors() {
        let mut entries = header();
        assert_eq!(validate_header(&entries).unwrap().len(), 208);
        let key = "diffusion_model.token_refiner.blocks.1.mlp.fc2.lora_A.weight";
        let entry = entries.remove(key).unwrap();
        assert!(validate_header(&entries)
            .unwrap_err()
            .to_string()
            .contains(key));
        entries.insert(key.into(), entry.clone());
        entries.insert("diffusion_model.unexpected.weight".into(), entry);
        assert!(validate_header(&entries)
            .unwrap_err()
            .to_string()
            .contains("unsupported"));
    }

    #[test]
    fn fused_qkv_rank_and_dtype_are_checked() {
        let mut entries = header();
        let key = "diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight";
        entries.get_mut(key).unwrap().shape[0] = 128;
        assert!(validate_header(&entries).is_err());
        let mut entries = header();
        entries.get_mut(key).unwrap().dtype = Dtype::F16;
        assert!(validate_header(&entries).is_err());
    }

    #[test]
    fn four_step_grids_match_trained_modality_shifts() {
        let (video, audio) = TurboPreset::Four.schedules();
        assert_eq!(
            video.sigmas,
            vec![1.0, 18.0 / 19.0, 6.0 / 7.0, 2.0 / 3.0, 0.0]
        );
        assert_eq!(audio.sigmas, vec![1.0, 0.9, 0.75, 0.5, 0.0]);
        assert_eq!(TurboPreset::Eight.schedules().0.timesteps.len(), 8);
    }
}

//! H3 LoRA checkpoints and pinned Turbo adapter descriptions.
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
    /// Multiplier applied to B(Ax): strength * alpha / rank, with missing alpha == rank.
    pub scale: f32,
    /// The base stack interleaves gate/up in 16-row runs; adapters store the halves contiguously.
    pub gate_up: bool,
    pub(crate) file_index: usize,
}

/// Mapped, fully accounted-for adapter checkpoints and their weighted projections.
pub struct Adapter {
    files: Vec<Checkpoint>,
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
                    file_index: 0,
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
        Ok(Self {
            files: vec![file],
            projections,
        })
    }

    /// First mapped checkpoint, for single-file adapter inspection.
    pub fn checkpoint(&self) -> &Checkpoint {
        &self.files[0]
    }
}

/// A local LoRA checkpoint with a finite signed weight. Zero disables loading.
#[derive(Clone, Debug)]
pub struct Lora {
    pub path: PathBuf,
    pub strength: f32,
}
impl Lora {
    pub fn new(path: impl Into<PathBuf>, strength: f32) -> Self {
        Self {
            path: path.into(),
            strength,
        }
    }
}

fn validate_lora_header(entries: &BTreeMap<String, Entry>) -> Result<Vec<LinearAdapter>> {
    let supported = specifications();
    let mut expected = BTreeSet::new();
    let mut projections = Vec::new();
    for mut spec in supported {
        let a_name = format!("{}.lora_A.weight", spec.prefix);
        let b_name = format!("{}.lora_B.weight", spec.prefix);
        let alpha_name = format!("{}.alpha", spec.prefix);
        let a = entries.get(&a_name);
        let b = entries.get(&b_name);
        if a.is_none() && b.is_none() {
            continue;
        }
        let (Some(a), Some(b)) = (a, b) else {
            return invalid(format!("{}: LoRA needs both A and B tensors", spec.prefix));
        };
        let float = |d| matches!(d, Dtype::F16 | Dtype::BF16 | Dtype::F32);
        if a.shape.len() != 2
            || b.shape.len() != 2
            || !float(a.dtype)
            || !float(b.dtype)
            || a.shape[1] != spec.input
            || b.shape[0] != spec.output
            || a.shape[0] == 0
            || a.shape[0] != b.shape[1]
            || a.shape[0] > spec.input.min(spec.output)
        {
            return invalid(format!("{}: invalid LoRA A/B shape or dtype", spec.prefix));
        }
        spec.rank = a.shape[0];
        spec.scale = 1.0; // Missing alpha means alpha == rank, as in ai-toolkit exports.
        expected.extend([a_name, b_name]);
        if let Some(alpha) = entries.get(&alpha_name) {
            if !float(alpha.dtype) || !(alpha.shape.is_empty() || alpha.shape == [1]) {
                return invalid(format!("{alpha_name}: expected a floating scalar"));
            }
            expected.insert(alpha_name);
        }
        projections.push(spec);
    }
    if let Some(name) = entries.keys().find(|name| !expected.contains(*name)) {
        return invalid(format!(
            "unsupported LoRA tensor {name}; refusing a partial load"
        ));
    }
    if projections.is_empty() {
        return invalid("LoRA contains no supported H3 projections");
    }
    Ok(projections)
}

/// Read a floating tensor element without assuming checkpoint precision.
pub(crate) fn float_at(dtype: Dtype, bytes: &[u8], index: usize) -> Result<f32> {
    let value = match dtype {
        Dtype::BF16 => {
            half::bf16::from_le_bytes(bytes[index * 2..index * 2 + 2].try_into().unwrap()).to_f32()
        }
        Dtype::F16 => {
            half::f16::from_le_bytes(bytes[index * 2..index * 2 + 2].try_into().unwrap()).to_f32()
        }
        Dtype::F32 => f32::from_le_bytes(bytes[index * 4..index * 4 + 4].try_into().unwrap()),
        _ => return invalid("unsupported LoRA precision"),
    };
    if !value.is_finite() {
        return invalid("LoRA contains a non-finite value");
    }
    Ok(value)
}

impl Adapter {
    /// Load and validate weighted H3 LoRAs. Unlisted projections retain the base
    /// path. All supplied tensors must be accounted for; no format guessing.
    ///
    /// # Safety
    /// All active checkpoint files must remain immutable while this object lives.
    pub unsafe fn open_loras(loras: &[Lora]) -> Result<Option<Self>> {
        let mut files = Vec::new();
        let mut projections = Vec::new();
        for lora in loras {
            if !lora.strength.is_finite() {
                return invalid("LoRA strength must be finite");
            }
            if lora.strength == 0.0 {
                continue;
            }
            // Safety: inherited from this function's caller.
            let file = unsafe { Checkpoint::open(&lora.path) }?;
            let mut specs = validate_lora_header(file.entries())?;
            for spec in &mut specs {
                if let Ok(alpha) = file.at(&format!("{}.alpha", spec.prefix)) {
                    spec.scale = float_at(alpha.dtype, file.bytes(alpha), 0)? / spec.rank as f32;
                }
                spec.scale *= lora.strength;
                if !spec.scale.is_finite() {
                    return invalid("LoRA scale overflows");
                }
                spec.file_index = files.len();
                // Fail before any device allocations, including non-finite payloads.
                for suffix in ["lora_A.weight", "lora_B.weight"] {
                    let e = file.at(&format!("{}.{suffix}", spec.prefix))?;
                    let data = file.bytes(e);
                    for i in 0..e
                        .elements()
                        .map_err(|e| crate::Error::Invalid(e.to_string()))?
                    {
                        let value = float_at(e.dtype, data, i)?
                            * if suffix == "lora_B.weight" {
                                spec.scale
                            } else {
                                1.0
                            };
                        if !half::bf16::from_f32(value).is_finite() {
                            return invalid("LoRA weight exceeds BF16 range");
                        }
                    }
                }
            }
            projections.extend(specs.into_iter().filter(|s| s.scale != 0.0));
            files.push(file);
        }
        let mut ranks = BTreeMap::<&str, usize>::new();
        for spec in &projections {
            let rank = ranks.entry(&spec.prefix).or_default();
            *rank = rank
                .checked_add(spec.rank)
                .ok_or_else(|| crate::Error::Invalid("combined LoRA rank overflows".into()))?;
            if *rank > 32768 {
                return invalid("combined LoRA rank exceeds the kernel limit of 32768");
            }
        }
        Ok((!projections.is_empty()).then_some(Self { files, projections }))
    }

    pub(crate) fn source(&self, spec: &LinearAdapter) -> &Checkpoint {
        &self.files[spec.file_index]
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
    fn general_loras_allow_partial_targets_variable_rank_and_optional_alpha() {
        let mut entries = header();
        entries.retain(|name, _| name.starts_with("diffusion_model.blocks.0.attn.qkv_proj."));
        entries.remove("diffusion_model.blocks.0.attn.qkv_proj.alpha");
        entries
            .get_mut("diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight")
            .unwrap()
            .shape[0] = 16;
        entries
            .get_mut("diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight")
            .unwrap()
            .shape[1] = 16;
        let spec = validate_lora_header(&entries).unwrap();
        assert_eq!(spec.len(), 1);
        assert_eq!(spec[0].rank, 16);
        assert_eq!(spec[0].scale, 1.0);
        for dtype in [Dtype::F16, Dtype::F32] {
            for entry in entries.values_mut() {
                entry.dtype = dtype;
            }
            assert!(validate_lora_header(&entries).is_ok());
        }
        entries
            .get_mut("diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight")
            .unwrap()
            .shape[1] = 17;
        assert!(validate_lora_header(&entries).is_err());
        entries.remove("diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight");
        assert!(validate_lora_header(&entries).is_err());
    }

    #[test]
    fn general_loras_reject_unknown_and_empty_files() {
        assert!(validate_lora_header(&BTreeMap::new()).is_err());
        let mut entries = header();
        entries.insert(
            "diffusion_model.blocks.0.adaln_proj.lora_A.weight".into(),
            entries.values().next().unwrap().clone(),
        );
        assert!(validate_lora_header(&entries)
            .unwrap_err()
            .to_string()
            .contains("unsupported"));
    }

    fn fixture(path: &Path, value: f32, alpha: Option<f32>) {
        use safetensors::tensor::{serialize, Dtype as Dt, TensorView};
        let a = vec![value; 5376];
        let b = vec![value; 21504];
        let scalar = alpha.map(|v| v.to_le_bytes());
        let prefix = "diffusion_model.blocks.0.attn.qkv_proj";
        let mut tensors = vec![
            (
                format!("{prefix}.lora_A.weight"),
                TensorView::new(Dt::F32, vec![1, 5376], bytemuck::cast_slice(&a)).unwrap(),
            ),
            (
                format!("{prefix}.lora_B.weight"),
                TensorView::new(Dt::F32, vec![21504, 1], bytemuck::cast_slice(&b)).unwrap(),
            ),
        ];
        if let Some(bytes) = &scalar {
            tensors.push((
                format!("{prefix}.alpha"),
                TensorView::new(Dt::F32, vec![], bytes).unwrap(),
            ));
        }
        std::fs::write(path, serialize(tensors, None).unwrap()).unwrap();
    }

    #[test]
    fn weighted_files_account_for_alpha_disable_zero_and_reject_nonfinite_values() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("lora.safetensors");
        fixture(&path, 0.125, Some(2.0));
        // Safety: each mapping is dropped before the fixture is rewritten.
        let adapter =
            unsafe { Adapter::open_loras(&[Lora::new(&path, 0.5), Lora::new(&path, -1.0)]) }
                .unwrap()
                .unwrap();
        assert_eq!(adapter.projections.len(), 2);
        assert_eq!(adapter.projections[0].scale, 1.0);
        assert_eq!(adapter.projections[1].scale, -2.0);
        assert_eq!(adapter.projections[1].file_index, 1);
        drop(adapter);
        assert!(unsafe { Adapter::open_loras(&[Lora::new("absent", 0.0)]) }
            .unwrap()
            .is_none());
        assert!(unsafe { Adapter::open_loras(&[Lora::new("absent", f32::NAN)]) }.is_err());
        fixture(&path, f32::INFINITY, None);
        assert!(unsafe { Adapter::open_loras(&[Lora::new(&path, 1.0)]) }.is_err());
        fixture(&path, 1.0, Some(f32::NAN));
        assert!(unsafe { Adapter::open_loras(&[Lora::new(&path, 1.0)]) }.is_err());
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

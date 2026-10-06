//! Portable H3 encoded references: standalone v4 and combined v5 safetensors.
//!
//! Visual values are CTHW model-space latents. Audio values are held in native
//! `[2,32,T]` order and transposed to `[1,32,2,T]` only at the file boundary.
//! No legacy sidecars or format fallbacks are supported.
use crate::{error::invalid, Error, LatentGrid, Reference, Result};
use half::{bf16, f16};
use safetensors::{
    tensor::{serialize, Dtype, TensorView},
    SafeTensors,
};
use serde_json::{json, Value};
use std::{collections::HashMap, io::Write, path::Path};

mod presentation;
pub use presentation::{entries_from_sources, RefModPresentationOptions, RefModSource};

fn err(e: impl std::fmt::Display) -> Error {
    Error::Invalid(format!("refmod: {e}"))
}

/// A validated, owned reference. Metadata is retained when re-exported.
#[derive(Clone, Debug)]
pub struct RefModMember {
    metadata: Value,
    shape: Vec<usize>,
    values: Vec<f32>,
    dtype: Dtype,
}

impl RefModMember {
    /// Construct visual model-space latents; one frame becomes an image member.
    pub fn visual(name: &str, values: Vec<f32>, grid: LatentGrid) -> Result<Self> {
        Self::new(
            json!({"_format_version":4,"name":name,
            "kind":if grid.frames == 1 {"image"} else {"video"},
            "latent_t":grid.frames,"latent_h":grid.height,"latent_w":grid.width,
            "mode":"encode","source":"stack","concept_type":"identity",
            "description":"","tags":[],"optimize_steps":0,"sample_rate":32000}),
            vec![1, 24, grid.frames, grid.height, grid.width],
            values
                .into_iter()
                .map(|x| f16::from_f32(x).to_f32())
                .collect(),
            Dtype::F16,
        )
    }

    /// Construct normalized native audio latents in `[2,32,T]` order.
    pub fn audio(name: &str, values: Vec<f32>, frames: usize) -> Result<Self> {
        Self::new(
            json!({"_format_version":4,"name":name,"kind":"audio",
            "latent_t":frames,"latent_h":0,"latent_w":0,"mode":"encode",
            "source":"audio","concept_type":"voice","description":"",
            "tags":[],"sample_rate":32000,"optimize_steps":0}),
            vec![1, 32, 2, frames],
            values,
            Dtype::F32,
        )
    }

    fn new(metadata: Value, shape: Vec<usize>, values: Vec<f32>, dtype: Dtype) -> Result<Self> {
        validate_shape(&metadata, &shape)?;
        let count = shape
            .iter()
            .try_fold(1usize, |a, b| a.checked_mul(*b))
            .ok_or_else(|| err("tensor size overflows"))?;
        if values.len() != count || values.iter().any(|x| !x.is_finite()) {
            return invalid("refmod: tensor length mismatch or non-finite latent");
        }
        Ok(Self {
            metadata,
            shape,
            values,
            dtype,
        })
    }
    pub fn metadata(&self) -> &Value {
        &self.metadata
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }
    pub fn values(&self) -> &[f32] {
        &self.values
    }
    pub fn is_audio(&self) -> bool {
        self.metadata["kind"] == "audio"
    }
    pub fn token_count(&self) -> usize {
        if self.is_audio() {
            self.shape[3] * 2
        } else {
            self.shape[2] * (self.shape[3] / 2) * (self.shape[4] / 2)
        }
    }
    pub fn reference(&self) -> Reference<'_> {
        if self.is_audio() {
            Reference::Audio {
                latents: &self.values,
                frames: self.shape[3],
            }
        } else {
            let grid = LatentGrid {
                frames: self.shape[2],
                height: self.shape[3],
                width: self.shape[4],
            };
            if self.metadata["kind"] == "image" {
                Reference::Image {
                    latents: &self.values,
                    grid,
                    presented: None,
                }
            } else {
                Reference::Video {
                    latents: &self.values,
                    grid,
                    audio: None,
                }
            }
        }
    }
}

fn validate_shape(meta: &Value, shape: &[usize]) -> Result<()> {
    if meta["_format_version"].as_u64() != Some(4) {
        return invalid("refmod: only standalone/member format version 4 is supported");
    }
    let valid = match meta["kind"].as_str() {
        Some("audio") => {
            matches!(shape, [1,32,2,t] if *t > 0)
                && meta
                    .get("sample_rate")
                    .is_none_or(|v| v.as_u64() == Some(32000))
        }
        Some("image" | "video") => matches!(shape, [1,24,t,h,w]
            if *t > 0 && *h > 0 && *w > 0 && h % 2 == 0 && w % 2 == 0
                && (meta["kind"] != "image" || *t == 1)),
        _ => false,
    };
    if !valid || shape.iter().any(|&n| n > i32::MAX as usize) {
        return invalid(format!(
            "refmod: invalid kind, sample rate, or shape {shape:?}"
        ));
    }
    let (t, h, w) = if meta["kind"] == "audio" {
        (shape[3], 0, 0)
    } else {
        (shape[2], shape[3], shape[4])
    };
    for (key, expected) in [("latent_t", t), ("latent_h", h), ("latent_w", w)] {
        if let Some(v) = meta.get(key) {
            if v.as_u64() != Some(expected as u64) {
                return invalid(format!("refmod: {key} disagrees with tensor shape"));
            }
        }
    }
    Ok(())
}

/// A standalone reference or an ordered bundle. Loads own their storage (no mmap).
#[derive(Clone, Debug)]
pub struct RefMod {
    metadata: Value,
    members: Vec<RefModMember>,
    header: HashMap<String, String>,
}
impl RefMod {
    /// One member exports as v4; two or more export as a v5 bundle.
    pub fn new(name: &str, members: Vec<RefModMember>) -> Result<Self> {
        if members.is_empty() || members.len() > 256 {
            return invalid("refmod: expected 1–256 members");
        }
        let metadata = if members.len() == 1 {
            members[0].metadata.clone()
        } else {
            json!({"_format_version":5,"kind":"bundle","name":name})
        };
        Ok(Self {
            metadata,
            members,
            header: HashMap::new(),
        })
    }
    pub fn members(&self) -> &[RefModMember] {
        &self.members
    }
    pub fn metadata(&self) -> &Value {
        &self.metadata
    }
    pub fn token_count(&self) -> Result<usize> {
        self.members.iter().try_fold(0usize, |a, m| {
            a.checked_add(m.token_count())
                .ok_or_else(|| err("token count overflows"))
        })
    }
    /// Load current embedded-metadata files. Errors include the source path.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        Self::read(path).map_err(|e| err(format!("{}: {e}", path.display())))
    }
    fn read(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(err)?;
        let (_, info) = SafeTensors::read_metadata(&bytes).map_err(err)?;
        let header = info.metadata().clone().unwrap_or_default();
        let meta: Value =
            serde_json::from_str(header.get("refmod_meta").ok_or_else(|| {
                err("missing embedded refmod_meta; legacy files are unsupported")
            })?)
            .map_err(err)?;
        let bundle = meta["kind"] == "bundle";
        let metas = if bundle {
            if meta["_format_version"].as_u64() != Some(5) {
                return invalid("refmod: unsupported bundle version");
            }
            meta["members"]
                .as_array()
                .ok_or_else(|| err("missing bundle members"))?
                .clone()
        } else {
            vec![meta.clone()]
        };
        if metas.is_empty() || metas.len() > 256 {
            return invalid("refmod: expected 1–256 members");
        }
        let tensors = SafeTensors::deserialize(&bytes).map_err(err)?;
        let mut members = Vec::new();
        for (i, m) in metas.into_iter().enumerate() {
            let key = if bundle {
                format!("ref_{i}")
            } else {
                "latent".into()
            };
            let tensor = tensors.tensor(&key).map_err(err)?;
            validate_shape(&m, tensor.shape())?;
            let mut values = decode(tensor.dtype(), tensor.data())?;
            if m["kind"] == "audio" {
                values = audio_order(&values, tensor.shape()[3], false);
            }
            members.push(RefModMember::new(
                m,
                tensor.shape().to_vec(),
                values,
                tensor.dtype(),
            )?);
        }
        Ok(Self {
            metadata: meta,
            members,
            header,
        })
    }
    /// Atomically export. Existing destinations require `overwrite`.
    pub fn save(&self, path: impl AsRef<Path>, overwrite: bool) -> Result<()> {
        let path = path.as_ref();
        let bundle = self.metadata["kind"] == "bundle";
        let mut meta = if bundle {
            self.metadata.clone()
        } else {
            self.members[0].metadata.clone()
        };
        if bundle {
            meta["members"] = self.members.iter().map(|m| m.metadata.clone()).collect();
        }
        let mut header = self.header.clone();
        header.insert("refmod_meta".into(), meta.to_string());
        let mut raw = Vec::new();
        for m in &self.members {
            let values = if m.is_audio() {
                audio_order(&m.values, m.shape[3], true)
            } else {
                m.values.clone()
            };
            let dtype = if m.is_audio() { Dtype::F32 } else { Dtype::F16 };
            let mut data = Vec::new();
            for x in values {
                if dtype == Dtype::F32 {
                    data.extend_from_slice(&x.to_le_bytes());
                } else {
                    let y = f16::from_f32(x);
                    if !y.is_finite() {
                        return invalid("refmod: latent exceeds F16 export range");
                    }
                    data.extend_from_slice(&y.to_le_bytes());
                }
            }
            raw.push((dtype, data));
        }
        let views = raw
            .iter()
            .zip(&self.members)
            .enumerate()
            .map(|(i, ((dtype, data), m))| {
                let name = if bundle {
                    format!("ref_{i}")
                } else {
                    "latent".into()
                };
                Ok((
                    name,
                    TensorView::new(*dtype, m.shape.clone(), data).map_err(err)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let bytes = serialize(views, Some(header)).map_err(err)?;
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let mut file = tempfile::NamedTempFile::new_in(parent).map_err(err)?;
        file.write_all(&bytes).map_err(err)?;
        file.as_file().sync_all().map_err(err)?;
        if overwrite {
            file.persist(path).map_err(err)?;
        } else {
            file.persist_noclobber(path).map_err(err)?;
        }
        Ok(())
    }
    /// Prepare constant-strength reference buffers, preserving member/copy order.
    pub fn prepare(&self, options: ApplyOptions) -> Result<PreparedRefMod> {
        options.check()?;
        let mut members = Vec::new();
        let mut source_indices = Vec::new();
        let mut indices = Vec::new();
        let mut tokens = 0usize;
        for (source_index, m) in self.members.iter().enumerate() {
            let strength = if m.is_audio() {
                options.audio_strength
            } else {
                options.visual_strength
            };
            if strength == 0.0 {
                continue;
            }
            tokens = m
                .token_count()
                .checked_mul(options.copies)
                .and_then(|n| tokens.checked_add(n))
                .ok_or_else(|| err("token count overflows"))?;
            if options.max_tokens.is_some_and(|limit| tokens > limit) {
                return invalid("refmod: effective token budget exceeded");
            }
            let mut member = m.clone();
            if strength < 1.0 {
                let blurred = blur(m);
                for (x, b) in member.values.iter_mut().zip(blurred) {
                    // Upstream PyTorch materializes each operation in the latent dtype.
                    *x = rounded(
                        rounded(strength * *x, m.dtype)
                            + rounded((1.0 - strength) * rounded(b, m.dtype), m.dtype),
                        m.dtype,
                    );
                }
            }
            if member.values.iter().any(|x| !x.is_finite()) {
                return invalid("refmod: strength transformation produced a non-finite latent");
            }
            indices.try_reserve(options.copies).map_err(err)?;
            indices.extend(std::iter::repeat_n(members.len(), options.copies));
            members.push(member);
            source_indices.push(source_index + 1);
        }
        Ok(PreparedRefMod {
            members,
            source_indices,
            indices,
            tokens,
        })
    }
}
fn rounded(value: f32, dtype: Dtype) -> f32 {
    match dtype {
        Dtype::F16 => f16::from_f32(value).to_f32(),
        Dtype::BF16 => bf16::from_f32(value).to_f32(),
        _ => value,
    }
}

/// Runtime controls; saved config is informational and is never applied implicitly.
#[derive(Clone, Copy, Debug)]
pub struct ApplyOptions {
    pub visual_strength: f32,
    pub audio_strength: f32,
    pub copies: usize,
    pub max_tokens: Option<usize>,
}
impl Default for ApplyOptions {
    fn default() -> Self {
        Self {
            visual_strength: 1.0,
            audio_strength: 1.0,
            copies: 1,
            max_tokens: None,
        }
    }
}
impl ApplyOptions {
    fn check(self) -> Result<()> {
        if self.copies == 0
            || [self.visual_strength, self.audio_strength]
                .iter()
                .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        {
            return invalid("refmod: strengths must be finite in [0,1] and copies positive");
        }
        Ok(())
    }
}
/// Owns each transformed latent once; copies borrow the same storage.
pub struct PreparedRefMod {
    members: Vec<RefModMember>,
    source_indices: Vec<usize>,
    indices: Vec<usize>,
    tokens: usize,
}
impl PreparedRefMod {
    /// Active members in conditioning order, including copies. Values already include strength.
    pub fn members(&self) -> impl Iterator<Item = &RefModMember> {
        self.indices.iter().map(|&i| &self.members[i])
    }

    /// One-based original file member numbers and active members, including copies.
    /// Numbers remain stable when other members are disabled.
    pub fn indexed_members(&self) -> impl Iterator<Item = (usize, &RefModMember)> {
        self.indices
            .iter()
            .map(|&i| (self.source_indices[i], &self.members[i]))
    }

    pub fn references(&self) -> Vec<Reference<'_>> {
        self.indices
            .iter()
            .map(|&i| self.members[i].reference())
            .collect()
    }
    pub fn token_count(&self) -> usize {
        self.tokens
    }
}
fn decode(dtype: Dtype, bytes: &[u8]) -> Result<Vec<f32>> {
    match dtype {
        Dtype::F32 => Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|x| f32::from_le_bytes(*x))
            .collect()),
        Dtype::F16 => Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|x| f16::from_le_bytes(*x).to_f32())
            .collect()),
        Dtype::BF16 => Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|x| bf16::from_le_bytes(*x).to_f32())
            .collect()),
        _ => invalid(format!("refmod: unsupported dtype {dtype:?}")),
    }
}
fn audio_order(values: &[f32], t: usize, to_file: bool) -> Vec<f32> {
    let mut out = vec![0.0; values.len()];
    for c in 0..32 {
        for s in 0..2 {
            for f in 0..t {
                let native = (s * 32 + c) * t + f;
                let file = (c * 2 + s) * t + f;
                if to_file {
                    out[file] = values[native];
                } else {
                    out[native] = values[file];
                }
            }
        }
    }
    out
}

// PyTorch adaptive average pooling, followed by align_corners=false interpolation.
fn pooled_plane(src: &[f32], h: usize, w: usize, ph: usize, pw: usize) -> Vec<f32> {
    let mut out = vec![0.0; ph * pw];
    for y in 0..ph {
        for x in 0..pw {
            let (y0, y1, x0, x1) = (
                y * h / ph,
                ((y + 1) * h).div_ceil(ph),
                x * w / pw,
                ((x + 1) * w).div_ceil(pw),
            );
            let mut sum = 0.0;
            for yy in y0..y1 {
                for xx in x0..x1 {
                    sum += src[yy * w + xx];
                }
            }
            out[y * pw + x] = sum / ((y1 - y0) * (x1 - x0)) as f32;
        }
    }
    out
}
fn coord(i: usize, from: usize, to: usize) -> (usize, usize, f32) {
    let p = ((i as f32 + 0.5) * from as f32 / to as f32 - 0.5).max(0.0);
    let a = (p.floor() as usize).min(from - 1);
    (a, (a + 1).min(from - 1), p - a as f32)
}
fn blur(m: &RefModMember) -> Vec<f32> {
    let (planes, h, w) = if m.is_audio() {
        (64, 1, m.shape[3])
    } else {
        (24 * m.shape[2], m.shape[3], m.shape[4])
    };
    let (ph, pw) = ((h / 8).max(1), (w / 8).max(1));
    let mut out = vec![0.0; m.values.len()];
    for p in 0..planes {
        let down = pooled_plane(&m.values[p * h * w..(p + 1) * h * w], h, w, ph, pw);
        for y in 0..h {
            for x in 0..w {
                let (y0, y1, fy) = coord(y, ph, h);
                let (x0, x1, fx) = coord(x, pw, w);
                out[p * h * w + y * w + x] = (1.0 - fy)
                    * ((1.0 - fx) * down[y0 * pw + x0] + fx * down[y0 * pw + x1])
                    + fy * ((1.0 - fx) * down[y1 * pw + x0] + fx * down[y1 * pw + x1]);
            }
        }
    }
    out
}

mod creation;
pub use creation::{AudioInput, CreateOptions, ImageInput};

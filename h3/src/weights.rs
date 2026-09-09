//! Turning a checkpoint's tensors into the exact byte layout each kernel reads.
//!
//! A recipe is layout only, bit for bit: rows concatenated, rows interleaved, rows permuted, zero
//! padding, and lossless widening of small float vectors to the f32 the CPU-side maths takes. Nothing
//! here rotates or quantises — the checkpoint's dtypes are what the kernels run.
//!
//! Recipes are declarative and resolved against the checkpoint when they are assembled, so a recipe
//! table can be built and validated at open without reading a byte of tensor data.
use crate::checkpoint::{Checkpoint, Entry};
use half::{bf16, f16};
use safetensors::tensor::Dtype;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Checkpoint(#[from] crate::checkpoint::Error),
    #[error("no recipe for tensor {0}")]
    NoRecipe(String),
    #[error("{0}")]
    Layout(String),
    #[error("{0}")]
    Device(String),
}

pub type Result<T> = std::result::Result<T, Error>;

fn layout<T>(message: impl Into<String>) -> Result<T> {
    Err(Error::Layout(message.into()))
}

/// A run of whole rows taken from one tensor, in destination order.
#[derive(Clone, Debug)]
pub struct Segment {
    pub tensor: String,
    pub row0: usize,
    pub rows: usize,
}

/// How a tensor is built for the device: either rows gathered from the mapping at a pitch, or a host
/// array computed from the checkpoint (widening, zero extension, or a permutation of elements).
/// Computes a host array from the checkpoint: widening, zero extension, or a permutation of elements.
pub type Build = Box<dyn Fn(&Checkpoint) -> Result<Vec<u8>> + Send + Sync>;

pub enum Recipe {
    Rows {
        rows: usize,
        row_bytes: usize,
        pitch_bytes: usize,
        segments: Vec<Segment>,
    },
    Built {
        bytes: usize,
        build: Build,
    },
}

impl Recipe {
    /// What the tensor occupies once it is on the device.
    pub fn device_bytes(&self) -> usize {
        match self {
            Recipe::Rows {
                rows, pitch_bytes, ..
            } => rows * pitch_bytes,
            Recipe::Built { bytes, .. } => *bytes,
        }
    }

    /// The recipe's bytes on the host: the built array, or the segments gathered at the pitch with the
    /// pad between rows left zero.
    pub fn assemble(&self, ck: &Checkpoint) -> Result<Vec<u8>> {
        match self {
            Recipe::Built { bytes, build } => {
                let out = build(ck)?;
                if out.len() != *bytes {
                    return layout(format!(
                        "a built tensor is {} bytes, its recipe says {bytes}",
                        out.len()
                    ));
                }
                Ok(out)
            }
            Recipe::Rows {
                rows,
                row_bytes,
                pitch_bytes,
                segments,
            } => {
                let mut out = vec![0u8; rows * pitch_bytes];
                let mut row = 0usize;
                for segment in segments {
                    let entry = ck.at(&segment.tensor)?;
                    let bytes = ck.bytes(entry);
                    for i in 0..segment.rows {
                        let src = (segment.row0 + i) * row_bytes;
                        if src + row_bytes > bytes.len() {
                            return layout(format!("a row run past {}", segment.tensor));
                        }
                        let dst = row * pitch_bytes;
                        out[dst..dst + row_bytes].copy_from_slice(&bytes[src..src + row_bytes]);
                        row += 1;
                    }
                }
                if row != *rows {
                    return layout(format!(
                        "a recipe's segments add up to {row} rows, not {rows}"
                    ));
                }
                Ok(out)
            }
        }
    }
}

/// One checkpoint, the recipe table a plan built over it, and whatever has reached the device.
pub struct Weights {
    file: Checkpoint,
    recipes: BTreeMap<String, Recipe>,
    /// Uploaded on first use and kept for the session, behind a lock so the session can be moved
    /// between threads.
    uploaded: Mutex<BTreeMap<String, Arc<hrx::Buffer>>>,
}

/// Bytes staged per transfer. Large enough that the per-call overhead disappears, small enough that a
/// 27 GB tensor never needs a host copy of itself.
const CHUNK: usize = 16 << 20;

impl Weights {
    /// Maps the file and builds its recipe table. The plan validates every source tensor's dtype and
    /// shape, so a checkpoint that does not match fails here rather than mid-generation.
    /// # Safety
    ///
    /// The checkpoint is mapped, not copied. See [`crate::Session::new`] for what that requires of
    /// the file for as long as the returned `Weights` lives.
    pub unsafe fn open(
        path: impl AsRef<std::path::Path>,
        plan: impl FnOnce(&Checkpoint, &mut BTreeMap<String, Recipe>) -> Result<()>,
    ) -> Result<Self> {
        // Safety: the caller's, and stated on Session and Checkpoint::open — the checkpoint must not
        // be modified while this Weights lives.
        let file = unsafe { Checkpoint::open(path) }?;
        let mut recipes = BTreeMap::new();
        plan(&file, &mut recipes)?;
        Ok(Self {
            file,
            recipes,
            uploaded: Mutex::new(BTreeMap::new()),
        })
    }

    pub fn file(&self) -> &Checkpoint {
        &self.file
    }
    pub fn has(&self, name: &str) -> bool {
        self.recipes.contains_key(name)
    }
    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.recipes.keys()
    }

    pub fn recipe(&self, name: &str) -> Result<&Recipe> {
        self.recipes
            .get(name)
            .ok_or_else(|| Error::NoRecipe(name.to_string()))
    }

    pub fn assemble(&self, name: &str) -> Result<Vec<u8>> {
        self.recipe(name)?.assemble(&self.file)
    }

    /// The tensor on the device, uploaded on first use and memoised.
    ///
    /// `expect_bytes` is what the caller has sized its kernel arguments for; a recipe that does not
    /// match is a wiring error and fails here rather than reading past an allocation later.
    ///
    /// Rows that are one straight run of the mapping at their own pitch go up chunk by chunk with no
    /// model-owned staging copy. HRX still copies each chunk into its upload staging. Anything gathered at a wider pitch is staged a chunk at a time, so the pad
    /// between rows stays zero. Either way the file's pages are released once their bytes are on the
    /// device: a tensor is read once, and tens of gigabytes of resident checkpoint would compete with
    /// the device allocations for the same memory on this part.
    pub fn at(
        &self,
        stream: &mut hrx::Stream,
        name: &str,
        expect_bytes: usize,
    ) -> Result<Arc<hrx::Buffer>> {
        // The size is checked on both paths. Checking only the miss would mean the guard against a
        // wrongly-sized kernel operand held for the first caller and vanished for every one after,
        // which is the worst shape for a check to have.
        let recipe = self.recipe(name)?;
        if recipe.device_bytes() != expect_bytes {
            return layout(format!(
                "tensor {name} is {} bytes on the device, expected {expect_bytes}",
                recipe.device_bytes()
            ));
        }
        if let Some(buffer) = self.uploaded.lock().expect("not poisoned").get(name) {
            return Ok(buffer.clone());
        }
        let buffer = Arc::new(
            stream
                .allocate(expect_bytes.max(1))
                .map_err(|e| Error::Device(e.to_string()))?,
        );
        self.fill(stream, recipe, &buffer)?;
        self.uploaded
            .lock()
            .expect("not poisoned")
            .insert(name.to_string(), buffer.clone());
        Ok(buffer)
    }

    /// As [`Weights::at`], additionally checking the shape the kernel will read it with.
    pub fn rows(
        &self,
        stream: &mut hrx::Stream,
        name: &str,
        rows: usize,
        row_bytes: usize,
        pitch_bytes: usize,
    ) -> Result<Arc<hrx::Buffer>> {
        match self.recipe(name)? {
            Recipe::Rows {
                rows: r,
                row_bytes: rb,
                pitch_bytes: pb,
                ..
            } if *r == rows && *rb == row_bytes && *pb == pitch_bytes => {}
            _ => {
                return layout(format!(
                    "tensor {name} is not {rows} rows of {row_bytes} bytes at pitch {pitch_bytes}"
                ))
            }
        }
        self.at(stream, name, rows * pitch_bytes)
    }

    fn fill(&self, stream: &mut hrx::Stream, recipe: &Recipe, buffer: &hrx::Buffer) -> Result<()> {
        let device = |e: hrx::Error| Error::Device(e.to_string());
        match recipe {
            Recipe::Built { .. } => {
                let bytes = recipe.assemble(&self.file)?;
                stream.upload(buffer.binding(), &bytes).map_err(device)?;
            }
            Recipe::Rows {
                rows,
                row_bytes,
                pitch_bytes,
                segments,
            } => {
                if segments.len() == 1 && pitch_bytes == row_bytes {
                    // one straight run: borrow the mapping directly for HRX to stage
                    let segment = &segments[0];
                    let entry = self.file.at(&segment.tensor)?;
                    let all = self.file.bytes(entry);
                    let from = segment.row0 * row_bytes;
                    let span = &all[from..from + segment.rows * row_bytes];
                    self.file.will_need(span);
                    for (i, chunk) in span.chunks(CHUNK).enumerate() {
                        crate::dispatch::upload_at(stream, buffer, i * CHUNK, chunk)
                            .map_err(|e| Error::Device(e.to_string()))?;
                    }
                } else {
                    // Gathered at the pitch, through a staging buffer zeroed once. Every row's first
                    // `row_bytes` are overwritten before that row is sent, and the pad past them is
                    // never written at all, so it stays zero for the life of the buffer — refilling
                    // it between chunks would rewrite every staged byte a second time.
                    let per = (CHUNK / (*pitch_bytes).max(1)).max(1);
                    let mut stage = vec![0u8; per * pitch_bytes];
                    let (mut staged, mut written) = (0usize, 0usize);
                    for segment in segments {
                        let entry = self.file.at(&segment.tensor)?;
                        let all = self.file.bytes(entry);
                        let from = segment.row0 * row_bytes;
                        self.file
                            .will_need(&all[from..from + segment.rows * row_bytes]);
                        for i in 0..segment.rows {
                            let src = from + i * row_bytes;
                            let dst = staged * pitch_bytes;
                            stage[dst..dst + row_bytes].copy_from_slice(&all[src..src + row_bytes]);
                            staged += 1;
                            if staged == per {
                                crate::dispatch::upload_at(
                                    stream,
                                    buffer,
                                    written * pitch_bytes,
                                    &stage[..staged * pitch_bytes],
                                )
                                .map_err(|e| Error::Device(e.to_string()))?;
                                written += staged;
                                staged = 0;
                            }
                        }
                    }
                    if staged > 0 {
                        crate::dispatch::upload_at(
                            stream,
                            buffer,
                            written * pitch_bytes,
                            &stage[..staged * pitch_bytes],
                        )
                        .map_err(|e| Error::Device(e.to_string()))?;
                        written += staged;
                    }
                    if written != *rows {
                        return layout(format!(
                            "a recipe's segments add up to {written} rows, not {rows}"
                        ));
                    }
                }
                // HRX owns a staging copy; the file's pages are not needed by queued work
                for segment in segments {
                    let entry = self.file.at(&segment.tensor)?;
                    let all = self.file.bytes(entry);
                    let from = segment.row0 * row_bytes;
                    self.file
                        .done_with(&all[from..from + segment.rows * row_bytes]);
                }
            }
        }
        Ok(())
    }

    /// A recipe's bytes as f32, for the tables the host itself reads.
    pub fn host_f32(&self, name: &str, count: usize) -> Result<Vec<f32>> {
        let bytes = self.assemble(name)?;
        if bytes.len() != count * 4 {
            return layout(format!(
                "tensor {name} is {} bytes, expected {}",
                bytes.len(),
                count * 4
            ));
        }
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }
}

// --- widening -------------------------------------------------------------------------------

/// Bytes per element for the dtypes a recipe can lay out.
pub fn elem_bytes(dtype: Dtype) -> usize {
    match dtype {
        Dtype::F32 => 4,
        Dtype::F16 | Dtype::BF16 => 2,
        Dtype::I8 | Dtype::U8 => 1,
        _ => 0,
    }
}

/// One float element widened to f32. F32 passes through; F16 and BF16 widen exactly.
fn widen_one(dtype: Dtype, bytes: &[u8]) -> Result<f32> {
    Ok(match dtype {
        Dtype::F32 => f32::from_le_bytes(bytes[..4].try_into().unwrap()),
        Dtype::F16 => f16::from_le_bytes(bytes[..2].try_into().unwrap()).to_f32(),
        Dtype::BF16 => bf16::from_le_bytes(bytes[..2].try_into().unwrap()).to_f32(),
        other => return layout(format!("cannot widen {other:?} to f32")),
    })
}

fn widenable(entry: &Entry, what: &str) -> Result<usize> {
    let eb = elem_bytes(entry.dtype);
    if eb == 0 || entry.dtype == Dtype::I8 || entry.dtype == Dtype::U8 {
        return layout(format!("{what} on {:?}", entry.dtype));
    }
    Ok(eb)
}

// --- recipe primitives: layout only, bit for bit ----------------------------------------------

/// The rows of one or more tensors of the same row width, in order, at `pitch_bytes` (0: the row width).
pub fn rows_of(ck: &Checkpoint, parts: &[&str], pitch_bytes: usize) -> Result<Recipe> {
    let (mut rows, mut row_bytes) = (0usize, 0usize);
    let mut segments = Vec::new();
    for (i, name) in parts.iter().enumerate() {
        let entry = ck.at(name)?;
        if i == 0 {
            row_bytes = entry.row_bytes();
        } else if entry.row_bytes() != row_bytes {
            return layout("concatenated tensors differ in row width");
        }
        segments.push(Segment {
            tensor: (*name).to_string(),
            row0: 0,
            rows: entry.rows(),
        });
        rows += entry.rows();
    }
    let pitch_bytes = if pitch_bytes != 0 {
        pitch_bytes
    } else {
        row_bytes
    };
    if pitch_bytes < row_bytes {
        return layout("a pitch narrower than the rows");
    }
    Ok(Recipe::Rows {
        rows,
        row_bytes,
        pitch_bytes,
        segments,
    })
}

/// 16-row runs alternating two halves: the gate|up operand of the fused SwiGLU GEMM. Each half is
/// `rows_each` rows starting at its tensor's given first row.
pub fn interleave16(
    ck: &Checkpoint,
    first: (&str, usize),
    second: (&str, usize),
    rows_each: usize,
    pitch_bytes: usize,
) -> Result<Recipe> {
    let a = ck.at(first.0)?;
    let b = ck.at(second.0)?;
    if a.row_bytes() != b.row_bytes()
        || !rows_each.is_multiple_of(16)
        || first.1 + rows_each > a.rows()
        || second.1 + rows_each > b.rows()
    {
        return layout("interleave: the halves do not fit");
    }
    let row_bytes = a.row_bytes();
    let pitch_bytes = if pitch_bytes != 0 {
        pitch_bytes
    } else {
        row_bytes
    };
    let mut segments = Vec::new();
    for i in (0..rows_each).step_by(16) {
        segments.push(Segment {
            tensor: first.0.to_string(),
            row0: first.1 + i,
            rows: 16,
        });
        segments.push(Segment {
            tensor: second.0.to_string(),
            row0: second.1 + i,
            rows: 16,
        });
    }
    Ok(Recipe::Rows {
        rows: 2 * rows_each,
        row_bytes,
        pitch_bytes,
        segments,
    })
}

/// A tensor's rows followed by `pad_rows` zero rows (a padded N: the vision MLP's 4304 -> 4352).
pub fn rows_padded(
    ck: &Checkpoint,
    name: &str,
    pad_rows: usize,
    pitch_bytes: usize,
) -> Result<Recipe> {
    let entry = ck.at(name)?;
    let row_bytes = entry.row_bytes();
    let pitch = if pitch_bytes != 0 {
        pitch_bytes
    } else {
        row_bytes
    };
    if pitch < row_bytes {
        return layout("a pitch narrower than the rows");
    }
    let rows = entry.rows();
    let (name, total) = (name.to_string(), rows + pad_rows);
    Ok(Recipe::Built {
        bytes: total * pitch,
        build: Box::new(move |ck| {
            let entry = ck.at(&name)?;
            let src = ck.bytes(entry);
            let mut out = vec![0u8; total * pitch];
            for i in 0..rows {
                out[i * pitch..i * pitch + row_bytes]
                    .copy_from_slice(&src[i * row_bytes..(i + 1) * row_bytes]);
            }
            Ok(out)
        }),
    })
}

/// `(first, count)` of rows or elements taken from a tensor, in destination order.
pub type Run = (usize, usize);

/// A tensor's rows in an explicit order (the VAE attention's head-interleaved rows as `[Q|K|V]`).
pub fn rows_permuted(
    ck: &Checkpoint,
    name: &str,
    runs: &[Run],
    pitch_bytes: usize,
) -> Result<Recipe> {
    let entry = ck.at(name)?;
    let row_bytes = entry.row_bytes();
    let pitch_bytes = if pitch_bytes != 0 {
        pitch_bytes
    } else {
        row_bytes
    };
    let mut segments = Vec::new();
    let mut rows = 0;
    for &(first, count) in runs {
        if first + count > entry.rows() {
            return layout(format!("a row run past {name}"));
        }
        segments.push(Segment {
            tensor: name.to_string(),
            row0: first,
            rows: count,
        });
        rows += count;
    }
    Ok(Recipe::Rows {
        rows,
        row_bytes,
        pitch_bytes,
        segments,
    })
}

/// A float vector's elements in an explicit order, as f32 (a bias permuted like its rows).
pub fn widen_runs(ck: &Checkpoint, name: &str, runs: &[Run]) -> Result<Recipe> {
    let entry = ck.at(name)?;
    widenable(entry, "widen_runs")?;
    let mut n = 0;
    for &(first, count) in runs {
        if first + count > entry.elements() {
            return layout(format!("an element run past {name}"));
        }
        n += count;
    }
    let (name, runs) = (name.to_string(), runs.to_vec());
    Ok(Recipe::Built {
        bytes: n * 4,
        build: Box::new(move |ck| {
            let entry = ck.at(&name)?;
            let eb = elem_bytes(entry.dtype);
            let src = ck.bytes(entry);
            let mut out = Vec::with_capacity(n * 4);
            for &(first, count) in &runs {
                for i in 0..count {
                    let at = (first + i) * eb;
                    out.extend_from_slice(&widen_one(entry.dtype, &src[at..])?.to_le_bytes());
                }
            }
            Ok(out)
        }),
    })
}

/// Runs of `size` alternating two halves of one tensor, `first` first.
pub fn interleaved_runs(first: usize, second: usize, count: usize, size: usize) -> Vec<Run> {
    let mut runs = Vec::new();
    for i in (0..count).step_by(size) {
        runs.push((first + i, size));
        runs.push((second + i, size));
    }
    runs
}

/// Each row's `groups` groups of `in_elems` zero-extended to `out_elems` (the attention output
/// projection's head dim 72 -> 128, or a plain K pad with one group). `out_rows` zero-pads the rows.
pub fn regroup(
    ck: &Checkpoint,
    name: &str,
    groups: usize,
    in_elems: usize,
    out_elems: usize,
    out_rows: usize,
) -> Result<Recipe> {
    let entry = ck.at(name)?;
    let eb = elem_bytes(entry.dtype);
    if eb == 0 {
        return layout(format!("regroup on {:?}", entry.dtype));
    }
    if entry.row_bytes() != groups * in_elems * eb || out_elems < in_elems {
        return layout("regroup: the row does not hold the groups");
    }
    let rows = entry.rows();
    if out_rows != 0 && out_rows < rows {
        return layout("regroup: fewer rows than the tensor");
    }
    let all = if out_rows != 0 { out_rows } else { rows };
    let (in_bytes, out_bytes) = (in_elems * eb, out_elems * eb);
    let (row_in, row_out) = (groups * in_bytes, groups * out_bytes);
    let name = name.to_string();
    Ok(Recipe::Built {
        bytes: all * row_out,
        build: Box::new(move |ck| {
            let src = ck.bytes(ck.at(&name)?);
            let mut out = vec![0u8; all * row_out];
            for i in 0..rows {
                for g in 0..groups {
                    let (d, s) = (i * row_out + g * out_bytes, i * row_in + g * in_bytes);
                    out[d..d + in_bytes].copy_from_slice(&src[s..s + in_bytes]);
                }
            }
            Ok(out)
        }),
    })
}

/// Float tensors as f32, element for element (F32 verbatim; F16 and BF16 widened, which is exact).
pub fn widen_f32(ck: &Checkpoint, parts: &[&str]) -> Result<Recipe> {
    let mut n = 0;
    for name in parts {
        let entry = ck.at(name)?;
        widenable(entry, "widen_f32")?;
        n += entry.elements();
    }
    let parts: Vec<String> = parts.iter().map(|s| (*s).to_string()).collect();
    Ok(Recipe::Built {
        bytes: n * 4,
        build: Box::new(move |ck| {
            let mut out = Vec::with_capacity(n * 4);
            for name in &parts {
                let entry = ck.at(name)?;
                let eb = elem_bytes(entry.dtype);
                let src = ck.bytes(entry);
                for i in 0..entry.elements() {
                    out.extend_from_slice(&widen_one(entry.dtype, &src[i * eb..])?.to_le_bytes());
                }
            }
            Ok(out)
        }),
    })
}

/// A float vector as f32, zero-extended to `n_out` elements (a bias beside a zero-padded weight).
pub fn widen_padded(ck: &Checkpoint, name: &str, n_out: usize) -> Result<Recipe> {
    let entry = ck.at(name)?;
    widenable(entry, "widen_padded")?;
    if n_out < entry.elements() {
        return layout("widen_padded: shorter than the tensor");
    }
    let (name, n) = (name.to_string(), entry.elements());
    Ok(Recipe::Built {
        bytes: n_out * 4,
        build: Box::new(move |ck| {
            let entry = ck.at(&name)?;
            let eb = elem_bytes(entry.dtype);
            let src = ck.bytes(entry);
            let mut out = vec![0u8; n_out * 4];
            for i in 0..n {
                out[i * 4..i * 4 + 4]
                    .copy_from_slice(&widen_one(entry.dtype, &src[i * eb..])?.to_le_bytes());
            }
            Ok(out)
        }),
    })
}

/// Rounds `n` up to a multiple of `m`.
pub fn up(n: usize, m: usize) -> usize {
    n.div_ceil(m) * m
}

/// A causal 3-D conv weight `[Cout][Cin][3][3][3]` as the implicit GEMM's operand rows
/// `[Cout_pad][taps * Cin_pad]`, taps in `(t, y, x)` order with the channel innermost, zero-padded.
///
/// `taps` is 27 for the clip form and 9 for the image form, which keeps only the last temporal tap.
pub fn conv3d_taps(
    ck: &Checkpoint,
    name: &str,
    cout: usize,
    cin: usize,
    taps: usize,
) -> Result<Recipe> {
    let entry = ck.at(name)?;
    if entry.dtype != Dtype::F16 {
        return layout(format!("conv3d_taps on {:?}", entry.dtype));
    }
    if taps != 27 && taps != 9 {
        return layout("conv3d_taps: taps must be 27 or 9");
    }
    let (cin_pad, cout_pad) = (up(cin, 8), up(cout, 64));
    let k = up(taps * cin_pad, 32);
    let name = name.to_string();
    Ok(Recipe::Built {
        bytes: cout_pad * k * 2,
        build: Box::new(move |ck| {
            let src = ck.bytes(ck.at(&name)?);
            let mut out = vec![0u8; cout_pad * k * 2];
            for oc in 0..cout {
                for ic in 0..cin {
                    for t in 0..3 {
                        if taps == 9 && t != 2 {
                            continue;
                        }
                        for y in 0..3 {
                            for x in 0..3 {
                                let tap = if taps == 27 {
                                    (t * 3 + y) * 3 + x
                                } else {
                                    y * 3 + x
                                };
                                let from = ((((oc * cin + ic) * 3 + t) * 3 + y) * 3 + x) * 2;
                                let to = (oc * k + tap * cin_pad + ic) * 2;
                                out[to..to + 2].copy_from_slice(&src[from..from + 2]);
                            }
                        }
                    }
                }
            }
            Ok(out)
        }),
    })
}

/// The f32 `[N, 1]` per-row scales of concatenated int8 operands, in the rows' order.
pub fn scales_rows(ck: &Checkpoint, parts: &[&str]) -> Result<Recipe> {
    widen_f32(ck, parts)
}

/// The per-row scales of an interleaved gate|up operand, in the interleaved rows' order.
pub fn scales_interleave16(
    ck: &Checkpoint,
    first: (&str, usize),
    second: (&str, usize),
    rows_each: usize,
) -> Result<Recipe> {
    for (name, _) in [first, second] {
        let entry = ck.at(name)?;
        if entry.dtype != Dtype::F32 || entry.row_bytes() != 4 {
            return layout("scales must be f32 [N, 1]");
        }
    }
    let runs = interleaved_runs(first.1, second.1, rows_each, 16);
    // The two halves alternate in 16-element runs, exactly as their rows do.
    let (a, b) = (first.0.to_string(), second.0.to_string());
    Ok(Recipe::Built {
        bytes: 2 * rows_each * 4,
        build: Box::new(move |ck| {
            let (ea, eb) = (ck.at(&a)?, ck.at(&b)?);
            let (sa, sb) = (ck.bytes(ea), ck.bytes(eb));
            let mut out = Vec::with_capacity(2 * rows_each * 4);
            for (i, &(first, count)) in runs.iter().enumerate() {
                let src = if i % 2 == 0 { sa } else { sb };
                out.extend_from_slice(&src[first * 4..(first + count) * 4]);
            }
            Ok(out)
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// A checkpoint holding the given tensors, laid out in the order they are listed.
    fn checkpoint(
        dir: &std::path::Path,
        tensors: &[(&str, &str, Vec<usize>, Vec<u8>)],
    ) -> Checkpoint {
        let path = dir.join("c.safetensors");
        let (mut header, mut offset, mut blob) = (String::from("{"), 0usize, Vec::new());
        for (i, (name, dtype, shape, bytes)) in tensors.iter().enumerate() {
            if i > 0 {
                header.push(',');
            }
            let shape_text = shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(",");
            header.push_str(&format!(
                "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{shape_text}],\"data_offsets\":[{},{}]}}",
                offset,
                offset + bytes.len()
            ));
            offset += bytes.len();
            blob.extend_from_slice(bytes);
        }
        header.push('}');
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header.as_bytes()).unwrap();
        file.write_all(&blob).unwrap();
        drop(file);
        unsafe { Checkpoint::open(&path) }.unwrap()
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn as_f32(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn rows_of_concatenates_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let ck = checkpoint(
            dir.path(),
            &[
                ("a", "I8", vec![2, 4], vec![1, 2, 3, 4, 5, 6, 7, 8]),
                ("b", "I8", vec![1, 4], vec![9, 10, 11, 12]),
            ],
        );
        let recipe = rows_of(&ck, &["a", "b"], 0).unwrap();
        assert_eq!(recipe.device_bytes(), 12);
        assert_eq!(
            recipe.assemble(&ck).unwrap(),
            (1..=12u8).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_pitch_pads_each_row_with_zeros() {
        let dir = tempfile::tempdir().unwrap();
        let ck = checkpoint(
            dir.path(),
            &[("a", "I8", vec![2, 4], vec![1, 2, 3, 4, 5, 6, 7, 8])],
        );
        let recipe = rows_of(&ck, &["a"], 6).unwrap();
        assert_eq!(recipe.device_bytes(), 12);
        assert_eq!(
            recipe.assemble(&ck).unwrap(),
            vec![1, 2, 3, 4, 0, 0, 5, 6, 7, 8, 0, 0]
        );
        // a pitch narrower than the row is a plan error, not a truncation
        assert!(rows_of(&ck, &["a"], 3).is_err());
        // and concatenating differing row widths is refused
        let ck2 = checkpoint(
            dir.path(),
            &[
                ("a", "I8", vec![1, 4], vec![0; 4]),
                ("b", "I8", vec![1, 8], vec![0; 8]),
            ],
        );
        assert!(rows_of(&ck2, &["a", "b"], 0).is_err());
    }

    #[test]
    fn interleave16_alternates_the_halves_in_sixteens() {
        let dir = tempfile::tempdir().unwrap();
        // 32 gate rows then 32 up rows, one byte each, valued by row so the order is readable
        let gate: Vec<u8> = (0..32u8).collect();
        let up: Vec<u8> = (100..132u8).collect();
        let ck = checkpoint(
            dir.path(),
            &[
                ("gate", "I8", vec![32, 1], gate),
                ("up", "I8", vec![32, 1], up),
            ],
        );
        let recipe = interleave16(&ck, ("gate", 0), ("up", 0), 32, 0).unwrap();
        let out = recipe.assemble(&ck).unwrap();
        assert_eq!(out.len(), 64);
        assert_eq!(&out[0..16], &(0..16u8).collect::<Vec<_>>()[..]);
        assert_eq!(&out[16..32], &(100..116u8).collect::<Vec<_>>()[..]);
        assert_eq!(&out[32..48], &(16..32u8).collect::<Vec<_>>()[..]);
        assert_eq!(&out[48..64], &(116..132u8).collect::<Vec<_>>()[..]);
        // a half count that is not a multiple of sixteen does not describe the operand
        assert!(interleave16(&ck, ("gate", 0), ("up", 0), 24, 0).is_err());
        assert!(interleave16(&ck, ("gate", 16), ("up", 0), 32, 0).is_err());
    }

    #[test]
    fn scales_interleave16_follows_the_rows() {
        let dir = tempfile::tempdir().unwrap();
        let gate: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let up: Vec<f32> = (0..32).map(|i| 100.0 + i as f32).collect();
        let ck = checkpoint(
            dir.path(),
            &[
                ("gs", "F32", vec![32, 1], f32_bytes(&gate)),
                ("us", "F32", vec![32, 1], f32_bytes(&up)),
            ],
        );
        let out = as_f32(
            &scales_interleave16(&ck, ("gs", 0), ("us", 0), 32)
                .unwrap()
                .assemble(&ck)
                .unwrap(),
        );
        assert_eq!(out.len(), 64);
        assert_eq!(&out[0..16], &gate[0..16]);
        assert_eq!(&out[16..32], &up[0..16]);
        assert_eq!(&out[32..48], &gate[16..32]);
        assert_eq!(&out[48..64], &up[16..32]);
        // scales must be f32 [N, 1]
        let ck2 = checkpoint(dir.path(), &[("gs", "F16", vec![32, 1], vec![0; 64])]);
        assert!(scales_interleave16(&ck2, ("gs", 0), ("gs", 0), 32).is_err());
    }

    #[test]
    fn widen_f32_is_exact_for_f16_and_bf16() {
        let dir = tempfile::tempdir().unwrap();
        let f16_bits: Vec<u8> = [1.0f32, -2.5, 0.5]
            .iter()
            .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
            .collect();
        let bf16_bits: Vec<u8> = [3.0f32, -0.25]
            .iter()
            .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
            .collect();
        let ck = checkpoint(
            dir.path(),
            &[
                ("h", "F16", vec![3], f16_bits),
                ("b", "BF16", vec![2], bf16_bits),
                ("f", "F32", vec![2], f32_bytes(&[7.5, -8.25])),
            ],
        );
        let out = as_f32(
            &widen_f32(&ck, &["h", "b", "f"])
                .unwrap()
                .assemble(&ck)
                .unwrap(),
        );
        assert_eq!(out, vec![1.0, -2.5, 0.5, 3.0, -0.25, 7.5, -8.25]);
        // int8 rows are not widened by this path: their scales are separate tensors
        let ck2 = checkpoint(dir.path(), &[("q", "I8", vec![4], vec![0; 4])]);
        assert!(widen_f32(&ck2, &["q"]).is_err());
    }

    #[test]
    fn widen_padded_zero_extends() {
        let dir = tempfile::tempdir().unwrap();
        let ck = checkpoint(
            dir.path(),
            &[("bias", "F32", vec![3], f32_bytes(&[1.0, 2.0, 3.0]))],
        );
        let out = as_f32(&widen_padded(&ck, "bias", 6).unwrap().assemble(&ck).unwrap());
        assert_eq!(out, vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0]);
        assert!(widen_padded(&ck, "bias", 2).is_err());
    }

    #[test]
    fn rows_padded_appends_zero_rows() {
        let dir = tempfile::tempdir().unwrap();
        let ck = checkpoint(
            dir.path(),
            &[("w", "I8", vec![2, 3], vec![1, 2, 3, 4, 5, 6])],
        );
        let recipe = rows_padded(&ck, "w", 2, 0).unwrap();
        assert_eq!(recipe.device_bytes(), 12);
        assert_eq!(
            recipe.assemble(&ck).unwrap(),
            vec![1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0]
        );
    }

    #[test]
    fn rows_permuted_reorders_whole_rows() {
        let dir = tempfile::tempdir().unwrap();
        // three rows of two bytes; ask for them as [row2, row0, row1]
        let ck = checkpoint(
            dir.path(),
            &[("qkv", "I8", vec![3, 2], vec![0, 1, 2, 3, 4, 5])],
        );
        let out = rows_permuted(&ck, "qkv", &[(2, 1), (0, 1), (1, 1)], 0)
            .unwrap()
            .assemble(&ck)
            .unwrap();
        assert_eq!(out, vec![4, 5, 0, 1, 2, 3]);
        assert!(rows_permuted(&ck, "qkv", &[(2, 2)], 0).is_err());
    }

    #[test]
    fn regroup_zero_extends_each_group() {
        let dir = tempfile::tempdir().unwrap();
        // two rows of two groups of two f16 elements, widened to three elements per group
        let ck = checkpoint(
            dir.path(),
            &[("w", "F16", vec![2, 4], (1..=16u8).collect())],
        );
        let out = regroup(&ck, "w", 2, 2, 3, 0)
            .unwrap()
            .assemble(&ck)
            .unwrap();
        assert_eq!(out.len(), 2 * 2 * 3 * 2);
        assert_eq!(&out[0..4], &[1, 2, 3, 4]); // group 0's two elements
        assert_eq!(&out[4..6], &[0, 0]); // the extension
        assert_eq!(&out[6..10], &[5, 6, 7, 8]); // group 1
        assert_eq!(&out[10..12], &[0, 0]);
        // out_rows pads whole rows too
        let padded = regroup(&ck, "w", 2, 2, 3, 4)
            .unwrap()
            .assemble(&ck)
            .unwrap();
        assert_eq!(padded.len(), 4 * 2 * 3 * 2);
        assert!(padded[24..].iter().all(|b| *b == 0));
        assert!(regroup(&ck, "w", 2, 2, 1, 0).is_err()); // narrower than the input
        assert!(regroup(&ck, "w", 3, 2, 3, 0).is_err()); // the row does not hold the groups
    }

    #[test]
    fn widen_runs_permutes_elements() {
        let dir = tempfile::tempdir().unwrap();
        let ck = checkpoint(
            dir.path(),
            &[("b", "F32", vec![4], f32_bytes(&[0.0, 1.0, 2.0, 3.0]))],
        );
        let out = as_f32(
            &widen_runs(&ck, "b", &[(2, 2), (0, 2)])
                .unwrap()
                .assemble(&ck)
                .unwrap(),
        );
        assert_eq!(out, vec![2.0, 3.0, 0.0, 1.0]);
        assert!(widen_runs(&ck, "b", &[(3, 2)]).is_err());
    }

    #[test]
    fn interleaved_runs_pairs_the_halves() {
        assert_eq!(
            interleaved_runs(0, 32, 32, 16),
            vec![(0, 16), (32, 16), (16, 16), (48, 16)]
        );
    }

    #[test]
    fn a_plan_that_fails_names_the_tensor_it_wanted() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("c.safetensors");
        {
            let ck = checkpoint(dir.path(), &[("present", "I8", vec![1, 4], vec![0; 4])]);
            drop(ck);
        }
        // Safety: a checkpoint this test wrote and nothing else touches.
        let result = unsafe {
            Weights::open(&path, |ck, out| {
                out.insert("kept".into(), rows_of(ck, &["present"], 0)?);
                out.insert(
                    "lost".into(),
                    rows_of(ck, &["blocks.0.attn.qkv_proj.weight"], 0)?,
                );
                Ok(())
            })
        };
        let message = result.err().unwrap().to_string();
        assert!(
            message.contains("blocks.0.attn.qkv_proj.weight"),
            "{message}"
        );
    }

    #[test]
    fn weights_expose_recipes_and_host_tables() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("c.safetensors");
        {
            let ck = checkpoint(dir.path(), &[("bias", "F16", vec![2], vec![0, 60, 0, 64])]);
            drop(ck);
        }
        // Safety: a checkpoint this test wrote and nothing else touches.
        let w = unsafe {
            Weights::open(&path, |ck, out| {
                out.insert("b".into(), widen_f32(ck, &["bias"])?);
                Ok(())
            })
        }
        .unwrap();
        assert!(w.has("b") && !w.has("nope"));
        assert_eq!(w.host_f32("b", 2).unwrap(), vec![1.0, 2.0]);
        assert!(w.host_f32("b", 3).is_err());
        assert!(matches!(w.recipe("nope"), Err(Error::NoRecipe(_))));
    }
}

//! A read-only view of one ComfyUI checkpoint, mapped rather than read.
//!
//! Nothing is read at open beyond the header: pages fault in when a tensor's bytes are first touched,
//! and are released again once they are on the device. The dtype strings are the file's own, so a plan
//! validates against what the checkpoint actually holds rather than against an expectation baked in here.
use hrx::artifacts::safetensors::{DType, Entry, FileView};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("cannot open {path}: {source}")]
    Open {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("{path} is not a safetensors file: {message}")]
    Header { path: PathBuf, message: String },
    #[error("missing tensor {name} in {path}")]
    Missing { name: String, path: PathBuf },
    #[error("{name} is {found}, expected {expected} in {path}")]
    Mismatch {
        name: String,
        found: String,
        expected: String,
        path: PathBuf,
    },
    #[error("{name} has an unsupported checkpoint dtype {dtype:?} in {path}")]
    Dtype {
        name: String,
        dtype: DType,
        path: PathBuf,
    },
}

pub type Result<T> = std::result::Result<T, Error>;

pub struct Checkpoint {
    file: FileView,
}

impl Checkpoint {
    /// Maps a checkpoint and reads its header.
    ///
    /// # Safety
    ///
    /// The file is mapped, not copied — a 30 GB checkpoint has to be, and every tensor is read
    /// through the mapping as the upload needs it. So the file must not be modified or truncated
    /// while the returned `Checkpoint` lives:
    ///
    /// - Modifying it changes bytes this crate has already validated. The header says a tensor is
    ///   `[5376][14336]` of bf16 and the reader trusts that from then on.
    /// - Truncating it turns a mapped page into a `SIGBUS`, which no `Result` can carry: the process
    ///   dies at the read.
    ///
    /// Nothing in the filesystem enforces this and nothing here can check it, which is why this is
    /// `unsafe` rather than a comment claiming the file is immutable. [`crate::Session`] states the
    /// same requirement in its own documentation, and is the safe entry point that relies on it.
    pub unsafe fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        std::fs::File::open(&path).map_err(|source| Error::Open {
            path: path.clone(),
            source,
        })?;
        let file = unsafe { FileView::map(&path) }.map_err(|error| Error::Header {
            path: path.clone(),
            message: error.to_string(),
        })?;
        Ok(Self { file })
    }

    pub fn entries(&self) -> &BTreeMap<String, Entry> {
        self.file.entries()
    }
    pub fn has(&self, name: &str) -> bool {
        self.file.contains(name)
    }

    pub fn at(&self, name: &str) -> Result<&Entry> {
        self.file.entries().get(name).ok_or_else(|| Error::Missing {
            name: name.to_string(),
            path: self.file.path().to_path_buf(),
        })
    }

    /// The checked lookup a plan uses. A `-1` dimension matches whatever the file has there.
    pub fn at_checked(&self, name: &str, dtype: DType, shape: &[i64]) -> Result<&Entry> {
        let entry = self.at(name)?;
        let shape_ok = entry.shape.len() == shape.len()
            && entry
                .shape
                .iter()
                .zip(shape)
                .all(|(&got, &want)| want < 0 || got as i64 == want);
        if entry.dtype != dtype || !shape_ok {
            return Err(Error::Mismatch {
                name: name.to_string(),
                found: format!("{:?} {:?}", entry.dtype, entry.shape),
                expected: format!("{dtype:?} {shape:?}"),
                path: self.file.path().to_path_buf(),
            });
        }
        Ok(entry)
    }

    /// A tensor's bytes, still in the mapping. Reading them faults their pages in.
    pub fn bytes(&self, entry: &Entry) -> &[u8] {
        self.file
            .bytes(entry)
            .expect("checkpoint entry belongs to this file")
    }

    /// Hint the kernel to read a range ahead of a sequential pass over it.
    pub fn will_need(&self, range: &[u8]) {
        self.file.will_need_bytes(range);
    }

    /// Release a range's pages once its bytes are on the device. A tensor is read once, and tens of
    /// gigabytes of checkpoint left resident compete with the device allocations for the same memory
    /// on a unified-memory part. `H3_KEEP_MAPPED=1` keeps them, which is what a repeated test or
    /// benchmark run wants: the page cache then serves the next run instead of re-reading the file.
    pub fn done_with(&self, range: &[u8]) {
        static KEEP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let keep = *KEEP.get_or_init(|| {
            std::env::var_os("H3_KEEP_MAPPED").is_some_and(|v| !v.is_empty() && v != "0")
        });
        if !keep {
            self.file.done_with_bytes(range);
        }
    }
}

/// Bytes per element for the dtypes a ComfyUI checkpoint holds. `U8` includes its quantisation
/// metadata blobs. Anything else is a checkpoint this build does not understand.
pub fn dtype_bytes(name: &str, dtype: DType, path: &Path) -> Result<usize> {
    match dtype {
        DType::BOOL | DType::U8 | DType::I8 => Ok(1),
        DType::U16 | DType::I16 | DType::F16 | DType::BF16 => Ok(2),
        DType::U32 | DType::I32 | DType::F32 => Ok(4),
        DType::U64 | DType::I64 | DType::F64 => Ok(8),
        other => Err(Error::Dtype {
            name: name.to_string(),
            dtype: other,
            path: path.to_path_buf(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hrx::artifacts::safetensors::DType as Dtype;
    use std::fs::File;
    use std::io::Write;

    /// Writes a safetensors file: `(name, dtype, shape, bytes)` entries laid out in order.
    fn write_checkpoint(
        path: &Path,
        tensors: &[(&str, &str, Vec<usize>, Vec<u8>)],
        header_override: Option<u64>,
    ) {
        let mut header = String::from("{");
        let mut offset = 0usize;
        let mut blob = Vec::new();
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
        let mut file = File::create(path).unwrap();
        let len = header_override.unwrap_or(header.len() as u64);
        file.write_all(&len.to_le_bytes()).unwrap();
        file.write_all(header.as_bytes()).unwrap();
        file.write_all(&blob).unwrap();
    }

    fn tmp() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    #[test]
    fn reads_dtypes_shapes_and_bytes() {
        let dir = tmp();
        let path = dir.path().join("c.safetensors");
        write_checkpoint(
            &path,
            &[
                ("gate", "I8", vec![2, 4], (0..8u8).collect()),
                ("scale", "F32", vec![2, 1], vec![0u8; 8]),
            ],
            None,
        );
        let ck = unsafe { Checkpoint::open(&path) }.unwrap();
        let gate = ck.at("gate").unwrap();
        assert_eq!(gate.dtype, Dtype::I8);
        assert_eq!(gate.shape, vec![2, 4]);
        assert_eq!(
            (
                gate.rows(),
                gate.row_bytes().unwrap(),
                gate.elements().unwrap()
            ),
            (2, 4, 8)
        );
        assert_eq!(ck.bytes(gate), &(0..8u8).collect::<Vec<_>>()[..]);
        assert!(ck.has("scale") && !ck.has("nope"));
        // entries are ordered, so error reporting is deterministic
        assert_eq!(
            ck.entries().keys().cloned().collect::<Vec<_>>(),
            vec!["gate", "scale"]
        );
    }

    #[test]
    fn a_missing_tensor_names_itself_and_the_file() {
        let dir = tmp();
        let path = dir.path().join("c.safetensors");
        write_checkpoint(&path, &[("a", "F16", vec![1], vec![0, 0])], None);
        let ck = unsafe { Checkpoint::open(&path) }.unwrap();
        let message = ck
            .at("blocks.0.attn.qkv_proj.weight")
            .unwrap_err()
            .to_string();
        assert!(
            message.contains("blocks.0.attn.qkv_proj.weight"),
            "{message}"
        );
        assert!(message.contains("c.safetensors"), "{message}");
    }

    #[test]
    fn a_type_or_shape_mismatch_reports_both_sides() {
        let dir = tmp();
        let path = dir.path().join("c.safetensors");
        write_checkpoint(&path, &[("gate", "I8", vec![32, 8], vec![0u8; 256])], None);
        let ck = unsafe { Checkpoint::open(&path) }.unwrap();
        let message = ck
            .at_checked("gate", Dtype::F16, &[32, 8])
            .unwrap_err()
            .to_string();
        assert!(
            message.contains("I8") && message.contains("F16"),
            "{message}"
        );
        // a free dimension matches anything, and the right type passes
        assert!(ck.at_checked("gate", Dtype::I8, &[-1, 8]).is_ok());
        assert!(ck.at_checked("gate", Dtype::I8, &[32, -1]).is_ok());
        assert!(ck.at_checked("gate", Dtype::I8, &[8, 32]).is_err());
        assert!(ck.at_checked("gate", Dtype::I8, &[32]).is_err());
    }

    #[test]
    fn rejects_files_that_are_not_checkpoints() {
        let dir = tmp();
        let short = dir.path().join("short.safetensors");
        File::create(&short).unwrap().write_all(b"abc").unwrap();
        assert!(matches!(
            unsafe { Checkpoint::open(&short) },
            Err(Error::Header { .. })
        ));

        let missing = dir.path().join("nope.safetensors");
        assert!(matches!(
            unsafe { Checkpoint::open(&missing) },
            Err(Error::Open { .. })
        ));

        // a header length that runs past the file
        let bad = dir.path().join("bad.safetensors");
        write_checkpoint(&bad, &[("a", "F16", vec![1], vec![0, 0])], Some(1 << 20));
        assert!(matches!(
            unsafe { Checkpoint::open(&bad) },
            Err(Error::Header { .. })
        ));
    }

    #[test]
    fn rejects_a_span_past_the_data_block() {
        let dir = tmp();
        let path = dir.path().join("c.safetensors");
        // declare more bytes than the blob holds
        let header = br#"{"a":{"dtype":"F32","shape":[64],"data_offsets":[0,256]}}"#;
        let mut file = File::create(&path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header).unwrap();
        file.write_all(&[0u8; 16]).unwrap(); // far short of 256
        drop(file);
        assert!(unsafe { Checkpoint::open(&path) }.is_err());
    }

    #[test]
    fn rejects_dimensions_that_are_not_whole_numbers() {
        // The C implementation parsed dimensions from their source spelling to reject these; serde's
        // usize deserialisation refuses them for the same reason.
        for spelling in ["-1", "1.5", "true", "null", "\"2\"", "1e999", "NaN"] {
            let dir = tmp();
            let path = dir.path().join("c.safetensors");
            let header =
                format!(r#"{{"a":{{"dtype":"F32","shape":[{spelling}],"data_offsets":[0,4]}}}}"#);
            let mut file = File::create(&path).unwrap();
            file.write_all(&(header.len() as u64).to_le_bytes())
                .unwrap();
            file.write_all(header.as_bytes()).unwrap();
            file.write_all(&[0u8; 4]).unwrap();
            drop(file);
            assert!(
                unsafe { Checkpoint::open(&path) }.is_err(),
                "shape [{spelling}] should be rejected"
            );
        }
    }

    #[test]
    fn scalars_and_empty_tensors_are_valid() {
        let dir = tmp();
        let path = dir.path().join("c.safetensors");
        write_checkpoint(
            &path,
            &[
                ("scalar", "F32", vec![], vec![0u8; 4]),
                ("empty", "BF16", vec![2, 0, 3], vec![]),
            ],
            None,
        );
        let ck = unsafe { Checkpoint::open(&path) }.unwrap();
        let scalar = ck.at("scalar").unwrap();
        assert_eq!(
            (scalar.elements().unwrap(), scalar.rows(), scalar.bytes),
            (1, 1, 4)
        );
        let empty = ck.at("empty").unwrap();
        assert_eq!((empty.elements().unwrap(), empty.bytes), (0, 0));
        assert!(ck.bytes(empty).is_empty());
    }

    #[test]
    fn dtype_sizes_cover_what_a_checkpoint_holds() {
        let path = Path::new("x.safetensors");
        for (dtype, want) in [
            (Dtype::I8, 1),
            (Dtype::U8, 1),
            (Dtype::BOOL, 1),
            (Dtype::F16, 2),
            (Dtype::BF16, 2),
            (Dtype::F32, 4),
            (Dtype::I64, 8),
        ] {
            assert_eq!(dtype_bytes("t", dtype, path).unwrap(), want, "{dtype:?}");
        }
        assert!(dtype_bytes("t", Dtype::F8_E5M2, path).is_err());
    }

    #[test]
    fn madvise_rounds_outward_to_read_and_inward_to_drop() {
        let dir = tmp();
        let path = dir.path().join("c.safetensors");
        write_checkpoint(&path, &[("a", "F32", vec![4096], vec![7u8; 16384])], None);
        let ck = unsafe { Checkpoint::open(&path) }.unwrap();
        let entry = ck.at("a").unwrap();
        let bytes = ck.bytes(entry);
        // Both hints are advisory and must leave the data readable either way.
        ck.will_need(bytes);
        assert_eq!(bytes[0], 7);
        ck.done_with(bytes);
        assert_eq!(ck.bytes(entry)[16383], 7);
        // An empty range is a no-op rather than a bad madvise call.
        ck.will_need(&[]);
        ck.done_with(&[]);

        // A buffer that is not part of the mapping is left alone. MADV_DONTNEED on private anonymous
        // memory zeroes it, so a range that is not checked is a way to erase a caller's data.
        let foreign = vec![0xABu8; 256 * 1024];
        ck.done_with(&foreign);
        assert!(
            foreign.iter().all(|b| *b == 0xAB),
            "foreign memory was touched"
        );
    }
}

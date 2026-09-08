//! A read-only view of one ComfyUI checkpoint, mapped rather than read.
//!
//! Nothing is read at open beyond the header: pages fault in when a tensor's bytes are first touched,
//! and are released again once they are on the device. The dtype strings are the file's own, so a plan
//! validates against what the checkpoint actually holds rather than against an expectation baked in here.
use memmap2::Mmap;
use safetensors::tensor::{Dtype, Metadata, SafeTensors};
use std::collections::BTreeMap;
use std::fs::File;
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
        dtype: Dtype,
        path: PathBuf,
    },
}

pub type Result<T> = std::result::Result<T, Error>;

/// One tensor's placement in the mapped data block.
#[derive(Clone, Debug)]
pub struct Entry {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    /// Offset from the start of the data block, not the file.
    pub offset: usize,
    pub bytes: usize,
}

impl Entry {
    /// A scalar counts as one row, so `row_bytes` is well defined for every tensor.
    pub fn rows(&self) -> usize {
        self.shape.first().copied().unwrap_or(1)
    }
    pub fn row_bytes(&self) -> usize {
        self.bytes.checked_div(self.rows()).unwrap_or(self.bytes)
    }
    pub fn elements(&self) -> usize {
        self.shape
            .iter()
            .product::<usize>()
            .max(usize::from(self.shape.is_empty()))
    }
    /// `I8 [32, 8]`, as it appears in a mismatch message.
    pub fn describe(&self) -> String {
        format!("{:?} {:?}", self.dtype, self.shape)
    }
}

pub struct Checkpoint {
    path: PathBuf,
    map: Mmap,
    /// Where the data block starts in the file.
    data: usize,
    /// Ordered, because a lookup failure reports the lexicographically first offender.
    entries: BTreeMap<String, Entry>,
}

impl Checkpoint {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path).map_err(|source| Error::Open {
            path: path.clone(),
            source,
        })?;
        // Safety: the checkpoint is treated as immutable for the process's lifetime. A concurrent
        // writer would be a violation, which is true of the C implementation this replaces as well.
        let map = unsafe { Mmap::map(&file) }.map_err(|source| Error::Open {
            path: path.clone(),
            source,
        })?;

        let (header, metadata): (usize, Metadata) =
            SafeTensors::read_metadata(&map).map_err(|e| Error::Header {
                path: path.clone(),
                message: e.to_string(),
            })?;
        let data = 8 + header;
        let data_bytes = map.len().checked_sub(data).ok_or_else(|| Error::Header {
            path: path.clone(),
            message: "header runs past the end of the file".into(),
        })?;

        let mut entries = BTreeMap::new();
        for (name, info) in metadata.tensors() {
            let (begin, end) = info.data_offsets;
            if end < begin || end > data_bytes {
                return Err(Error::Header {
                    path: path.clone(),
                    message: format!("tensor {name} spans past the data block"),
                });
            }
            entries.insert(
                name.clone(),
                Entry {
                    dtype: info.dtype,
                    shape: info.shape.clone(),
                    offset: begin,
                    bytes: end - begin,
                },
            );
        }
        Ok(Self {
            path,
            map,
            data,
            entries,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
    pub fn entries(&self) -> &BTreeMap<String, Entry> {
        &self.entries
    }
    pub fn has(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    pub fn at(&self, name: &str) -> Result<&Entry> {
        self.entries.get(name).ok_or_else(|| Error::Missing {
            name: name.to_string(),
            path: self.path.clone(),
        })
    }

    /// The checked lookup a plan uses. A `-1` dimension matches whatever the file has there.
    pub fn at_checked(&self, name: &str, dtype: Dtype, shape: &[i64]) -> Result<&Entry> {
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
                found: entry.describe(),
                expected: format!("{dtype:?} {shape:?}"),
                path: self.path.clone(),
            });
        }
        Ok(entry)
    }

    /// A tensor's bytes, still in the mapping. Reading them faults their pages in.
    pub fn bytes(&self, entry: &Entry) -> &[u8] {
        &self.map[self.data + entry.offset..self.data + entry.offset + entry.bytes]
    }

    /// Hint the kernel to read a range ahead of a sequential pass over it.
    pub fn will_need(&self, range: &[u8]) {
        self.advise(range, libc::MADV_WILLNEED);
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
            self.advise(range, libc::MADV_DONTNEED);
        }
    }

    /// The whole pages a range covers, so a partial page at either end is never advised away under a
    /// neighbouring tensor's bytes: WILLNEED rounds outward, DONTNEED inward. Dropping a shared page
    /// would cost the neighbour a re-read, not correctness, so the asymmetry is the safe direction.
    ///
    /// The range must lie inside this checkpoint's mapping, and anything else is ignored. Both
    /// callers are safe functions taking a `&[u8]`, and `MADV_DONTNEED` on private anonymous memory
    /// *zeroes* it — so without this check a caller could hand over an unrelated buffer and have it
    /// silently erased. Every real call site slices `bytes()`, which is always inside.
    fn advise(&self, range: &[u8], how: i32) {
        if range.is_empty() {
            return;
        }
        let map_start = self.map.as_ptr() as usize;
        let map_end = map_start + self.map.len();
        let first = range.as_ptr() as usize;
        let last = first + range.len();
        if first < map_start || last > map_end {
            return;
        }
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        let (begin, end) = if how == libc::MADV_WILLNEED {
            (first & !(page - 1), (last + page - 1) & !(page - 1))
        } else {
            ((first + page - 1) & !(page - 1), last & !(page - 1))
        };
        if end > begin {
            unsafe { libc::madvise(begin as *mut libc::c_void, end - begin, how) };
        }
    }
}

/// Bytes per element for the dtypes a ComfyUI checkpoint holds. `U8` includes its quantisation
/// metadata blobs. Anything else is a checkpoint this build does not understand.
pub fn dtype_bytes(name: &str, dtype: Dtype, path: &Path) -> Result<usize> {
    match dtype {
        Dtype::BOOL | Dtype::U8 | Dtype::I8 => Ok(1),
        Dtype::U16 | Dtype::I16 | Dtype::F16 | Dtype::BF16 => Ok(2),
        Dtype::U32 | Dtype::I32 | Dtype::F32 => Ok(4),
        Dtype::U64 | Dtype::I64 | Dtype::F64 => Ok(8),
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
        let ck = Checkpoint::open(&path).unwrap();
        let gate = ck.at("gate").unwrap();
        assert_eq!(gate.dtype, Dtype::I8);
        assert_eq!(gate.shape, vec![2, 4]);
        assert_eq!((gate.rows(), gate.row_bytes(), gate.elements()), (2, 4, 8));
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
        let ck = Checkpoint::open(&path).unwrap();
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
        let ck = Checkpoint::open(&path).unwrap();
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
            Checkpoint::open(&short),
            Err(Error::Header { .. })
        ));

        let missing = dir.path().join("nope.safetensors");
        assert!(matches!(
            Checkpoint::open(&missing),
            Err(Error::Open { .. })
        ));

        // a header length that runs past the file
        let bad = dir.path().join("bad.safetensors");
        write_checkpoint(&bad, &[("a", "F16", vec![1], vec![0, 0])], Some(1 << 20));
        assert!(matches!(Checkpoint::open(&bad), Err(Error::Header { .. })));
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
        assert!(Checkpoint::open(&path).is_err());
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
                Checkpoint::open(&path).is_err(),
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
        let ck = Checkpoint::open(&path).unwrap();
        let scalar = ck.at("scalar").unwrap();
        assert_eq!((scalar.elements(), scalar.rows(), scalar.bytes), (1, 1, 4));
        let empty = ck.at("empty").unwrap();
        assert_eq!((empty.elements(), empty.bytes), (0, 0));
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
        let ck = Checkpoint::open(&path).unwrap();
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
        // and a range that starts inside the mapping but runs past its end is refused, not clamped
        let inside = ck.bytes(ck.at("a").unwrap());
        let past = unsafe { std::slice::from_raw_parts(inside.as_ptr(), 1 << 30) };
        ck.done_with(past);
    }
}

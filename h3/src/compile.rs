//! Compiling Loom kernels through `loom-compile` into a shared cache, and loading them.
//!
//! One kernel per (stem, source text, symbol, backend, target, config, compiler binary): the cache file
//! name carries a hash of all of them, so neither an edited kernel source nor a replaced `loom-compile`
//! ever reuses a stale binary. `tools/kernel_cache.py` keeps the same policy for the Python harnesses,
//! and this must keep it too — the tag is the only thing that makes a cache entry reusable across
//! implementations.
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::OnceLock;

pub const BACKEND: &str = "amdgpu-hal";
pub const TARGET: &str = "gfx1151";

/// A kernel's configuration: ordered, because the order is part of the cache tag and of the argv.
pub type Cfg = Vec<(String, String)>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("missing kernel source {0}")]
    MissingSource(PathBuf),
    #[error("loom-compile not found: {0}")]
    NoCompiler(String),
    #[error("loom-compile failed for {stem}: {command}")]
    Failed { stem: String, command: String },
    #[error("{0}")]
    Io(String),
    #[error(transparent)]
    Runtime(#[from] hrx::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

/// A float as C's `%.17g`, which is how every float kernel config is spelled.
///
/// The spelling matters twice over: it is hashed into the cache tag, and it is handed to the compiler
/// as the constant itself. Rust's own formatting produces `0.00001` where this produces
/// `1.0000000000000001e-05`, so a shortest-round-trip formatter would both miss every existing cache
/// entry and change the text the compiler sees.
pub fn num(v: f64) -> String {
    if v.is_nan() {
        return (if v.is_sign_negative() { "-nan" } else { "nan" }).to_string();
    }
    if v.is_infinite() {
        return (if v < 0.0 { "-inf" } else { "inf" }).to_string();
    }
    const P: i32 = 17;
    // %g chooses between %e and %f from the decimal exponent the %e form would use.
    let exponent = if v == 0.0 {
        0
    } else {
        let e = format!("{:.*e}", (P - 1) as usize, v);
        e[e.find('e').unwrap() + 1..].parse::<i32>().unwrap_or(0)
    };
    if !(-4..P).contains(&exponent) {
        let s = format!("{:.*e}", (P - 1) as usize, v);
        let (mantissa, exp) = s.split_at(s.find('e').unwrap());
        // C writes at least two exponent digits, with a sign; Rust writes neither.
        let value: i32 = exp[1..].parse().unwrap_or(0);
        format!(
            "{}e{}{:02}",
            trim(mantissa),
            if value < 0 { '-' } else { '+' },
            value.abs()
        )
    } else {
        trim(&format!("{:.*}", (P - 1 - exponent).max(0) as usize, v))
    }
}

/// Drops the trailing zeros of a fractional part, and the point if nothing is left after it, which is
/// what `%g` does without the `#` flag.
fn trim(s: &str) -> String {
    if !s.contains('.') {
        return s.to_string();
    }
    let t = s.trim_end_matches('0');
    t.strip_suffix('.').unwrap_or(t).to_string()
}

/// FNV-1a 64 as the C implementation has it. The offset basis below is one digit short of the
/// canonical 14695981039346656037; it is kept because every cache tag on disk was computed with it.
fn fnv(bytes: &[u8]) -> u64 {
    let mut h: u64 = 1469598103934665603;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(1099511628211);
    }
    h
}

/// The first `digits` characters of the 16-digit hex, which is the high nibbles.
fn hex(h: u64, digits: usize) -> String {
    format!("{h:016x}")[..digits].to_string()
}

pub struct Compiler {
    exe: String,
    sources: PathBuf,
    cache: PathBuf,
    compiler_id: OnceLock<String>,
    /// Loaded once per process. A session is driven from one thread at a time — the C interface
    /// serialises on its own mutex — so this needs no lock, and `hrx::Kernel` is not `Send` anyway.
    loaded: RefCell<HashMap<String, Rc<hrx::Kernel>>>,
}

impl Compiler {
    pub fn new(
        exe: impl Into<String>,
        sources: impl Into<PathBuf>,
        cache: impl Into<PathBuf>,
    ) -> Self {
        Self {
            exe: exe.into(),
            sources: sources.into(),
            cache: cache.into(),
            compiler_id: OnceLock::new(),
            loaded: RefCell::new(HashMap::new()),
        }
    }

    fn source_path(&self, stem: &str) -> PathBuf {
        self.sources.join(format!("{stem}.loom"))
    }

    fn source_hash(&self, stem: &str) -> Result<String> {
        let path = self.source_path(stem);
        let text = std::fs::read(&path).map_err(|_| Error::MissingSource(path))?;
        Ok(hex(fnv(&text), 10))
    }

    /// The compiler binary as the spawn would find it — a bare name searches `PATH` — with its size and
    /// modification time, so replacing it invalidates every entry it produced.
    fn compiler_id(&self) -> Result<&str> {
        if let Some(id) = self.compiler_id.get() {
            return Ok(id);
        }
        let mut path = PathBuf::from(&self.exe);
        if !self.exe.contains('/') {
            let dirs = std::env::var("PATH").unwrap_or_default();
            for dir in dirs.split(':') {
                let candidate = Path::new(if dir.is_empty() { "." } else { dir }).join(&self.exe);
                // X_OK, the same test the spawn will make
                if unsafe { libc::access(cstring(&candidate)?.as_ptr(), libc::X_OK) } == 0 {
                    path = candidate;
                    break;
                }
            }
        }
        let meta = std::fs::metadata(&path).map_err(|_| Error::NoCompiler(self.exe.clone()))?;
        use std::os::unix::fs::MetadataExt;
        let id = format!(
            "{}:{}:{}.{}",
            path.display(),
            meta.size(),
            meta.mtime(),
            meta.mtime_nsec()
        );
        let _ = self.compiler_id.set(id);
        Ok(self.compiler_id.get().expect("just set"))
    }

    /// The cache file's name, which is also the in-process key.
    pub fn tag(&self, stem: &str, symbol: &str, cfg: &Cfg) -> Result<String> {
        let mut tag = format!("{stem}__s{}", self.source_hash(stem)?);
        let mut identity = format!("{BACKEND}\n{TARGET}\n{symbol}\n{}\n", self.compiler_id()?);
        for (key, value) in cfg {
            // the tag carries only the last dotted component, the identity the whole key
            let short = key
                .rfind('.')
                .map(|i| &key[i + 1..])
                .unwrap_or(key.as_str());
            tag.push_str(&format!("__{short}_{value}"));
            identity.push_str(&format!("{key}={value}\n"));
        }
        tag.push_str(&format!("__i{}", hex(fnv(identity.as_bytes()), 10)));
        Ok(tag
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.' {
                    c
                } else {
                    '_'
                }
            })
            .collect())
    }

    /// The kernel, compiling it into the cache first if it is not there. Loaded once per process.
    pub fn get(
        &self,
        gpu: &hrx::Gpu,
        stem: &str,
        symbol: &str,
        cfg: &Cfg,
    ) -> Result<Rc<hrx::Kernel>> {
        let tag = self.tag(stem, symbol, cfg)?;
        if let Some(kernel) = self.loaded.borrow().get(&tag) {
            return Ok(kernel.clone());
        }
        let path = self.cache.join(format!("{tag}.hsaco"));
        if !path.exists() {
            self.compile(stem, symbol, cfg, &path)?;
        }
        let kernel = Rc::new(gpu.load(&path, symbol)?);
        self.loaded.borrow_mut().insert(tag, kernel.clone());
        Ok(kernel)
    }

    /// Compile into a unique temporary and rename it into place under a lock on `<path>.lock`, so
    /// several sessions asking for the same kernel at once neither load a half-written binary nor
    /// clobber each other's temporary.
    fn compile(&self, stem: &str, symbol: &str, cfg: &Cfg, path: &Path) -> Result<()> {
        std::fs::create_dir_all(&self.cache).map_err(|e| Error::Io(e.to_string()))?;
        let lock_path = path.with_extension("hsaco.lock");
        let lock = File::create(&lock_path)
            .map_err(|e| Error::Io(format!("cannot create {}: {e}", lock_path.display())))?;
        let _guard = FlockGuard::exclusive(&lock)?;
        if path.exists() {
            return Ok(()); // another session published it while we waited
        }

        let tmp = path.with_extension(format!(
            "hsaco.tmp.{}.{:016x}",
            std::process::id(),
            fnv(&std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos().to_le_bytes().to_vec())
                .unwrap_or_default())
        ));
        let mut args: Vec<String> = vec![
            self.source_path(stem).display().to_string(),
            format!("--backend={BACKEND}"),
            format!("--target={TARGET}"),
            format!("--root=@{symbol}"),
            format!("--output={}", tmp.display()),
        ];
        for (key, value) in cfg {
            args.push(format!("--config={key}={value}"));
        }
        let status = std::process::Command::new(&self.exe)
            .args(&args)
            .status()
            .map_err(|e| Error::Io(format!("cannot spawn {}: {e}", self.exe)))?;
        let produced = std::fs::metadata(&tmp)
            .map(|m| m.len() > 0)
            .unwrap_or(false);
        if !status.success() || !produced {
            let _ = std::fs::remove_file(&tmp);
            let mut command = self.exe.clone();
            for a in &args {
                command.push(' ');
                command.push_str(a);
            }
            return Err(Error::Failed {
                stem: stem.into(),
                command,
            });
        }
        std::fs::rename(&tmp, path).map_err(|e| {
            let _ = std::fs::remove_file(&tmp);
            Error::Io(format!("cannot rename {}: {e}", tmp.display()))
        })
    }
}

fn cstring(path: &Path) -> Result<std::ffi::CString> {
    std::ffi::CString::new(path.as_os_str().as_encoded_bytes())
        .map_err(|_| Error::Io(format!("{} contains a NUL", path.display())))
}

/// Holds an exclusive `flock` for its lifetime, released even if the compile throws.
struct FlockGuard(i32);

impl FlockGuard {
    fn exclusive(file: &File) -> Result<Self> {
        let fd = file.as_raw_fd();
        if unsafe { libc::flock(fd, libc::LOCK_EX) } != 0 {
            return Err(Error::Io("cannot take the compile lock".into()));
        }
        Ok(Self(fd))
    }
}

impl Drop for FlockGuard {
    fn drop(&mut self) {
        unsafe { libc::flock(self.0, libc::LOCK_UN) };
    }
}

/// Writes a file, creating its directory. Used by the tests and by callers staging sources.
pub fn write(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    File::create(path)?.write_all(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv_matches_the_c_implementation_basis_and_all() {
        // Note the offset basis is 1469598103934665603, not FNV-1a 64's canonical
        // 14695981039346656037 — a digit went missing in the C source. It is kept exactly, because
        // every cache tag on disk was computed with it; "fixing" it would orphan the whole cache.
        // These are printf("%016llx") of the C function on the same inputs.
        assert_eq!(format!("{:016x}", fnv(b"")), "14650fb0739d0383");
        assert_eq!(format!("{:016x}", fnv(b"a")), "44bd8ad473cd9906");
        assert_eq!(format!("{:016x}", fnv(b"foobar")), "88fad7c0a8ff07f2");
        assert_eq!(
            format!("{:016x}", fnv(b"amdgpu-hal\ngfx1151\n")),
            "57bdb1be4ae3dabc"
        );
    }

    #[test]
    fn hex_takes_the_high_nibbles() {
        assert_eq!(hex(0x0123456789abcdef, 10), "0123456789");
        assert_eq!(hex(0xf, 16), "000000000000000f");
    }

    // The literals are written with as many digits as printf emitted, so each pair reads as
    // input-and-output. They denote the same doubles a shorter spelling would.
    #[allow(clippy::excessive_precision)]
    #[test]
    fn num_matches_printf_percent_17g() {
        // Every answer here is printf("%.17g") of the same double, generated by C and pasted in.
        // These are the values that appear as kernel configs plus the shapes that separate the %f and
        // %e forms; Rust's own formatting agrees with none of them.
        for (value, want) in [
            (0.0f64, "0"),
            (-0.0, "-0"),
            (1.0, "1"),
            (-1.0, "-1"),
            (0.5, "0.5"),
            (0.125, "0.125"),
            (2.0, "2"),
            (100.0, "100"),
            (1e-5, "1.0000000000000001e-05"),
            (1e-6, "9.9999999999999995e-07"),
            (1e-4, "0.0001"),
            (1e17, "1e+17"), // exponent == precision: the %e form takes over
            (1e18, "1e+18"),
            (1e16, "10000000000000000"), // one below it, and %f still wins
            (0.0625, "0.0625"),
            (0.088388347648318447, "0.088388347648318447"),
            (3.0517578125e-05, "3.0517578125e-05"),
            (0.0078125, "0.0078125"),
            (std::f64::consts::FRAC_1_SQRT_2, "0.70710678118654757"),
            (1e-30, "1.0000000000000001e-30"),
            (1e30, "1e+30"),
            (123456789.0, "123456789"),
        ] {
            assert_eq!(num(value), want, "{value}");
        }
        assert_eq!(num(f64::INFINITY), "inf");
        assert_eq!(num(f64::NEG_INFINITY), "-inf");
    }

    #[test]
    fn the_tag_carries_source_symbol_and_config() {
        let dir = tempfile::tempdir().unwrap();
        let sources = dir.path().join("kernels");
        write(&sources.join("gemm.loom"), b"kernel v1").unwrap();
        // a compiler that exists, so compiler_id resolves
        let exe = dir.path().join("loom-compile");
        write(&exe, b"#!/bin/sh\nexit 0\n").unwrap();
        let c = Compiler::new(
            exe.display().to_string(),
            &sources,
            dir.path().join("cache"),
        );

        let cfg: Cfg = vec![("h3.gemm.k_size".into(), "2048".into())];
        let base = c.tag("gemm", "h3_gemm", &cfg).unwrap();
        assert!(base.starts_with("gemm__s"), "{base}");
        assert!(base.contains("__k_size_2048"), "{base}"); // only the last dotted component
        assert!(base.contains("__i"), "{base}");

        // the symbol is in the identity, not the visible tag, so it changes the __i suffix
        let other_symbol = c.tag("gemm", "h3_other", &cfg).unwrap();
        assert_ne!(base, other_symbol);
        assert_eq!(base.split("__i").next(), other_symbol.split("__i").next());

        // a different config value changes both halves
        let other_cfg: Cfg = vec![("h3.gemm.k_size".into(), "4096".into())];
        assert_ne!(base, c.tag("gemm", "h3_gemm", &other_cfg).unwrap());

        // an edited source changes the __s hash
        write(&sources.join("gemm.loom"), b"kernel v2").unwrap();
        let edited = Compiler::new(
            exe.display().to_string(),
            &sources,
            dir.path().join("cache"),
        )
        .tag("gemm", "h3_gemm", &cfg)
        .unwrap();
        assert_ne!(base, edited);
    }

    #[test]
    fn a_replaced_compiler_invalidates_the_tag() {
        let dir = tempfile::tempdir().unwrap();
        let sources = dir.path().join("kernels");
        write(&sources.join("k.loom"), b"src").unwrap();
        let exe = dir.path().join("loom-compile");
        write(&exe, b"v1").unwrap();
        let first = Compiler::new(exe.display().to_string(), &sources, dir.path().join("c"))
            .tag("k", "s", &vec![])
            .unwrap();
        // a different size and mtime
        std::thread::sleep(std::time::Duration::from_millis(10));
        write(&exe, b"a longer v2").unwrap();
        let second = Compiler::new(exe.display().to_string(), &sources, dir.path().join("c"))
            .tag("k", "s", &vec![])
            .unwrap();
        assert_ne!(first, second);
    }

    #[test]
    fn every_tag_is_a_safe_file_name() {
        let dir = tempfile::tempdir().unwrap();
        let sources = dir.path().join("kernels");
        write(&sources.join("k.loom"), b"src").unwrap();
        let exe = dir.path().join("loom-compile");
        write(&exe, b"v").unwrap();
        let c = Compiler::new(exe.display().to_string(), &sources, dir.path().join("c"));
        // a float config carries '.', '-' and '+', and the sanitiser must keep the name usable
        let cfg: Cfg = vec![("h3.k.eps".into(), num(1e-6))];
        let tag = c.tag("k", "s", &cfg).unwrap();
        assert!(
            tag.chars()
                .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' || ch == '.'),
            "{tag}"
        );
        assert!(!tag.contains('/') && !tag.contains(' '));
    }

    #[test]
    fn a_missing_source_names_itself() {
        let dir = tempfile::tempdir().unwrap();
        let exe = dir.path().join("loom-compile");
        write(&exe, b"v").unwrap();
        let c = Compiler::new(
            exe.display().to_string(),
            dir.path().join("kernels"),
            dir.path(),
        );
        let message = c.tag("nope", "s", &vec![]).unwrap_err().to_string();
        assert!(message.contains("nope.loom"), "{message}");
    }

    #[test]
    fn a_missing_compiler_names_itself() {
        let dir = tempfile::tempdir().unwrap();
        let sources = dir.path().join("kernels");
        write(&sources.join("k.loom"), b"src").unwrap();
        let c = Compiler::new("definitely-not-on-path-h3", &sources, dir.path());
        let message = c.tag("k", "s", &vec![]).unwrap_err().to_string();
        assert!(message.contains("definitely-not-on-path-h3"), "{message}");
    }
}

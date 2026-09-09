//! Compiling Loom kernels through `loom-compile` into a shared cache, and loading them.
//!
//! One kernel per (stem, source text, symbol, backend, target, config, compiler binary): the cache
//! file name is a digest of all of them, so neither an edited kernel source nor a replaced
//! `loom-compile` ever reuses a stale binary. `hrx::loom` computes the digest and owns the cache; what
//! is here is the choice of source and configuration.
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

pub const BACKEND: &str = "amdgpu-hal";
pub const TARGET: &str = "gfx1151";

/// A kernel's configuration. `hrx::loom` canonicalises it into a `BTreeMap` before hashing it and
/// before spelling out the argv, so the order these pairs are built in is not observable — but a
/// key repeated in one `Cfg` would silently lose all but one of its values, which `config` refuses.
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

/// Model-specific source selection and loaded exports. Compilation, integrity,
/// process locks and publication are provided by the shared HRX crate.
pub struct Compiler {
    exe: String,
    sources: PathBuf,
    source_cache: Mutex<HashMap<String, Arc<str>>>,
    cache: PathBuf,
    compiler: OnceLock<hrx::loom::Compiler>,
    loaded: Mutex<HashMap<String, Arc<hrx::Kernel>>>,
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
            source_cache: Mutex::new(HashMap::new()),
            cache: cache.into(),
            compiler: OnceLock::new(),
            loaded: Mutex::new(HashMap::new()),
        }
    }
    fn compiler(&self) -> Result<&hrx::loom::Compiler> {
        if let Some(c) = self.compiler.get() {
            return Ok(c);
        }
        let explicit = if self.exe.is_empty() || self.exe == "loom-compile" {
            None
        } else {
            Some(Path::new(&self.exe))
        };
        let compiler = hrx::loom::Compiler::resolve(explicit)?;
        let _ = self.compiler.set(compiler);
        Ok(self.compiler.get().expect("compiler initialized"))
    }
    fn source(&self, stem: &str) -> Result<Arc<str>> {
        if let Some(source) = self
            .source_cache
            .lock()
            .expect("source cache poisoned")
            .get(stem)
        {
            return Ok(source.clone());
        }
        let path = self.sources.join(format!("{stem}.loom"));
        let source: Arc<str> = if self.sources.as_os_str().is_empty() {
            embedded_source(stem)
                .ok_or_else(|| Error::MissingSource(path.clone()))?
                .into()
        } else {
            std::fs::read_to_string(&path)
                .map_err(|_| Error::MissingSource(path.clone()))?
                .into()
        };
        self.source_cache
            .lock()
            .expect("source cache poisoned")
            .insert(stem.into(), source.clone());
        Ok(source)
    }
    /// Where compiled artifacts go. An empty `cache` means the caller expressed no preference, and
    /// gets the shared per-user one; resolving it here rather than at the call sites keeps a
    /// `Compiler::new(_, _, "")` from quietly writing a cache into the process's current directory.
    fn cache_dir(&self) -> Result<PathBuf> {
        if self.cache.as_os_str().is_empty() {
            Ok(hrx::bundle::cache_root()?.join("h3/kernels/hrx-v1"))
        } else {
            Ok(self.cache.join("hrx-v1"))
        }
    }

    pub fn tag(&self, stem: &str, symbol: &str, cfg: &Cfg) -> Result<String> {
        let source = self.source(stem)?;
        let mut request = hrx::loom::Request::new(&source, symbol);
        request.config = config(stem, cfg)?;
        Ok(self.compiler()?.key(&request)?)
    }
    pub fn get(
        &self,
        gpu: &hrx::Gpu,
        stem: &str,
        symbol: &str,
        cfg: &Cfg,
    ) -> Result<Arc<hrx::Kernel>> {
        let source = self.source(stem)?;
        let mut request = hrx::loom::Request::new(&source, symbol);
        request.config = config(stem, cfg)?;
        let compiler = self.compiler()?;
        let tag = compiler.key(&request)?;
        if let Some(kernel) = self
            .loaded
            .lock()
            .expect("compiler cache poisoned")
            .get(&tag)
        {
            return Ok(kernel.clone());
        }
        let path = compiler.compile(&request, &self.cache_dir()?)?;
        // Safety: verified artifact from trusted model source and configured compiler.
        let kernel = Arc::new(unsafe { gpu.load(&path, symbol)? });
        let mut loaded = self.loaded.lock().expect("compiler cache poisoned");
        Ok(loaded.entry(tag).or_insert(kernel).clone())
    }
}

/// A `Cfg` as the map the compiler hashes and spells out, rejecting a repeated key.
///
/// The map keeps one value per key, so a builder that pushed the same key twice would have all but
/// the last of them disappear — into the cache tag as well as into the argv, which means the kernel
/// that ran and the name it was filed under would agree with each other and with nothing else. It
/// costs one comparison per kernel built to know that never happens.
fn config(stem: &str, cfg: &Cfg) -> Result<std::collections::BTreeMap<String, String>> {
    let map: std::collections::BTreeMap<String, String> = cfg.iter().cloned().collect();
    if map.len() != cfg.len() {
        return Err(Error::Failed {
            stem: stem.to_string(),
            command: "a configuration key was given twice".into(),
        });
    }
    Ok(map)
}

include!(concat!(env!("OUT_DIR"), "/kernel_sources.rs"));

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
    fn embedded_sources_need_no_checkout_and_explicit_directories_are_honored() {
        let c = Compiler::new("unused", PathBuf::new(), PathBuf::new());
        assert!(c.source("gn_silu_f16").unwrap().contains("h3_gn_silu_f16"));
        let directory = tempfile::tempdir().unwrap();
        let c = Compiler::new("unused", directory.path(), PathBuf::new());
        assert!(c.source("gn_silu_f16").is_err());
    }

    #[test]
    fn a_key_given_twice_is_refused_rather_than_silently_dropped() {
        let one: Cfg = vec![
            ("h3.gemm.k_size".into(), "2048".into()),
            ("h3.gemm.n_size".into(), "4096".into()),
        ];
        assert_eq!(config("gemm", &one).unwrap().len(), 2);
        // the map would keep 4096 and lose 2048, in the argv and in the cache tag alike
        let twice: Cfg = vec![
            ("h3.gemm.k_size".into(), "2048".into()),
            ("h3.gemm.k_size".into(), "4096".into()),
        ];
        assert!(config("gemm", &twice).is_err());
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
        assert_eq!(base.len(), 64);

        // the symbol is in the identity, not the visible tag, so it changes the __i suffix
        let other_symbol = c.tag("gemm", "h3_other", &cfg).unwrap();
        assert_ne!(base, other_symbol);

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

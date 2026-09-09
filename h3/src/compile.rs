//! Compiling Loom kernels in process through the shared HRX library.
//!
//! One kernel per (stem, source text, symbol, backend, target, config, compiler binary): the cache
//! file name is a digest of all of them, so neither an edited kernel source nor a replaced
//! `libloomc` ever reuses a stale binary. `hrx::loom` computes the digest and owns the cache; what
//! is here is the choice of source and configuration.
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

/// A kernel's configuration. `hrx::loom` canonicalises it into a `BTreeMap` before hashing it and
/// before passing it to the native compiler, so the order these pairs are built in is not observable — but a
/// key repeated in one `Cfg` would silently lose all but one of its values, which `config` refuses.
pub type Cfg = Vec<(String, String)>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("missing kernel source {0}")]
    MissingSource(PathBuf),
    #[error("invalid configuration for {stem}: {message}")]
    Configuration { stem: String, message: String },
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
    library: Option<PathBuf>,
    modules: Mutex<HashMap<String, hrx::loom::Module>>,
    sources: PathBuf,
    source_cache: Mutex<HashMap<String, Arc<str>>>,
    cache: PathBuf,
    compiler: OnceLock<hrx::loom::Compiler>,
    loaded: Mutex<HashMap<String, Arc<hrx::Kernel>>>,
}
impl Compiler {
    pub fn new(
        library: Option<PathBuf>,
        sources: impl Into<PathBuf>,
        cache: impl Into<PathBuf>,
    ) -> Self {
        Self {
            library,
            modules: Mutex::new(HashMap::new()),
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
        let compiler = hrx::loom::Compiler::resolve(self.library.as_deref())?;
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

    fn module(&self, stem: &str) -> Result<hrx::loom::Module> {
        let mut modules = self.modules.lock().expect("module cache poisoned");
        if let Some(module) = modules.get(stem) {
            return Ok(module.clone());
        }
        let source = self.source(stem)?;
        let module = self.compiler()?.module(&source);
        modules.insert(stem.into(), module.clone());
        Ok(module)
    }
    pub fn tag(&self, stem: &str, symbol: &str, cfg: &Cfg) -> Result<String> {
        let mut request = hrx::loom::Specialization::new(symbol);
        request.config = config(stem, cfg)?;
        Ok(self.module(stem)?.key(&request)?)
    }

    pub fn get(
        &self,
        stream: &mut hrx::Stream,
        stem: &str,
        symbol: &str,
        cfg: &Cfg,
    ) -> Result<Arc<hrx::Kernel>> {
        let module = self.module(stem)?;
        let mut request = hrx::loom::Specialization::new(symbol);
        request.config = config(stem, cfg)?;
        let tag = module.key(&request)?;
        if let Some(kernel) = self
            .loaded
            .lock()
            .expect("compiler cache poisoned")
            .get(&tag)
        {
            return Ok(kernel.clone());
        }
        let artifact = module.compile(&request, &self.cache_dir()?)?;
        // Safety: verified artifact from trusted model source and configured compiler.
        let kernel = Arc::new(unsafe { stream.load_artifact(&artifact)? });
        let mut loaded = self.loaded.lock().expect("compiler cache poisoned");
        Ok(loaded.entry(tag).or_insert(kernel).clone())
    }
}

/// A `Cfg` as the map the compiler hashes and spells out, rejecting a repeated key.
///
/// The map keeps one value per key, so a builder that pushed the same key twice would have all but
/// the last of them disappear — into the cache tag as well as into specialization, which means the kernel
/// that ran and the name it was filed under would agree with each other and with nothing else. It
/// costs one comparison per kernel built to know that never happens.
fn config(stem: &str, cfg: &Cfg) -> Result<std::collections::BTreeMap<String, String>> {
    let map: std::collections::BTreeMap<String, String> = cfg.iter().cloned().collect();
    if map.len() != cfg.len() {
        return Err(Error::Configuration {
            stem: stem.to_string(),
            message: "a configuration key was given twice".into(),
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
        let c = Compiler::new(None, PathBuf::new(), PathBuf::new());
        assert!(c.source("gn_silu_f16").unwrap().contains("h3_gn_silu_f16"));
        let directory = tempfile::tempdir().unwrap();
        let c = Compiler::new(None, directory.path(), PathBuf::new());
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
    fn a_missing_source_names_itself() {
        let dir = tempfile::tempdir().unwrap();
        let library = dir.path().join("libloomc.so");
        write(&library, b"v").unwrap();
        let c = Compiler::new(
            Some(library.clone()),
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
        let c = Compiler::new(
            Some(PathBuf::from("definitely-not-on-path-h3")),
            &sources,
            dir.path(),
        );
        let message = c.tag("k", "s", &vec![]).unwrap_err().to_string();
        assert!(message.contains("definitely-not-on-path-h3"), "{message}");
    }
}

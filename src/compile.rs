//! Compiling Loom kernels in process through the shared HRX library.
//!
//! One kernel per (stem, source text, symbol, backend, target, config, compiler binary): the cache
//! file name is a digest of all of them, so neither an edited kernel source nor a replaced
//! `libloomc` ever reuses a stale binary. `hrx::loom` computes the digest and owns the cache; what
//! is here is the choice of source, configuration and target — the target being whatever the stream's
//! device reports, so a machine with a different GPU compiles for itself and files the result apart.
//!
//! Asking for a kernel and building it are separate steps. [`Compiler::get`] only records the
//! request; the outstanding set is compiled in parallel by [`Compiler::flush`], or by the first
//! launch that needs one of them. A stack asks for its forty-odd kernels before it builds any, so a
//! cold cache spends its time on several threads rather than one.
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

/// One kernel, asked for and not yet built.
///
/// Compiling is what a cold cache costs, and the requests are independent of one another, so
/// [`Compiler::get`] records the request and returns this rather than stopping to build it. The
/// outstanding set is then compiled together, on as many threads as the compiler allows, by
/// [`Compiler::flush`] or by the first launch that needs any one of them.
///
/// The queue behind it is `hrx::loom::Kernels`, which is where this crate's version of it went.
#[derive(Clone)]
pub struct Kernel(hrx::loom::Pending);

impl Kernel {
    /// The loaded kernel, if its batch has already been built.
    ///
    /// For callers with no stream to build one on — recording into a graph, which cannot load an
    /// executable. Everywhere else wants [`Kernel::resolve`].
    pub(crate) fn built(&self) -> Option<&hrx::Kernel> {
        self.0.built()
    }

    /// The loaded kernel, building the outstanding batch if this is the first call that needs it.
    pub(crate) fn resolve(&self, stream: &mut hrx::Stream) -> Result<&hrx::Kernel> {
        // Safety: every requested source is this repository's own, checked in or embedded.
        Ok(unsafe { self.0.resolve(stream) }?)
    }
}

/// How many kernels `hrx::loom::Compiler::compile_all` may build at once.
///
/// Each worker holds its own compiler workspace, so this trades memory for latency at a point where
/// the process is otherwise about to sit on the checkpoint's page faults anyway. Bounded rather than
/// taken whole: past a handful of threads the artifact cache and the allocator dominate.
fn workers() -> usize {
    std::thread::available_parallelism()
        .map_or(1, std::num::NonZeroUsize::get)
        .min(8)
}

#[derive(Hash, PartialEq, Eq)]
struct RequestKey {
    stem: String,
    symbol: String,
    config: Cfg,
}

/// Model-specific source selection and pending requests. HRX owns compilation,
/// artifact integrity and the loaded executable cache.
pub struct Compiler {
    library: Option<PathBuf>,
    sources: PathBuf,
    source_cache: Mutex<HashMap<String, Arc<str>>>,
    /// Built on the first request, for the target that request's stream reported.
    kernels: OnceLock<hrx::loom::KeyedKernels<RequestKey>>,
}
impl Compiler {
    pub fn new(library: Option<PathBuf>, sources: impl Into<PathBuf>) -> Self {
        Self {
            library,
            sources: sources.into(),
            source_cache: Mutex::new(HashMap::new()),
            kernels: OnceLock::new(),
        }
    }
    /// The cache, built on first use for `target`.
    ///
    /// The target selects the Loom profile — which descriptor sets a kernel's hand-written low asm
    /// may name — and it is part of the artifact cache key, so it has to be the architecture the
    /// kernels will actually run on. [`Compiler::get`] passes the stream's own, which is what the
    /// device reports; only [`Compiler::tag`], which has no device to ask, leaves it unset and takes
    /// the crate's default. Whichever comes first fixes it for this `Compiler`.
    fn kernels(
        &self,
        target: Option<&hrx::Target>,
    ) -> Result<&hrx::loom::KeyedKernels<RequestKey>> {
        if let Some(kernels) = self.kernels.get() {
            let built = kernels.kernels().compiler().target();
            if target.is_some_and(|target| target != built) {
                return Err(Error::Io(format!(
                    "compiler target {} differs from stream target {}",
                    built.as_str(),
                    target.expect("checked").as_str()
                )));
            }
            return Ok(kernels);
        }
        let options = hrx::loom::CompilerOptions {
            target: target.cloned().unwrap_or_default(),
            workers: std::num::NonZeroUsize::new(workers()).expect("at least one worker"),
            ..Default::default()
        };
        let compiler = hrx::loom::Compiler::shared(self.library.as_deref(), options)?;
        let _ = self.kernels.set(hrx::loom::Kernels::new(compiler).keyed());
        self.kernels(target)
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
    pub fn tag(&self, stem: &str, symbol: &str, cfg: &Cfg) -> Result<String> {
        let mut request = hrx::loom::Specialization::new(symbol);
        request.replace_config(config(stem, cfg)?);
        let source = self.source(stem)?;
        Ok(self
            .kernels(None)?
            .kernels()
            .compiler()
            .module(&source)
            .key(&request)?)
    }

    /// Ask for a kernel. What comes back is a handle, not yet a kernel: see [`Kernel`].
    ///
    /// The stream is read for its target and not otherwise touched, so a builder can ask for its
    /// whole set of kernels before the first of them is built. The same kernel asked for twice is
    /// compiled once and shared, which is why the stack's fifty blocks cost what one does.
    pub fn get(
        &self,
        stream: &mut hrx::Stream,
        stem: &str,
        symbol: &str,
        cfg: &Cfg,
    ) -> Result<Kernel> {
        let mut canonical = cfg.clone();
        canonical.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        if canonical.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Err(Error::Configuration {
                stem: stem.into(),
                message: "a configuration key was given twice".into(),
            });
        }
        let key = RequestKey {
            stem: stem.into(),
            symbol: symbol.into(),
            config: canonical,
        };
        // Sources are immutable within this Compiler. Equal keys include every
        // specialization input; hits skip source lookup and hashing entirely.
        let pending = unsafe {
            self.kernels(Some(stream.target()))?
                .request_or_insert_with(stream, key, |key| {
                    let source = self
                        .source(&key.stem)
                        .map_err(|e| hrx::Error::Message(e.to_string()))?;
                    let mut spec = hrx::loom::Specialization::new(&key.symbol);
                    spec.replace_config(key.config.iter().cloned().collect());
                    Ok((source, spec))
                })
        }?;
        Ok(Kernel(pending))
    }

    /// Build every kernel asked for so far.
    ///
    /// A builder calls this once it has asked for its whole set, so that a kernel that will not
    /// compile is reported where it was configured rather than at the launch that first needs it.
    pub fn flush(&self, stream: &mut hrx::Stream) -> Result<()> {
        // Safety: every requested source is this repository's own, checked in or embedded.
        Ok(unsafe { self.kernels(Some(stream.target()))?.kernels().build(stream) }?)
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
        let c = Compiler::new(None, PathBuf::new());
        assert!(c.source("gn_silu_f16").unwrap().contains("h3_gn_silu_f16"));
        let directory = tempfile::tempdir().unwrap();
        let c = Compiler::new(None, directory.path());
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

    /// A batch is not all-or-nothing for the requests behind the one that failed.
    ///
    /// The failing kernel is asked for first here, so everything after it is still queued when the
    /// batch gives up. Those handles are held by their callers and are still wanted: losing them
    /// would turn a kernel that builds perfectly well into a handle that can never be resolved and
    /// cannot say why.
    #[test]
    #[ignore = "requires gfx1151 and provisioned HRX"]
    fn a_failed_batch_still_builds_the_requests_behind_it() {
        let mut stream = hrx::Stream::open().expect("stream");
        let compiler = Compiler::new(None, "");
        let cfg: Cfg = vec![
            ("h3.gn_silu_f16.channels".into(), "128".into()),
            ("h3.gn_silu_f16.groups".into(), "32".into()),
            ("h3.gn_silu_f16.plane".into(), "64".into()),
            ("h3.gn_silu_f16.rows_bound".into(), "64".into()),
            ("h3.gn_silu_f16.eps".into(), num(1e-5)),
        ];
        let doomed = compiler
            .get(&mut stream, "gn_silu_f16", "h3_no_such_export", &cfg)
            .expect("a request is recorded without being built");
        let wanted = compiler
            .get(&mut stream, "gn_silu_f16", "h3_gn_silu_f16", &cfg)
            .expect("a request is recorded without being built");

        let message = compiler.flush(&mut stream).unwrap_err().to_string();
        assert!(message.contains("h3_no_such_export"), "{message}");
        // the one behind it builds, and the failure is still the failure it was
        assert!(wanted.resolve(&mut stream).is_ok(), "{message}");
        let again = doomed.resolve(&mut stream).unwrap_err().to_string();
        assert!(again.contains("h3_no_such_export"), "{again}");
    }

    #[test]
    #[ignore = "requires the provisioned Loom compiler"]
    fn a_built_cache_rejects_a_different_stream_target() {
        let compiler = Compiler::new(None, "");
        let first = hrx::Target::new("gfx1151").unwrap();
        let other = hrx::Target::new("gfx1100").unwrap();
        compiler.kernels(Some(&first)).unwrap();
        let error = match compiler.kernels(Some(&other)) {
            Err(error) => error,
            Ok(_) => panic!("a built cache concealed a target mismatch"),
        };
        assert!(error.to_string().contains("differs from stream target"));
    }

    #[test]
    fn a_missing_source_names_itself() {
        let dir = tempfile::tempdir().unwrap();
        let library = dir.path().join("libloomc.so");
        write(&library, b"v").unwrap();
        let c = Compiler::new(Some(library.clone()), dir.path().join("kernels"));
        let message = c.tag("nope", "s", &vec![]).unwrap_err().to_string();
        assert!(message.contains("nope.loom"), "{message}");
    }

    #[test]
    fn a_missing_compiler_names_itself() {
        let dir = tempfile::tempdir().unwrap();
        let sources = dir.path().join("kernels");
        write(&sources.join("k.loom"), b"src").unwrap();
        let c = Compiler::new(Some(PathBuf::from("definitely-not-on-path-h3")), &sources);
        let message = c.tag("k", "s", &vec![]).unwrap_err().to_string();
        assert!(message.contains("definitely-not-on-path-h3"), "{message}");
    }
}

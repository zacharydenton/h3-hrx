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
/// [`Compiler::flush`] or by the first launch that needs any one of them. A handle is a pair of
/// pointers: copy it into as many structures as the kernel is used from.
#[derive(Clone)]
pub struct Kernel {
    cell: Arc<OnceLock<hrx::Kernel>>,
    queue: Arc<Queue>,
}

impl Kernel {
    /// The loaded kernel, if its batch has already been built.
    ///
    /// For callers with no stream to build one on — recording into a graph, which cannot load an
    /// executable. Everywhere else wants [`Kernel::resolve`].
    pub(crate) fn built(&self) -> Option<&hrx::Kernel> {
        self.cell.get()
    }

    /// The loaded kernel, building the outstanding batch if this is the first call that needs it.
    pub(crate) fn resolve(&self, stream: &mut hrx::Stream) -> Result<&hrx::Kernel> {
        if let Some(kernel) = self.cell.get() {
            return Ok(kernel);
        }
        // A batch reports the first kernel in it that would not build, which is not necessarily this
        // one — so ask for the cell again before passing that failure on as though it were ours.
        let outcome = self.queue.flush(stream);
        if let Some(kernel) = self.cell.get() {
            return Ok(kernel);
        }
        outcome?;
        Err(Error::Io(
            "a kernel outlived the compiler that queued it".into(),
        ))
    }
}

/// One outstanding request: what to compile, and the handle waiting for it.
struct Request {
    tag: String,
    module: hrx::loom::Module,
    spec: hrx::loom::Specialization,
    cells: Vec<Arc<OnceLock<hrx::Kernel>>>,
}

/// The requests not yet built, and the kernels already loaded.
///
/// Shared by the compiler and every handle it has issued, so a handle can build its own batch
/// without holding a borrow of the compiler and without a lifetime of its own.
struct Queue {
    /// Built on the first request, for the target that request's stream reported.
    compiler: OnceLock<hrx::loom::Compiler>,
    pending: Mutex<Vec<Request>>,
    loaded: Mutex<HashMap<String, hrx::Kernel>>,
}

impl Queue {
    /// Build everything outstanding: compile off the device, then load in order.
    ///
    /// Compilation is pure — source, configuration and target to bytes — so it runs on several
    /// threads at once. Loading is not: it takes the stream, and the stream is not shared.
    fn flush(&self, stream: &mut hrx::Stream) -> Result<()> {
        let requests = std::mem::take(&mut *self.pending.lock().expect("kernel queue poisoned"));
        if requests.is_empty() {
            return Ok(());
        }
        let batch: Vec<(&hrx::loom::Module, &hrx::loom::Specialization)> =
            requests.iter().map(|r| (&r.module, &r.spec)).collect();
        let artifacts = self
            .compiler
            .get()
            .expect("a request was recorded, so its compiler exists")
            .compile_all(&batch);
        let mut outcome = Ok(());
        let mut unbuilt = Vec::new();
        {
            let mut loaded = self.loaded.lock().expect("compiler cache poisoned");
            for (request, artifact) in requests.into_iter().zip(artifacts) {
                // Safety: verified artifact from trusted model source and configured compiler.
                let built = artifact
                    .map_err(Error::from)
                    .and_then(|a| Ok(unsafe { stream.load_artifact(&a) }?));
                let kernel = match built {
                    Ok(kernel) => kernel,
                    Err(error) => {
                        // A kernel that will not build is that kernel's failure and no one else's, so
                        // the batch carries on and the rest of it is loaded. This one goes back on the
                        // queue still wanted: dropping it would leave the handles that asked for it
                        // holding an empty cell with nothing to say about why, where returning it
                        // means the next attempt reports the same failure again.
                        unbuilt.push(request);
                        if outcome.is_ok() {
                            outcome = Err(error);
                        }
                        continue;
                    }
                };
                let kernel = loaded.entry(request.tag).or_insert(kernel).clone();
                for cell in request.cells {
                    let _ = cell.set(kernel.clone());
                }
            }
        }
        if !unbuilt.is_empty() {
            let mut pending = self.pending.lock().expect("kernel queue poisoned");
            unbuilt.append(&mut pending);
            *pending = unbuilt;
        }
        outcome
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

/// Model-specific source selection and loaded exports. Compilation, integrity,
/// process locks and publication are provided by the shared HRX crate.
pub struct Compiler {
    library: Option<PathBuf>,
    modules: Mutex<HashMap<String, hrx::loom::Module>>,
    sources: PathBuf,
    source_cache: Mutex<HashMap<String, Arc<str>>>,
    queue: Arc<Queue>,
}
impl Compiler {
    pub fn new(library: Option<PathBuf>, sources: impl Into<PathBuf>) -> Self {
        Self {
            library,
            modules: Mutex::new(HashMap::new()),
            sources: sources.into(),
            source_cache: Mutex::new(HashMap::new()),
            queue: Arc::new(Queue {
                compiler: OnceLock::new(),
                pending: Mutex::new(Vec::new()),
                loaded: Mutex::new(HashMap::new()),
            }),
        }
    }
    /// The compiler, built on first use for `target`.
    ///
    /// The target selects the Loom profile — which descriptor sets a kernel's hand-written low asm
    /// may name — and it is part of the artifact cache key, so it has to be the architecture the
    /// kernels will actually run on. [`Compiler::get`] passes the stream's own, which is what the
    /// device reports; only [`Compiler::tag`], which has no device to ask, leaves it unset and takes
    /// the crate's default. Whichever comes first fixes it for this `Compiler`.
    fn compiler(&self, target: Option<&hrx::Target>) -> Result<&hrx::loom::Compiler> {
        if let Some(c) = self.queue.compiler.get() {
            if target.is_some_and(|target| target != c.target()) {
                return Err(Error::Io(format!(
                    "compiler target {} differs from stream target {}",
                    c.target().as_str(),
                    target.unwrap().as_str()
                )));
            }
            return Ok(c);
        }
        let options = hrx::loom::CompilerOptions {
            target: target.cloned().unwrap_or_default(),
            workers: std::num::NonZeroUsize::new(workers()).expect("at least one worker"),
            ..Default::default()
        };
        let compiler = hrx::loom::Compiler::with_options(self.library.as_deref(), options)?;
        let _ = self.queue.compiler.set(compiler);
        self.compiler(target)
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
    fn module(&self, target: Option<&hrx::Target>, stem: &str) -> Result<hrx::loom::Module> {
        let mut modules = self.modules.lock().expect("module cache poisoned");
        if let Some(module) = modules.get(stem) {
            self.compiler(target)?;
            return Ok(module.clone());
        }
        let source = self.source(stem)?;
        let module = self.compiler(target)?.module(&source);
        modules.insert(stem.into(), module.clone());
        Ok(module)
    }
    pub fn tag(&self, stem: &str, symbol: &str, cfg: &Cfg) -> Result<String> {
        let mut request = hrx::loom::Specialization::new(symbol);
        request.config = config(stem, cfg)?;
        Ok(self.module(None, stem)?.key(&request)?)
    }

    /// Ask for a kernel. What comes back is a handle, not yet a kernel: see [`Kernel`].
    ///
    /// The stream is read for its target and not otherwise touched, so a builder can ask for its
    /// whole set of kernels before the first of them is built.
    pub fn get(
        &self,
        stream: &mut hrx::Stream,
        stem: &str,
        symbol: &str,
        cfg: &Cfg,
    ) -> Result<Kernel> {
        let module = self.module(Some(stream.target()), stem)?;
        let mut spec = hrx::loom::Specialization::new(symbol);
        spec.config = config(stem, cfg)?;
        let tag = module.key(&spec)?;
        let cell = Arc::new(OnceLock::new());
        let handle = Kernel {
            cell: cell.clone(),
            queue: self.queue.clone(),
        };
        if let Some(kernel) = self
            .queue
            .loaded
            .lock()
            .expect("compiler cache poisoned")
            .get(&tag)
        {
            let _ = cell.set(kernel.clone());
            return Ok(handle);
        }
        let mut pending = self.queue.pending.lock().expect("kernel queue poisoned");
        // The same kernel asked for twice within one batch is compiled once and shared, which is why
        // the stack's fifty blocks cost what one does.
        if let Some(request) = pending.iter_mut().find(|r| r.tag == tag) {
            request.cells.push(cell);
        } else {
            pending.push(Request {
                tag,
                module,
                spec,
                cells: vec![cell],
            });
        }
        Ok(handle)
    }

    /// Build every kernel asked for so far.
    ///
    /// A builder calls this once it has asked for its whole set, so that a kernel that will not
    /// compile is reported where it was configured rather than at the launch that first needs it.
    pub fn flush(&self, stream: &mut hrx::Stream) -> Result<()> {
        self.queue.flush(stream)
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
    fn cached_modules_reject_a_different_stream_target() {
        let compiler = Compiler::new(None, "");
        let first = hrx::Target::new("gfx1151").unwrap();
        let other = hrx::Target::new("gfx1100").unwrap();
        compiler.module(Some(&first), "prepare_i8_family").unwrap();
        let error = match compiler.module(Some(&other), "prepare_i8_family") {
            Err(error) => error,
            Ok(_) => panic!("a cached module concealed a target mismatch"),
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

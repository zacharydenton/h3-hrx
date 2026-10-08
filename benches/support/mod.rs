//! Shared benchmark inputs and measurement boundaries; no production behavior lives here.
#![allow(dead_code)] // Each benchmark executable uses a different subset.
use criterion::{Bencher, Criterion};
use h3_hrx::{Config, DenoiseParams, ResidencyPolicy, Session, SessionOptions};
use std::{
    path::PathBuf,
    sync::OnceLock,
    time::{Duration, Instant},
};

pub const PROMPT: &str = "A red fox walks through a snowy forest at dawn. The camera slowly follows. Soft wind and footsteps in snow.";

pub fn prompt() -> &'static str {
    static TEXT: OnceLock<String> = OnceLock::new();
    TEXT.get_or_init(|| match std::env::var_os("H3_BENCH_PROMPT_FILE") {
        Some(path) => {
            let text = std::fs::read_to_string(path).expect("read H3_BENCH_PROMPT_FILE");
            assert!(
                !text.trim().is_empty(),
                "benchmark prompt must not be empty"
            );
            text
        }
        None => PROMPT.into(),
    })
}

pub fn criterion() -> Criterion {
    // Keep filters unchanged, but never compare different queue/compiler modes
    // against one another automatically. Match Criterion's output-root lookup.
    let root = std::env::var_os("CRITERION_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let target = std::env::var_os("CARGO_TARGET_DIR")
                .map(PathBuf::from)
                .or_else(|| {
                    let output = std::process::Command::new(std::env::var_os("CARGO")?)
                        .args(["metadata", "--no-deps", "--format-version", "1"])
                        .output()
                        .ok()?;
                    let metadata: serde_json::Value =
                        serde_json::from_slice(&output.stdout).ok()?;
                    Some(PathBuf::from(metadata["target_directory"].as_str()?))
                })
                .unwrap_or_else(|| PathBuf::from("target"));
            target.join("criterion")
        });
    let io = weight_io();
    let storage = hrx::storage::StorageConfig::default();
    let mut output = root.join(engine_settings().id()).join(format!(
        "{:?}-{:?}-slots{}-bytes{}-stats{}",
        io.mode,
        io.progress,
        io.slots.map_or(storage.slots, std::num::NonZeroUsize::get),
        io.slot_bytes
            .map_or(storage.slot_bytes, std::num::NonZeroUsize::get),
        io.statistics,
    ));
    // Instrumentation changes dispatch ordering and adds waits/markers. Never
    // overwrite or compare ordinary latency estimates with diagnostic samples.
    let profile = std::env::var_os("H3_PROFILE");
    if profile.as_ref().is_some_and(|v| !v.is_empty() && v != "0") {
        output = output.join(
            if profile.as_deref() == Some(std::ffi::OsStr::new("device")) {
                "profile-device"
            } else {
                "profile-host"
            },
        );
    }
    Criterion::default()
        .output_directory(&output)
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
}

pub fn checkpoint(relative: &str) -> PathBuf {
    let path = match std::env::var_os("H3_BENCH_MODELS") {
        Some(root) => PathBuf::from(root).join(relative),
        None => h3_hrx::models::Resolver::new().find(relative).unwrap(),
    };
    assert!(path.is_file(), "missing checkpoint {}", path.display());
    path
}

pub fn compiler() -> h3_hrx::compile::Compiler {
    h3_hrx::compile::Compiler::with_options(None, sources(), compiler_options())
}

struct EngineSettings {
    gpu: i32,
    compute: String,
    copy: String,
    private_bytes: u32,
    processor: String,
    workers: Option<std::num::NonZeroUsize>,
}

impl EngineSettings {
    fn id(&self) -> String {
        format!(
            "gpu{}-{}-{}-scratch{}-{}-workers{}",
            self.gpu,
            self.compute,
            self.copy,
            self.private_bytes,
            self.processor,
            self.workers
                .map_or_else(|| "auto".into(), |n| n.to_string())
        )
    }
}

fn engine_settings() -> &'static EngineSettings {
    static SETTINGS: OnceLock<EngineSettings> = OnceLock::new();
    SETTINGS.get_or_init(|| {
        let value = |key, default: &str| std::env::var(key).unwrap_or_else(|_| default.into());
        let settings = EngineSettings {
            gpu: value("H3_BENCH_GPU", "0")
                .parse()
                .expect("integer GPU ordinal"),
            compute: value("H3_BENCH_COMPUTE_ENGINE", "pm4"),
            copy: value("H3_BENCH_COPY_ENGINE", "compute"),
            private_bytes: value("H3_BENCH_AQL_PRIVATE_BYTES", "4096")
                .parse()
                .expect("unsigned AQL scratch ceiling"),
            processor: value("H3_BENCH_PROCESSOR_MODE", "default"),
            workers: std::env::var("H3_BENCH_COMPILE_WORKERS")
                .ok()
                .map(|n| n.parse().expect("positive compiler worker count")),
        };
        assert!(settings.gpu >= 0, "H3_BENCH_GPU must be nonnegative");
        assert!(
            matches!(settings.compute.as_str(), "pm4" | "aql"),
            "H3_BENCH_COMPUTE_ENGINE must be pm4 or aql"
        );
        assert!(
            matches!(settings.copy.as_str(), "compute" | "sdma"),
            "H3_BENCH_COPY_ENGINE must be compute or sdma"
        );
        assert!(
            matches!(settings.processor.as_str(), "default" | "cu" | "wgp"),
            "H3_BENCH_PROCESSOR_MODE must be default, cu or wgp"
        );
        settings
    })
}

pub fn compiler_options() -> h3_hrx::compile::Options {
    let settings = engine_settings();
    h3_hrx::compile::Options {
        workers: settings.workers,
        processor_mode: match settings.processor.as_str() {
            "cu" => hrx::loom::ProcessorMode::ComputeUnit,
            "wgp" => hrx::loom::ProcessorMode::WorkgroupProcessor,
            _ => hrx::loom::ProcessorMode::Default,
        },
        ..Default::default()
    }
}

pub fn weight_io() -> h3_hrx::weights::WeightIo {
    use h3_hrx::weights::{WeightIo, WeightIoMode};
    let capacity = |key: &str, unit: usize| {
        std::env::var(key).ok().map(|value| {
            let n: usize = value.parse().expect("integer storage capacity");
            assert!((1..=64).contains(&n), "{key} must be 1..=64");
            std::num::NonZeroUsize::new(n * unit).unwrap()
        })
    };
    WeightIo {
        mode: match std::env::var("H3_BENCH_WEIGHT_IO")
            .as_deref()
            .unwrap_or("mapped")
        {
            "mapped" => WeightIoMode::Mapped,
            "native-buffered" => WeightIoMode::NativeBuffered,
            "native-direct" => WeightIoMode::NativeDirect,
            _ => panic!("H3_BENCH_WEIGHT_IO must be mapped, native-buffered, or native-direct"),
        },
        progress: match std::env::var("H3_BENCH_STORAGE_PROGRESS")
            .as_deref()
            .unwrap_or("sqpoll")
        {
            "sqpoll" => hrx::storage::StorageProgress::Sqpoll,
            "wait" => hrx::storage::StorageProgress::Wait,
            _ => panic!("H3_BENCH_STORAGE_PROGRESS must be sqpoll or wait"),
        },
        statistics: std::env::var_os("H3_BENCH_STORAGE_STATISTICS").is_some(),
        slots: capacity("H3_BENCH_STORAGE_SLOTS", 1),
        slot_bytes: capacity("H3_BENCH_STORAGE_SLOT_MIB", 1 << 20),
    }
}

pub fn configure_weights(mut weights: h3_hrx::weights::Weights) -> h3_hrx::weights::Weights {
    weights.set_io(weight_io(), &compiler()).unwrap();
    weights
}

pub fn runtime_options(
    memory_budget: Option<hrx::residency::MemoryBudget>,
) -> hrx::execution::RuntimeOptions {
    let settings = engine_settings();
    hrx::execution::RuntimeOptions {
        gpu_index: settings.gpu,
        native_lifetime: weight_io().native_lifetime(),
        compute_engine: if settings.compute == "aql" {
            hrx::execution::ComputeEngine::Aql {
                maximum_private_bytes: settings.private_bytes,
            }
        } else {
            hrx::execution::ComputeEngine::Pm4
        },
        copy_engine: if settings.copy == "sdma" {
            hrx::execution::CopyEngine::Sdma
        } else {
            hrx::execution::CopyEngine::Compute
        },
        memory_budget,
        ..Default::default()
    }
}

pub fn stream(budget: Option<hrx::residency::MemoryBudget>) -> hrx::Stream {
    let options = runtime_options(budget);
    hrx::Device::open_with_lifetime(options.gpu_index, options.native_lifetime)
        .unwrap()
        .stream_with_options(hrx::StreamOptions {
            compute_engine: options.compute_engine,
            copy_engine: options.copy_engine,
            memory_budget: options.memory_budget,
            ..Default::default()
        })
        .unwrap()
}

/// Validation readback uses the same completed host access as H3 stages. Avoid
/// a second GPU allocation/copy, whose raw SDMA command may exceed the ring.
pub fn read(stream: &mut hrx::Stream, view: hrx::View<'_>) -> Vec<u8> {
    let mut bytes = vec![0; view.len()];
    stream.read_blocking(view, &mut bytes).unwrap();
    bytes
}

pub fn configure_cli(command: &mut std::process::Command) {
    let settings = engine_settings();
    command
        .arg("--gpu")
        .arg(settings.gpu.to_string())
        .args([
            "--compute-engine",
            &settings.compute,
            "--copy-engine",
            &settings.copy,
            "--processor-mode",
            &settings.processor,
        ])
        .arg("--aql-private-bytes")
        .arg(settings.private_bytes.to_string());
    command
        .arg("--weight-io")
        .arg(std::env::var("H3_BENCH_WEIGHT_IO").unwrap_or_else(|_| "mapped".into()))
        .arg("--storage-progress")
        .arg(std::env::var("H3_BENCH_STORAGE_PROGRESS").unwrap_or_else(|_| "sqpoll".into()));
    if let Some(workers) = settings.workers {
        command.arg("--compile-workers").arg(workers.to_string());
    }
    let io = weight_io();
    if let Some(slots) = io.slots {
        command.arg("--storage-slots").arg(slots.to_string());
    }
    if let Some(bytes) = io.slot_bytes {
        command
            .arg("--storage-slot-mib")
            .arg((bytes.get() >> 20).to_string());
    }
    if io.statistics {
        command.arg("--weight-io-statistics");
    }
}
pub fn sources() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("kernels")
}

pub fn config(references: bool) -> Config {
    // Resolve before timing, including stages that load weights lazily.
    Config {
        dit: Some(checkpoint(if references {
            h3_hrx::models::DIT_REF2VA
        } else {
            h3_hrx::models::DIT_FL2VA
        })),
        te: Some(checkpoint(h3_hrx::models::TE)),
        video_vae: Some(checkpoint(h3_hrx::models::VIDEO_VAE)),
        audio_vae: Some(checkpoint(h3_hrx::models::AUDIO_VAE)),
        kernel_sources: sources(),
        attention: attention(),
        compiler: compiler_options(),
        weight_io: weight_io(),
        ..Default::default()
    }
}

pub fn attention() -> h3_hrx::Attention {
    match std::env::var("H3_BENCH_ATTN").as_deref().unwrap_or("i8") {
        "i8" => h3_hrx::Attention::from_bits(8).unwrap(),
        "f16" => h3_hrx::Attention::from_bits(16).unwrap(),
        "i4" => h3_hrx::Attention::from_bits(4).unwrap(),
        _ => panic!("H3_BENCH_ATTN must be i8, f16 or i4"),
    }
}

pub fn budget_bytes() -> usize {
    budget_bytes_or(64)
}

pub fn budget_bytes_or(default_gib: usize) -> usize {
    let gib: usize = std::env::var("H3_BENCH_BUDGET_GIB")
        .map(|s| s.parse().expect("integer GiB budget"))
        .unwrap_or(default_gib);
    assert!((1..=1024).contains(&gib));
    gib << 30
}

pub fn residency() -> ResidencyPolicy {
    match std::env::var("H3_BENCH_RESIDENCY")
        .as_deref()
        .unwrap_or("budgeted")
    {
        "stage-scoped" => ResidencyPolicy::StageScoped,
        "budgeted" => ResidencyPolicy::Budgeted,
        "retain" => ResidencyPolicy::Retain,
        _ => panic!("H3_BENCH_RESIDENCY must be stage-scoped, budgeted or retain"),
    }
}

pub fn execution_id() -> String {
    let graph = std::env::var("H3_GRAPH").is_ok_and(|s| !s.is_empty() && s != "0");
    format!(
        "{}bit/{}/{}GiB",
        attention().bits(),
        if graph { "graph" } else { "eager" },
        budget_bytes() >> 30
    )
}

pub fn params() -> (String, DenoiseParams) {
    let profile = std::env::var("H3_BENCH_PROFILE").unwrap_or_else(|_| "smoke".into());
    let (width, height, frames, steps) = match profile.as_str() {
        "smoke" => (64, 64, 5, 4),
        "480p" => (864, 480, 124, 21),
        "768p" => (1344, 768, 124, 21),
        _ => panic!("H3_BENCH_PROFILE must be smoke, 480p or 768p"),
    };
    let steps: usize = std::env::var("H3_BENCH_STEPS")
        .map(|s| s.parse().expect("integer sigma-point count"))
        .unwrap_or(steps);
    assert!(
        (2..=1000).contains(&steps),
        "H3_BENCH_STEPS must be 2..=1000"
    );
    let seed: u64 = std::env::var("H3_BENCH_SEED")
        .map(|s| s.parse().expect("unsigned integer seed"))
        .unwrap_or(7);
    let digest = hrx::bundle::digest(prompt().as_bytes());
    (
        format!("{profile}/steps{steps}/seed{seed}/prompt-{}", &digest[..12]),
        DenoiseParams {
            width,
            height,
            frames,
            steps,
            seed,
            ..Default::default()
        },
    )
}

pub struct Runtime {
    pub manager: hrx::residency::ResidencyManager,
    pub context: hrx::inference::ModelContext,
}
impl Runtime {
    pub fn new() -> Self {
        let manager = hrx::residency::ResidencyManager::new(budget_bytes()).unwrap();
        let context =
            hrx::inference::ModelContext::new(runtime_options(Some(manager.budget()))).unwrap();
        Self { manager, context }
    }
    pub fn session(&self, mut config: Config, options: SessionOptions) -> Session {
        config.compiler = compiler_options();
        // SAFETY: benchmark checkpoints and adapters must remain immutable for the process lifetime.
        unsafe { Session::new_in(config, options, &self.context) }.unwrap()
    }
    pub fn warm_session(&self, references: bool) -> Session {
        self.session(
            config(references),
            SessionOptions {
                residency: residency(),
                ..Default::default()
            },
        )
    }
}

pub fn values(n: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i.wrapping_mul(37) % 257) as f32 / 128. - 1.) * scale)
        .collect()
}
pub fn pixels(width: usize, height: usize, frames: usize) -> Vec<f32> {
    (0..width * height * frames * 3)
        .map(|i| ((i * 17 + i / (width * 3) * 13) % 251) as f32 / 250.)
        .collect()
}
pub fn finite(values: &[f32]) {
    assert!(!values.is_empty());
    assert!(
        values.iter().all(|x| x.is_finite()),
        "nonfinite benchmark output"
    );
}
pub fn digest(values: &[f32]) -> String {
    finite(values);
    hrx::bundle::digest(bytemuck::cast_slice(values))
}

/// Outside timing: compare identical workloads across native engines/compilers.
pub fn report_digest(name: &str, bytes: &[u8]) {
    if std::env::var_os("H3_BENCH_DETAILS").is_some() {
        eprintln!("{name} output sha256: {}", hrx::bundle::digest(bytes));
    }
}

pub fn report_elapsed(name: &str, iterations: u64, elapsed: Duration) {
    if std::env::var_os("H3_BENCH_DETAILS").is_some() {
        eprintln!(
            "{name}: {iterations} completed forwards in {:.6} s",
            elapsed.as_secs_f64()
        );
    }
}

/// The API returns completed host output. Validation and disposal are outside timing.
/// Factories are kept in an Option by callers so setup runs once, after Criterion filtering.
pub fn measure<T>(b: &mut Bencher, mut run: impl FnMut() -> T, mut check: impl FnMut(&T)) {
    b.iter_custom(|iterations| {
        let mut total = Duration::ZERO;
        for _ in 0..iterations {
            let start = Instant::now();
            let output = run();
            total += start.elapsed();
            check(&output);
        }
        total
    });
}

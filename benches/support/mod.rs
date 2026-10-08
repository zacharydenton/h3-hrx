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
    Criterion::default()
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
    h3_hrx::compile::Compiler::new(None, sources())
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
        let context = hrx::inference::ModelContext::new(hrx::execution::RuntimeOptions {
            memory_budget: Some(manager.budget()),
            ..Default::default()
        })
        .unwrap();
        Self { manager, context }
    }
    pub fn session(&self, config: Config, options: SessionOptions) -> Session {
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

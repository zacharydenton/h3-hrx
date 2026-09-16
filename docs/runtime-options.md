# Session lifecycle and experimental acceleration

The `h3` CLI releases completed model stages by default. Conditioning runs with
the text encoder, sampling retains the full DiT, and final decoding loads the
VAEs after releasing the DiT. A stream fence precedes each release. No per-layer
weight eviction occurs during sampling. Use `--residency retain` to keep models
resident within the process.

Existing library callers preserve their behavior: `Session::new(Config)` retains
models for subsequent requests. Choose stage-scoped ownership explicitly:

```rust,no_run
use h3_hrx::{Config, ResidencyPolicy, Session, SessionOptions};

// Safety: keep every resolved checkpoint immutable while the session lives.
let mut session = unsafe {
    Session::new_with_options(Config::default(), SessionOptions {
        residency: ResidencyPolicy::StageScoped,
        ..SessionOptions::default()
    })?
};
# Ok::<(), h3_hrx::Error>(())
```

Input images/audio can be encoded before sampling. The session releases those
input VAEs when denoising begins. Cancellation releases completed stage owners
and permits a later request on the same session. An unsuccessful stream fence
retains allocations that may still be in flight.

## Turbo qualification

The experimental `SessionOptions::turbo` path keeps the existing INT8 base and
adds BF16 low-rank branches. Adapters resolve through the standard Hugging Face
Hub cache at pinned revisions; `h3-dev inspect-adapter --offline turbo-768p-4`
validates the cached four-evaluation checkpoint without opening the GPU.

Four/eight-evaluation presets require 1344×768, 124 frames, Euler, and video/audio
shifts 6/3. They support text or one first-frame keyframe. The corresponding sigma
grids contain five/nine points including terminal zero. Both presets include
all fifty DiT blocks and both refiner blocks. They reject unsupported checkpoint,
attention, reference, and cache combinations before loading models.

The CLI presets remain hidden during numerical and perceptual qualification.
The base preset, default sampler, and default evaluation count are unchanged.
Do not infer quality equivalence or a measured speedup from the evaluation count.

## Cache qualification

`SessionOptions::cache` accepts `Off`, `Observe`, or `Conservative` with separate
conditioning/audio/video thresholds. `Observe` measures changes and runs every
block. Conservative reuse requires all three accumulated relative L1 changes to
pass their thresholds; the first two and last two evaluations always run in full,
and consecutive skips are forbidden. Nonfinite metrics force full computation.
Caching remains off by default and cannot be combined with Turbo.

The legacy positive `DenoiseParams::cache_threshold` maps one value to all three
thresholds under the hardened policy. Its old early/consecutive-skip behavior is
intentionally removed. Explicit policies cannot be combined with a positive
legacy threshold. Thresholds require calibration against saved noise, motion,
and audio before a public CLI preset is recommended.

## Reproducible diagnostics

`H3_STAGE_TRACE=1` emits JSON stage events for checkpoint identity, actual packed
rows, sigma arrays, cache decisions, and model release. It does not enable the
intrusive kernel profiler. Host submission timestamps can overlap GPU work;
release events follow synchronization.

`scripts/optimization_benchmark.py` records source/binary/prompt identity,
complete process time, and separate GPU-residency, process-PSS, and system-memory
views. These UMA views overlap and must not be added.

`h3-dev tune-gemm` screens groups 2/3/4/8 at the actual 38,048/40,047-row shapes
using four real layer weights, exact output checks, warmup, and alternating
measurements. `h3-dev compile-probes` checks the operand/attention experiments
without opening the GPU. Developer switches `H3_FUSED_OPERANDS=1` and
`H3_REUSE_SCRATCH=1` enable pending operand-fusion and scratch-lifetime experiments;
neither changes the default until its parity and timing gates pass.

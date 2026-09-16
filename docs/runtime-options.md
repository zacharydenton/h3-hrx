# Session lifecycle and experimental acceleration

The `h3` CLI releases completed model stages by default. Conditioning runs with
the text encoder, sampling retains the full DiT, and final decoding loads the
VAEs after releasing the DiT. A stream fence precedes each release. No per-layer
weight eviction occurs during sampling. Use `--residency retain` to keep models
resident within the process.

`Session::new(Config)` retains
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

Input images/audio can be encoded before sampling. Stage-scoped encode and decode
calls release their finished VAE units after fencing. Cancellation releases completed stage owners
and permits a later request on the same session. An unsuccessful stream fence
retains allocations that may still be in flight.

## Shared allocation budgets

`Session::new_in(config, options, &context)` selects the context's GPU and
optional `RuntimeOptions::memory_budget`. All four lazy units (text encoder,
DiT, video VAE and audio VAE) charge native weights, growing workspace and
upload/readback staging before allocation. Charges survive queued uses and
recorded graphs, and are released only when their storage is safe to destroy.
Budget exhaustion returns an error; it does not offload work to CPU.

Use `ResidencyManager::budget()` to share the ceiling with other HRX models or
corpora. `Retain` pins session models; `StageScoped` releases completed stages.
`Budgeted` retains each idle model in the shared manager's LRU cache and reloads
it if another allocation evicts it. Active stages are never evicted, and there
is no per-layer paging. CLI callers can choose an explicit ceiling:

```sh
h3 --residency budgeted --memory-budget-mib 81920 --prompt "a red fox in snow"
```

The ceiling is a caller choice, not an estimate of every workload's requirements.
Library callers select `ResidencyPolicy::Budgeted` with `Session::new_in` and a
live `ResidencyManager`-backed budget. Cache units retain their original private
stream identity; they are not shared mutable models across sessions. Completed
or cancelled stages fence before returning units to the cache. Session teardown
unregisters all four units so mapped checkpoints do not outlive its safety contract.

StageScoped leaves the native stream's bounded staging cache resident between
calls; dropping the session releases it too. Compiler/code-object memory, native
allocator rounding and checkpoint mappings are not part of this byte ceiling.
Each public inference stage uses HRX's `NativeSession` to reserve the context's
compute lane and replay its original stream-bound graphs. Earlier submissions
drain first; later compute cannot overtake the stage. Other transfer/NPU lanes
remain available. Native transfers inside the stage remain on its private stream
and are not included in coordinated Runtime transfer statistics. Runtime traces
include host-observed stage latency, including loading and compilation.

Inputs and output buffers are borrowed directly, and progress callbacks run on
the calling thread. A callback may enqueue other compute but must not wait for
it while holding the lane. Nested sessions in the same context return `Busy`.
Admission also returns `Busy` when the context's submission capacity is full.
Cancellation, ordinary errors and panics all fence before releasing the lane;
an uncertain fence quarantines the entire owner and prevents session reuse.
This is synchronous stage scheduling, not asynchronous per-layer submission.
`profile_report()` now returns `Result<Option<String>>`, so admission or device
failures are reported rather than discarded.

The pinned-checkpoint replay (`examples/qualify_session.rs`, budgeted mode)
matches the frozen audio/video encode, decode and denoise outputs byte-for-byte.
It also checks caller-thread progress, deferred competing compute, cancellation/
retry, pressure eviction/reload of all four units, and zero retained budget after
session teardown. These are correctness checks, not a claim of faster generation.

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

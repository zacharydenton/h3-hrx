# Runtime options

The `h3` CLI releases completed model stages by default. Conditioning runs with
the text encoder, sampling retains the full DiT, and final decoding loads the
VAEs after releasing the DiT. A stream fence precedes each release. No per-layer
weight eviction occurs during sampling. Use `--residency retain` to keep models
resident within the process.

`Session::new(Config)` retains models for subsequent requests. Choose
stage-scoped ownership explicitly:

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
calls release their finished VAE units after fencing. Cancellation releases
completed stage owners and permits a later request on the same session. A failed fence
retains allocations that may still be in flight.

## Shared allocation budgets

`Session::new_in(config, options, &context)` selects the context's GPU, compute
and copy engines, and optional `RuntimeOptions::memory_budget`. All five lazy units (text encoder,
DiT, video VAE, audio VAE and latent upscaler) charge native weights, growing workspace and
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

Library callers select `ResidencyPolicy::Budgeted` with `Session::new_in` and a
live `ResidencyManager`-backed budget. Cache units retain their original private
stream identity; they are not shared mutable models across sessions. Completed
or cancelled stages fence before returning units to the cache. Session teardown
unregisters all model units so mapped checkpoints do not outlive its safety contract.

StageScoped leaves the native stream's bounded staging cache resident between
calls; dropping the session releases it too. Ordinary compiler/code-object memory, native
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
`profile_report()` returns `Result<Option<String>>` and propagates admission
or device failures.

The pinned-checkpoint replay (`examples/qualify_session.rs`, budgeted mode)
matches the frozen audio/video encode, decode and denoise outputs byte-for-byte.
It also checks caller-thread progress, deferred competing compute, cancellation/
retry, pressure eviction/reload of model units, and zero retained budget after
session teardown.

## Turbo qualification

The experimental `SessionOptions::turbo` path keeps the existing INT8 base and
adds BF16 low-rank branches. Adapters resolve through the standard Hugging Face
Hub cache at pinned revisions. Use `cargo test --lib stack::adapter::tests::` to validate
cached adapters.

Four/eight-evaluation presets require 1344×768, 124 frames, Euler, and video/audio
shifts 6/3. They support text or one first-frame keyframe. The corresponding sigma
grids contain five/nine points including terminal zero. Both presets include
all fifty DiT blocks and both refiner blocks. They reject unsupported checkpoint,
attention, reference, and cache combinations before loading models.

The CLI presets remain hidden during numerical and perceptual qualification.
Quality and end-to-end speed remain unqualified.

## Cache qualification

`SessionOptions::cache` accepts `Off`, `Observe`, or `Conservative` with separate
conditioning/audio/video thresholds. `Observe` measures changes and runs every
block. Conservative reuse requires all three accumulated relative L1 changes to
pass their thresholds; the first two and last two evaluations always run in full,
and consecutive skips are forbidden. Nonfinite metrics force full computation.
Caching remains off by default and cannot be combined with Turbo.

The legacy positive `DenoiseParams::cache_threshold` maps one value to all three
thresholds under the hardened policy. Explicit policies cannot be combined with
a positive legacy threshold. Calibrate thresholds against saved noise, motion
and audio before using them as a preset.

## Reproducible diagnostics

`H3_STAGE_TRACE=1` emits JSON stage events for checkpoint identity, actual packed
rows, sigma arrays, cache decisions, and model release. It does not enable the
intrusive kernel profiler. Host submission timestamps can overlap GPU work;
release events follow synchronization.

Use [Criterion benchmarks](performance.md) for repeatable timing comparisons.
`cargo run --example cache_calibrate -- TRACE` reads a captured stage trace from `--cache-observe`
and proposes cache thresholds to evaluate on the same inputs.

`H3_FUSED_OPERANDS=0` disables fused attention preparation for comparisons.
`H3_COMPACT_QKV=0` disables direct V publication from eligible INT8 projections.
`H3_REUSE_SCRATCH=1` enables the optional QKV/feed-forward scratch reuse experiment.

## Native engines and compiler controls

PM4 compute with compute copies remains the default. Every inference CLI command
accepts `--gpu`, `--compute-engine pm4|aql`, `--copy-engine compute|sdma`,
`--aql-private-bytes`, `--processor-mode default|cu|wgp`, and `--compile-workers`.
These settings apply to the session's actual stream, including recorded graphs.
SDMA uses coherent allocation backing and host fences at engine transitions;
measure end-to-end latency before choosing it for a workload.

For library callers, set `RuntimeOptions::compute_engine` and `copy_engine` on the
`ModelContext` passed to `Session::new_in`. Compiler choices live in
`Config::compiler`. `Session::new` uses default runtime engines; use `new_in` for
AQL, SDMA, or a nondefault GPU.

`--sanitize access,value,operation,race --compute-engine aql` enables report-only
instrumentation. Completed stages drain reports and return an error for any
failure or dropped feedback, including graph replay. Library callers use
`Config::compiler.sanitizer` and can bound report/shadow storage through
`sanitizer_runtime`; CLI equivalents are `--sanitizer-report-bytes` and
`--sanitizer-shadow-bytes`. AQL scratch and diagnostic allocations share the context's
budget. Instrumentation changes compilation/cache identity and adds substantial
runtime cost. Address checks cover whole bound allocations, not individual
slices. Race checks cover workgroup/LDS accesses, not cross-workgroup or
cross-queue races.

Use `--compile-reports target/compiler-reports` for detailed compiler reports or
`--compile-traces target/compiler-traces` for bounded text pass traces. Neither
writes files unless requested. CU/WGP mode changes need their own numerical and
performance checks; selecting a mode does not establish a speedup.

## Checkpoint I/O

`--weight-io mapped|native-buffered|native-direct` selects the loading route.
Mapped loading remains the default. Native modes use HRX's GPU-authored io_uring
session and four retained 16 MiB payload slots. Contiguous, gathered, interleaved
and pitched row recipes preserve their encoded bytes; built recipes continue to
run on the CPU. Loaded tensors remain resident under the selected residency policy.
Adjacent file fragments share a read and a descriptor-driven GPU gather, keeping
the read lease alive until the gather completes. Each gather uses a bounded,
rebased destination view; layouts that exceed its indexing limits use individual
copies or row gathers.
Row gathers also zero the padding of every completed row, including rows split
across reads, so padded tensors do not need a separate full-buffer clear.
The first bounded set of reads overlaps destination allocation; accepted reads
retain their slots through completion even if the destination exceeds the budget.

`--storage-slots 1..64` and `--storage-slot-mib 1..64` override the native
payload capacity. Larger slots reduce read/copy command counts; additional slots
allow more overlap but retain more memory. Their product is the payload memory
charge, excluding commands and resident weights. Rust callers use
`WeightIo::slots` and `WeightIo::slot_bytes`; byte sizes must be page aligned.

`--storage-progress sqpoll|wait` selects kernel polling or eventfd/deferred-work
service. Direct mode requires filesystem direct-I/O support and fails explicitly
when unavailable. This uses registered system pages, not discrete VRAM peer DMA.
`--weight-io-statistics` collects host-observed loading intervals and reports
counters for the current model units. Native storage capacity is charged to the
shared allocation budget in addition to resident weights and execution scratch.
The timings separate allocation, read submission/wait, consumer submission/wait,
and completed-consumer retirement. They measure host time, including any blocking
inside those calls, rather than isolated GPU copy duration.
`consumer_commands` counts logical copies and gathers, allowing fragmented
recipes to be compared with the number of storage requests.

Rust callers set `Config::weight_io`. `Session::new` selects the required native
lifetime; applications supplying a `ModelContext` must choose
`RuntimeOptions::native_lifetime = NativeLifetime::Process` for native loading.
Low-level model users pass their compiler to `set_weight_io` before preparing
weights; row kernels share its source selection, reports and sanitizer options. Query
`Weights::io_statistics` or `Session::weight_io_statistics` for typed counters.

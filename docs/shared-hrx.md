# Loom kernels and HRX execution

H3 expresses its GPU operations as custom Loom kernels and executes them through
HRX from Rust. This keeps kernel arithmetic, scheduling and memory ownership
available for [measurement and optimization](performance.md).

| Layer | Responsibility |
| --- | --- |
| H3 | Model shapes, checkpoint layouts, conditioning and sampler semantics |
| Loom kernels | Attention, projections, normalization, convolutions and other GPU operations |
| HRX | Compilation, artifact caches, native buffers, transfers, dispatch, graphs and execution admission |
| `Session` | The application-facing encoding, denoising and decoding API |

These changes require the matching HRX development checkout with
`NativeSession::run_cooperative` (based on 0.12.0). Until that API is released,
use the local Cargo override below. Applications sharing
a `ModelContext` must resolve the same HRX crate instance. The current kernels
target Strix Halo; H3 does not use HRX's NPU feature.

## Provisioning and overrides

HRX downloads and verifies its native bundle on first GPU use.
See [offline provisioning](setup.md#offline-runtime).

| Setting | Purpose |
| --- | --- |
| `HRX_RUNTIME_DIR` | Use a trusted local native-library directory |
| `HRX_BUNDLE_MANIFEST` | Select a local manifest for a mirror or custom bundle |
| `HRX_OFFLINE=1` | Disable network provisioning |
| `HRX_LOOM_LIBRARY` | Select a developer `libloomc.so` |
| `Config::loom_library` | Set the compiler library explicitly |

Caches live under `$XDG_CACHE_HOME/hrx`, or `~/.cache/hrx` when unset.
`hrx gc [DAYS]` removes obsolete bundles and kernels unused for that many days
(default 30). Cache hits refresh last-use timestamps.

For local HRX development, override the registry dependency without changing the
manifest or committing a local lockfile:

```sh
cargo check --config 'patch.crates-io.hrx-rs.path="../hrx-rs"'
```

CI uses the version and checksum recorded in `Cargo.lock`.

## Compilation and dispatch

Installed binaries embed `kernels/` and the tokenizer. Tests and diagnostics use
the working tree. A `Compiler` reads each source once and is fixed to the first
stream's GPU target; create another compiler to use edited sources or another
target.

Kernel requests are batched until `Compiler::flush` or the first launch.
HRX owns parallel compilation and verifies cached artifacts. Cache identity
covers compiler, source, export, target, processor mode, sanitizer settings and configuration. Warm requests reuse
prepared exports and bindings. H3 evaluates the artifact's native launch function
for each distinct workload and caches the result; eager and graph dispatch use
the same compiler-authored workgroup dimensions. Model code supplies workload
indices and verifies tensor extents.

`Config::compiler` controls worker count, CU/WGP scheduling, sanitizer classes,
and optional report/trace destinations. Reports include the native manifest,
resource guidance and source expansions. Pass traces use bounded text output and
force a fresh compilation even for a cached specialization. Diagnostic output is
opt-in; put it under `target/` or outside the checkout.

Weights use HRX's `allocate_from` to initialize and publish owned GPU buffers
directly. Repacking uses at most 256 MiB of temporary host storage for row recipes;
larger gathered tensors retain 16 MiB staging chunks. Gathered tensors within that
limit prepare and release their source pages in merged intervals.
Packing arrays of at least 16 MiB uses two CPU workers writing disjoint row ranges
of the same host buffer; smaller arrays use one worker.
`Stream::read_blocking` waits for readback completion. Normal inference does not
synchronize around every kernel; `H3_PROFILE=1` does, changing timing.

Euler and ResMultistep keep evolving latents on the device. ResMultistep also
retains its denoised history, preserving FP32 first/final updates and separately
rounded FP64 intermediate arithmetic. Schedule coefficients remain on the CPU.
ErSDE retains its CPU solver and noise sequence. Progress callbacks wait for each
reported evaluation to complete.

## Graph execution

`H3_GRAPH=1` records the video decoder stack and DiT block loop as one replay per
transformer block. With step caching, block 0 and its metric run eagerly; the host
chooses between replaying blocks 1–49 and adding the cached residual. Each replay
preserves its internal dependencies; stream order preserves dependencies between
blocks, including shared scratch and mixed-precision boundaries.

`H3_PROFILE` and `H3_DUMP_BLOCKS` select eager execution even after a graph is
cached. Replacing sequence/block storage or a VAE grid invalidates the associated
recordings. A model and its graphs remain bound to their creating stream. The stream inherits
the context's compute and copy engines as well as its GPU and memory budget.
PM4 retains native graph batching. AQL and SDMA replay prepared commands in order;
engine transitions wait for completion on the host. Device-clock graph profiling
requires PM4 with compute copies; host profiling works with either engine.

H3 submits logical fills, copies and uploads directly to HRX. HRX 0.10.1
splits native commands to fit SDMA rings and compute dispatch limits,
including ranges larger than 4 GiB, and bounds host-upload staging internally.

Addresses, constants and launch geometry are fixed at recording time; tensor
contents can change in the same allocations. Dependencies order scratch reuse.
Q preparation, K preparation and V transpose branch from their producer and
feed attention directly.

Graphs are opt-in. [Criterion benchmarks](performance.md) compare eager and
recorded dispatch and DiT execution, including submission and completion waits.

## Session ownership

HRX's `NativeSession` handles compute-lane admission, completion fencing and
quarantine after uncertain completion. H3's lazy model units share a memory
budget and residency manager. See [session lifecycle](runtime-options.md) for
cancellation, eviction and callback constraints.

`Session` opts into cooperative admission on its single private stream. It fences
after every eager or recorded transformer block, allowing other queued components
to run before H3 resumes. Eager kernel dispatch and lazy weight loading also check
for waiters. This keeps a denoising loop from reserving compute for the entire
render while preserving exclusive admission between checkpoints. Kernels are not
preempted; a large block, tensor load or compilation can still delay a handoff.
Standalone low-level model calls outside `Session` do not install checkpoints.

Run the warm-render correctness and latency probe with the same local override:

```sh
H3_GRAPH=1 cargo run --release --example scheduler_probe \
  --config 'patch.crates-io.hrx-rs.path="../hrx-rs"'
```

It compares baseline and contended latent bytes and reports short-graph latency
and completed render times. It uses the cached default DiT/text checkpoints.
Pass `-- --decode` for a smaller video-decoder workload. Run separately with
`H3_GRAPH=0` and `H3_GRAPH=1` to compare eager and block-recorded output hashes.

On the local gfx1151 GPU (2026-10-26), the 256×256, five-frame decode probe
completed 34 short jobs in eager mode and 32 with block replay, with maximum
latencies of 25.8 ms and 23.3 ms respectively. Baseline and contended output bytes
matched in each mode, and both modes produced the same SHA-256
`031da6ef1ee919d32debcce1198db36112e2cb6d7dd9177dbb4cd83eff27151e`.
These are decoder measurements under the host's existing load, not denoising
latency guarantees or a render throughput comparison.

[Rustler](../clients/rustler/README.md) workers own their sessions and accept jobs
through a bounded queue. See [client integration](../clients/README.md) for examples.

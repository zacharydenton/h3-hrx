# Shared HRX integration

H3 uses `hrx-rs` for native loading, allocation, dispatch, graphs, compilation and
artifact caching. H3 requires HRX 0.10.1 or newer for native large-transfer support.
Model code owns tensor
layouts, source selection and numerical behavior; `Session` is the public API.
H3 does not enable HRX's optional NPU feature.
Consumers sharing HRX contexts must resolve the same HRX crate instance.

## Provisioning and overrides

HRX downloads and verifies its native bundle on first GPU use.
See [offline provisioning](setup.md#toolchain-and-build).

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

For local HRX development, override the pinned dependency without changing the
manifest or committing a local lockfile:

```sh
cargo check --config 'patch."https://github.com/zacharydenton/hrx-rs.git".hrx-rs.path="../hrx-rs"'
```

CI uses the revision recorded in `Cargo.lock`.

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

`H3_GRAPH=1` records the video decoder stack and DiT block loop. With step caching,
block 0 and its metric run eagerly; the host chooses between replaying blocks
1–49 and adding the cached residual.

`H3_PROFILE` and `H3_DUMP_BLOCKS` select eager execution even after a graph is
cached. Replacing sequence/block storage or a VAE grid invalidates the associated
recordings. A model and its graphs remain bound to their creating stream. The stream inherits
the context's compute and copy engines as well as its GPU and memory budget.
PM4 retains native graph batching. AQL and SDMA replay prepared commands in order;
engine transitions wait for completion on the host. Device-clock graph profiling
requires PM4 with compute copies; host profiling works with either engine.

H3 submits logical fills, copies and uploads directly to HRX. The pinned HRX
revision splits native commands to fit SDMA rings and compute dispatch limits,
including ranges larger than 4 GiB, and bounds host-upload staging internally.

Addresses, constants and launch geometry are fixed at recording time; tensor
contents can change in the same allocations. Dependencies order scratch reuse.
Q preparation, K preparation and V transpose branch from their producer and
feed attention directly.

Graphs remain opt-in. [Criterion benchmarks](performance.md) compare eager and
recorded dispatch and DiT execution, including submission and completion waits.

## Session ownership

HRX's `NativeSession` handles compute-lane admission, completion fencing and
quarantine after uncertain completion. H3's lazy model units share a memory
budget and residency manager. See [session lifecycle](runtime-options.md) for
cancellation, eviction and callback constraints.

[Rustler](../clients/README.md) workers own their sessions and accept jobs through
a bounded queue. The Python parity tools invoke an ignored Rust fixture test; neither
integration adds a Python or Rustler dependency to the inference library.

## Model-specific engine boundaries

H3 keeps immutable checkpoint mappings and directly initializes repacked weights
through HRX. GPU-authored storage rings do not replace these model-aware packing
recipes or provide automatic weight paging. Likewise, resident XDNA sessions
require NPU kernels and tensor layouts that H3 does not currently provide. H3 does
not expose storage-ring or NPU switches that would leave inference unchanged.

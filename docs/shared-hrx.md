# Shared HRX integration

H3 uses `hrx-rs` for native loading, allocation, dispatch, graphs, compilation and
artifact caching. `Cargo.toml` pins commit `8f0e084` for direct weight initialization,
which is newer than the 0.8.16 release. Model code owns tensor
layouts, source selection and numerical behavior; `Session` is the public API.
H3 does not enable HRX's optional NPU feature.
Consumers sharing HRX contexts should use the same Git revision.

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

For local HRX development, add this patch to `.cargo/config.toml`:

```toml
[patch."https://github.com/zacharydenton/hrx-rs"]
hrx-rs = { path = "../hrx-rs" }
```

Restore the pinned dependency before committing lockfiles. HRX's
`scripts/check-consumers.py` tests committed consumer snapshots against a
candidate checkout.

## Compilation and dispatch

Installed binaries embed `kernels/` and the tokenizer. Tests and diagnostics use
the working tree. A `Compiler` reads each source once and is fixed to the first
stream's GPU target; create another compiler to use edited sources or another
target.

Kernel requests are batched until `Compiler::flush` or the first launch.
HRX owns parallel compilation and verifies cached artifacts. Cache identity
covers compiler, source, export, target and configuration. Warm requests reuse
prepared exports and bindings.

Weights use HRX's `allocate_from` to initialize and publish owned GPU buffers
directly. Repacking uses at most 256 MiB of temporary host storage for row recipes;
larger gathered tensors retain 16 MiB staging chunks. Gathered tensors within that
limit prepare and release their source pages in merged intervals.
Packing arrays of at least 16 MiB uses two CPU workers writing disjoint row ranges
of the same host buffer; smaller arrays use one worker.
`Stream::read_blocking` waits for readback completion. Normal inference does not
synchronize around every kernel; `H3_PROFILE=1` does, changing timing.

## Graph execution

`H3_GRAPH=1` records the video decoder stack and DiT block loop. With step caching,
block 0 and its metric run eagerly; the host chooses between replaying blocks
1–49 and adding the cached residual.

`H3_PROFILE` and `H3_DUMP_BLOCKS` select eager execution even after a graph is
cached. Replacing sequence/block storage or a VAE grid invalidates the associated
recordings. A model and its graphs remain bound to their creating stream.

Addresses, constants and launch geometry are fixed at recording time; tensor
contents can change in the same allocations. Dependencies order scratch reuse.
Q preparation, K preparation and V transpose branch from their producer and
feed attention directly.

Graphs remain opt-in. [Criterion benchmarks](performance.md) compare eager and
recorded dispatch and DiT execution, including submission and completion waits.

## Session ownership

HRX's `NativeSession` handles compute-lane admission, completion fencing and
quarantine after uncertain completion. H3's four lazy model units share a memory
budget and residency manager. See [session lifecycle](runtime-options.md) for
cancellation, eviction and callback constraints.

[Rustler](../clients/README.md) workers own their sessions and accept jobs through
a bounded queue. The Python parity tools invoke an ignored Rust fixture test; neither
integration adds a Python or Rustler dependency to the inference library.

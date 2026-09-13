# Shared HRX integration

H3 depends on the shared `hrx-rs` crate. Its former local `hrx` package and
binary rpath build scripts are removed. The model still exposes `Session` as its
Rust API. It does not require Python, Torch,
ROCm development headers or an LLVM build.

The workspace uses `hrx-rs` 0.4 from crates.io, with the exact release pinned
in the workspace lockfile. It includes the coordinated GPU/NPU APIs, compiler
cache fixes and keyed pending requests. H3 enables only `download` and `loom`;
ordinary inference does not initialize an NPU.

For local HRX development, put this in an ignored `.cargo/config.toml`:

```toml
[patch.crates-io]
hrx-rs = { path = "../hrx.rs" }
```

Restore the pinned dependency before committing lockfiles. HRX's
`scripts/check-consumers.py` checks committed consumer snapshots against a
candidate HRX tree without modifying the original checkouts.

`hrx gc [DAYS]` collects what provisioning leaves behind: runtime bundles the
crate's manifest no longer pins, and kernel artifacts unused for longer than
DAYS, 30 by default. Cache hits refresh an artifact's timestamp, so the sweep
tracks last use rather than creation. Nothing is evicted implicitly.

The shared runtime supplies per-session streams, allocator-based allocation,
checked buffer ranges, dynamic native loading, and prepared binding dispatch.
`h3_hrx::compile` now selects model sources/exports and delegates compilation,
SHA-256 identity, integrity checks and atomic publication to `hrx::loom`. It
compiles for the architecture the stream's device reports, and asking for a kernel
does not build it: requests accumulate, and `Compiler::flush` hands the whole
outstanding set to `hrx::loom::Compiler::compile_all`, which owns the thread
budget. `KeyedKernels::request_or_insert_with` identifies requests by source name,
export and sorted configuration. Warm requests skip source lookup, text hashing
and specialization-map construction. The first miss still joins the batch.
Sources are immutable within a compiler session and are read once. The source files and tokenizer sit at the
repository root, in `kernels/` and `assets/`, and are embedded into an installed binary; tests and
examples read them from the working tree.
Compiled kernels use HRX’s shared per-user kernel cache.

The pinned native release includes the VOPD register-bank fix and Loom's
fragment-repack and SMEM storage-reuse fixes on the public upstream compiler. The
release is public and the crate carries its manifest, so the first run that opens
the GPU downloads and verifies it; [preparing the cache by hand](setup.md#toolchain-and-build)
is for provisioning ahead of time or for a machine that will be offline later.

That compiler is also the reason the four kernels with hand-written low asm —
`attention_mha64t32`, `attention_mha64hm32` and the two `attention_i8qkhm` — name
`gfx1151` where the other hundred name `gfx11-generic`. A Loom target is a bare
architecture, and the profile it selects is what registers the descriptor set the
asm is spelled in: `amdgpu.rdna3_5.core`, which no generic profile provides.

Default first-use provisioning is implemented in HRX. `HRX_RUNTIME_DIR` overrides
the native directory; `HRX_BUNDLE_MANIFEST` selects a pinned mirror; `HRX_OFFLINE`
refuses network access. Explicit model compiler arguments or `HRX_LOOM_LIBRARY`
select a developer compiler. See the shared crate README for the bundle contract.

`Config::loom_library` selects a shared compiler library, or `HRX_LOOM_LIBRARY`
does; either empty takes the pinned bundle's. The crate's `Session` is the interface,
`clients/rustler` provides Elixir interop, and
`scripts/parity.py` reaches the host through `h3-dev parity-dump` rather
than by linking it.

`clients/rustler` is a minimal, working Rustler adapter owned by an Elixir app.
Its worker owns an H3 Rust session; jobs contain owned data and a bounded queue.
It calls the Rust API directly. No Rustler dependency is added to H3 or HRX.

`loomrun` is gone: it existed to launch one kernel for the Python test harness,
and the kernel tests dispatch in process now. The equivalent runner lives in the
shared crate for anyone who wants it.

## Execution costs

Weight uploads borrow checkpoint chunks or reuse a host buffer for padded rows.
HRX copies them into owned staging before returning, so file pages can be released
without waiting for GPU completion. These are staged transfers, not zero-copy DMA.

Prepared kernel handles borrow their loaded export during each dispatch; they do
not clone an `Arc` or take the compiler queue lock after the first resolution.
A compiler is fixed to one target and rejects a different stream target even on
a module-cache hit. `Stream::read_blocking` supplies the completion wait for host readback;
normal inference does not enable per-kernel profiling.

## Graph execution

`H3_GRAPH=1` enables cached recordings of the video decoder's stack and the DiT
block loop. With the step cache enabled, block 0 and its metric remain eager;
the host decides whether to replay a separate recording of blocks 1–49 or add
the cached residual. The host branch stays outside the graph.

Profiling (`H3_PROFILE`) and intermediate block dumps (`H3_DUMP_BLOCKS`) select
eager execution on every call, including after a graph has been cached. Video
VAE and DiT objects require their creating stream. Replacing DiT sequence or
block storage clears both the full-loop and cache-miss recordings; changing the
VAE grid replaces its workspace and recording together.

The adapters share the eager kernel builders. Most launches form a chain because
stages reuse scratch. Integer attention preparation branches from its producer:
Q preparation, K preparation and V transpose write separate spans. Attention
names those three nodes directly, without an empty join. In the pinned runtime,
empty joins create separate queue-barrier partitions. Edges have no universal
price; several dependencies can be discharged by one barrier.

Addresses, constants and geometry are fixed at recording time. Modulation and
residual **contents** change in their existing allocations before replay. Native
retention keeps resources allocated; recording a pooled workspace would also
require keeping its pool owners alive to prevent reuse.

## Performance evidence

Graphs remain opt-in. Earlier full-decode runs were too noisy to establish an
end-to-end benefit: the same configuration ranged from 31 s to 45 s on a busy
machine. Those results are not evidence that concurrent execution cannot help.
GPU timestamps and native partition/workstream counters are not exposed by the
pinned API.

`h3-dev dispatch-cost` compares the same prepared kernels and allocations eagerly and
as a serial graph. Both arms receive three warmups and nine measured batches,
with alternating A/B and B/A order. Samples end with a completion wait. Each
arm must overwrite poisoned output with the expected zeros; reset and readback
are outside the timer. The two prepare sizes and two audio-convolution shapes separate small launches from
larger workloads. These are wall-clock measurements of submission, execution
and waiting together, not isolated GPU timings or model-wide predictions.

A 2026-09-10 run with the local HRX checkout measured these completed times per
kernel. Both paths passed the output checks. System load varied during this
pass, so repeat the paired benchmark before using it to select an execution path.

| Kernel and shape | Eager | Graph |
| --- | ---: | ---: |
| Prepare 256×1 | 4.10 µs | 4.34 µs |
| Prepare 4096×1024 | 45.89 µs | 46.03 µs |
| Conv1d4 1024×5 | 7.17 ms | 7.23 ms |
| Conv1d4 8×165600 | 251.3 µs | 186.0 µs |

Repeated on 2026-09-10 at load average 1.3, three consecutive runs, after the
diagnostics moved into `h3-dev`. Every absolute figure roughly halves, which is
the earlier pass measuring contention rather than either path:

| Kernel and shape | Eager | Graph | Ratio |
| --- | ---: | ---: | ---: |
| Prepare 256×1 | 2.35 µs | 2.07 µs | 0.88–0.89× |
| Prepare 4096×1024 | 21.2 µs | 21.1 µs | 0.91–1.13× |
| Conv1d4 1024×5 | 4.83 ms | 4.83 ms | 1.00× three times |
| Conv1d4 8×165600 | 111.2 µs | 110.8 µs | 0.99–1.00× |

This resolves the one row that had looked like a win. Conv1d4 8×165600 came back
251 µs against 186 µs on the busy machine, and 111 µs both ways on the idle one:
that 26% was load, not the graph. What survives is a saving of roughly 280 ns per
node, which shows against a kernel doing nothing and disappears against one doing
microseconds of arithmetic.

The historical decoder profiles are useful for locating costs: a warm video
tile spent about 20 ms enqueueing, 271 ms in blocking readback and 3 ms
unpatchifying; a warm stereo audio decode of 500 latents took 1.49 s. Profiling
synchronizes around each launch, so its stage durations also include submission
and waiting. Kernel grids and these elapsed times alone do not prove hardware
resource saturation.

Audio and video decode use disjoint inputs and outputs and could run on separate
streams with separate workspaces. Measuring the benefit requires paired
completed runs, output parity and peak-memory measurements. Resident weights
make transfer overlap mainly a first-use question, which should be measured
separately.

## Integration checks, 2026-09-10

With HRX 0.2.0, all 22 native tests passed. Locked workspace builds, CPU tests,
clippy and workspace rustdoc with warnings denied also passed using the crates.io
package. Coverage includes graph replay with changed input bytes, direct
branch dependencies, invalidation of both DiT recordings when sequence storage
grows, compiler target checks, conditioning, sampling, GEMMs, attention and
convolutions. Repeat the native checks with:

```sh
HRX_OFFLINE=1 cargo test -- --ignored --test-threads=1
HRX_OFFLINE=1 cargo run --release --bin h3-dev -- dispatch-cost
```

Checkpoint-backed `denoise_cases` runs compared `H3_GRAPH=0` with `H3_GRAPH=1`
at 32×32, five frames, Euler sampler and seed 7. Three steps exercised the full
loop; four steps with cache thresholds `1000000` and `0.000000001` exercised
cache hits and forced misses. All six video/audio latent files were finite and
byte-identical. A further run with `H3_GRAPH=1`, `H3_PROFILE=1` and
`H3_DUMP_BLOCKS` produced identical latents, reported stage timings and wrote
all 50 DiT block dumps. These check replay correctness and diagnostics; visual
quality and throughput require separate measurements.

Three historical release runs on Ryzen AI MAX+ 395 / gfx1151 with Rust 1.95 nightly and
hrx-rs 0.1.0 measured 159–176 ns of host time per `Prepare::run`, including binding
checks, kernel resolution and scalar packing. Each run takes the median of nine
2,048-launch batches after three warmups and checks the output. Completed batches
averaged 2.15–2.21 µs per kernel. This is a small prepared-kernel benchmark; full
video/audio latency, checkpoint-scale load time and peak memory were not remeasured.

## NPU scope

Audio remains on the GPU. No NPU artifact, buffer import or runtime initialization
is added to H3's inference path. A future audio port needs its own measured stage
boundary and quality qualification; enabling the HRX NPU feature alone does not
provide an offload.

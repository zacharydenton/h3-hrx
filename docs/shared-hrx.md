# Shared HRX integration

H3 depends on the shared `hrx-rs` crate. Its former local `hrx` package and
binary rpath build scripts are removed. The model still exposes `Session` as its
Rust API. It does not require Python, Torch,
ROCm development headers or an LLVM build.

The workspace takes it from crates.io by version, under the name `hrx`:

```toml
hrx = { package = "hrx-rs", version = "0.2.0", default-features = false, features = ["download", "loom"] }
```

So a clean clone builds with `cargo build` and nothing else — no credentials, no
sibling checkout, no `[patch]` table.

**0.2.0 is not yet on crates.io.** Until it is, point Cargo at a local checkout
from an uncommitted `.cargo/config.toml`, which keeps the override off the
dependency itself:

```toml
[patch.crates-io]
hrx-rs = { path = "../hrx.rs" }
```

`hrx gc [DAYS]` collects what provisioning leaves behind: runtime bundles the
crate's manifest no longer pins, and kernel artifacts unused for longer than
DAYS, 30 by default. Cache hits refresh an artifact's timestamp, so the sweep
tracks last use rather than creation. Nothing is evicted implicitly.

The shared runtime supplies per-session streams, allocator-based allocation,
checked buffer ranges, dynamic native loading, and prepared binding dispatch.
`h3::compile` now selects model sources/exports and delegates compilation,
SHA-256 identity, integrity checks and atomic publication to `hrx::loom`. It
compiles for the architecture the stream's device reports, and asking for a kernel
does not build it: requests accumulate, and `Compiler::flush` hands the whole
outstanding set to `hrx::loom::Compiler::compile_all`, which owns the thread
budget. Sources are read once per compiler session. The source files and tokenizer are
inside the `h3` package at `h3/kernels` and `h3/assets`. Tests and examples use
these paths directly.
Compiled kernels go to an `hrx-v1` cache subdirectory.

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
does; either empty takes the pinned bundle's. There is no C ABI any more — the
crate's `Session` is the interface, `examples/rustler` is the interop, and
`scripts/parity.py` reaches the host through the `parity_dump` example rather
than by linking it.

`examples/rustler` is a minimal, working Rustler adapter owned by an Elixir app.
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
a module-cache hit. `Stream::read` supplies the completion wait for host readback;
normal inference does not enable per-kernel profiling.

## Video decode is compute-bound, measured 2026-09-09

A 480x864x22 decode runs as fifteen 256x256 spatial tiles. Timed per phase, a warm tile costs about
20 ms of host enqueue, 271 ms blocked in `read_blocking` -- which drains the queue before it
transfers 10 MiB -- and 3 ms of unpatchify. Across the decode, every host phase together (latent
gather, unpatchify, pixel blend) is **106 ms of 25.2 s, 0.4%**. The first tile carries about 21 s of
one-time weight upload and kernel setup, which amortises over a real clip's 105 tiles.

That is roughly 1.08 ms of GPU time per dispatch against 175 ns to enqueue one, so the decoder is
bound by its kernels and not by the host. Queuing the readback to hide the host phase is worth about
1%, and recording the fifteen independent tiles as concurrent workstreams competes for the same
margin, since the device is already busy for 92% of a tile's wall time.

Larger tiles are not the answer either: one 480x864 tile takes 26.2 s where fifteen 256x256 tiles
take 25.2 s, despite covering 2.4x less total tile area, because the decoder's attention is quadratic
in a tile's token count. The 256-pixel tiling is the efficient configuration.

## The audio VAE is compute-bound too, measured 2026-09-09

`docs/archive/notes.md` calls BigVGAN launch-bound, which was true of the torch implementation this
one replaced. It is not true here. A warm stereo decode of 500 latents takes 1.49 s in process, and
one latent -- where the ~836 dispatches are the same but the work is not -- takes 121 ms. Under
`H3_PROFILE=1` that short case reports 126 ms of kernel time against 128 ms of wall, so 145 us per
dispatch is the kernel running, not the host enqueuing it, and one enqueue costs 175 ns.

The kernels are inefficient at short lengths -- `audio res conv2` averages 635 us over a 1024x5
tensor -- which is a kernel problem worth its own look, and not one a recording addresses.

## Recording, and what it is worth

`Sink` in `h3::dispatch` decides where a launch goes: onto the stream now, or into a graph to replay
later. `Prepare::emit`, `Gemm::emit` and `Stack::emit` take one, and `run`/`forward` are the eager
wrappers, so the dispatch path is written once and the recorded path cannot drift from it. A
recording chains every launch behind the one before it, because the stacks reuse one scratch pair
between stages and consecutive launches therefore conflict even where their operands do not. The one
exception is declared: `prep_q`, `prep_k` and the V transpose read `q`, `k` and `v`, share `zmean`
read-only, and write six allocations no other two of them touch, so each waits on what came before
rather than on its neighbours.

`H3_GRAPH=1` records and replays two paths: the video decoder's stack, which a clip repeats over a
hundred times with identical bindings, and the DiT's fifty blocks, whose every binding, grid and
constant is the same at every step -- only the modulation table's contents and the residual stream
change, and the recording already points at both. The step cache stays eager, since the host decides
after block 0 whether the rest of the step runs at all and a recording cannot branch.

It is off by default because it is not faster. Timed in process, interleaved, median of nine
batches: replaying a 256-node chain costs **0.58x** an eager dispatch when each node is a 256-wide
prepare that does nothing, and **0.99x** when each node is a 4096x1024 prepare that does the work a
real stack's node does. The per-node cost is around 2 us, which is a win against a 4.5 us launch of
nothing and invisible against 130 us of arithmetic. Whole-decode comparisons on this machine were
too noisy to separate the two at all -- a contended box swung the same configuration between 31 s
and 45 s -- which is its own reason to trust the in-process measurement and not the wall clock.

What the recording does prove is that it is a faithful one. Under `H3_GRAPH=1` the seven video
decodes and the ten reference denoise cases are byte-identical to the eager path, the latter
exercising the concurrent operand prepares in all fifty blocks.

All three candidate paths were measured before any of this: a denoise step averages tens of
milliseconds of GPU work per dispatch, a video decode tile about 1.08 ms, and an audio decode 145 us.
Against 175 ns to enqueue, none is bound by the host, and no arrangement of dependencies changes the
arithmetic they are waiting on.

A second stream to overlap the two decodes is not used either, for the same arithmetic. Video and
audio decode are independent -- different checkpoints, disjoint inputs, disjoint outputs -- but both
are bound by the one GPU, so running them together does not reduce the work, only fills whatever
idle the other leaves. Video decode leaves about 7 ms of idle per tile, the host's unpatchify and
blend, which is 0.74 s across a 124-frame clip's 105 tiles. Audio decode needs 1.54 s of GPU. So the
ceiling is about 0.8 s of 35.85 s, for a second stream, a thread, and an end to `Session`'s
single-threaded contract.

A separate copy stream is not used either. Weights become resident on first use, and later steps
reuse them; overlapping those first-use uploads would need its own measurement, including peak
memory, rather than a claim from the dispatch cost alone.

## Integration checks, 2026-09-09

Workspace CPU tests, clippy with warnings denied, library rustdoc, and all 19
native tests passed. The native suite covers compiler target checks, recovery from
a batch holding a kernel that will not build, conditioning, resident sampling,
GEMMs, attention and convolutions. Repeat it with:

```sh
HRX_OFFLINE=1 cargo test -p h3 -- --ignored --test-threads=1
HRX_OFFLINE=1 cargo run --release -p h3 --example dispatch_cost
```

Three release runs on Ryzen AI MAX+ 395 / gfx1151 with Rust 1.95 nightly and
hrx-rs 0.1.0 measured 159–176 ns of host time per `Prepare::run`, including binding
checks, kernel resolution and scalar packing. Each run takes the median of nine
2,048-launch batches after three warmups and checks the output. Completed batches
averaged 2.15–2.21 µs per kernel. This is a small prepared-kernel benchmark; full
video/audio latency, checkpoint-scale load time and peak memory were not remeasured.

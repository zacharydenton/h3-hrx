# Shared HRX integration

H3 depends on the shared `hrx-rs` crate. Its former local `hrx` package and
binary rpath build scripts are removed. The model still exposes `Session` as its
Rust API. It does not require Python, Torch,
ROCm development headers or an LLVM build.

The workspace takes it from crates.io by version, under the name `hrx`:

```toml
hrx = { package = "hrx-rs", version = "0.1.0", default-features = false, features = ["download", "loom"] }
```

So a clean clone builds with `cargo build` and nothing else — no credentials, no
sibling checkout, no `[patch]` table. A local development override is still
Cargo's `[patch]` mechanism when you want one.

The shared runtime supplies per-session streams, allocator-based allocation,
checked buffer ranges, dynamic native loading, and prepared binding dispatch.
`h3::compile` now selects model sources/exports and delegates compilation,
SHA-256 identity, integrity checks and atomic publication to `hrx::loom`. It
compiles for the architecture the stream's device reports, and asking for a kernel
does not build it: requests accumulate and the whole outstanding set is compiled
at once, across threads. Sources are read once per compiler session. The source files and tokenizer are
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

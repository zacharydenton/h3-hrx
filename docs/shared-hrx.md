# Shared HRX integration

H3 depends on the shared `hrx.rs` crate. Its former local `hrx` package and
binary rpath build scripts are removed. The model still exposes `Session` as its
Rust API. It does not require Python, Torch,
ROCm development headers or an LLVM build.

The workspace pins HRX to a Git revision with explicit features. A clean clone
needs access to the private `hrx.rs` repository, but no sibling checkout.
A local development override can be supplied with Cargo's `[patch]` mechanism.

The shared runtime supplies per-session streams, allocator-based allocation,
checked buffer ranges, dynamic native loading, and prepared binding dispatch.
`h3::compile` now selects model sources/exports and delegates compilation,
SHA-256 identity, integrity checks and atomic publication to `hrx::loom`.
Sources are read once per compiler session. The source files and tokenizer are
inside the `h3` package at `h3/kernels` and `h3/assets`. Tests and examples use
these paths directly.
Compiled kernels go to an `hrx-v1` cache subdirectory.

The pinned native release includes the VOPD register-bank fix and Loom's
fragment-repack and SMEM storage-reuse fixes on the public upstream compiler. Because `hrx.rs` is
private, [download it with an authenticated GitHub CLI](setup.md#toolchain-and-build)
and prepare the local cache before running the model. A sibling checkout is
optional.

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

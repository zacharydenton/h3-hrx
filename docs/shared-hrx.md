# Shared HRX integration

H3 depends on the shared `hrx.rs` crate. Its former local `hrx` package and
binary rpath build scripts are removed. The model still exposes `Session` as its
Rust API, and `libh3.so` as its portable C ABI. Neither requires Python, Torch,
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
encoding-configuration and dependent-inline-type fixes. Because `hrx.rs` is
private, [download it with an authenticated GitHub CLI](setup.md#toolchain-and-build)
and prepare the local cache before running the model. A sibling checkout is
optional.

Default first-use provisioning is implemented in HRX. `HRX_RUNTIME_DIR` overrides
the native directory; `HRX_BUNDLE_MANIFEST` selects a pinned mirror; `HRX_OFFLINE`
refuses network access. Explicit model compiler arguments or `LOOM_COMPILE`
select a developer compiler. See the shared crate README for the bundle contract.

The C ABI keeps version 8 and exactly the symbols it had. Headers are generated
by cbindgen as a Cargo build dependency into `OUT_DIR`, with failures treated as
build errors. `scripts/build_host.sh` copies that header to `include/h3.h`;
ordinary Cargo builds leave the checkout untouched, and the CPU tests check for
header drift. Foreign slices check alignment and arithmetic before becoming Rust
references. Session panics remain contained and poison the session.

`examples/rustler` is a minimal, working Rustler adapter owned by an Elixir app.
Its worker owns an H3 Rust session; jobs contain owned data and a bounded queue.
It calls the Rust API directly. No Rustler dependency is added to H3 or HRX.

`loomrun` is gone: it existed to launch one kernel for the Python test harness,
and the kernel tests dispatch in process now. The equivalent runner lives in the
shared crate for anyone who wants it.

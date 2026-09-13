# Contributing to h3-hrx

h3-hrx runs MiniMax H3 on AMD Strix Halo through Loom and HRX. Contributions
to kernels, model correctness, documentation, and reproducible measurements
are welcome. For larger changes, open an issue to discuss the approach first.

The model host, tests, and tooling are Rust. Checked-in `.loom` files are the
source of truth: edit them directly and test the affected operation against an
independent CPU reference. The old Python generators, wrappers, and one-off
measurement scripts are retired; their history remains in Git.

## Tests

Use a current stable Rust toolchain. From the repository root:

```sh
scripts/test.sh --cpu
scripts/test.sh --gpu
```

The CPU suite runs formatting, Clippy and workspace tests. GPU tests are explicitly
ignored by default and must be requested on gfx1151; once requested, missing
hardware, compiler, or runtime is a failure, never a silent pass. HRX provisions
and caches the compiler and runtime. `HRX_OFFLINE=1` requires an existing bundle.

Use the shared HRX crate for native loading, allocation, scalar packing, dispatch,
compilation and caching. Model code owns its source selection, shapes,
weight layout and numerical semantics. `Session` is the interface; keep Rustler
adapters in the consuming application.

See [test coverage](docs/testing.md) for the numerical cases and remaining limits.
Record dimensions, precision, toolchain and GPU when reporting performance.

## Reporting bugs and sending changes

Include the commit, OS, GPU, memory size, Rust version, HRX version, command,
and relevant error output in bug reports. For numerical problems, include
dimensions, seed, sampler, attention mode, and a minimal prompt you can share.

Keep pull requests focused and explain the resulting behavior and validation.
Performance claims need the measurement conditions and an output correctness
check. CPU CI does not establish GPU correctness or model parity; use the
[release checklist](docs/releasing.md) for those gates.

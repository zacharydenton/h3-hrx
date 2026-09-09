# Contributing

The model host, tests, and tooling are Rust. Checked-in `.loom` files are the
source of truth: edit them directly and test the affected operation against an
independent CPU reference. The old Python generators, wrappers, and one-off
measurement scripts are retired; their history remains in Git.

```sh
scripts/test.sh --cpu
scripts/test.sh --gpu
```

The CPU suite runs formatting, Clippy and workspace tests. GPU tests are explicitly
ignored by default and must be requested on gfx1151; once requested, missing
hardware, compiler, or runtime is a failure, never a silent pass. HRX provisions
and caches the compiler and runtime. `HRX_OFFLINE=1` requires an existing bundle.

Use the shared HRX crate for native loading, allocation, scalar packing, dispatch,
compilation, caching and FFI guards. Model code owns its source selection, shapes,
weight layout and numerical semantics. Preserve the generated C ABI and keep
Rustler adapters in the consuming application.

See [test coverage](docs/testing.md) for the numerical cases and remaining limits.
Record dimensions, precision, toolchain and GPU when reporting performance.

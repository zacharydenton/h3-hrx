# Contributing to h3-hrx

Edit checked-in `.loom` kernels directly and validate numerical changes against
an independent reference. Retired generators and experiments remain in Git history.

Use HRX for native loading, allocation, dispatch, compilation and caching. H3 owns
model shapes, weight layouts and numerical semantics. `Session` is the inference
API; language adapters belong in consuming applications.

## Tests

From the repository root:

```sh
scripts/test.sh --cpu
scripts/test.sh --gpu
```

The CPU tier runs formatting, Clippy and workspace tests. GPU tests require
`gfx1151` and a provisioned HRX bundle. Checkpoint-backed and parity checks are
listed in [test coverage](docs/testing.md).

## Changes and bug reports

Explain the behavior changed and how it was checked. For numerical bugs, include
shape, seed, sampler, attention mode and a minimal prompt. For performance claims,
include hardware, precision, timing boundaries and an output correctness check.

Bug reports should include the commit, OS, GPU, memory, Rust/HRX versions,
command and error output. See the [release checks](docs/releasing.md) before
publishing a build.

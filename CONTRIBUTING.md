# Contributing to h3-hrx

Edit checked-in `.loom` kernels directly and validate numerical changes against
an independent reference. Retired generators and experiments remain in Git history.

Use HRX for native loading, allocation, dispatch, compilation and caching. H3 owns
model shapes, weight layouts and numerical semantics. `Session` is the inference
API; language adapters belong in consuming applications.

## Tests

From the repository root:

```sh
cargo test
cargo bench
```

Both commands include GPU workloads and require `gfx1151` and a provisioned HRX
bundle. Checkpoints resolve through the standard Hub cache. Tests run serially
by default. CPU-only CI and independent reference checks are described in
[test coverage](docs/testing.md).

Use [Criterion benchmarks](docs/performance.md) for performance work. Keep generated
reports and baselines under `target/`; commit benchmark code, not results.

## Changes and bug reports

Explain the behavior changed and how it was checked. For numerical bugs, include
shape, seed, sampler, attention mode and a minimal prompt. For performance claims,
include hardware, precision, timing boundaries and an output correctness check.

Bug reports should include the commit, OS, GPU, memory, Rust/HRX versions,
command and error output. See the [release checks](docs/releasing.md) before
publishing a build.

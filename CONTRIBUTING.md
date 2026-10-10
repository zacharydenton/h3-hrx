# Contributing

Improve the completed render: its quality, speed, memory use or reliability.
H3 owns model layouts and numerical behavior; custom Loom kernels perform GPU
work, and HRX handles native resources and execution.

## Make a change

Edit the checked-in `.loom` sources directly. Validate numerical changes against
an independent reference and inspect rendered output when arithmetic changes.
Keep the public inference API in `Session`; language adapters belong in clients.
See [Loom and HRX integration](docs/shared-hrx.md) for execution and ownership.

```sh
cargo test
cargo bench
```

Both commands include GPU workloads and require Strix Halo, the HRX bundle and
model checkpoints. Tests run serially by default. Use the [test guide](docs/testing.md)
for CPU-only checks and independent references.

For optimization, follow the [performance guide](docs/performance.md): profile a
real workload, validate the candidate, then compare completed timings. Keep
reports, scratch renders and Criterion baselines under `target/`; commit the
benchmark code, not generated results.

## Send a change or report a bug

Explain the problem, resulting behavior and checks performed. Include:

- For numerical issues: shape, seed, sampler, attention mode and a minimal prompt.
- For performance claims: hardware, workload, precision, timing boundaries,
  variation and output validation.
- For failures: commit, OS, GPU, memory, Rust/HRX versions, command and error output.

See [release checks](docs/releasing.md) before publishing a build.

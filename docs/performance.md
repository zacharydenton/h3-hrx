# Benchmarks

Benchmarks use [Criterion](https://criterion-rs.github.io/book/). GPU workloads
require AMD Strix Halo (`gfx1151`), a provisioned HRX bundle and an idle GPU.
Compilation, weight loading and initial warmup happen before measurement.

```sh
cargo bench --locked --features bench-gpu --bench kernels
```

| Benchmark | Workload | Timed boundary |
| --- | --- | --- |
| `prepare_f32_i8` | FP32 Hadamard preparation at 14,336 and 25,600 columns | Dispatch and GPU completion |
| `gemm` | INT8/BF16 matrices, FP32 projection and SwiGLU output | Dispatch and GPU completion; cached or rotating weights exceeding 64 MiB |
| `dispatch_256` | 256 ordered FP16 preparation launches | Eager submission or recorded graph replay, through GPU completion |
| `audio/roundtrip/3200` | Warm session encode/decode, 3,200 stereo samples per channel | Complete API calls, including transfers, allocation and output disposal |
| `dit_stack` | One resident DiT block, 256 or 2,048 tokens | Eager forward or graph replay through completion; input reset and readback excluded |

Kernel inputs are deterministic synthetic data. Preparation includes values
above FP16 range. Model benchmarks use real weights and deterministic synthetic
inputs. Setup checks finite output; replay checks run outside timing. These are
performance checks, not independent numerical oracles; run the
[correctness tests](testing.md) before interpreting a speedup.

## Checkpoint-backed workloads

Set `H3_BENCH_MODELS` to an immutable local Comfy-Org/MiniMax-H3 snapshot containing
the audio VAE and FL2VA DiT files at their published relative paths. Benchmarks
never download weights. Audio and DiT run separately within a 32 GiB allocation
budget; keep the snapshot unchanged until the process exits.

```sh
export H3_BENCH_MODELS=/path/to/models--Comfy-Org--MiniMax-H3/snapshots/REVISION
cargo bench --locked --features bench-gpu --bench models
```

Use a filter to select a workload. Listing names does not open the GPU or load
checkpoints. `--test` executes each selected workload once without collecting a
statistical report:

```sh
cargo bench --locked --features bench-gpu --bench kernels -- --list
cargo bench --locked --features bench-gpu --bench kernels -- prepare_f32_i8 --test
cargo bench --locked --features bench-gpu --bench models -- audio --test
```

## Comparing revisions

Save a baseline before editing, then compare with the same hardware, toolchain,
inputs and runtime environment:

```sh
cargo bench --locked --features bench-gpu --bench kernels -- --save-baseline before
# Make the change and run correctness tests.
cargo bench --locked --features bench-gpu --bench kernels -- --baseline before
```

Criterion stores reports and baselines under `target/criterion/`, which is ignored
by Git. Keep results, logs, comparison videos and timing tables out of the repo.
The default is ten samples, a one-second warmup and a three-second measurement
window; Criterion may extend sampling for slow workloads. Use `--sample-size`,
`--warm-up-time` and `--measurement-time` for longer runs.

These workloads do not measure complete generation, cold loading or peak memory.
Use identical model inputs and complete outputs when evaluating those costs.
Keep `H3_PROFILE` and stage tracing disabled during benchmarks; use
`examples/profile_audio.rs` and `H3_STAGE_TRACE=1` separately for diagnosis.

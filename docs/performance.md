# Measure render performance

Measure the workload you intend to ship. A faster Loom kernel matters when it
reduces completed render time; HRX profiling helps locate where that time goes.
The [featured 768p renders](showcase.md#wyvern-chase--anime-and-cinematic) include
loading, conditioning, sampling, decoding and media encoding.

## Time a complete render

Build once, then time the installed binary with a fixed prompt and settings:

```sh
cargo install --locked --path . --bin h3
/usr/bin/time -f 'Elapsed: %e seconds' \
  h3 --width 1344 --height 768 --frames 124 --steps 21 --seed 0 \
  --weight-io native-direct --memory-budget-mib 49152 \
  --out clip.mp4 < docs/prompts/wyvern_cinematic.txt
```

This produces 124 frames at 24 fps with 20 model evaluations. Run once to fetch
missing checkpoints and compile kernels before measuring local inference.
A fresh process still benefits from filesystem and compiler caches; it does
not establish cold-disk performance. Record the commit, HRX bundle, Loom compiler,
hardware, settings and cache state with your results.

## Run the benchmarks

```sh
cargo bench
```

The Criterion suite includes host work, kernels, complete model stages, renders,
loading and memory. GPU targets require Strix Halo, the HRX bundle and cached or
downloadable checkpoints; media targets also need ffmpeg and ffprobe. Cargo runs
benchmark executables sequentially. Results stay in `target/criterion/`, with
separate directories for native engine/compiler configurations.

`H3_BENCH_MODELS` overrides the base-model snapshot directory. `CRITERION_HOME`
or Cargo's target directory changes the results location. Keep checkpoint files
immutable during a run.

Start with a filtered case. `--list` needs no models or GPU; `--test` executes
validation and one measured iteration without a sampling campaign:

```sh
cargo bench --locked --bench pipeline -- --list
cargo bench --locked --bench host -- --test
cargo bench --locked --bench pipeline -- text/res_multistep/warm_session --test
```

| Profile | Canvas | Frames | Model evaluations |
| --- | --- | --- | --- |
| `smoke` (default) | 64×64 | 5 | 3 |
| `480p` | 864×480 | 124 | 20 |
| `768p` | 1344×768 | 124 | 20 |

Use `H3_BENCH_PROFILE=768p` for render-performance claims. Smoke runs exercise
the complete network at a short sequence length. Kernel and model targets also
include production shapes regardless of this profile; `cargo bench` includes
repeated 50-block, 37,977-token DiT runs. The full suite can take hours.
Turbo cases always use their trained 1344×768×124 shape and four/eight evaluations.

To benchmark the featured cinematic workload in fresh CLI processes:

```sh
H3_BENCH_PROFILE=768p H3_BENCH_STEPS=21 H3_BENCH_SEED=0 \
  H3_BENCH_PROMPT_FILE=docs/prompts/wyvern_cinematic.txt \
  H3_BENCH_WEIGHT_IO=native-direct H3_BENCH_BUDGET_GIB=48 \
  cargo bench --locked --bench pipeline -- 'cli/768p/.*/text/cold_process' --test
```

Omit `--test` for Criterion samples. It requires at least ten samples; full-size
renders can take hours per case. Prompt, seed, step count, attention, graph mode
and budget are included in render case names to keep baselines separate.
The defaults are ten samples, one second of warmup and a three-second target
window; slow cases exceed that window. Use `--sample-size`, `--warm-up-time` and
`--measurement-time` to extend runs. Network downloads, external prompt services
and long World save/resume rollouts are outside benchmark timing.

## Find the bottleneck with HRX and Loom

Profile a representative 50-block DiT pass:

```sh
H3_BENCH_BUDGET_GIB=32 H3_PROFILE=device \
  cargo bench --locked --bench models -- 'dit_stack/eager/37977/50_layers$' --test
```

`H3_PROFILE=device` reports completed GPU dispatch intervals, shapes and retained
allocations. `H3_COMPILE_REPORT_DIR=target/compiler-reports` adds Loom resource
and wait reports. `H3_BENCH_DETAILS=1` prints output digests, completed forward
counts and loading details where supported.

Collect diagnostics separately from latency. Profiling adds barriers and
serializes work; compiler occupancy estimates are not measured GPU utilization.
Device-clock capture requires PM4 with compute copies. `H3_PROFILE=1` provides
synchronized host timing for other engine combinations.
See [report formats and kernel coverage](benchmark-reference.md#hrx-diagnostics).

## Compare a change

```sh
cargo bench --locked --bench pipeline -- text/res_multistep/warm_session --save-baseline before
# Make the change and validate its output.
cargo bench --locked --bench pipeline -- text/res_multistep/warm_session --baseline before
```

Keep the workload and environment fixed. Validate numerical output, then measure
at the kernel, block, full stack and render levels. Attention grows quadratically
with sequence length; projection work grows linearly. A small-shape speedup can
reverse at 768p. Synthetic inputs and rotating weights do not reproduce every
activation or memory access in a real render.

Alternate baseline/candidate order and report variation, especially on shared
hardware. Reuse the same allocations for in-process kernel comparisons. For
separate builds, keep both worktrees: benchmark executables read the Loom sources
from their build checkout, so copying only a binary does not freeze its kernels.
Disable profiling, compiler reports and stage tracing for latency runs.

Check [numerical correctness](testing.md) independently of replay equality.
Compare rendered motion, detail and audio when arithmetic changes. A matching
output digest proves repeatability; it does not establish visual quality.

## Memory and runtime settings

`H3_BENCH_BUDGET_GIB` caps native allocations, not total system memory. Leave
headroom for mapped files, packing, compiler work and media encoding. Run GPU
benchmarks serially and watch available RAM as well as reservations.

Stage/API render benchmarks default to budgeted residency; fresh CLI cases use
stage-scoped residency. Select it explicitly for a smaller complete-render cap:

```sh
H3_BENCH_RESIDENCY=stage-scoped H3_BENCH_BUDGET_GIB=32 \
  cargo bench --locked --bench pipeline -- text/res_multistep/warm_session --test
```

| Control | Default | Use |
| --- | --- | --- |
| `H3_BENCH_ATTN` | `i8` | `f16` or experimental `i4` for stage/render cases |
| `H3_GRAPH` | Off | Compare graph replay in a separate process |
| `H3_BENCH_GPU` | `0` | GPU ordinal |
| `H3_BENCH_COMPUTE_ENGINE` | `pm4` | `pm4` or `aql` |
| `H3_BENCH_COPY_ENGINE` | `compute` | `compute` or `sdma` |
| `H3_BENCH_PROCESSOR_MODE` | `default` | `default`, `cu` or `wgp` |
| `H3_BENCH_COMPILE_WORKERS` | Automatic, at most 8 | Compiler parallelism |
| `H3_BENCH_AQL_PRIVATE_BYTES` | `4096` | AQL scratch ceiling per workitem |

[Benchmark coverage and timing boundaries](benchmark-reference.md) ·
[Storage comparisons](benchmark-reference.md#storage-and-profiler-comparisons) ·
[Memory measurements](benchmark-reference.md#memory) ·
[Runtime configuration](runtime-options.md)

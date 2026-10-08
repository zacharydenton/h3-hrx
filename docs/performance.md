# Benchmarks

The [Criterion](https://criterion-rs.github.io/book/) suite covers host preparation, individual model stages, complete
renders, loading, eviction, and sampled peak memory. Run `cargo bench` for the
whole suite. Cargo runs benchmark executables sequentially; use an idle machine.
Results and baselines belong in ignored `target/criterion/`, never in Git.
Each native engine/compiler configuration has its own subdirectory there.
`CRITERION_HOME` or Cargo's target directory overrides the output root.

## Targets and coverage

| Target | Coverage |
| --- | --- |
| `host` | Short/long tokenization, mixed-media presentation, image resizing, RefMod loading/strength/copies, packed sequence layout |
| `attention` | Head-128 FP16 four/eight-wave tile comparisons across ragged lengths and the 4096-token boundary, checked against sampled FP64 attention |
| `kernels` | FP32 Hadamard preparation, INT8 attention preparation in both layouts, V transpose, INT8/BF16 GEMMs including all four DiT projections through 37,977 rows, cached/rotating weights, eager/graph dispatch |
| `models` | Resident INT8 DiT eager/graph comparison through 37,977 tokens including partial tiles, a complete 50-block 768p sequence in both modes, and short audio roundtrip |
| `stages` | Complete text encoder plus token refiner, vision tower, video encode/decode, audio encode/decode, complete 50-block denoising trajectories with Euler and ResMultistep |
| `lifecycle` | Mapping/planning, tensor packing and completed uploads, block loading throughput, cold-file loading, forced audio eviction/reload |
| `pipeline` | Complete text, first/last-frame, image/audio reference, video/audio reference, RefMod, LoRA, Turbo and World renders; cache observation/reuse; WAV and H.264/AAC output; fresh CLI processes |
| `memory` | Complete text and reference renders under stage-scoped and budgeted residency, measuring sampled peak reservations and process RSS separately |

Host resize cases cover 1080p to 480p/768p, enlargement, unchanged dimensions,
and a large downscale. They exercise the reference/keyframe bilinear path:

```sh
cargo bench --bench host -- resize/
```

Video stage cases cross spatial tile overlaps horizontally and vertically and
use 5/22/39/56 frames across temporal chunks. Audio cases cover the 800-sample
hop and 255/256/257 latent-frame boundaries, plus a full 124-frame soundtrack.
The 480p/768p profiles add their full-size vision and video-decoder workloads.

## Requirements and quick checks

`host` needs only the Rust toolchain. Other targets require gfx1151 and a
provisioned HRX bundle. No feature flags, wrapper scripts or environment variables
are required:

```sh
cargo bench
```

FL2VA, Ref2VA, Qwen3-VL, both VAEs and pinned adapters resolve through the standard
Hugging Face cache. Missing files are downloaded during setup, outside timing.
`H3_BENCH_MODELS` optionally overrides the base-model snapshot directory.
Keep checkpoint files immutable during a run. `pipeline` and `memory` require the `cli` feature
(enabled by default), FFmpeg with libx264/AAC, and ffprobe. They reuse the CLI's
actual WAV writer and muxer. Temporary media is removed after each workload.

List cases before running them. Filters apply before model loading; `--list`
works without checkpoints or a GPU. `--test` checks the selected workload with
one measured iteration and any setup/validation runs, without a timing report:

```sh
cargo bench --locked --bench stages -- --list
cargo bench --locked --bench pipeline -- --list
cargo bench --locked --bench host -- --test
cargo bench --locked --bench stages -- video_decode --test
cargo bench --locked --bench lifecycle -- --test
cargo bench --locked --bench pipeline -- text/res_multistep/warm_session --test
cargo bench --locked --bench pipeline -- cli/ --test
```

## Workload profiles and overrides

All GPU targets, including fresh CLI processes, accept the same HRX controls:

| Variable | Default | Values |
| --- | --- | --- |
| `H3_BENCH_GPU` | `0` | Nonnegative GPU ordinal |
| `H3_BENCH_COMPUTE_ENGINE` | `pm4` | `pm4`, `aql` |
| `H3_BENCH_COPY_ENGINE` | `compute` | `compute`, `sdma` |
| `H3_BENCH_AQL_PRIVATE_BYTES` | `4096` | Scratch ceiling per workitem |
| `H3_BENCH_PROCESSOR_MODE` | `default` | `default`, `cu`, `wgp` |
| `H3_BENCH_COMPILE_WORKERS` | Automatic, at most 8 | Positive worker count |

These controls apply before stream allocation and compilation. AQL scratch is
charged to the workload's budget. Configuration-specific result directories keep
Criterion from comparing different engines or processor modes automatically;
benchmark names and filters stay the same. Native library overrides still need
separate saved baselines, as do different compiler/runtime releases.

```sh
H3_BENCH_COMPUTE_ENGINE=aql H3_BENCH_COPY_ENGINE=sdma \
  cargo bench --locked --bench models -- dit_stack/eager/37977/1_layer
H3_BENCH_PROCESSOR_MODE=cu \
  cargo bench --locked --bench kernels -- attention_i8qkhm/37977
```

`H3_BENCH_DETAILS=1` prints attention and DiT output hashes outside timing for
cross-configuration comparisons, plus completed DiT forward counts and latency
(also in `--test` mode). Every run checks repeated execution for identical output.
Use `H3_COMPILE_REPORT_DIR` for HRX's resource/wait reports and
`H3_PROFILE=device` for GPU profiling; collect latency separately from diagnostics.
Device-clock profiling requires PM4 with compute copies; `H3_PROFILE=1` uses
synchronized host timing with either engine.
Queue or compiler changes must pass numerical checks and full-workload timing
before becoming defaults. SDMA's coherent allocation policy can affect compute
latency as well as transfers.

| `H3_BENCH_PROFILE` | Canvas | Frames | Sigma points / model evaluations |
| --- | --- | --- | --- |
| `smoke` (default) | 64×64 | 5 | 4 / 3 |
| `480p` | 864×480 | 124 | 21 / 20 |
| `768p` | 1344×768 | 124 | 21 / 20 |

Smoke exercises the complete network, including every DiT layer and all selected
stages, at a small sequence length. Use production profiles for performance
claims. Criterion requires at least ten samples: full-size renders can take
hours per workload. The default command includes every variant; filters are
available for focused work.

The kernel and `models` targets have their own shape matrices, including
production sequence lengths regardless of this profile. The default `models`
suite includes `dit_stack/{eager,graph}/37977/50_layers`; its repeated full-stack
measurements can take tens of minutes. This case streams all 50 checkpoint
blocks through one activation workspace. It measures the resident backbone
with synthetic inputs and modulation tables; conditioning, sampler updates,
decoding and model loading remain covered by the stage and render targets.

```sh
H3_BENCH_PROFILE=480p cargo bench --locked --bench pipeline -- text/res_multistep/warm_session
H3_BENCH_PROFILE=768p cargo bench --locked --bench stages -- denoise/
H3_GRAPH=1 cargo bench --locked --bench pipeline -- text/res_multistep/warm_session
H3_BENCH_ATTN=f16 cargo bench --locked --bench stages -- denoise/
```

To reproduce a particular ordinary H3 render, `H3_BENCH_PROMPT_FILE` supplies
the verbatim prompt to denoise/render benchmarks, `H3_BENCH_STEPS` overrides
the sigma-point count, and `H3_BENCH_SEED` overrides the seed (default 7).
Profile-based benchmark names include the step count, seed and prompt-content
digest so these workloads do not share a baseline accidentally. For example:

```sh
H3_BENCH_PROFILE=768p H3_BENCH_STEPS=31 H3_BENCH_SEED=0 \
  H3_BENCH_PROMPT_FILE=/path/to/prompt.txt H3_BENCH_BUDGET_GIB=48 \
  cargo bench --locked --bench pipeline -- 'cli/768p/.*/text/cold_process' --test
```

`--test` executes the case once without Criterion's sampling/warmup campaign;
use the CLI under an external wall-clock timer for a single end-to-end timing.
Omit `--test` to collect Criterion samples. The usual profiles still use 20
evaluations; the CLI's unqualified default uses 30.

## Interpreting improvements

Choose the target render before choosing a kernel. Rank its profiled stages by
their share of elapsed time: video encoding is absent from text-only generation,
and a large decoder speedup can have little effect when denoising dominates.
Attention scales quadratically with sequence length, while projection work
scales linearly. Small-sequence rankings therefore do not predict 768p rankings.

Validate a candidate at the full-size kernel, block, 50-block stack and render
levels. Rotating matrices help expose weight traffic, but do not reproduce the
full model's working set or its sequence of activation reads and writes. Require
a repeatable improvement at the next level before attributing a local gain to
render performance. Preserve numerical checks and compare the same attention,
sampler, prompt, references, evaluation count, residency policy and memory cap.

Report startup and resident sampling separately. `cold_session` and
`cold_process` recreate their named object; they do not evict the filesystem or
compiler caches. Setup runs, replay checks and Criterion warmups can warm both.
The `lifecycle` cold-file cases measure DiT and text-encoder QKV/gate-up packing
and uploads from private evicted files, including multi-tensor recipes. They
do not measure whole-model cold startup. Device profiling
serializes launches and adds timestamp overhead, so use unprofiled runs for
latency comparisons. Alternate baseline/candidate runs on shared hardware and
report timing variation; a noisy full render does not establish a regression
or a speedup by itself.

Keep the Rust dependency revision, native HRX bundle and Loom compiler fixed
within a comparison. Compiler changes can alter register pressure and occupancy
even when the kernel source is unchanged. Record those identities with results
outside the repository. When comparing kernels in one process, reuse the same
input, weight and output allocations and alternate execution order; separate
workspaces introduce another variable on this UMA device. Include sustained
full-stack measurements: a brief warm microbenchmark does not capture the
clocks, memory pressure or contention of a long render.

## Runtime controls

`H3_BENCH_ATTN` selects `i8` (default), `f16`, or experimental `i4` attention
for stage/render targets. The `models` DiT cases use INT8 attention.
`H3_GRAPH` is read once by the runtime: compare it in separate processes.
Stage/render case names include attention, graph mode and the allocation budget to keep
incompatible baselines separate. `H3_BENCH_BUDGET_GIB` defaults to 64 for the
stage/render/lifecycle/memory suite and 32 for the `models` benchmarks. Profile
a single DiT block with a smaller cap:

```sh
H3_BENCH_BUDGET_GIB=2 H3_PROFILE=device cargo bench --locked --bench models -- dit_stack/eager/2048 --test
```

For a production-size bottleneck ranking, profile the 768p block or all 50
blocks. The full stack needs additional host-memory headroom beyond its cap:

```sh
H3_BENCH_BUDGET_GIB=12 H3_PROFILE=device cargo bench --locked --bench models -- 'dit_stack/eager/37977/1_layer$' --test
H3_BENCH_BUDGET_GIB=32 H3_PROFILE=device cargo bench --locked --bench models -- 'dit_stack/eager/37977/50_layers$' --test
```

DiT fuses Q/K normalization, rotary embedding and INT8 preparation from 4,096
tokens unless K smoothing is enabled. `H3_FUSED_OPERANDS=0` selects the separate
path for comparisons; the fused path preserves the intermediate FP16 rounding.

`H3_BENCH_RESIDENCY` selects `budgeted` (default), `retain`, or `stage-scoped` for
stage and API render benchmarks. A reused stage-scoped session reloads weights
between stages; its name records that policy. CLI cases always use stage-scoped
residency. Budgeted runs can evict idle models, but cannot evict a pinned DiT
while loading the text encoder for another prompt: 32 GiB is insufficient for
that retained pair. Use stage-scoped ownership for a 32 GiB complete render:

```sh
H3_BENCH_RESIDENCY=stage-scoped H3_BENCH_BUDGET_GIB=32 cargo bench --locked --bench pipeline -- text/res_multistep/warm_session --test
```

The allocation budget does not cap total process or system memory. Leave room
for mapped checkpoints, host packing, compiler memory, media encoding and other
applications. On UMA these compete with GPU allocations for physical RAM. Use
stage-scoped residency and a smaller budget on a shared machine; run GPU
benchmarks serially and watch system available memory as well as reservations.

`lora` defaults to the pinned Orbit adapter; `H3_BENCH_LORA` can override it.
`world` defaults to the pinned H3-World adapter; `H3_BENCH_WORLD_ADAPTER` can
override it. World uses Euler and a left-pan schedule. Turbo resolves the pinned
four/eight-step adapters. Turbo always uses its trained 1344×768×124
configuration and four/eight evaluations, even with the smoke profile.

Cache cases use at least eight evaluations so the middle steps exercise the
reuse policy. `forced_reuse` deliberately uses very large thresholds to measure
that execution path; it is not a quality-qualified cache preset.

## Measurement boundaries

- **Vision rotary tables:** `rotary/vision` measures table construction into reused
  buffers, including any per-call coordinate cache. Cases span 32px through
  landscape/portrait 768p, and the 3584px square maximum vision area. GPU uploads
  and output-buffer allocation are excluded.
- **Vision tokens:** `vision_tokens` measures copying prebuilt patch projections,
  FP32 position interpolation/addition, and allocation/conversion of FP16 residual
  rows. Cases span 32px through 768p in both orientations. GPU projection,
  readback and upload are excluded.
- **Video encoder input:** `video_encoder_input` measures ImageNet normalization
  and FP16 packing into reused eight-channel staging rows. Cases cover stills,
  full 17-frame tiles, cropped source frames and narrow edge tiles. Allocation
  and GPU upload are excluded.
- **Spatial video assembly:** `video_spatial` measures tile-buffer allocation,
  copying prebuilt decoded fixtures, overlap blending and planar output writes.
  Cases cover each overlap axis and full 28-frame windows at 480p/768p, with
  fresh and reused output buffers. Model execution is excluded; benchmark
  buffers stay below 1 GiB.
- **Temporal video assembly:** `video_temporal` measures window-buffer allocation,
  copying prebuilt decoded fixtures, temporal trimming, cross-fades and float RGB
  output writes. Cases cover short and multiple-window clips through 768p; peak
  benchmark buffers stay below 2 GiB. Model execution and RGB byte conversion are excluded.
- **Video decoder output:** `video_decoder_output` measures FP16 conversion and
  unpacking into channel-major float frames, from single tokens through a full
  7×16×16 tile. `reused` keeps the output allocation across calls, as tiled decoding
  does; `fresh` includes allocating it. GPU readback is excluded.
- **Video decoder input:** `video_decoder_input` measures the host 24×24
  post-quant projection and FP16 conversion into reused staging rows. Cases range
  from single voxels through a full 7×16×16 decode tile. Allocation, latent
  normalization and GPU uploads are excluded.
- **Host rotary tables:** `rotary/dit` includes allocation and construction of both
  tables, from a prebuilt packed layout and inverse frequencies. Cases cover the
  smoke grid, 480p/768p clips, references with a keyframe, and a 15-second 2x grid.
  GPU uploads and the rest of conditioning are excluded.
- **Stages:** setup and initial warmup excluded; API work, allocations, transfers
  and completed host output included. The denoising stage includes prompt
  conditioning and sampling, but excludes VAE decode and media encoding.
- **Warm render:** same session and runtime across samples. Each iteration
  tokenizes/presents inputs, encodes selected references, conditions, samples,
  decodes both modalities, writes WAV, and finishes H.264/AAC muxing.
- **Cold session:** the same render plus session construction, model loading,
  packing/uploads, and session teardown on each iteration. Compiler and OS
  filesystem caches remain available; this is not a disk-cache flush.
- **Cold CLI process:** process startup through successful output completion,
  including actual reference-file decoding, tokenizer/session construction and
  teardown. Compilation of the Rust binary is outside timing.
- **Lifecycle:** checkpoint mapping/plan construction, host packing, and packing
  plus completed upload are separate cases. Upload uses a fresh weight owner to
  prevent a cached lookup from replacing the transfer. `load_block` measures the
  four large projections of one DiT, text or video transformer block, with byte
  throughput and exact readback checks. OS file caches remain available.
  `checkpoint/dit/load_model` loads every planned DiT tensor under one weight
  owner and keeps the full set resident through validation. It includes native
  storage startup, allocation, packing and completed transfers, but excludes
  mapping/plan construction, inference kernel compilation and tensor validation.
  It uses a 32 GiB allocation budget by default (`H3_BENCH_BUDGET_GIB` overrides
  it); allow additional host memory for one tensor's reference and readback.
  Validation releases row-source mappings after each tensor, unless
  `H3_KEEP_MAPPED` is set. Recipes are visited in name order, not inference
  preparation order.
  On Linux, `H3_BENCH_DETAILS=1` also reports process CPU time, page-fault and
  input-block deltas for the loading interval, excluding validation. These help
  distinguish cache effects from changes in the loader; a fresh owner does not
  imply cold files.
  On Linux, `cold_file` copies representative DiT and text-encoder tensors to a private checkpoint
  under `target/`, flushes it, and evicts only that fixture before each iteration.
  It times allocation, disk faults, packing and completed upload; fixture setup,
  eviction and validation are excluded. Use a disk-backed workspace for this case.
  Eviction timing includes
  applying pressure, releasing it, and reloading/decoding the audio model.

Replay digests, finite checks and ffprobe validation run outside latency timing.
Model inputs are deterministic synthetic fixtures; full rendering defaults to seed 7.
Native output replay is checked for exact equality. Encoded frame counts follow
the production muxer's `-shortest` behavior: rounded audio can trim a partial
final video-frame interval (the five-frame smoke render encodes four frames).
These checks complement the
[independent correctness tests](testing.md); they do not establish visual quality.
Keep `H3_PROFILE`, `H3_COMPILE_REPORT_DIR` and stage tracing disabled during timing runs.

## HRX diagnostics

Collect diagnostics separately from Criterion latency samples:

```sh
H3_PROFILE=device H3_COMPILE_REPORT_DIR=target/h3-reports cargo bench --bench stages -- audio_encode/165600 --test 2>target/h3-profile.log
H3_PROFILE=device H3_COMPILE_REPORT_DIR=target/h3-reports cargo bench --bench kernels -- gemm/i8/swiglu/cached/2048x5376x28672 --test 2>target/h3-gemm-profile.log
```

GEMM cases include DiT QKV, FFN and down-projection dimensions at 256 and 2,048
rows. FP32 SwiGLU also covers 4,096 and 8,192 rows with cached and rotating
weights, using a 1 GiB GPU allocation budget. Residual down projections cover
4,096 rows; their residual is reset outside timing before each iteration. Other
GEMM cases use a 512 MiB budget. All use the runtime's operand pitches and direct
buffer initialization. Setup and output readback stay outside timing.
`prepare_f32_i8` covers tiny rows, the tiled FFN dispatch boundary, and batches
through 4,096 tokens, plus wider 25,600-value rows. It supports the same HRX
profiling and allocation cap, with direct readback outside timing.
`prepare_qk_i8` cases cover token-major and head-major operands at 1, 257, 2,048
and 8,192 tokens with the same allocation cap and HRX profiling support.
`qk_rotary_quantization` compares fused and separate Q/K preparation at 1, 257,
4,096 and 8,192 tokens. It checks exact codes and scales against the separate
path outside timing, with a 1 GiB allocation cap for the reference intermediates.
`attention_i8qkhm` runs the DiT attention kernel with all 56 heads at 1–8,192
tokens, including partial query and key tiles. It uses head-major INT8 Q/K,
transposed FP16 V, a 512 MiB allocation cap, and direct output checks outside
timing. HRX profiling is available for these cases too.
`rope_qknorm_f16` covers Q/K normalization with and without copying V.
`transpose_v_f16` covers separate V and fused QKV inputs at 1, 257, 2,048 and
8,192 tokens, checking exact half bits and zeroed padding across reused capacity.
`prepare_attention_output_i8` compares 128/224/448-thread workgroups for the
7,168-value FP16 attention output. Every case checks packed codes and scales
against the 128-thread kernel outside timing and uses a 512 MiB allocation cap.

`H3_PROFILE=device` uses HRX's owned graphs and device-clock markers. Each
`H3_GPU_PROFILE` JSON line records the stage, symbol, launch geometry, scalar
arguments, binding sizes, distinct retained allocation bytes, device intervals
and enclosing replay host time. Retained allocation bytes count shared backing
once and can exceed the sum of sliced binding sizes.
The diagnostic graph serializes each dispatch and adds markers/barriers; these
intervals locate costly kernels but do not measure ordinary graph overlap or
hardware utilization. Unsupported timestamp capture fails explicitly.
`H3_PROFILE=1` retains synchronized host timing instead.

For `models` graph cases, `H3_PROFILE=device` instruments the whole recorded
stack. `H3_GPU_GRAPH_PROFILE` reports all dispatch intervals in one device-clock
timeline, its span and gaps, distinct retained allocation bytes, and enclosing
replay host time. The graph keeps its dependency edges and avoids a separate
host submission/readback for every dispatch. HRX still inserts completion
barriers around each marker pair, so this remains a serialized diagnostic;
use the same case without profiling to measure ordinary graph replay.
Each timed batch's output is checked byte-for-byte against the eager reference
outside timing. Both one-block and 50-block cases support this capture:

```sh
H3_BENCH_BUDGET_GIB=12 H3_PROFILE=device cargo bench --locked --bench models -- 'dit_stack/graph/37977/1_layer$' --test
H3_BENCH_BUDGET_GIB=32 H3_PROFILE=device cargo bench --locked --bench models -- 'dit_stack/graph/37977/50_layers$' --test
```

Instrumented Criterion results use separate `profile-device` or `profile-host`
subdirectories, so they cannot replace ordinary latency baselines.

`H3_COMPILE_REPORT_DIR` writes one detailed Loom report per specialization,
including resources, complete wait reasons and evidence-backed guidance.
It compiles separate ordinary and analysis artifacts and requires identical
executable bytes before dispatch. Normal execution keeps its usual artifact.
Compiler occupancy and wait counts are models, not measured utilization or stall
time; unavailable evidence remains null. Reports and profiling logs stay ignored.

The `audio_qkv_f32` kernel cases compare row-major and packed FP32 projections
at one and 207 rows, checking byte-identical output outside timing.
`audio_encoder_conv_f32` isolates all five encoder levels: one- and seven-tap
residual convolutions, dilations 1/3/9, and stride 2/4/5 downsampling. Scalar,
four-sample and packed-channel kernels are compared at the same shapes where
supported, with exact output checks before and after timing. These cases use
synthetic weights and a 1 GiB allocation cap, without loading checkpoints.
The final 2048-channel projection also compares packed prefetch and three-tap
kernels at 1/4/207/500 latent frames, including the short-window dispatch choices.

## Memory

```sh
cargo bench --locked --bench memory -- text/res_multistep/StageScoped/sampled_peak_reservations --test
cargo bench --locked --bench memory -- sampled_peak_reservations
```

These are Criterion measurements in **MiB/render**, using a fresh session per
iteration. A separate sampling thread reads the residency manager's reservations
and Linux `/proc/self/status` RSS every 2 ms while the complete render runs.
Shorter allocation spikes may be missed. Reservations include native GPU buffers
and loader reservations; RSS covers the current process and excludes FFmpeg child
processes. RSS includes pre-existing process memory. These views overlap on UMA
and must not be summed. They are separate from latency runs because sampling
adds overhead. Session teardown must return reservations to zero.

## Baselines

```sh
cargo bench --locked --bench pipeline -- text/res_multistep/warm_session --save-baseline before
# Make the change and run correctness tests.
cargo bench --locked --bench pipeline -- text/res_multistep/warm_session --baseline before
```

Keep hardware, checkpoints, adapter files, profile, budget and runtime options
fixed. Defaults are ten samples, one second of warmup and a three-second target
measurement window; slow cases necessarily exceed that window. Criterion's
`--sample-size`, `--warm-up-time` and `--measurement-time` control longer runs.
Benchmarks read Loom sources from their build checkout's `kernels/` directory.
For alternating kernel comparisons, build in separate worktrees and retain both
source trees; copying only the executable does not freeze its kernels.

The suite covers the local inference and media pipeline. External prompt/LLM
services, network downloads, disk-cache eviction and long World save/resume
rollouts are outside these benchmarks.

## Latent upscaler

`cargo bench` includes the `upscale` Criterion target: checkpoint metadata,
CPU convolution-weight packing, cold/warm learned-network execution,
reference-conditioned refinement, and both generation passes through video/audio decoding.
Network cases honor `H3_BENCH_PROFILE` (`smoke`, `480p`, or `768p`) with 2× spatial
upscaling; refinement and pipeline cases use 64×64 → 128×128 and five frames.
Existing `H3_BENCH_BUDGET_GIB`, residency,
attention and HRX profiling controls apply.

```sh
cargo bench --bench upscale
cargo bench --bench upscale -- upscale/pack
H3_PROFILE=device cargo bench --bench upscale -- upscale/network/warm_weights --test
H3_BENCH_PROFILE=480p H3_BENCH_BUDGET_GIB=4 cargo bench --bench upscale -- upscale/network/warm_weights
cargo bench --bench upscale -- upscale/groupnorm_stats
cargo bench --bench upscale -- upscale/groupnorm_apply
cargo bench --bench upscale -- upscale/conv3d
cargo bench --bench upscale -- upscale/temporal
```

Network profiles separate convolutions, whole-volume normalization, temporal
convolutions, interpolation and residual updates. Results remain under `target/`.
GroupNorm statistics cases sweep whole-volume row counts and workgroup sizes
within a 256 MiB device budget, check against FP64 mean and centered variance,
and verify deterministic replay outside timing.
GroupNorm apply cases use up to 1 GiB, check modulation with FP16 rounding at
each intermediate boundary, and verify finite, deterministic output outside timing.
Spatial and temporal convolution cases cover small tensors and large 512-channel
volumes within a 1 GiB residency budget, including readback. They check sampled interior and
boundary outputs against FP64 and verify finite, deterministic output outside timing.
The `conv3d_residual` cases include the fused residual addition and use a 2 GiB
budget to accommodate the extra input and readback.
Packing cases use the released input, residual-block and output convolution
weights without opening a GPU. They check every packed byte and padding byte
against the original checkpoint layout outside timing.

## Storage and profiler comparisons

Set `H3_BENCH_WEIGHT_IO=mapped|native-buffered|native-direct` and
`H3_BENCH_STORAGE_PROGRESS=sqpoll|wait` for lifecycle and model loading tests.
`H3_BENCH_STORAGE_SLOTS` and `H3_BENCH_STORAGE_SLOT_MIB` each accept 1..64;
unset values retain HRX's four 16 MiB slots. These controls reach fresh CLI
processes too. Each route, capacity and statistics setting has a separate
Criterion directory. `H3_BENCH_STORAGE_STATISTICS=1`
collects native loading intervals; `H3_BENCH_DETAILS=1` prints completed tensor
loading times and counters outside the measured interval. Cold-file tests evict
only their private disk fixtures. Keep at least seven alternating independent
samples before comparing medians; measure warm loading and resident inference
as well as cold loading before changing defaults.
The completed `weight-load`/`weight-block` durations include native session
startup; `WeightStatistics::total` starts after that session is constructed.
Do not treat fresh-session single-tensor latency as sustained whole-model
throughput.

Run comparisons directly through Criterion:

```sh
for mode in mapped native-buffered native-direct; do
  H3_BENCH_WEIGHT_IO="$mode" cargo bench --locked --bench lifecycle -- checkpoint/cold_file
done
```

Repeat in rotating route order, and use `pack_and_upload` for warm loading.
For a separate diagnostic run, enable storage statistics and print the phases:

```sh
H3_BENCH_WEIGHT_IO=native-direct H3_BENCH_STORAGE_PROGRESS=wait \
  H3_BENCH_STORAGE_STATISTICS=1 H3_BENCH_DETAILS=1 \
  cargo bench --locked --bench lifecycle -- checkpoint/cold_file --test
```

`/usr/bin/time -v` can wrap a built benchmark executable to collect process CPU
time, peak RSS and page faults. Use `H3_PROFILE=device` for HRX dispatch timing.
Collect these separately from latency samples. Raw native HRX queues may have
no ROCprofiler dispatch coverage; an empty trace is not evidence of idle GPU
time. Kernel replay is unsuitable for file I/O because its external side
effects cannot be restored between passes.

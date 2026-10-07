# Benchmarks

The [Criterion](https://criterion-rs.github.io/book/) suite covers host preparation, individual model stages, complete
renders, loading, eviction, and sampled peak memory. Run `cargo bench` for the
whole suite. Cargo runs benchmark executables sequentially; use an idle machine.
Results and baselines belong in ignored `target/criterion/`, never in Git.

## Targets and coverage

| Target | Coverage |
| --- | --- |
| `host` | Short/long tokenization, mixed-media presentation, image resizing, RefMod loading/strength/copies, packed sequence layout |
| `kernels` | FP32 Hadamard preparation, INT8 attention preparation in both layouts, V transpose, INT8/BF16 GEMMs with FP32 outputs including DiT projection dimensions, cached/rotating weights, eager/graph dispatch |
| `models` | Resident one-block DiT eager/graph comparison through 8,192 tokens, covering both integer attention layouts, and short audio roundtrip |
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

```sh
H3_BENCH_PROFILE=480p cargo bench --locked --bench pipeline -- text/res_multistep/warm_session
H3_BENCH_PROFILE=768p cargo bench --locked --bench stages -- denoise/
H3_GRAPH=1 cargo bench --locked --bench pipeline -- text/res_multistep/warm_session
H3_BENCH_ATTN=f16 cargo bench --locked --bench stages -- denoise/
```

`H3_BENCH_ATTN` selects `i8` (default), `f16`, or experimental `i4` attention.
`H3_GRAPH` is read once by the runtime: compare it in separate processes.
Case names include attention, graph mode and the allocation budget to keep
incompatible baselines separate. `H3_BENCH_BUDGET_GIB` defaults to 64 for the
stage/render/lifecycle/memory suite and 32 for the `models` benchmarks. Profile
a single DiT block with a smaller cap:

```sh
H3_BENCH_BUDGET_GIB=2 H3_PROFILE=device cargo bench --locked --bench models -- dit_stack/eager/2048 --test
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
  On Linux, `cold_file` copies representative DiT tensors to a private checkpoint
  under `target/`, flushes it, and evicts only that fixture before each iteration.
  It times allocation, disk faults, packing and completed upload; fixture setup,
  eviction and validation are excluded. Use a disk-backed workspace for this case.
  Eviction timing includes
  applying pressure, releasing it, and reloading/decoding the audio model.

Replay digests, finite checks and ffprobe validation run outside latency timing.
Model inputs are deterministic synthetic fixtures; full rendering uses seed 7.
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
rows. Residual down projections also cover 4,096 rows, with cached and rotating
weights; their residual is reset outside timing before each iteration. Cases use
the runtime's operand pitches, direct buffer initialization and a 512 MiB GPU
allocation budget. Setup and output readback stay outside timing.
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

The suite covers the local inference and media pipeline. External prompt/LLM
services, network downloads, disk-cache eviction and long World save/resume
rollouts are outside these benchmarks.

## Latent upscaler

`cargo bench` includes the `upscale` Criterion target: checkpoint metadata,
CPU convolution-weight packing, cold/warm learned-network execution,
reference-conditioned refinement, and both generation passes through video/audio decoding. Inputs are bounded to
64×64 → 128×128 and five frames. Existing `H3_BENCH_BUDGET_GIB`, residency,
attention and HRX profiling controls apply.

```sh
cargo bench --bench upscale
cargo bench --bench upscale -- upscale/pack
H3_PROFILE=device cargo bench --bench upscale -- upscale/network/warm_weights --test
```

Network profiles separate convolutions, whole-volume normalization, temporal
convolutions, interpolation and residual updates. Results remain under `target/`.
Packing cases use the released input, residual-block and output convolution
weights without opening a GPU. They check every packed byte and padding byte
against the original checkpoint layout outside timing.

# Benchmark reference

Criterion coverage, timing boundaries and HRX controls. For a first measurement
or a performance comparison, start with the [benchmark guide](performance.md).

## Targets and coverage

| Target | Coverage |
| --- | --- |
| `host` | Short/long tokenization, mixed-media presentation, image resizing, RefMod loading/strength/copies, packed sequence layout |
| `attention` | Head-128 FP16 four/eight-wave tiles and 16-head vision attention with 72 populated channels, including 480p/768p token counts and ragged lengths, checked against sampled FP64 attention |
| `kernels` | FP32 Hadamard preparation, INT8 attention preparation in both layouts, V transpose, INT8/BF16 GEMMs including all four DiT projections through 37,977 rows and all vision projections at 64px/480p/768p, cached/rotating weights, eager/graph dispatch |
| `models` | Resident INT8 DiT eager/graph comparison through 37,977 tokens including partial tiles, a complete 50-block 768p sequence in both modes, and short audio roundtrip |
| `stages` | Complete text encoder plus token refiner, vision tower, video encode/decode, audio encode/decode, complete 50-block denoising trajectories with Euler and ResMultistep |
| `lifecycle` | Mapping/planning, tensor packing and completed uploads, block loading throughput, cold-file loading, forced audio eviction/reload |
| `pipeline` | Complete text, first/last-frame, image/audio reference, video/audio reference, RefMod, LoRA, Turbo and World renders; cache observation/reuse; WAV and H.264/AAC output; fresh CLI processes |
| `memory` | Complete text and reference renders under stage-scoped and budgeted residency, measuring sampled peak reservations and process RSS separately |
| `upscale` | Weight packing, learned-network execution, reference-conditioned refinement and two-pass generation |

Host resize cases cover 1080p to 480p/768p, enlargement, unchanged dimensions,
and a large downscale. They exercise the reference/keyframe bilinear path:

```sh
cargo bench --bench host -- resize/
```

`cargo bench --bench kernels -- vision_gemm/` isolates the vision patch, QKV,
attention projection, MLP and merger/DeepStack GEMMs. It uses model dimensions,
sampled FP64 checks with BF16 operand rounding, and exact replay checks. Both
cached weights and a rotating weight set exceeding 64 MiB are measured within
a 512 MiB residency budget. Allocations and the ring position persist across
Criterion samples, including single-iteration samples. Compilation, allocation,
correctness readback and residual restoration are outside timing.
The attention projection uses the production zero-padded heads and skips their
inactive fragments; throughput counts only the active K extent. Native tests
compare this specialization bit for bit with the general residual GEMM.
Use the complete `stages` vision cases
to verify whether an isolated kernel improvement helps the tower.

`cargo bench --bench kernels -- vision_layernorm/` covers the block and merger
widths at 64px/480p/768p row counts, plus small and maximum-width cases. Its
resident fixture persists across samples, with readback and exact replay checks
outside timing. Native tests check FP64 results, constant and low-variance rows,
both sides of the register-cache cutoff, scalar/vector stores and output guards.

`cargo bench --bench attention` keeps each attention fixture resident across
warmup and samples within a 512 MiB budget. A sampled FP64 oracle checks the
initial output, and every sample checks exact replay outside timing. Use full
vision stages to confirm that an isolated attention gain helps the tower.

Video stage cases cross spatial tile overlaps horizontally and vertically and
use 5/22/39/56 frames across temporal chunks. Audio cases cover the 800-sample
hop and 255/256/257 latent-frame boundaries, plus a full 124-frame soundtrack.
The 480p/768p profiles add their full-size vision and video-decoder workloads.

`cargo bench --bench stages -- conditioning/` measures a fresh stage-scoped
session through text encoding and refinement, including model loading and
compiler setup. It covers the selected prompt and the longer `tidal_sky` prompt,
checking output digests across repetitions. The warm `text_refiner` cases also
use the selected prompt length. Whole-model loading alone does not reproduce
the scattered embedding reads during conditioning.

`models` stack cases use synthetic inputs and modulation tables with all 50
checkpoint blocks and a shared activation workspace. Completion waits are
included; conditioning, solver updates and decoding are measured by the stage
and render targets.

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
Eligible INT8 DiT projections also write V directly in attention order, keeping
compact Q/K rows and V in the existing allocation. `H3_COMPACT_QKV=0` selects
the separate V transpose for comparisons. Adapter-enabled stacks retain the
original projection layout.

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
- **Vision rotary kernel:** `vision_rotary` measures resident FP32 QKV rotation,
  FP16 conversion and head padding with the tower's 16 heads. Cases include
  480p/768p token counts and partial workgroups. Compilation, allocation, uploads
  and output verification are outside timing; complete tower timing is in `stages`.
- **Vision tokens:** `vision_tokens_cpu` measures the CPU reference: copying
  prebuilt patch projections, FP32 position interpolation/addition, and
  allocation/conversion of FP16 residual rows. Cases span 32px through 768p in
  both orientations. GPU projection,
  readback and upload are excluded. The runtime keeps position addition and FP16
  conversion on the device; the complete `stages/.../vision` cases include this path.
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
  `checkpoint/{dit,text}/load_model` loads every planned tensor of the selected
  model under one weight owner and keeps the full set resident through validation.
  It includes native storage startup, allocation, packing and completed transfers,
  but excludes mapping/plan construction, inference kernel compilation and tensor
  validation. The text plan also includes the vision tower's weights.
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
weights, using a 1 GiB GPU allocation budget. All four DiT projections also cover
15,666 and 37,977 rows with a 4 GiB budget. Long INT8 FP32 SwiGLU uses the
192×256 tile from 32,768 rows; smaller sequences use the smaller tile.
Compact QKV, INT8 residual projections, and long-sequence SwiGLU specialize
address arithmetic for at most 65,536 rows. Dispatch checks the actual row count
and retains a general kernel for larger calls, including when an operator is
reused with more rows.
Residual down projections cover 4,096 rows; their residual is reset outside
timing before each iteration. Other GEMM cases use a 512 MiB budget. All use
the runtime's dispatch choices, operand pitches and direct buffer initialization.
Setup and output readback stay outside timing.
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

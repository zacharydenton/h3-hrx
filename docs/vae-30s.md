# Decoder target: 30 seconds

The goal remains **at most 30 seconds on the original workload and timing
boundary**. It has not been reached. The current representative resident
video-plus-audio measurement is **33.02 seconds**, with preserved output parity.
This is one uncontended resident comparison on a saved 124-frame clip; a second
confirmation run was invalidated by unrelated GPU jobs starting during decode.
These measurements are from the Radeon 8060S after GPU use became available.

## Current implementation

The f16 video decoder now uses 128x256 GEMM workgroup tiles, eight 64x64 waves
and 64-element K stages. The accumulation order stays ascending in 16-element
WMMA steps. A compiler scheduling fence keeps one A packet and one W packet
prefetched before the multiplies; other loads remain freely scheduled. SwiGLU writes directly at the padded down-projection input pitch,
eliminating the intermediate repacking launch. The host selects row groups and
allocates activations using the chosen tile geometry and output pitch.

The standard 2048-channel video decoder fuses the QKV projection with per-head
normalization and rotary embedding. It retains the original saturated f16
projection boundary before f32 RMSNorm, and rotates the same 48 channels. V
bypasses normalization and rotation. The fused kernel writes
`[Q/K/V][32 heads][capacity][64 channels]`; attention reads those three component
bases from one allocation. This removes the separate rotary launch and QKV
intermediate traffic. Other stack configurations retain the separate path.

Head-64 attention stages 32 keys at a time. All 128 lanes load distinct K/V
packets, eliminating the duplicated staging of the previous 16-key mapping.
It prefetches the next K/V tile into registers while computing the current tile,
with a safe unused reload of tile zero on the last iteration. The transposed formulation computes K Q^T and V^T P^T, so each lane pair
tracks one query with scalar max/sum statistics. Softmax uses base 2 and masks
only a partial final key tile. Probabilities are narrowed before the lane
exchange and packed directly into the PV operand; the same packing writes
contiguous output channels. This removes the probability and output LDS round
trips. The current kernel uses 9,728 bytes of LDS, 192 VGPRs and zero private
scratch at the decoder configuration. It retains f32 accumulators and f16 output.

Audio convolutions compute four samples per thread with an unrolled tap loop.
They retain the original float32 accumulation order and output bits.

CPU unpatchifying converts 16 contiguous half values per row with F16C on
supporting x86 CPUs, with the scalar implementation as a fallback. RGB rounding
uses the same f32 scaled product and halfway-away-from-zero rule as before.
Checkpoint precision, spatial and temporal tiling, rotary coordinates and
blending remain unchanged.

`H3_VAE_FAST=0` restores the original video decoder GPU kernels for comparisons.
`H3_VAE_WIDE=1` explicitly selects the slower experimental 256x256 tile.
The CPU conversion and four-sample f32 audio convolution improvements apply to all variants.

## Initial video comparison, before prefetch changes

| 864x480, 124 frames | Original GPU kernels | Initial 128x256 GPU kernels |
| --- | ---: | ---: |
| First decode in a fresh decoder-only session | 41.07 s | 35.91 s |
| Second decode, resident weights | 40.34 s | 35.37 s |

The resident improvement is 12.3%. Output differs by at most one RGB byte,
with 68.33 dB PSNR between variants over all 124 frames. Logs are in
`build/vae-wide/base-cpu124.log` and `fast-cpu-coop124.log`.

This is **video-only** timing. The input repeats the real saved 22-frame latent
`build/py22_video.npy` along time and truncates it to 37 latent tokens. It matches
the original output geometry but is not the original 124-frame latent input.
The seven temporal chunks each contain seven latent tokens; the spatial split
is 5 by 3 tiles, each with 1797 transformer tokens including registers and CLS.

The old `build/review/warm.log` records 44.1 seconds unprofiled and 45.5 seconds
profiled, with 39.6 seconds of video kernel stages. The CLI's `decoded in` interval
includes video, audio and session destruction, including lazy decoder loading.
It can also free resident DiT/text weights. Neither the video-only measurements
above nor a sum of profiled kernel stages establishes that this interval meets
30 seconds. `H3_PROFILE` synchronizes around launches and measures host wall time;
use unprofiled runs to judge latency.

## Video plus audio

`tools/bench_vae_decode.py` on repeated `build/lat_hip.npz` at the same 124-frame
geometry measured **41.71 → 37.97 seconds resident**, including audio decode.
First-call totals were 42.54 and 37.94 seconds. Teardown of the decoder-only
sessions took 0.062 and 0.080 seconds, reported separately. The fast video stage
in this run took 36.30–36.40 seconds; the results show some workload/run variation
relative to the video-only comparison. RGB parity was 68.37 dB, maximum one byte.
The machine-readable record is `build/vae-wide/decode-av.json`. These results
still leave roughly eight seconds to remove from video-plus-audio decode.

## Prefetch follow-up

The compiled 128x256 kernel originally sank its nominal next-chunk loads after
all four WMMA substeps. Keeping two packets early with `scf.schedule.fence`
improved the rotating-weight measurements (mean of two alternating rounds):

| Projection | Before scheduling fence | Two packets prefetched early |
| --- | ---: | ---: |
| Gate/up | 3.598 ms | 3.368 ms |
| Down | 1.772 ms | 1.615 ms |
| QKV | 1.389 ms | 1.316 ms |
| Out | 0.479 ms | 0.457 ms |

The selected gate/up kernel uses 224 VGPRs and zero private scratch bytes.
The all-packet variant uses 232 VGPRs and also reports zero private scratch;
its initial large slowdown did not repeat, so register spilling is not an
established explanation. A two-round recheck averaged 3.406 ms for all packets
versus 3.386 ms for the selected pair.
Keeping every packet early, blocking the operand loads by column, and batching
all 15 tiles did not establish a larger consistent gain. A follow-up with
sixteen 32x64 waves and early prefetch also lost: gate/up averaged 3.571 ms
versus 3.331 ms for the selected eight-wave kernel, and down averaged
1.741 ms versus 1.612 ms (`build/vae-wide/wave32-prefetch.json`). The attention prefetch
passed at 13, 32, 40, 64, 65, 96, 517 and 1797 tokens; its 1797-token test took
about 1.02 ms versus 1.07 ms before prefetching.

On the repeated `build/lat_hip.npz` video-plus-audio workload, the prefetch version
measured **35.85 seconds resident** (34.31 video + 1.54 audio), versus 37.97 seconds
before the prefetch changes and 41.71 seconds with the original GPU kernels.
The first call took 36.76 seconds and teardown took 0.063 seconds separately.
See `build/vae-wide/decode-av-prefetch.json`. The original 30-second target is
still unproven and unmet on this representative input. That version
produces bit-identical RGB frames to the previously validated candidate on
`build/py22_video.npy`, preserving its 63.94 dB comparison with diffusers
(`build/vae-wide/parity-prefetch.log`).

Sampled gate/up hardware counters before the scheduling change reported about
7.8 resident waves per CU, 33–35% `MemUnitBusy`, 44–47% L2 hits and zero
`LDSBankConflict`. These counters have different scopes and are not a direct
DRAM-bandwidth measurement. The records are under `build/vae-wide/counters-*`;
the sampled decoder runs used ROCProfiler only for diagnosis, not acceptance
timing. GPU instruction dumps in `build/vae-wide/*-isa.txt` document the load
ordering. Loom must support `scf.schedule.fence` to compile the current kernels.

## Transposed attention follow-up

A 32-query-per-wave, 16-key candidate was correct but slower. The initial
version used 140 private scratch bytes; scheduling fences alone still spilled
160 bytes. Moving half the query fragments to LDS reduced this to 28 bytes,
and staging all query fragments eliminated spills (232 VGPRs, 27,904 LDS bytes),
but still took 1.383 ms at 1797 tokens. A native subgroup max reduction compiled
to fused DPP maximum instructions but did not establish a timing improvement.
These candidates are archived locally under `build/vae-wide/rejected/`.

The retained four-wave transposed 32-key kernel avoids those costs. Its short
1797-token test took 0.864 ms, versus about 1.02 ms for the previous formulation.
An eight-wave variant passed numerically but took 1.028 ms, so it was rejected.
These microbenchmarks are directional evidence, not decoder acceptance timing.

On the saved 22-frame video, consecutive resident runs measured 4.758/4.793 s
before and 4.712/4.712 s after the attention change. Output differed by at most
one RGB byte, with 69.07 dB PSNR between variants. The independent diffusers
comparison remained **63.94 dB**, including decoder-only isolation and five-frame
decoding (`build/vae-wide/parity-transpack.log`). The generated production
kernel also passed at 13, 32, 40, 64, 65, 96, 517 and 1797 tokens, and the CPU
suite passed (`attention-transpack-final.log`, `cpu-transpack-final.log`). The final production
output is bit-identical to that independently validated candidate on the saved
22-frame input (`parity-transpack-production.log`).
Extra query waves keep clamped reads but no longer duplicate a valid wave's
output writes.

At that stage, the repeated `build/lat_hip.npz` workload measured **34.58 seconds
resident** (33.042 video + 1.541 audio), versus 35.85 seconds before this attention
change. The first call took 35.15 seconds; decoder-only teardown was 0.063 seconds
separately (`build/vae-wide/decode-av-transpack.json`). This is still the synthetic
124-frame geometry described above, rather than the input used for the original 44.1-second measurement. This left another 4.6 seconds to remove from the representative resident
interval before the subsequent fusion change below.

## Fused QKV and head-major attention

Fusion with row-major output passed the numerical checks but regressed the
22-frame decoder: 4.803/4.812 seconds versus 4.646/4.727 seconds current. Profiling
showed attention rising from 0.60 to 0.71 seconds after the allocation change;
the fused projection itself was roughly tied with GEMM plus rotary embedding.
An isolated sum of separate kernel timings had predicted a small improvement,
so it was insufficient evidence for promotion. Skipping unused V normalization
did not materially improve that version.

Publishing head-major Q/K/V and consuming it directly in attention produced a
full-decoder improvement. Resident 22-frame runs measured 4.611/4.590 seconds,
against 4.660/4.730 seconds before and 4.728/4.718 seconds when current code was
rerun afterward. RGB differed by at most one byte, with 69.08 dB PSNR. The fused
kernel preserves the f16 projection rounding, but its f32 normalization sum has
a different reduction order from the separate rotary kernel.

On the repeated 124-frame NPZ, resident video plus audio improved from **34.47
to 33.80 seconds** in the paired prototype comparison. RGB maximum difference
was one byte, PSNR 68.41 dB over all 124 frames. The rebuilt production library
then measured **33.81 seconds resident** (32.269 video + 1.544 audio), with
34.37 seconds on its first call and 0.062 seconds decoder-only teardown reported
separately. This is the same representative geometry and timing boundary used
above, not the input used for the original 44.1-second measurement.

Independent diffusers parity passed at **63.95 dB**, including decoder-only
session isolation and five-frame decoding. Production RGB was bit-identical to
that independently validated candidate on the saved 22-frame input. The full
CPU suite, fused projection numerical gates and head-major attention gates all
passed. Host checks cover component offsets, padded allocation, row-group tails,
the shared attention capacity and omission of the separate rotary launch. GPU
checks include zero heads, saturated projections, ragged tokens, exact V output
and untouched padding, with an independent float64 normalization/rotation oracle.
At the 1797-token configuration, fused QKV uses 231 VGPRs and 55,296 bytes LDS;
head-major attention uses 192 VGPRs and 9,728 bytes LDS. Both have zero private
scratch. Prototype sources are retained in `build/vae-wide/prototypes/qkv-fusion/`.

Evidence is under `build/vae-wide/`: `decode-qkvropehm22.json`,
`decode-qkvropehm-av124-{current,fused_headmajor}.json`,
`decode-qkvhm-production.json`, `parity-qkvropehm.log`,
`qkvhm-production-parity.log`, `qkvhm-cpu-final.log`, `qkvhm-host-tests.log`,
`qkvhm-gpu-final.log` and `qkvhm-attention-final.log`.


## What the GEMM experiments showed

The operand traffic estimate `2*M*K*N*(1/tile_n + 1/tile_m)` is a useful comparison
of workgroup loads; it is not a measurement of physical DRAM traffic. Enlarging
256x128 to 256x256 cuts this term by one third. Exchanging it for 128x256 leaves
the term unchanged. Cache reuse, occupancy, barriers, register pressure and tail
work determine whether either change saves time.

Representative separately allocated, rotating-weight measurements at M=1797:

| Projection | Original 256x128 | 128x256, before padded SwiGLU output |
| --- | ---: | ---: |
| Gate/up | 3.789 ms | 3.644 ms |
| Down | 2.181 ms | 1.745 ms |
| QKV | 1.487 ms | 1.381 ms |
| Out | 0.561 ms | 0.468 ms |

The original gate/up kernel already sustained roughly 32 TFLOP/s while rotating
through 36 weight allocations. These measurements do not reproduce the proposed
1.9x single-weight cache advantage for the current padded kernels.

The 256x256 tile passed numerical checks but lost: its 32-element K stage was
about 10–16% slower in the initial comparison. A 64-element bank-swizzled stage
was also slower (gate/up 4.928 ms versus 3.757 ms). A 64-key attention tile
passed numerical checks but took 1.370 ms at 1797 tokens, versus 1.071 ms for
the retained 32-key tile. Splitting K/V staging across waves at 16 keys also
lost (1.222 ms). Smaller K stages, larger row
groups and additional LDS prefetching did not provide a consistent improvement.
The retained wide generator can generate either K stage; its checked-in kernels
use K=64. Other unsuccessful prototypes and logs are archived locally in
`build/vae-wide/rejected/`.

## Follow-up experiments: no production change

The next round tested fixed WMMA schedules, unrolled output loops, direct output
stores and a 192x256 intermediate workgroup tile. None supplied a useful decoder
improvement. Rotating-weight measurements below use 36 weight copies, 144
launches and three alternating rounds at 1797 tokens. Each pair is its own
comparison; do not compare baselines from different pairs as an improvement.

| Experiment | Gate/up, current / candidate | Down | QKV | Out |
| --- | ---: | ---: | ---: | ---: |
| Fully unrolled output loops | 3.421 / 3.381 ms | 1.594 / 1.597 | 1.305 / 1.271 | 0.452 / 0.448 |
| Direct scalar output stores | 3.361 / 3.579 ms | 1.612 / 1.685 | 1.311 / 1.431 | 0.456 / 0.531 |
| 192x256, 12 waves, swizzled K64 stage | 3.386 / 4.139 ms | 1.585 / 1.956 | 1.307 / 1.540 | 0.459 / 0.526 |

Four fixed multiply-group schedules measured 3.975–4.022 ms for gate/up against
3.384 ms current. The 16-multiply row-order helper used 253 VGPRs and 32 bytes of
private scratch; the production kernel used 224 VGPRs and zero scratch. The
unrolled and direct-store candidates also used 224 VGPRs and zero scratch.

The first 192x256 prototype used a 64,512-byte padded operand stage and masked
the final partial weight packet. It compiled with 256 VGPRs and 32 bytes of
scratch. Its timing run was contaminated by another GPU process and is
inconclusive. Moving that masked prefetch later increased scratch to 192 bytes;
using masked memory operations retained the 32-byte spill. The final swizzled
revision uses 61,440 bytes LDS, 240 VGPRs and zero scratch. It stores the extra
unused weight rows separately, avoiding duplicate LDS writes. After fixing an
initial generator error, it passed all numerical and padding checks, but the
clean rerun above still lost. Removing a spill alone did not make the tile win.

The output-loop and final intermediate-tile candidates passed plain, residual
and SwiGLU tests across ragged tile boundaries and all four production
projection shapes. Direct stores and fixed schedules passed the smaller
ragged-shape gates; they were not promoted or independently checked through the
full decoder. Logs and JSON are `build/vae-wide/gemm-{locked,epilogue,direct,
midwide,midswz}*`; compiled resource reports are `*-schedule-gu-metadata.txt`.

Other follow-ups also remain experimental. Sustained 64-key transposed attention
tied the current 32-key kernel (991 versus 992 microseconds); extra scheduling
fences changed this by at most about 2%. Head-major Q/K/V storage produced
bit-identical attention output and about a 3.8% isolated improvement (935 versus
972 microseconds). Its later integration with fused QKV output is described
above. A 96x64 wave tile with 192x128 workgroups passed numerical checks
but lost to the selected 64x64 waves. These prototypes and the scheduling
sources are archived in `build/vae-wide/rejected/followup-schedules/`, with their
drivers and measurement records retained under `build/vae-wide/`.

## Bounded load scheduling follow-up

Fixed WMMA helpers prevented four global prefetch packets from moving past the
helper. Moving those four loads explicitly after the multiplies removed the
32-byte scratch spill and reduced usage to 232 VGPRs. Four helper orderings then
measured 3.354–3.404 ms gate/up versus 3.397 ms current, without a useful verified
full-decoder gain (`gemm-late.json`). Removing accumulator copies altogether
failed tied-register allocation; changing copies to moves retained the spill.

A native LDS helper kept only two weight fragments live at once and prefetched
next-step A fragments as old ones died. It used 216 VGPRs with zero scratch, but
was 1–4% slower on the four projection shapes (`gemm-lds-native.json`). Spending
that register space on more global prefetching produced these rotating-weight
means, again with 36 copies, 144 launches and three alternating rounds:

| Early packets | Gate/up | Down | QKV | Out |
| --- | ---: | ---: | ---: | ---: |
| Current | 3.370 ms | 1.620 ms | 1.317 ms | 0.462 ms |
| Three | 3.376 | 1.617 | 1.325 | 0.459 |
| Four | 3.327 | 1.638 | 1.294 | 0.449 |
| Six | 3.368 | 1.659 | 1.315 | 0.456 |

The four-packet candidate used 224 VGPRs and zero scratch. These are small,
projection-dependent differences; none has replaced the integrated GEMM.
The three-, four- and six-packet families passed actual gate/up and down shapes
at 13 and 517 tokens. One small three-packet plain configuration (K128/N512,
M-group 2) failed the compiler's cyclic wait-frontier fixed-point check before
GPU launch. Logs and numerical gates are `gemm-lds-prefetch*`; prototype sources
are archived under `build/vae-wide/rejected/lds-schedules/`.

## Audio convolution follow-up

Profiling the 207-latent stereo audio decode put about 1.25 of 1.54 seconds in
residual convolutions. The new `conv1d4_f32` gives each lane four independent
output samples, so 64 lanes still cover 256 samples per workgroup. It unrolls
the short tap loop and skips per-tap padding checks only when the complete
workgroup is interior. Boundary groups preserve conditional loads and FMAs;
each sample retains ascending input-channel/tap accumulation and the original
residual-plus-bias rounding.

Three alternating resident audio comparisons measured 1.542–1.543 seconds
before, 0.893–0.895 seconds with four accumulators, and **0.762–0.764 seconds**
with four accumulators and unrolled taps (`audio-multi.json`). Complete audio
outputs were bit-identical on the representative repeated 207-latent input and
random inputs of length 1 and 7. This is an audio-only measurement; combined
decoder acceptance remains separate.

The integrated kernel passes 66 CPU compile configurations, GPU checks of
ragged sample counts and actual decoder convolution shapes, and output guards.
The small cases also use an independent float64 oracle. The full CPU suite
passes (`audio4-cpu-final.log`); GPU gates are in `conv4-gpu.log`. The rebuilt
decoder is bit-identical to the previous audio implementation, and the existing
diffusers audio reference check passes at **109.4 dB SNR**
(`audio4-reference.log`).

A saved genuine 124-frame input is available in `build/fox_480p_5s_latents.pt`.
Its video tensor `[1,24,37,30,54]` is finite. Its stereo audio contains 32
nonfinite values at time zero; the benchmark replaces those with zero, as the
existing `tests/test_pipe.py` audio reference does. This treatment is explicit
in `decode-audio4-fox124.json`; the derived NPZ is
`build/vae-wide/fox124-audio-cleaned.npz`. It is not established that these are
the latents used for the older 44.1-second CLI measurement.

| First uncontended resident pair, 864x480x124 | Before audio change | Integrated audio kernel |
| --- | ---: | ---: |
| Video | 32.506 s | 32.259 s |
| Audio | 1.543 s | 0.763 s |
| Combined decode | 34.048 s | **33.022 s** |

Video output is byte-identical and audio output bit-identical between libraries.
The small video timing difference is run variation; only audio implementation
changed in this pair. The first call of the new library took 34.010 seconds,
and decoder-only teardown took 0.071 seconds separately. Two unrelated GPU
processes started during the second new-library resident run: that 98.739-second
timing is explicitly marked invalid in the JSON, while its output checks still
passed. Repeat uncontended combined timing when the GPU is available.

Prototype sources are retained in `build/vae-wide/prototypes/audio-conv/`.
The diagnostic host build
`build/vae-wide/libh3pipe_video_hostprofile.so` separates CPU assembly and RGB
conversion from GPU time; its findings are below.

## Remaining-cost diagnosis and pending candidates

A diagnostic 22-frame run on the first seven saved fox latent tokens attributes
4.364 of 4.510 resident video seconds to GPU execution and copies. Measured CPU
phases are 0.003 seconds for clip input preparation, 0.036 for unpatchifying,
0.068 for spatial assembly and 0.029 for RGB conversion
(`video-hostprofile22.log`). The first call included compilation and is not a
resident timing. These measurements put the remaining large opportunity in the
GPU stages: gate/up 1.81 seconds, down 0.87, fused QKV 0.80 and attention 0.51
on that sample. CPU conversion cannot supply the remaining three seconds alone.

An experimental extension of the GEMM row-group range tests scheduling order
without changing tile shape, arithmetic or layouts. The following means use
36 rotating weight copies, 144 launches and three alternating rounds at 1797
tokens; the harness checks for unrelated GPU clients before and after each
measurement (`gemm-groups.json`).

| Row group | Gate/up | Down | Plain QKV | Out |
| --- | ---: | ---: | ---: | ---: |
| Current, 3 | 3.422 ms | 1.626 ms | 1.329 ms | 0.458 ms |
| 1 | 4.867 | 1.602 | 1.331 | 0.454 |
| 5 | 3.423 | 1.622 | 1.326 | 0.460 |
| 8 | 3.536 | 1.941 | 1.425 | 0.550 |
| 15 | 3.306 | 1.638 | 1.288 | 0.452 |

Groups 1, 5, 8 and 15 pass ragged numerical/padding tests. Group 15 gate/up and
out, group 1 down, and group 15 fused QKV also pass actual projection shape
gates (`gemm-groups-test.log`, `gemm-groups-actual-test.log`). The fused QKV
tests include the independent normalization/rotation oracle. Their one-launch
times are not acceptance measurements; unrelated jobs affected some of them.
A private host (`libh3pipe_groups.so`, `groups-overlay/`) selects group 15 for
1797-token gate/up, QKV and out, and group 1 for down. Its guarded decoder
comparison stopped when another GPU client appeared, before a candidate decode
measurement was accepted (`decode-groups-22.log`). This selection is **not the
production default** and requires a complete uncontended comparison.

Two transposed audio convolution candidates are CPU-ready. One unrolls the
existing tap loop. The other uses the output's stride phase to visit only
contributing taps, in their original order. An exhaustive independent tap
relation check and all candidate compile configurations pass
(`convt-phase-cpu.log`, driver `build/vae-wide/test_convt_phase.py`). GPU
numerical gates and timings are still pending. The current production audio
upsampling kernel remains unchanged.

## Reproduction and validation

CPU-only checks, without GPU initialization:

```sh
bash scripts/test.sh --cpu
python3 tools/bench_vae_gemm.py --compile-only
```

The CPU suite checks generated-source identity, compilation of all four decoder
projection configurations and row groups, and fake-runtime stack dispatch at
1, 517 and 1797 tokens. It checks allocation sizes and row routing through padded
SwiGLU output. Fake HIP tests verify rotating-input addresses, cleanup and early
argument rejection under ASan/UBSan. Host tests check every half bit pattern and
pixel rounding boundaries, including adjacent representable float values.

GPU numerical gates and streaming timings:

```sh
bash scripts/build_host.sh
python3 tests/test_gemm_f16_wide.py --gpu --fast
python3 tests/test_gemm_qkv_rope.py --gpu
python3 tests/test_conv1d4.py --gpu
ATTN_D=64 ATTN=attention_mha64t32_lds_f16_wmma \
  python3 tests/test_attention.py --tokens 13 517 1797
ATTN_D=64 ATTN=attention_mha64hm32_lds_f16_wmma \
  python3 tests/test_attention.py --tokens 13 517 1797
python3 tools/bench_vae_gemm.py --gpu --output build/vae-wide/streaming-final.json
```

The GEMM tests include ragged token counts, garbage operand padding, output
padding guards, residual additions and all four production projection shapes.
The streaming benchmark rotates 36 copies with identical values at different
addresses; allocation, copying and compilation happen before timing. Activations
are reused, so it remains a microbenchmark rather than a full-decoder simulation.

Use a Python environment with torch and diffusers for the independent oracle:

```sh
python3 tests/test_decoder_tiles.py --latents build/py22_video.npy
```

The rebuilt default passed at **63.95 dB PSNR** against the f32 diffusers decoder
at 864x480x22, including decoder-only session isolation and the five-frame case
(`build/vae-wide/parity-qkvropehm.log`, `qkvhm-production-parity.log`).

Repeatable unprofiled decoder comparisons on saved video latents:

```sh
python3 tools/bench_vae_decode.py --gpu --latents saved_video.npy \
  --output build/vae-wide/decode.json
```

An NPZ with `video` and `audio` arrays measures both decoders. The tool reports
first-call and resident timings and session teardown separately. To explicitly
synthesize the 124-frame geometry from a shorter saved input:

```sh
python3 tools/bench_vae_decode.py --gpu --latents build/lat_hip.npz \
  --frames 124 --repeat-latents --repeat 2 \
  --output build/vae-wide/decode-av.json
```

Do not interpret repeated input data, warm kernel sums or an interval excluding
audio as proof that the original 30-second target is met. Further work must save
at least another 3.0 seconds from representative resident video-plus-audio decode,
while preserving parity.

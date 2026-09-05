# Notes

Every decision and every measurement, won or lost, in order. The sibling repos'
conventions apply: ship only the measured-best configuration; losing variants go to
`experiments/` with their number here.

## Day 0: what the model is (2026-09-05)

MiniMax H3 (Hailuo 3.0), open weights since 2026-08-03 (`MiniMaxAI/MiniMax-H3`,
diffusers layout; Comfy-Org repack exists). One packed sequence of text, video and audio
rows through 50 identical blocks:

| | H3 (FL2VA) | Krea 2 Turbo |
| --- | ---: | ---: |
| hidden | 5376 | 6144 |
| blocks | 50 | 28 |
| attention | 56 heads x 128, MHA, separate q/k/v, RMSNorm q/k, RoPE on 96 of 128 (rotate-half, 3 axes x 16) | 48/12 GQA, fused qkv, interleaved-pair RoPE |
| MLP | SwiGLU 14336 | SwiGLU 16384 |
| modulation | AdaLN shift/scale/gate per (timestep, modality) row, from a 2688-d time embedding through a 96768-wide projection per block (13B of the 33B, precomputable per schedule) | per-block table + shared time projection |
| residual gates | table row per token class | sigmoid of a per-token projection (attention) / table (MLP) |
| block params | 385M -> 19.3B active | 460M -> 12.9B |
| latents | video f16 t4 d24, patch 1x2x2 (96 ch/token); audio 32 ch at 40 Hz | image f8 16 ch |
| encoder | Qwen3-VL-32B, hidden states of layer 50 (5120-d), 2 refiner blocks | Qwen3-VL-4B taps + fusion |
| scheduler | flow, sigma shift 12 (video) / 3 (audio) | flow, mu 1.15, 8 Turbo steps |

The checkpoint is mixed precision: patch projections, timestep MLP and output heads f32,
everything else bf16. The pipeline class (`MiniMaxH3Pipeline` / modular) is not in the
diffusers checkout of 2026-09-05 that has the transformer; the packing contract is spelled
out in the transformer's docstring (rows, `(t, h, w)` positions, per-row modality tags and
timestep indices, no padding).

## The arithmetic that sets expectations

At 768p 16:9, 5 s at 24 fps: 24 x 43 = 1032 tokens per latent frame, 31 latent frames,
about 35k rows with text and audio. Per block: GEMMs 2 x 35k x 385M = 27 TFLOP, attention
4 x 35k^2 x 128 x 56 = 35 TFLOP. Per step over 50 blocks: 1.35 PFLOP of int4 GEMM and
1.76 PFLOP of fp16 attention. At the rates krea2-loom reaches on this part (75 TOPS int4,
20 TFLOP/s attention): roughly 18 s + 88 s per step. Attention, not the int4 GEMMs, is
the wall at video lengths; H3's native sparse attention is not in the open release. At
480p and 4 s (about 10k rows) the same step is about 12 s.

## Day 0, continued: the checkpoint on the box and what the kernels needed

The weights were already here in ComfyUI form (`~/comfy-models`): the FL2VA and Ref2VA
transformers as `pruned_int8_convrot` (21 GB each), the Qwen3-VL-32B encoder cut to its
first 50 layers in int8 ConvRot (27 GB), both VAEs. `comfy_quant` says
`{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}` with per-row
`weight_scale`; comfy-kitchen's rotation is the Kronecker power of the *regular* H4 (one
negative entry per row) normalised by 1/sqrt(256), exactly what krea2-loom's kernels compute,
so the export is a per-row int8 -> int4 requantisation and nothing is rotated again
(`tools/export_weights.py`, 9.65 GB in 63 s). ComfyUI's int8 path does not quantise the
activations (`quantize_input: False`), so "none" in the reference is the production baseline.
"Pruned": the AdaLN projections read an 8-d curve (`adaln_t_table` [1025][8], lerp on
t in [0, 1], no silu) instead of the 2688-d time embedding.

`reference/h3_ref.py` is an own torch implementation (no ComfyUI import; its runtime
packages only exist in the podman image and its code is GPL). It matches ComfyUI's
`MiniMaxH3Model` to 2e-7 relative at toy size, run inside the image with
`PYTHONPATH=/opt/ComfyUI` and ComfyUI's `--cpu` (argument parsing must be enabled first).

Kernels from krea2-loom by configuration plus four deltas: the attention generator's MHA
mode (four query tiles of one head per workgroup, grid ceil(T/64) x 56, 240 VGPRs, passes
at 100/1000/5504 rows, 14.5 TFLOP/s at 5504); prepare-norm with a plain norm weight and
per-row (scale, shift) from a class table (class = timestep class * 3 + modality), lane
count as a config because 5376 and 7168 are not multiples of 2048 (96 / 128 / 256 lanes);
the residual GEMM's gate from a [classes][N] table; rotate-half RoPE on 96 channels, the
partner quad fetched by `kernel.subgroup.shuffle<index>` from 12 lanes away (the source
lane needs an `index.assume` range before the cast). Two harness lessons: the test harness
cast every plain input to f32 (an `in_i32` kind was added), and a wrong Hadamard in the
reference cannot be caught by a test that uses the same reference on both sides.

## Day 0, evening: the stack runs; what int4 costs H3

`tests/test_blocks.py` on the 2097-row fixture, native against the reference:

| blocks | stream update cosine vs W4A4 ref | vs int8 | ref W4A4 vs int8 | final velocity cosine vs W4A4 ref | vs int8 | ref W4A4 vs int8 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.99886 | 0.99275 | 0.99167 | 0.99839 | 0.99377 | 0.99370 |
| 16 | 0.99824 | 0.99669 | 0.99665 | 0.99957 | 0.99837 | 0.99838 |
| 50 | 0.93470 | 0.87301 | 0.87548 | 0.99387 | 0.97803 | 0.97860 |

The stream columns collapse past block 24 for every quantisation, the reference's own
included, because the residual stream carries a few channels at 1e6 against an rms of 1e4
and the cosine of the update is theirs. The final layer's RMSNorm removes them; on the
velocity, int4 costs 2.2% of cosine over 50 blocks (Krea 2's 28 blocks cost 2.8% on its
metric) and the native kernels sit 0.6% from the reference's arithmetic, the f16
intermediates' share. Stage profile at 2097 rows, 50 blocks, 2.57 s: gate|up 681 ms, qkv
543, attention 516, down 352, out 187, prepare norm 98, prepare down input 82, rope 57,
prepare out input 50.

Overflows found the hard way: the f16 residual stream (inf from block ~18) and the f16 LDS
Hadamard in the plain prepare (the gate|up products reach 5e4; four unnormalised radix-4
stages multiply the range by 256). Both are f32 now; the gate|up output itself stays f16
with a 5e4 peak on this fixture against 65504, to be revisited (bf16 storage) if a real
prompt exceeds it.

## The quantisation study, and the fork it forces (2026-09-05)

`tools/quant_study.py`: 50 blocks on the fixture, the final-layer velocity against the int8
checkpoint with float activations (ComfyUI's path):

| weights | activations | cosine | rel rms err |
| --- | --- | ---: | ---: |
| int4 per row (the kernels today) | int4 per token | 0.97860 | 0.2091 |
| int4 per 256-group | int4 per token | 0.98795 | 0.1572 |
| int4 per 128-group | int4 per token | 0.98757 | 0.1587 |
| int4 per 128-group | int8 per token | 0.99141 | 0.1323 |
| int4 per 64-group | int8 per token | 0.99221 | 0.1261 |
| int4 per row; fc2 int8 | int4; int8 | 0.98122 | 0.1953 |
| int4 per row; fc2 and out_proj int8 | int4; int8 | 0.98939 | 0.1458 |
| int4 per 128-group; fc2 int8 | int8 | 0.99176 | 0.1297 |
| int8 as shipped | int8 per token | 0.99972 | 0.0238 |

The loss is the 4-bit weights, not the activations: int8 activations on the int8 weights cost
2.4% of velocity, while every int4-weight configuration costs 12.6% or more, and finer weight
groups buy only a third of the gap back. H3 is an order of magnitude more sensitive to 4-bit
weights than Krea 2 was (its W4A4 reference sat 2.8% from bf16). Note the published int8
checkpoint is itself the quantiser's choice for this model.

Two paths, to be chosen:
- W8A8: the checkpoint's int8 rows unchanged, int8 per-token activations, on the iu8 WMMA
  (loom-gemm's int8 kernel: 36 TOPS measured, 67% of the 54 peak, against 75 for int4). At
  video lengths attention dominates and the step grows by roughly a fifth; at short sequences
  the GEMMs take twice as long. Quality 0.9997.
- int4 per 64/128-group weights with int8 activations: keeps the int4 rate minus the per-group
  float accumulation in the GEMM (each k-group's int32 partial converted and scaled, an
  estimated 15-25% on the GEMM), at 12.6-13.2% velocity error per step.

## Study v4: can the group scales leave the k-loop? (2026-09-05)

The (row, K-group) int4 scales factored as r[n] * t[g] with t folded into the activations,
so the GEMM stays plain int4: no gain at any group size (w4r256a4 0.97874, w4r64a4 0.97783
against w4a4 0.97860) -- the scale variation is not separable. True per-group weight scales
help modestly and monotonically: g64 0.98773, g32 0.98904 with int4 per-token activations;
g64 with int8 activations 0.99221. The 4-bit weight floor on H3 is about 0.99 cosine /
13% relative velocity error, and reaching it costs per-group float accumulation inside the
GEMM's k-loop. Target from the user: 100 TOPS with reasonable quality; the int8 path (54
TOPS peak) cannot reach it, so this is int4 with group scales, on a GEMM that has to run at
86% of the iu4 peak.

## Study v5: per-group both sides

| weights | activations | cosine | rel rms err |
| --- | --- | ---: | ---: |
| int4 per 128-group | int4 per 256-group | 0.99056 | 0.1381 |
| int4 per 64-group | int4 per 256-group | 0.98915 | 0.1484 |
| int4 per 32-group | int4 per 256-group | 0.99120 | 0.1329 |
| int4 per 32-group | int8 per token | 0.99314 | 0.1175 |

So "reasonable" int4 on H3 means per-group weight scales (32-128) with per-group activation
scales, at 0.99 cosine / 13% relative velocity error. The kernel cost of group scales is a
second, f32, accumulator set: the group's int32 partials are converted and scaled into it
once per group. With 64x64 wave tiles that is 128 + 128 accumulator VGPRs and does not fit;
with 32x64 tiles (the 128x128 kernel) it does, at 0.75 LDS reads per multiply. The
throughput and quality mechanisms compete for the same registers.

## GPTQ at export time (2026-09-05, running)

The per-row int4 format is the plain GEMM's, so the only lever that costs no registers is a
better quantiser: `tools/gptq_export.py` runs GPTQ block by block on the fixture's
activations (each block calibrated on the output of the already-quantised blocks before
it), per-row scales from the unquantised row, Hessians damped in f64 because the 2097
calibration rows are fewer than K. It writes `build/weights_gptq/` in the runtime's format
and reports the 50-block velocity cosine against the int8 checkpoint (round-to-nearest:
0.97860). 37 s per block.

## GPTQ result, and the 64x64-wave GEMM in the runtime (2026-09-06)

GPTQ per-row int4 (sequential over the 50 blocks, per-row scales, damped f64 Hessians):
final-layer velocity cosine 0.99220 / rel rms 0.1250 in the reference against 0.97860 /
0.2091 for round-to-nearest -- the same quality the best per-group configuration reached
(0.99221) with no kernel cost, so the plain int4 GEMM format stays. Through the native
runtime (`tests/test_blocks.py --weights build/weights_gptq`): 0.99120 at 50 blocks against
the int8 path, 0.99614 at one block. `build/weights_gptq` is now the default export to use.

`tools/gen_gemm.py` puts the three H3 epilogues on loom-gemm's 256x128 tile (lever 9); all
six GEMM kernels pass `tests/test_gemm.py` against float64. Block-level A/B at 2097 rows,
50 blocks, same contended window (another session's job on the GPU):

| stage | 128-row tile | 256-row tile |
| --- | ---: | ---: |
| gate/up + swiglu | 880 ms | 714 ms |
| qkv | 610 | 518 |
| down + residual | 380 | 338 |
| out + residual | 244 | 203 |
| forward | 2974 | 2518 |

`H3_GEMM_TILE` selects the tile in the builder (default 256); the host reads it from the
kernel directory. Calm-box numbers to follow.

## Calm-box numbers, 2026-09-06

2097 rows, GPTQ weights, 256-row GEMM tile, 50 blocks in 1.62 s (32 ms per block):

| stage | ms | share | rate |
| --- | ---: | ---: | ---: |
| gate/up + swiglu | 447 | 27.6% | 72 TOPS |
| qkv | 336 | 20.7% | 72 TOPS |
| attention | 301 | 18.6% | 21 TFLOP/s |
| down + residual | 265 | 16.3% | 61 TOPS |
| out + residual | 123 | 7.6% | 66 TOPS |
| prepare norm, rope, prepare inputs | 150 | 9.2% | |

At 2097 rows the 256-row tile pads to 9 tiles (9% ghost rows), so the GEMMs run at about
79 TOPS on real rows. The 100 TOPS target is 86% of the iu4 peak; the GEMM sits at 62-68%.

## 10317 rows (480p, 4 s): attention is bandwidth-bound

Fixture 32 text + 25 x 405 video + 160 audio rows, GPTQ weights, 256-row tile, 50 blocks in
23.8 s: attention 18.0 s (75.7%, 8.5 TFLOP/s against 21 at 2097 rows), gate/up 1.95 s
(82 TOPS), qkv 1.50 (80), down 1.12 (71), out 0.51 (78), the rest 0.7 s. Attention's rate
falls with length because each workgroup of 64 queries streams its head's whole K and V
(10317 x 128 x 2 x 2 B = 5.3 MB, past L2) -- 162 workgroups x 56 heads x 5.3 MB = 48 GB per
block, 2.4 TB per step, 133 GB/s over the 18 s: the memory system's streaming rate. The
lever is queries per K/V pass: eight-wave workgroups (128 queries) halve the traffic.

## Eight-wave attention workgroups -- won 1.89x at 10317 rows (2026-09-06)

`ATTN_WAVES=8` in the generator: 128 queries share each staged K/V tile, the 256 lanes split
the staging (lanes 0..127 K, 128..255 V transposed), scratch and Q regions per wave, LDS
33 KB, 256 VGPRs without spills. Interleaved A/B: 10317 rows 173 ms against 329 (17.6 vs
9.3 TFLOP/s, 1.894x); 2097 rows 6.85 against 5.86 ms (0.855x: too few workgroups). The
builder picks 8 waves from 4096 rows and writes `attention_waves.txt`; the host reads it
for the grid and block. The remaining gap to the 21 TFLOP/s the kernel reaches when K/V
fit L2 is the same streaming traffic at half the volume; the next step in that direction
is 256-query workgroups or K/V shared across the two heads a WGP runs.

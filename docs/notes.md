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

## Sixteen-wave workgroups -- lost (0.919x at 10317 rows)

`ATTN_WAVES=16` (256 queries per K/V pass, lanes 256..511 idle in staging, 55.5 KB LDS, 256
VGPRs): 188.6 against 173.3 ms for eight waves. One workgroup per WGP is too little
occupancy for the halved traffic to pay. Eight waves is the point of the curve; the kernel
sits at 17.6 TFLOP/s at 10k rows against 21 when K/V fit L2.

## The clip (2026-09-06)

`tools/encode_prompt.py`: transformers' Qwen3-VL built on the meta device with 50 layers,
the final norm replaced by identity (the conditioning is layer 50's unnormalised state),
the vision tower and head dropped, the rotary module recreated on the device, the int8
rows dequantised per row and every block linear wrapped to rotate its input by the same
Hadamard the rows were rotated by; 72 s to load, 28 tokens for the fox prompt, rms 40.
`tools/pipeline.py`: the t2va layout from the reference, diffusers' MiniMaxH3Scheduler
twice (shift 12 and 3), noise drawn in diffusers' order, the head's raw data-ward velocity
to the scheduler (ComfyUI negates it, diffusers does not), diffusers' VAEs from the
official repo (their weight names differ from the ComfyUI files), ffmpeg mux. The 22-frame
8-step smoke clip is a coherent backlit fox on snow (`docs/media/smoke_fox_22f_8steps.jpg`).

The full clip: `tools/pipeline.py "A red fox trots through fresh snow at dawn, ..." --frames
124 --steps 50` (49 after the shift collapses a duplicate sigma): 37 x 30 x 54 latents, 207
audio latents, 28 text rows, 15427 packed rows; 29.4 s per step, attention 20.1 s of it
(17 TFLOP/s), gate/up 2.9 s, qkv 2.2 s, down 1.7 s, out 0.8 s; 1440 s of denoising, then
the two VAE decodes in torch (the video decoder is the slow one, several minutes at this
size). The frames are temporally coherent and on prompt (`docs/media/fox_480p_5s_strip.jpg`).
Attention at video length is the next lever; the budget that resists a 256-query pass is
LDS (Q tiles for 16 waves) and registers (two query tiles per wave).

## The video VAE decoder in Loom (2026-09-06)

diffusers keeps the decoder in fp32 (`_keep_in_fp32_modules`), which is why the torch
decode is slow: 161 s per 7-latent-frame clip (11345 tokens through 36 blocks of width 2048,
32 heads of 64, SwiGLU 8192), about 19 minutes for the 5-second clip. In Loom it is the
H3 block session with the decoder's shapes: the int4 GEMM family gained biases and
diffusers' SwiGLU order (silu on the second half), the attention generator a head size of
64 (176 VGPRs, 7-9 KB of LDS), the RoPE kernel a two-channels-per-lane variant (the
rotate-half partner sits 12 lanes away in both), and prepare-norm with a zero one-class
AdaLN table is the decoder's RMSNorm. `tools/decode_loom.py` keeps diffusers' post_quant_conv,
proj_in, register/cls tokens, rotary grid, norm_out, proj_out, unpatchify and temporal
chunk blending around the native blocks.

Quantisation study on one clip (`tools/vae_quant_study.py`, frame PSNR against fp32):

| weights | activations | PSNR |
| --- | --- | ---: |
| int4 per row (RTN) | int4 per token | 28.31 dB |
| int4 per row (RTN) | float | 30.73 dB |
| int8 per row | int8 per token | 52.03 dB |

The native int4 decoder reproduces the study exactly (28.28 dB through the head at 36
blocks, update cosine 0.9868): the kernels are faithful, the int4 weights are the loss.
Since the decoder is a small share of a clip, the generators gained an int8 mode (the iu8
WMMA: fragments of vector<4xi32>, 64-k steps, 228 VGPRs; int8 prepare kernels packing four
bytes per word -- the per-lane chunk count must come from elements, not words) and the
session, builder, export and wrapper take `bits`. GPTQ for the int4 decoder runs alongside.

W8A8 through the native session (`tests/test_vae_blocks.py --bits 8`): frame PSNR 52.04 dB at
36 blocks, update cosine 0.99992 -- lossless class, the same number as the torch study.
20.8 s per clip against torch's 161 s while GPTQ shared the GPU; attention (head 64) took
59% of it at a poor 3 TFLOP/s, the head-64 kernel doing half the multiplies per tile for
the same barriers and staging. Calm numbers and the two-query-tiles-per-wave variant follow.

## Calm-box decoder, and the text encoder in Loom (2026-09-06)

The W8A8 decoder on a quiet GPU: 6.05 s of block time per 11345-token clip, 45.7 s for the
124-frame clip (seven chunks, diffusers' head and blending around them) against torch fp32's
~19 minutes. Attention is 72% of it (4.4 s per clip, 8.6 TFLOP/s: the head-64 kernel does
half the multiplies per tile for the same staging and barriers). The GEMMs run at
14-19 TOPS on the decoder's shapes; a two-query-tiles-per-wave attention variant is the
remaining lever, worth up to ~2x on the decoder.

The text encoder (Qwen3-VL-32B's language model cut to 50 layers, ComfyUI's int8 ConvRot
file) is the same session shape with three deltas:

- attention: a GQA-8 causal mode of the LDS kernel. A workgroup is one 16-row query tile and
  one kv head; its eight waves are that kv head's eight query heads, so K/V are staged once
  for all eight. The key loop stops at the query tile, and the diagonal tile takes an
  additive mask built from the WMMA accumulator layout probed on gfx1151
  (`experiments/probe_wmma_layout.loom`: lane l holds column l%16; lanes 0-15 hold rows
  0,2,..,14 and lanes 16-31 rows 1,3,..,15, element e at row 2e + l/16). Passes vs SDPA
  (`is_causal`, `enable_gqa`) at 28..1000 tokens, 240 VGPRs.
- RoPE: a `kv_heads` config (k and v at 8 heads; q at 64), the 128-channel rotate-half
  variant, the q/k RMSNorm weights from the file.
- GEMMs: the int8 family without biases (`gemm_i8_256`, `_resid_256`, `_swiglu_256`),
  n_size bound raised to 65536 for the 51200-wide gate|up; and prepare_plain16_i8, an
  f16-LDS variant for the 25600-wide down-projection input (25600 f32 does not fit 64 KB;
  the f16 input is pre-scaled by 1/16 at staging so the four unnormalised Hadamard stages
  land exactly on the normalised rotation -- codes identical to the f32 kernel, the token
  scale to f16 precision).

Accuracy vs transformers bf16 (the same weights dequantised; `tests/test_te_blocks.py`, the
33-token fox prompt): hidden cosine 0.99997 / 0.99996 / 1.0000 / 1.0000 after 1 / 4 / 12 / 50
layers, median per-token relative error 1.7% at layer 50 (one token 18%, whose residual is
small next to the 15168-magnitude massive-activation channels). A torch reference with
per-token int8 activations gives the same numbers, so the loss is W8A8's, not the kernels'.

Two bugs on the way, both in the harness, not the kernels: the numpy view of a CPU f32 input
shared its memory, so the session's result overwrote the input and the test compared the
output with itself ("update cosine 0.00000"); and a prompt is a single 256-row tile, for which
the raster group of two streamed every weight twice -- `m_group = 1` for one tile took the
50 layers from 564 to 306 ms (the GEMMs at 33 rows still move the 24 GB of int8 weights at
~90 GB/s of 200; a skinny-M weight-streaming kernel would halve it again, for a once-per-
prompt 0.3 s).

The pipeline now encodes uncached prompts in Loom and decodes video through the W8A8 Loom
session by default (`--torch-decode`, `--vae-bits 4` with the GPTQ export for the int4
decoder). Left in torch: the audio VAE (BigVGAN: 65 M parameters of narrow 1-D
convolutions, Snake activations and anti-aliasing filters across seven upsampling stages;
launch-bound, a different kernel family), the scheduler step, the VAE heads and the prompt
embedding lookup.

## The pipeline as one C library, every kernel in Loom (2026-09-06)

`host/h3pipe.cpp` + `host/h3pipe.h` (`build/libh3pipe.so`): prompt token ids in, latents /
RGB8 frames / stereo samples out, for callers in any language, the same shape as
dinov3-loom's and scrfd-loom's runners. Nothing between the kernels is Python any more:

- one generic transformer stack serves the text encoder, the token refiner, the 50 DiT
  blocks and the video decoder (dims, bits, bias, causal / GQA, gate tables, rope variant
  are parameters);
- the embedders and the final layer are the int8 GEMM family with padded K (96 -> 256,
  32 -> 256, 24 -> 256) and N (96 + 32 -> 128, the video and audio output projections
  stacked into one GEMM); the final norm + AdaLN is the prepare-norm kernel with a
  two-class table; the decoder's LayerNorm is a new `lnorm` prepare form with the bias in
  the table's shift row; the refiner's final norm is `norm_mod_f32` (in place, f32);
- the audio vocoder (BigVGAN) is five f32 SIMT Loom kernels: `conv1d_f32` (dilation,
  optional accumulate), `convt1d_f32`, `up2_snake_f32` (2x Kaiser-sinc + SnakeBeta),
  `down2_f32`, `axpy_f32`; weight norm folded and Snake parameters exponentiated at export;
- the host does the layout, rope tables, AdaLN curves (the 96768x8 projections per layer
  per step, ~30 ms), the scheduler, its own RNG (splitmix64 + Box-Muller: a seed does not
  reproduce a torch seed), patchify / unpatchify, the decoder's chunk loop and cross-fades,
  the ImageNet pixel mapping, and the post_quant_conv (24x24 per voxel);
- kernels compile on first use for a shape by spawning `loom-compile` into
  `build/kernel_cache` (the config string is the cache key; Loom compiles in milliseconds);
- `host/h3tok.cpp`: the Qwen2 byte-level BPE from tokenizer.json (a small JSON reader, the
  split regex evaluated by hand on code points), identical to transformers on the test
  prompts including accents, CJK, emoji, contractions and whitespace runs.

Verification (`tests/test_pipe.py`, `tests/test_tokenizer.py`): the refined text rows
against the reference's `text_in` at cosine 0.9998; one denoising step from shared noise
against the Python pipeline at update cosine 0.991 (video) / 0.990 (audio), where the int4
blocks' own sensitivity to a 0.1% input change is 0.994 (`--attrib`); the video decoder
against the Python Loom decoder at 49.4 dB; the audio decoder against diffusers at 108.9 dB
SNR. Two bugs found by those tests, both in the host: the sequence buffers were reallocated
underneath the refiner's identity rope tables (0.982 -> 0.991), and the single-class
embedder GEMMs indexed their gate table with the layout's class ids.

Timing at 22 frames 864x480 (2931 rows): 2.3 s per step (the Python-driven session: 2.4),
video decode 7.6 s (Python: 10.6), audio decode 0.5 s (torch: 1.95), session creation 10 s
plus 17 s the first time a prompt length compiles its kernels and uploads the 24 GB encoder.
`build/h3pipe --prompt "..."` writes raw RGB + WAV; ffmpeg muxes.

Fixed on the way: the Python clip writer mapped the decoder output as [-1, 1]; diffusers'
decode block does `clamp(v * imagenet_std + imagenet_mean, 0, 1)`, which both writers now do.
The attention kernels' `tokens` config allowed 16 at least; prompts shorter than a tile now
compile (tests at 13 tokens).

Runtime dependency: the HIP runtime API for module load, memory and launch (no device code;
the hosts build with hipcc as a plain C++ compiler). An HSA-only host is a swap of those
calls, not a rewrite.

## The 30-step target: where the step's time is, and what did not move it (2026-09-06)

30 steps of the 5-second 864x480 clip is 29 evaluations at 29.4 s = 14.2 minutes, plus 17 s
of prompt encoding and 47 s of decoding. The step is 70% attention, 26% GEMMs. The GEMMs run
at 78-82 TOPS, 80% of the 100 TOPS target and 70% of the measured 117 peak; even at the
target they would save 1.5 s of the 29.

Attention at 15427 rows, 8 waves, 56 heads (`tools/ab_attention.py`, calm box):

| variant | TFLOP/s | vs shipped |
| --- | ---: | ---: |
| shipped: 8 waves, 16-key tiles, 4 Q fragments hoisted (256 VGPRs, 33 KB LDS) | 17.6 | 1.00 |
| the same kernel at 2097 rows (a head's K/V fits L2) | 18.6 | its own ceiling |
| 4 waves at 15427 rows / at 2097 rows | 5.2 / 21.2 | K/V re-streaming |
| 4 waves, 32-key tiles, at 15427 rows | 7.6 | 1.47x over 4 waves |
| 16 waves | 16.3 | 0.93x |
| 8 waves, 32-key tiles (spills 76 B, 45 KB LDS) | 11.3 | 0.64x |
| 8 waves, no Q hoisting (208 VGPRs, 49 KB LDS) | 15.7 | 0.89x |
| 8 waves, no hoisting, 32-key tiles (248 VGPRs, 62 KB LDS) | 17.0 | 0.97x |
| 8 waves, hoist 2, 32-key tiles (spills) | 16.1 | 0.91x |

The reading: at clip length the 8-wave kernel is within 6% of what it does with K/V resident
in L2, so traffic is no longer the limiter (a split over keys to make concurrent workgroups
share an L2-sized chunk, with a merge, is worth that 6% at most). The limiter is the
kernel's inner loop at a third of the fp16 peak: per 16-key tile a wave does 16 WMMAs
against two barriers, 16 KB of LDS fragment reads, the 64-register accumulator rescale and
the softmax pass, with two waves per SIMD to hide any of it. Every knob that trades
registers for fewer barriers loses at the 256-VGPR ceiling. Moving the ceiling means a
different kernel (lazy rescaling, fragment reuse across query tiles), not a config.

The generator's 8-wave post-pass now classifies the 32-key tile's second K row correctly
(it produced an undefined name before); the variants live in experiments/.

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

## What hrx-demos and torch say about the ceiling (2026-09-06)

Calibration against the two other attention implementations on hand, at the pipeline's own
shapes, on this box (calm):

| attention, f16, 56 heads x 128 | 15427 rows | 10317 rows |
| --- | ---: | ---: |
| ours (8-wave LDS kernel) | 17.6 TFLOP/s | 17.6 |
| torch 2.13 ROCm SDPA, flash backend (aotriton / CK) | 18.2 | 17.8 |
| torch SDPA, efficient backend | 18.3 | 18.1 |
| AMD's hrx-demos Loom kernel (Ideogram, 18 heads x 256, on gfx1100) | 18.6 (12.4 ms "schedule": 24.8) | -- |

The decoder's head-64 shape is the exception: torch flash 13.7 TFLOP/s against our 8.6 at
11350 rows x 32 heads, so the head-64 kernel has 1.6x of headroom (the two-query-tiles
lever), worth about 15 s of the 47 s decode.

GEMMs at M = 15427: torch's fp16 hipBLASLt does 33-37 TFLOP/s (18 on the down projection);
our int4 WMMA family does 78-82 TOPS on the same shapes.

Reading: three independent implementations of f16 WMMA attention on RDNA3.x land at 18 +- 1
TFLOP/s for head 128 at these lengths, a third of the part's MMA peak. That is the
architecture's attention rate (a wave32 WMMA does 16x16x16 per 16 CU cycles, and the exp,
row reductions, accumulator rescale and fragment layout changes cannot hide behind it with
two waves per SIMD), not a property of one kernel. AMD's demo kernel is a single wave per
workgroup with K^T and V read as fragments straight from global memory through view layouts
and the probability tile bounced through LDS to change layout; it reaches the same rate on a
96-CU part, i.e. half ours per CU, and their case study projects their own best case at
0.75x of PyTorch's request time with attention accounting for 7 s of 90.

Consequences for the 30-step target: an attention rewrite is bounded by roughly 1.1x here;
the levers that remain are the head-64 decoder kernel (a decode-only 15 s), and
step-skipping caches on the DiT (TeaCache / first-block cache: reuse the previous step's
residual when the first block's delta is small, typically 1.6-2x at 30 steps at a small
quality cost), which is host logic on the C pipeline.

Two things from the hrx-demos runtime worth copying regardless: benchmarking with rotating
device-local buffers (their "rotation prevents a reused allocation from making a cache-hot
microbenchmark look like model execution"), and the Plan phase (dry-run the request to size
capacities, live ranges and kernel specialisations before touching model data).

## Native 768: the attention arithmetic has to change (2026-09-06)

1344x768 for 5 s packs ~37.7k rows: per step 1.35 PFLOP of int4 GEMM (17 s at 80 TOPS) and
2.0 PFLOP of attention (113 s at the 18 TFLOP/s every f16 implementation reaches here): ~65
minutes for 30 steps, 87% attention. The two concepts on the table:

**SageAttention** (low-precision QK^T with smoothing). Int8 WMMA runs at the f16 rate on
gfx1151 (54 vs 54), so only int4 (117) changes the MMA time. Per-block attention output error
on real activations (`tools/attn_quant_study.py`: the fox latents noised to sigma 0.9 / 0.5,
the real prompt, reference blocks): int8 per token 0.2-3%; int4 per token 4-62%; int4 with the
same 128-channel Hadamard on q and k (q.k unchanged), K smoothed by its token mean (softmax
invariant) and 32-channel scale groups 2-22%, the last layer worst. V int8 rotated: 0.5%.
Per block that looks fatal; over the stack it is not (`tools/quant_study.py attn:...`, final
velocity vs the int8 checkpoint at 50 blocks): int8 attention 0.99976 / 2.2%, int4 attention
0.99745 / 7.1%, int4 GEMMs (RTN) 0.97860 / 20.9%, both 0.97625 / 22.0%. Int4 QK^T costs one
point on top of the int4 GEMMs. Adopted: QK^T in int4 (per-token, per-head scales, the head
rotated by a Hadamard in the prepare kernel, K mean-smoothed), PV in f16 as now. Q resident
becomes 16 VGPRs instead of 64, K tiles 1 KB instead of 4 KB.

**FlashAttention-3** (overlap). The applicable part is ping-pong: two wave groups a tile apart
on triple-buffered K/V tiles so one group's softmax overlaps the other's MMAs instead of the
whole workgroup running in lockstep behind each barrier. Warp specialisation with async
copies has no RDNA3 equivalent. Estimated 1.2-1.3x; after the int4 kernel.

## Int4 QK^T attention: built, measured, and what it took (2026-09-06)

`kernels/prepare_qk_i4.loom` (per token, the eight waves over the heads: subtract the K
column mean, Sylvester H_128 by butterflies through the wave's LDS row, per-(token, head)
int4 with absmax/7, the attention scale and the Hadamard's 1/128 folded into Q's scale),
`kernels/colmean_f32.loom`, and `tools/gen_attention_i4qk.py`, which derives the int4-QK
kernel from the f16 generator's text: Q as eight int4 fragments in registers (16 VGPRs, no
Q LDS), K tiles of 16 x 16 words, i32 accumulation, scores = sitofp(acc) * q_scale[row] *
k_scale[key]; softmax and the f16 PV unchanged. 240 VGPRs, 11.5 KB LDS (from 33).

| int4 QK^T vs f16 (8 waves) | 5504 | 10317 | 15427 | 37743 rows |
| --- | ---: | ---: | ---: | ---: |
| f16 kernel, TFLOP/s-equivalent | 18.6 | 17.6 | 17.6 | ~16 |
| int4 QK^T kernel | 28.5 | 26.9 | 25.3 | 18.6 |

Pipeline at 10317 rows: attention 8.9 -> 5.9 s per step, the step 14.6 -> 11.9 s. At the
768 layout (37743 rows) the int4 kernel slides back to 18.6: the K/V stream's latency with
two waves per SIMD, not its bandwidth, which is what the prefetch variant
(`ATTN_PREFETCH=1`: double-buffered tiles, the next tile's loads issued before this tile's
compute, one barrier per tile; 256 VGPRs, 19 KB LDS, no spills) is for -- untested at the
time of writing because the GPU was busy with a game.

Quality through the native stack (fixture, GPTQ int4 GEMMs): final velocity cosine 0.9881
with int4 attention against 0.9912 with f16 (0.9786 for RTN int4 GEMMs alone). The C and
Python paths agree at 0.981 / 0.988 (video / audio) on one step, the int4 floor.

The prepare kernel cost a day: its first form, one wave per (token, head) with the pairs
spread over a flat grid and cross-lane traffic through `kernel.subgroup.shuffle<xor>` and
`kernel.subgroup.reduce`, was bit-exact against its replica on a single launch yet
non-deterministic across launches; without the wave-uniform `scf.if` guard it also faulted
(memory aperture violations) on one launch in seven, more with overflowed inputs; through
LDS instead of shuffles it still skipped one (token, head) in a few thousand at random.
Restructured to the rope kernel's shape -- one workgroup per token, the waves looping over
the heads, exchanges through LDS -- it is deterministic, fault-free over dozens of launches,
and bit-exact. The mechanism is unexplained (the wave id is workitem >> 5, the argument
slots are 32-bit and zeroed); the lesson is the pattern: give each workgroup a token, not a
flat pair index, and keep subgroup ops out of per-wave branches.

## Saturation, smoothing off, prefetch by length (2026-09-06, later)

The non-finite audio row at 50 layers was not the attention: the stream scanner
(`H3_DEBUG_NAN`) found an infinite token scale before the down projection at layer 49, i.e.
one silu(gate) * up value crossed 65504 in the f16 gate|up buffer (the notes already had them
at 5e4) and the prepare kernel's absmax became inf. The f16 attention path had been sitting
on that edge; int4 attention nudged one row over it. The GEMM epilogues now saturate to
+-65472 before every f16 store (`tools/gen_gemm.py`), which costs nothing and removes the
cliff for every path.

K mean smoothing, on the fixture's velocity with GPTQ blocks: 0.9881 with, 0.9899 without
(f16 attention 0.9912). Off by default (`H3_KSMOOTH=1` keeps it); the column-mean pass goes
with it.

The double-buffered kernel (`attention_i4qkp_mha8`, ATTN_PREFETCH=1: the next tile's loads
issued before this tile's compute, one barrier per tile, 256 VGPRs, 19 KB LDS) against the
shipped int4 kernel, interleaved best-of-4: 0.88x at 15427 rows, 1.39x at 37743 (2.12 vs
2.94 s per block). The builder and both hosts pick it from 20000 rows.

Where 768 stands: 37743 rows, 50 blocks, attention 2.24 s per block with the prefetch
kernel (19.3 TFLOP/s-equivalent), the step about 130 s of which attention is 82%; 30 steps
about 65 minutes. The GEMMs are 6% of it. The remaining attention levers are ping-pong wave
groups (the register budget at 256 makes it a redesign) and work reduction (step caching,
sparse attention); the int4 QK^T was the last one with a hardware rate behind it.

## SageAttention2's smoothing, measured on H3 (2026-09-06)

krea2-loom has an SA2-style kernel (`kernels/attention_sage_i4.loom`, `tools/gen_sage_attention.py`,
`docs/native-attention.md`) of the same structure as ours plus SA2's Q half: Q centred per 16-row
query tile, K centred over the sequence, and a precomputed correction table q_mean . K_c per
(head, query tile, key) added to the scores. Its probe on Krea 2 found centring far better than
rotation (block 0: 0.9989 vs 0.9362). On H3 it is the other way round.

Per block (`tools/attn_quant_study.py`, real activations, sigma 0.9, rel err / worst head):

| layer | int8 | int4 rotated (shipped) | SA2 centred + correction | centred + rotation |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.4% / 1% | 4.7% / 8.9% | 6.9% / 24% | 3.5% / 7.0% |
| 25 | 0.6% / 1% | 10.3% / 14.6% | 10.0% / 23.6% | 8.4% / 19% |
| 49 | 2.9% / 8.9% | 22.9% / 52% | 48.7% / 120% | 23.8% / 59% |

The granularity of Q's centring (16 rows, 128 rows, the whole sequence) barely matters. Full
stack (`tools/quant_study.py attn:...`, velocity cosine vs the int8 checkpoint): centred
0.9922, centred + rotation 0.9973, rotation alone 0.9975; with int4 GEMMs 0.9747 / 0.9781 /
0.9763 against 0.9786 for the GEMMs alone. So on H3 the rotation carries the accuracy, SA2's
centring adds at most 0.2 cosine points on the full stack, and a correction table costs
(heads x query tiles x keys) floats: 3.3 GB at 480p, 20 GB at 768, or a 128-MAC dot per key
per tile inside the kernel. Not adopted. What SA2 gave this part is what is already in: int4
QK^T on WMMA with per-token, per-head scales. Its FP8 P.V half has no gfx11 equivalent (no
FP8 WMMA before gfx12); our P.V stays f16. Its "per-thread" scales are 8 tokens per scale,
coarser than ours.

H3's late layers are the limit of int4 QK^T under any smoothing (layer 49 at 23% per block,
|k| to 378, |q| to 187); keeping the last layers in f16 attention did not move the velocity
(0.9972 vs 0.9975 with layers 45+ exact), because the velocity error is set by the middle.

## The 2x: what the ablations found in the int4 kernel (2026-09-06, night)

`tools/ablate_attention_i4.py` removes one component at a time from the shipped kernel and
times the variants interleaved. First round at 15427 rows: P.V MMAs and V fragment reads 32%,
**the V transpose staging 20%** (16 scalar f16 LDS stores per lane per tile), QK^T and K reads
13%, exp / max butterflies / accumulator rescale 3% each, and capping the LDS to one workgroup
per CU made it 3.5% faster. The softmax was never the cost.

Landed from that:
- `kernels/transpose_f16.loom`: V^T once per block (32x32 LDS tiles, headroom columns zeroed
  -- stale tile rows gave NaN x 0 = NaN in the masked keys the first time). The attention
  stages V by channel rows with one 32-byte load and one store. 768 attention 2.94 -> 1.69 s
  per block (interleaved). The register-prefetch variant, which spilled, is retired.
- double-buffered K/V tiles (ATTN_DBUF, now the shipped form): the trailing barrier goes;
  1.02-1.05x.
- `attention_i4qkl_mha8` for 20k+ rows: the same kernel with its LDS request padded so one
  workgroup runs per CU (fewer concurrent K/V streams); 1.09x at 37743 rows, 0.97x at 15427.
- 16 waves per workgroup, retried with the int4 kernel: 0.76-0.82x again.

Interleaved at 37743 rows (best of 6): f16 kernel ~3.2 s per block; int4 QK^T 2.94; +V^T
1.65; +double buffering 1.61; +one workgroup per CU 1.51 (27.1 TFLOP/s-equivalent). That is
2.1x on attention against the f16 kernel at 768; the session's 50-block mean under the day's
load is 1.7 s per block, the 768 step about 110 s (attention 75%), 30 steps about 55 minutes.
Velocity unchanged at 0.9899 (V^T is exact).

Second ablation round (noisier, the box shared): K/V staging 31%, the trailing barrier 15%,
the epilogue's publish loop 12%, P.V 37%, QK^T 26%. Left on the table, in order: the
epilogue (write the accumulator layout straight to global instead of 8 LDS round trips with
16 subgroup barriers), a lazy accumulator rescale, the P round trip through LDS (an
in-register layout change), and two query tiles per wave if the registers can be found.

Epilogue follow-up: a direct-store epilogue (`ATTN_DIRECT_OUT`, 64 scalar f16 stores per lane
from the accumulator layout, no LDS round trips or subgroup barriers) is correct and a wash
(1.03x at 15427 rows, 0.97x at 37743): the ablation's 12% was the output writes themselves,
which any epilogue pays. Kept in experiments/.

## Toward 4x: what H3's attention can skip (2026-09-07)

The MMA floor of the int4 kernel at 768 is 0.55 s per block (QK^T at the int4 rate, P.V at
the f16 rate) against 1.51 s now, so 4x on attention (0.8 s) is not reachable by overhead
alone; P.V has to be skipped where it contributes nothing. `tools/attn_sparsity_study.py`
measures that on the full 480p fox clip (15412 rows), per (head, 16-row query tile, 16-key
tile): a tile is skippable when every row's max score in it sits more than tau below the
row's running max (what an online-softmax kernel can know) or its final max (an upper bound):

| running-max rule, tau 6 | layer 0 | 10 | 25 | 40 | 49 |
| --- | ---: | ---: | ---: | ---: | ---: |
| sigma 0.9 | 13% | 18% | 38% | 22% | 51% |
| sigma 0.5 | 16% | 21% | 44% | 35% | 60% |
| sigma 0.2 | 20% | 23% | 46% | 41% | 62% |

Per-block output error 0.4-2.4% at tau 6, 0.04-0.35% at tau 8 (which skips about two thirds
as much). The final-max rule reaches 81% at layer 49 but only helps a two-pass kernel, whose
second QK^T pass costs what it saves. So the kernel's tile skip is worth about 1.2-1.3x on
attention averaged over the stack, more on the late layers: real but not the 2x. The rest
of a 4x on the step has to come from the step cache (blocks 1..49 reused when block 0's
output moves little between steps), which is the other lever built here.

Implemented: `skip_tau` config on the int4 kernels (one reduction and one cross-half shuffle
per tile give the wave-uniform gap; eight per-fragment branches skip the V loads, P.V MMAs and
rescale, a select skips the sum update; 240 VGPRs, no spills; 1e30 disables it; the builder
and the C stack read H3_ATTN_SKIP_TAU), and the first-block cache in the C pipeline
(`cache_threshold` in h3pipe_params, ABI 2; `absdiff_sum_f32` partial sums for the relative
L1; accumulated across skipped steps as TeaCache does; `H3_CACHE_TRACE=1` prints decisions).

## The 4x accounting, measured (2026-09-07)

The two levers above were measured on the fox clip through the C pipeline, GPU otherwise idle
(the earlier contended numbers are withdrawn):

| 480p, 124 frames (15427 rows), s per step | steps 2 and 3 |
| --- | ---: |
| f16 attention | 28.8 / 28.7 |
| int4 attention, skip off | 23.3 / 24.0 |
| int4, skip tau 8 | 24.5 / 24.5 |
| int4, skip tau 6 | 23.6 / 24.1 |

The tile skip buys nothing at 480p: the skipped tiles' P.V work is replaced by the gap
reduction and the branches, and the kernel is not MMA-bound (below). The step cache
(`tools/cache_study.py`, 22 frames, 30 steps, 480p, latents against the uncached run):

| threshold | time | video latent cosine | audio | frame PSNR |
| --- | ---: | ---: | ---: | ---: |
| 0 | 89.6 s (includes first-launch costs) | | | |
| 0.05 | 71.1 s | 1.0000 | 1.0000 | 138 dB (no evaluation skipped) |
| 0.10 | 45.6 s | 0.9389 | 0.9707 | 16.5 dB |
| 0.15 | 45.5 s | 0.9741 | 0.9538 | 22.5 dB |
| 0.20 | 31.4 s | 0.9710 | 0.9662 | 23.0 dB |

So the first-block criterion has no useful setting on H3: the first threshold that skips
anything (0.10) drops the latent cosine to 0.94 and the frames to 16.5 dB. The cache stays
in the API (0 = off) as a knob for previews, not a default. Neither lever delivers its
planned share of the 4x; the kernel's own ceiling is the remaining question.

### krea2-loom's prefetch schedule, tried here (item 4 of the krea2 review)

krea2-loom's native attention found that at 8k-16k tokens a four-wave kernel with explicit
next-tile prefetch (global loads issued at the top of the iteration, held in registers
across the compute, stored into the other LDS slot after it, one barrier per tile; no
loop-carried values) beat the eight-wave double buffer. `ATTN_PINGPONG=1` in
`tools/gen_attention_i4qk.py` builds that schedule on the int4 kernels
(`experiments/attention_i4qkpp*`); all variants compile at 256 VGPRs with no spills and
pass `tests/test_attention_i4.py` (cosine 0.99999996 against the replica). Interleaved
best-of-5 (`tools/ab_attention_i4.py`), GPU idle:

| 37743 rows (768) | ms | vs shipped long form |
| --- | ---: | ---: |
| `attention_i4qkl_mha8` (8 waves, double buffer, 1 WG/CU), shipped | 1607-1630 | 1.000x |
| 4 waves, double buffer, 3 WG/CU | 3209 | 0.54x |
| 4 waves, ping-pong, 3 WG/CU (`i4qkpp`) | 3149 | 0.55x |
| 4 waves, double buffer, LDS-padded to 2 WG/CU (`i4qkd2`) | 2009 | 0.81x |
| 4 waves, ping-pong, 2 WG/CU (`i4qkpp2`) | 1588-1596 | 1.012-1.022x |
| 4 waves, ping-pong, 1 WG/CU (`i4qkpp3`) | 1795 | 0.91x |
| 8 waves, ping-pong, every lane loads K and V (`i4qkppl`) | 1643 | 0.98x |
| 8 waves, ping-pong, loads in scf.if branches (`i4qkppy`) | 2085 | 0.77x |

At 15427 rows the shipped 8-wave kernel, the padded long form and `i4qkpp2` are within 1%
of each other (267-270 ms). The prefetch is real (1.26x over the four-wave double buffer at
equal occupancy) but the eight-wave kernel already shares each K/V tile across twice the
queries, and the two end up within 1-2%: every schedule saturates at 25.4-25.7 TFLOP/s-eq,
a third of the 74 TFLOP/s-eq MMA mix ceiling (int4 QK^T, f16 P.V). The kernel is not
latency-bound on staging; the ceiling is elsewhere (the eight chained QK^T MMAs per score
tile, the softmax VALU work, and the LDS V reads are the candidates). Not shipped: about 1%
of the step for a third host wiring of the long form, and neutral at 480p.

### The remaining dependency, and the runtime that removes it

`libh3pipe.so` runs no HIP device code and no vendor math library, but its host side still
uses eight HIP runtime calls (hipMalloc/Free, Memset, MemcpyHtoD/DtoD/DtoH, DeviceSynchronize,
ModuleLoad/ModuleLaunchKernel), so it links libamdhip64 and through it ROCr. hrx-system builds
`libhrx.so` (`libhrx/include/hrx_runtime.h`, `build-cuda/libhrx/src/libhrx/libhrx.so`) whose only
shared dependencies are libc and libm: it embeds IREE's AMDGPU HAL and drives the kernel driver
directly, with no HIP and no ROCr. It has every call the pipeline needs: `hrx_gpu_initialize`
/ `hrx_gpu_device_get`, `hrx_stream_create`, `hrx_buffer_allocate` (device-local) with
`hrx_buffer_get_device_ptr` and `hrx_buffer_lookup` (device pointer -> buffer + offset, so the
blob-offset pointers the pipeline passes around keep working), `hrx_stream_fill_buffer`,
`hrx_stream_copy_h2d/d2h/copy_buffer`, `hrx_stream_synchronize`, `hrx_executable_load_file`
(target family "amdgpu") with `hrx_executable_lookup_export_by_name` and
`hrx_executable_export_info` (constant byte length, binding count, workgroup size), and
`hrx_stream_dispatch(stream, executable, export, {workgroup_count, workgroup_size, subgroup_size},
constants, constants_size, bindings[], n, flags)`: scalar launch arguments go in the constants
block, buffer arguments as (buffer, offset, length) bindings. llama.cpp's `hrx-rfc` branch
(`ggml/src/ggml-hrx/runtime/command-program-executor.cpp`) dispatches Loom kernels exactly this
way. Porting `h3pipe.cpp`'s `Kernel`, `launch`, allocation and copy helpers onto it is a
contained change (one wrapper layer, validated per kernel by the export info) and leaves the
library with no ROCm userspace dependency at all: the kernel driver, loom-compile at cache-fill
time, and nothing else. That is the shape "independent inference" asks for, and krea2-loom's
native path (HIP kernels plus hipBLAS) is the divergence to fold back into it.

### The skip's idle cost, and what ships (2026-09-07, later)

The 768 step through the C pipeline (after fixing a buffer the final norm overran at that
shape: the pipe's int8 activation buffer was sized for text-width rows), GPU idle:

| 768x1344, 124 frames (37723 rows), s per step | step 1 / step 2 |
| --- | ---: |
| int4 attention, skip off | 125.8 / 130.7 |
| int4 attention, skip tau 6 | 114.8 / 113.8 |

So the tile skip is worth 1.13x on the 768 step, nothing at 480p. But the disassembly of the
skip kernel's key-tile loop (`llvm-objdump` on the HSACO) shows 897 instructions per tile
against 487 without the skip: the eight per-fragment `scf.if`s cost about 400 register moves,
selects and branch bookkeeping per tile even when no tile is ever skipped, and RDNA3 issues
WMMA through the same pipeline as that VALU work. Interleaved A/B, skip compiled but disabled
vs the plain kernel: 1620 vs 1463 ms at 37743 rows (plain 1.107x), 273 vs 241 ms at 15427
(plain 1.132x). A single branch around all eight fragments spills (`st_key`, 100 bytes of
private) and runs at 0.42x, so the eight-branch form stays as the skip form. Through the C
pipeline at 480p the two land within run-to-run noise (plain 25.0 / 23.8 s, skip twin never
skipping 23.3 / 23.3 s), so the kernel-level difference does not show at the step there.

Shipped: the plain kernels keep their names (`attention_i4qk_mha8`, `attention_i4qk_mha`,
`attention_i4qkl_mha8`) and each has a tile-skip twin (`attention_i4qks_mha8`,
`attention_i4qks_mha`, `attention_i4qksl_mha8`; `scripts/gen_attention_i4.sh` regenerates all
six). Both builders choose the twin only when a tau applies: `H3_ATTN_SKIP_TAU=<tau>` forces
it, `off` disables it, and unset means tau 6 on the long form (rows >= 20000) and off below.
The builder records the choice in `attention_stem.txt` (the Python host loads that symbol and
the attention HSACO is rebuilt when it changes). `H3_PROFILE=1` prints synchronized per-stage
times after each C-pipeline step (they inflate the step about 1.6x because the GPU clocks down
between synchronized launches; use them for proportions only: attention is about half of the
480p step), `H3_TRACE=1` prints and synchronizes every launch.

Where the int4 kernel's time goes now, from the plain loop's 487 instructions per key tile:
16 WMMAs; 88 register moves (the ten loop-carried 8-wide vectors); 72 DPP moves and 20 maxes
for the four cross-lane rounds of the row max plus the same for the row sum; 16 exps; 16 stores
and 8 loads to turn the score accumulator layout into the P operand layout through LDS; 88
selects; 64 `s_delay_alu` stalls. The formulation that removes most of it computes S^T = K Q^T
instead: the accumulator then holds a query column per lane, the row max and sum become seven
in-lane maxes plus one cross-half shuffle instead of four DPP rounds each, and the P operand
comes from one cross-half exchange instead of the LDS round trip. That is the next kernel lever
(about a third of the loop), and it is a regeneration of the kernel, not a tweak.

### 32-key tiles on the int4 kernel: lost (2026-09-07)

`tools/gen_attention_i4_32.py` builds two forms from the shipped 16-key kernel: a fused 32-key
tile (two K rows and two V chunks per lane per barrier, one max over both sub-tiles, two P round
trips, 16 int4 + 16 f16 MMAs per barrier; 248 VGPRs, no spills) and a plain unroll by two (the
16-key body twice under one barrier; 256 VGPRs, 72 bytes spilled). Both are correct
(`tests/test_attention_i4.py` cosine 0.99999996) and both lose, interleaved best-of-5:

| 15427 rows | ms | vs 16-key |
| --- | ---: | ---: |
| 16-key, shipped | 218-227 | 1.000x |
| fused 32-key | 279-282 | 0.78-0.81x |
| unrolled by two | 385 | 0.57x |

At 37743 rows the fused form is 0.77x of the long form. The f16 kernel's 1.47x from 32-key
tiles came from a four-wave kernel with LDS headroom; the int4 kernel is already at the
register ceiling (240 of 256, one workgroup per CU by VGPRs alone, so the LDS pad of the long
form is not what its 11% came from), and every per-tile cost it would amortise was measured
small. The key-scale prefetch (1.036x / 1.068x) is the one lever of this round that shipped.

Where this leaves the kernel: 30 TFLOP/s-eq is 40% of the int4/f16 mixed MMA ceiling; the f16
P.V half alone runs at about 50% of the part's f16 peak, which is the range published
attention kernels reach on RDNA3. The ablations put every remaining overhead below 7% and the
rest in overlapping latency that no single removal exposes. A further 2x on the kernel is not
on this list; the remaining step-level levers cut attention work instead: a lower skip tau
(quality trade), and a two-pass final-max skip for the late layers where 60-80% of tiles are
skippable under the final max but only 40-60% under the running max.

### The key scale's four placements, and a barrier the dbuf pass had missed (2026-09-07, later)

The per-tile global load of the key scale (5% in the ablation) was tried four ways on the
plain 8-wave kernel, interleaved A/Bs (the GPU was shared with another process for the later
rows, so only the ratios hold):

| key scale | 15427 rows | 37743 rows | notes |
| --- | ---: | ---: | --- |
| loaded after the QK^T chain (original) | 1.000x | 1.000x | |
| carried: next tile's loaded at the end, kept in a register | 1.036x | 1.068x | shipped |
| staged into LDS with the K tile (16 lanes store, loop reads LDS) | | 0.79x | no spills; lost anyway |
| loaded before the QK^T chain (nothing carried) | | 0.87x of carried | |

The skip twins spill 60 bytes with the carried form and run at 0.39x, and 0.77x with the
early form, so the twins take the scale the original way (`uncarry` in the generator) and are
byte-identical to the pre-prefetch twins. The carried form also exposed a generator bug: the
dbuf pass removes the loop's trailing barrier by matching the barrier followed by the yield,
and the carried scale's lines sat between them, so the shipped carried kernels ran two
barriers per tile and were still 1.07x. Measured directly at 15427 rows (shared GPU): the
trailing barrier costs 5% on the original kernel; the carry without it is 1.023x over the
original on that run. The pass now asserts one barrier per tile.

The reference stack's final velocity cosine with the skip (`tools/quant_study.py
attn:w4a4:a4rs<tau>`): tau 6 0.97508, tau 5 0.97506, tau 4 0.97529, against 0.9751 without
the skip: no measurable cost down to tau 4 on this metric, so the long form's default tau is
chosen by step time alone (below).

### The long form's default, decided at the step (2026-09-06)

768x1344, 124 frames, three steps through the C pipeline, the three settings interleaved
twice (the GPU showed 37-38% use from elsewhere before every run after the first):

| round | skip off (plain, carried scale) | tau 6 twin | tau 4 twin |
| --- | ---: | ---: | ---: |
| 1 | 110.1 / 113.1 s | 127.9 / 126.8 | 126.6 / 126.2 |
| 2 | 128.8 / 129.1 s | 128.7 / 128.0 | 124.9 / 125.4 |

With the carried key scale in the plain kernel the twins no longer earn their idle cost: tau 6
is a wash and tau 4 is about 3% at the step. The plain kernel is the default at every length;
`H3_ATTN_SKIP_TAU=4` is the opt-in (no measured velocity cost). On an idle GPU the 768 step
is 110-114 s, from 126-131 s at the start of this round and about 160 s with f16 attention:
30 steps at 768 are about 57 minutes.

Two process lessons from the round. The C kernel cache was keyed by stem and config only, so
an edited kernel source reused a stale binary: a spilling twin ran the 768 step at 263 s
before the cause was found. Both caches now carry a source hash. And a variant must be A/B'd
in every form it ships in: the carried scale was measured on the plain kernel and shipped to
the twins untested, where it spilled.

## Head to head with ComfyUI at 768 (2026-09-06)

`tools/bench_comfyui_h3.py` runs ComfyUI's own MiniMax H3 path in the Strix Halo image
(`podman run --rm ... --entrypoint /opt/venv/bin/python docker.io/kyuz0/amd-strix-halo-comfyui:latest`,
ComfyUI 62b3c94 of 2026-08-11 inside it, torch 2.14.0a0+rocm7.15, "pytorch attention",
DynamicVRAM on): the int8 ConvRot checkpoints from `~/comfy-models`, bf16 compute, the minimax
CLIP (Qwen3-VL-32B int8), the AV latent node, stock Euler on the 'simple' schedule with the
model's shifts 12/3, cfg 1, no decode. The same prompt, 1344x768, 124 frames, three steps,
timed per step from the sampler callback; our pipeline ran right after on the same idle GPU.

| | per step | text encode |
| --- | ---: | ---: |
| ComfyUI, int8 ConvRot, bf16 compute, pytorch attention | 772.7 / 772.4 / 768.8 s | 28.7 s |
| this pipeline, int4 blocks, int4-QK attention (same session) | 127.1 / 127.9 / 128.6 s | |
| this pipeline, best idle runs earlier in the day | 110 / 113 s | |

Six times faster per step (6.0x in the same session, 6.8x against our best idle runs); 30
steps are 6.4 hours in ComfyUI against 57 to 64 minutes here. The estimate before measuring
(1.5x) assumed torch SDPA at 18 TFLOP/s; at 37k rows ComfyUI's attention runs far below that.
Two runs before this one died: with `--disable-mmap` (the toolbox's recommendation, used by
krea2's bench) the 48 GB of int8 weights sit in RAM next to their device copies and the run
was OOM-killed at sampling; the script now mmaps and unloads the text encoder before sampling.
And a container launched from a shell wrapper loses its output if the wrapper dies, so the
bench runs detached (`podman run -d --name h3bench`, `podman logs`). Another session's GPU job
(`build/down-component-repeat.py`) overlapped the second attempt; the numbers above are from a
run with nothing else on the GPU (rocm-smi 0% before, checked).

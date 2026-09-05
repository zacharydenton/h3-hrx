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

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

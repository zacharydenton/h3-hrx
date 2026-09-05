# minimax-h3-loom

MiniMax H3's (Hailuo 3.0) transformer blocks in **Loom**, AMD's kernel language from
[ROCm/hrx-system](https://github.com/ROCm/hrx-system), for the Radeon 8060S (gfx1151), in
**W4A4 ConvRot** int4 on the part's `iu4` WMMA, the one 2x path this silicon has (117 TOPS
measured against 54 for int8 and fp16). The goal is int4 throughput: the best a 20B-active
video model can do on this part. A sibling of
[krea2-loom](https://github.com/zacharydenton/krea2-loom), whose kernels, runtime and
test discipline this repo starts from.

## The model

H3 is a 33B dense single-stream transformer over one packed sequence of text, video and
audio rows: 50 blocks of hidden 5376, 56 attention heads of 128 with no key-value sharing,
SwiGLU 14336, AdaLN modulation per (timestep, modality) row from tables that are
precomputed once per schedule (13B of the 33B never run at inference). Qwen3-VL-32B is
the encoder, a causal f16 t4 24-channel video VAE and a 40 Hz audio VAE the tokenizers.
`docs/notes.md` has the side-by-side with Krea 2 and the FLOP arithmetic.

## Plan

1. `reference/h3_ref.py`: the block stack on the checkpoint's names, bit-exact against
   diffusers' `MiniMaxH3Transformer3DModel` at toy size, with a `w4a4` mode that is the
   kernels' exact arithmetic. A fixture captured from a real denoising step.
2. Kernels, from krea2-loom's set by configuration where the shapes allow: the int4 GEMM
   (qkv fused to N = 21504, out 7168 -> 5376, gate|up with the SwiGLU product in the
   epilogue, down), the residual GEMM with a per-row-class gate table, the prepare kernels
   with per-row-class AdaLN shift/scale, rotate-half 3-axis RoPE with q/k RMSNorm, and the
   LDS-staged attention with four query tiles of one head per workgroup (no GQA to share).
3. The C runtime and ctypes wrapper, the end-to-end clip through the official pipeline
   with the blocks in Loom, PSNR against bf16, then the lever loop.

## What to expect

At 480p and 4 s (about 10k rows) a step is about 12 s at the rates krea2-loom reaches on
this part; at 768p and 5 s (about 35k rows) attention alone is about 90 s per step.
Attention, not the int4 GEMMs, is the wall at video lengths.

## Weights and license

`scripts/download.sh` fetches the FL2VA partition (66 GB transformer, 64 GB encoder,
VAEs) into `~/h3-models`. The MiniMax H3 Community License carries regional terms with
an application process for the USA, EU, UK and South Korea; read `LICENSE` in the
checkpoint before downloading.

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

## Status (2026-09-05, day one)

The 50 blocks run in Loom end to end. On the 2097-row fixture built from the real
checkpoint (32 text rows, a 5-frame 480p-class latent grid, 20 audio latents) every
kernel passes its test against the reference and the native stack reaches a final-layer
velocity cosine of 0.991 against the int8 checkpoint with the GPTQ weights
(`tests/test_blocks.py --weights build/weights_gptq`). A 50-block step takes 1.62 s at
that size on a calm box; at 10317 rows (480p, 4 s of video) it takes 14.6 s, of which
attention is 8.9 s at 17 TFLOP/s and the four GEMMs 5.0 s at 78-82 TOPS. The 2097-row
profile:

| stage | share |
| --- | ---: |
| gate/up GEMM with the SwiGLU product | 27% |
| qkv GEMM | 21% |
| attention (56 heads, 14.5 TFLOP/s) | 20% |
| down GEMM with the class-gated residual | 14% |
| out GEMM with the class-gated residual | 7% |
| prepare kernels, RoPE | 11% |

Two things H3 needed that Krea 2 did not: an f32 residual stream (H3's passes f16's range
by block 23 and reaches 3e6 by block 36) and f32 LDS in the prepare kernels (the gate/up
products reach 5e4 and the unnormalised Hadamard stages overflow f16). The raw residual
stream is dominated by a few huge channels, so cosines on it mislead deep in the stack;
the final layer's RMSNorm removes them, which is why the velocity is the metric.

The end-to-end clip runs from one command, `tools/pipeline.py "<prompt>"`: the prompt
through the text encoder in Loom (Qwen3-VL-32B's 50 layers, W8A8 from the int8 file, layer
50's raw state, 0.3 s for a 33-token prompt), the layout and the two schedules through
diffusers' `MiniMaxH3Scheduler`, the 50 blocks in Loom, the video VAE's 36 decoder blocks
in Loom (W8A8, 46 s for the 124-frame clip against about 19 minutes for diffusers' fp32
decoder), and diffusers' audio VAE plus ffmpeg. The first full clip: 124 frames (5.2 s) at
864x480 with stereo audio, 49 denoising steps with the GPTQ int4 blocks, frames 5, 60 and
118:

![fox](docs/media/fox_480p_5s_strip.jpg)

That render packs 15427 rows and takes 29.4 s per step (attention 70%, the four GEMMs
26%): 24 minutes of denoising. `docs/media/smoke_fox_22f_8steps.jpg` is the 22-frame smoke
clip; `docs/media/smoke_sailboat_22f_8steps_fullloom.jpg` the same size through the full-Loom path
(prompt encoded in Loom in 17 s including the 24 GB weight upload, 2.4 s per step at 2931 rows,
video decoded in Loom in 10.6 s).

**The C library.** `host/h3pipe.h` / `build/libh3pipe.so` is the whole pipeline behind a C ABI
in the shape of dinov3-loom's and scrfd-loom's runners: `h3pipe_create` (weight dirs, the
kernel sources, a cache dir, the loom-compile path), `h3pipe_denoise` (token ids -> model-space
latents, optional caller noise, a progress callback), `h3pipe_decode_video` (latents -> RGB8
frames), `h3pipe_decode_audio` (latents -> stereo samples) and `h3tok_encode` (text -> ids, the
Qwen2 byte-level BPE in C). Every kernel is Loom, compiled on first use for a shape into the
cache; the host does the layout, AdaLN curves, scheduler, RNG, patching and blending. Python
is only the oracle: `tests/test_pipe.py` checks the refined text rows (cosine 0.9998), one
denoising step from shared noise (update cosine 0.991, the int4 blocks' own floor), the video
decoder (49.4 dB) and the audio decoder (108.9 dB SNR) against the Python stages;
`tests/test_tokenizer.py` checks the tokenizer against transformers. `build/h3pipe --prompt
"..." --frames 22 --steps 8 --out build/clip` writes raw RGB and WAV; `tools/pipeline_c.py`
drives the library from Python and muxes with ffmpeg. The runtime dependency is the HIP
runtime API (module load, memory, launch); there is no device code outside Loom.

**What is in Loom and what is not.** In Loom: everything with tensors in it -- the DiT blocks,
the text encoder's layers (`tests/test_te_blocks.py`: hidden cosine 1.0000 after 50 layers
against transformers bf16, median per-token error 1.7%, the W8A8 floor), the token refiner,
the embedders and final layer (int8 GEMMs with padded K / N), the video decoder's blocks and
heads (`tests/test_vae_blocks.py --bits 8`: frame PSNR 52 dB; the int4 GPTQ decoder is
`vae_bits 4`) and the audio vocoder (five f32 SIMT kernels). On the host in C: the layout,
the AdaLN curve tables, the scheduler, noise, patchify and unpatchify, chunk blending and the
pixel mapping, together well under a second of a clip.

**Quantisation.** Round-to-nearest int4 per row cost 21% relative error on the final
velocity over 50 blocks; GPTQ at export time (`tools/gptq_export.py`, block by block on the
fixture's activations) brings the same format to 12.5%, velocity cosine 0.992 (0.991
through the native runtime), matching the best per-group scheme at no kernel cost. Int8
weights would cost 2.4% but cap the GEMMs at the part's 54 TOPS int8 peak; the target is
100 TOPS. `build/weights_gptq` is the export to use.

**GEMM tile.** The four GEMMs run on a 256x128 workgroup tile of 64x64 wave tiles (half the
LDS operand reads per multiply of the 128x128 tile), 15-17% faster per stage at 2097 rows;
`H3_GEMM_TILE=128` builds the smaller tile for comparison.

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

The blocks are exported from the ComfyUI `pruned_int8_convrot` checkpoint in
`~/comfy-models` (`tools/export_weights.py`, 9.65 GB of int4): the int8 rows are already
rotated by the same group-256 regular Hadamard the kernels use, so each row is just
requantised to int4. `scripts/download.sh` fetches the original bf16 release from
Hugging Face if it is ever needed. The MiniMax H3 Community License permits open-weight
use in the USA, EU, UK and South Korea; other regions apply to MiniMax for a licence.

## Environment

The torch side shares `~/code/krea2-loom/.venv` (ROCm torch 2.13, a diffusers dev checkout
that carries the H3 transformer, VAEs, scheduler and modular pipeline); `scripts/env.sh`
points at the Loom toolchain.

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

**Int4 QK^T attention.** SageAttention's idea with this part's arithmetic: int8 WMMA runs at the
f16 rate on gfx1151, int4 at 2.2x, and the 50-block velocity study showed int4 Q and K (per
token and head, the head rotated by a Hadamard in the prepare kernel) cost one point on top of
the int4 GEMMs (velocity cosine 0.990 against 0.991 with f16 attention). `prepare_qk_i4` and
`tools/gen_attention_i4qk.py` (QK^T in int4 with i32 accumulation, PV in f16) give attention
1.44x at the 480p clip length and, with the double-buffered variant chosen past 20k rows,
1.39x at 1344x768: the 480p step goes 14.6 -> 11.9 s, the 768 step is about 130 s (attention
82%), so 30 steps at 768 are about 65 minutes. `H3_ATTN_QK=f16` restores the f16 kernel.
Torch's SDPA and AMD's hrx-demos attention both sit at ~18 TFLOP/s f16 here; the int4 kernel
is the one attention change on this part with a hardware rate behind it.

**Tile skip and step cache (the levers past 2x).** The int4 kernels have tile-skip twins
(`attention_i4qks*`: a key tile whose scores sit more than tau below every row's running
max skips its P.V work, SpargeAttn-style). Measured on the fox clip through the C pipeline the
twin's idle branch machinery costs 10-13% and the skipping wins it back: against the plain
kernel with its key scale carried one tile ahead, tau 6 is a wash and tau 4 about 3% on the
768 step, so the plain kernel is the default at every length and `H3_ATTN_SKIP_TAU=4` the
opt-in (no measured velocity cost). On an idle GPU the 768 step is 110-114 s, so 30 steps at
768 are about 57 minutes. The first-block step cache (`cache_threshold` in `h3pipe_params`,
`--cache-threshold` in `tools/pipeline_c.py`) is a preview knob, not a default: the first
threshold that skips anything (0.10) drops the latent cosine to 0.94 and the frames to
16.5 dB against the uncached run (`tools/cache_study.py`).
`H3_PROFILE=1` prints per-stage times after each step (proportions only; the synchronization
inflates the step), `H3_TRACE=1` prints and synchronizes every launch. Every measured lever,
won or lost, is in `docs/notes.md`.

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

## Reference conditioning: fl2va and ref2va, all in Loom

The three encoders ComfyUI's graph needs for references now run in Loom: the audio VAE encoder
(`h3pipe_encode_audio`), Qwen3-VL's vision tower with its DeepStack features into the text encoder
(`h3pipe_vision_embed`, used by the prompt path), and the video VAE's causal 3-D conv encoder
(`h3pipe_encode_video`, images and 17-frame clips, tiled as ComfyUI does). `h3pipe_denoise_refs`
packs keyframes (fl2va) and reference images, videos and audio (ref2va) between the text and the
target streams with ComfyUI's layout. Against ComfyUI's own encoders: audio 1.0000000, vision
0.99996, VAE 0.9994; one-step network velocities 0.94-0.99 across t2va, fl2va and ref2va (the gap is
the int4 blocks). Weights: `tools/export_audio_encoder.py`, `tools/export_vision.py`,
`tools/export_vae_encoder.py`, and for ref2va `H3_CKPT=<ref2va checkpoint> H3_GLUE_OUT=... tools/export_glue.py`
plus `tools/gptq_export.py --out ...`. Run:

```sh
python3 tools/pipeline_c.py "<Picture 1> is the fox. A red fox ... with the sound of <Audio 1>" \
  --ref-image fox.png --ref-audio fox.wav        # ref2va: picks build/weights_i8_ref2va + weights_glue_ref2va when exported
python3 tools/pipeline_c.py "A red fox ..." --first-frame fox.png      # fl2va with the default weights
```

The defaults are the stock ComfyUI workflows' settings: `res_multistep` on the `simple`
schedule, 30 evaluations (`--steps 31` grid points; ComfyUI's workflows use 20), no CFG, shifts 12/3, and
`--precision int8` (the checkpoint's int8 rows, `tools/export_weights.py --bits 8`, with int8
QK^T attention, `tools/gen_attention_i8qk.py`; `--attn f16` is the same quality, slightly slower). `--precision bf16` runs the pruned bf16 checkpoint's rows in f16 (`tools/export_weights.py --bits 16
--source .../minimax_h3_fl2va_pruned_bf16.safetensors`, f16 attention, `tools/gen_gemm_f16.py` kernels):
no weight quantisation in the blocks, at about the int8 speed. `--precision int4` is the 2x-per-step path (GPTQ int4 blocks, int4 QK^T attention):
fine for text-only previews, but it ghosts keyframe and reference clips, because its deep-block
error drifts the moving frames away from the pinned anchor (`docs/notes.md`, "ComfyUI's
sampler, and why int4 ghosts conditioned clips"). At parity precision the host reproduces
ComfyUI's residual stream to 0.999 through all 50 blocks (`tests/test_comfy_parity.py`).

Details and every gate: `docs/plan-refs.md`, `docs/notes.md` ("Reference conditioning").

## What to expect

At 480p and 4 s (about 10k rows) a step is about 12 s at the rates krea2-loom reaches on
this part; at 768p and 5 s (about 37k rows) a step is 110-114 s (124 frames), attention about 75% of it.
Measured head to head against ComfyUI's own H3 path (int8 ConvRot checkpoints, bf16 compute,
pytorch attention, `tools/bench_comfyui_h3.py` in the Strix Halo image): 771 s per step at
1344x768 and 124 frames against 128 s here with the int4 path in the same session, six times
faster; 30 steps are 6.4 hours there and about an hour here (`docs/notes.md`, "Head to head
with ComfyUI"). The int8 path, the one whose clips match ComfyUI's, costs 146 s per
evaluation at that size with int8 QK^T attention (161 s with f16 attention), 5.3x faster than
ComfyUI; at 864x480 and 124 frames (5 s of video) 33 s against ComfyUI's 103 s, 3.1x; at
864x480 and 22 frames 4.0 s against 5.2 s. Attention is 70% of a 768p evaluation and runs at
17 TFLOP/s against the part's 54 peak; the remaining large lever is that kernel's schedule.

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

Run `bash scripts/test_host.sh` for CPU-only regressions without model weights or a GPU
context. After `bash scripts/build_host.sh`, `python3 tests/test_kernel_regressions.py`
checks GroupNorm and the experimental 32-key attention kernels on small synthetic inputs
without loading torch or checkpoints. `scripts/test.sh` includes these checks; its full
mode also runs the much larger model comparisons. The host build now produces the kernel
test runner at `build/loomrun`.

`python tests/test_decoder_tiles.py` in the torch venv checks C decoding against Python's
spatial and temporal blending. It loads only decoder weights and small projection tables,
without the DiT, text encoder or full floating-point VAE. Pass `--latents video.npy` to
check saved `[24,T,H,W]` latents.

`python tests/test_vae_blocks.py --curve 1,36 --bits 8` checks independent floating-point
accuracy on a real decoder tile, loading one reference block at a time. It enforces both
update cosine and frame PSNR; `--full-clip` expands the spatial test while keeping attention
memory bounded. The normal Python Loom decoding tools also load only the small VAE heads.

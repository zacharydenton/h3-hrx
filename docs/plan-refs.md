# Reference conditioning in Loom: fl2va, ref2va, audio (plan, 2026-09-06)

Goal: every H3 task the ComfyUI graph offers, with every kernel in Loom: text-to-video-audio (done),
first/last-frame keyframes (fl2va), and reference images, videos and audio (ref2va).

## What ComfyUI does (comfy/ldm/minimax/model.py, comfy_extras/nodes_minimax_h3.py, text_encoders/minimax.py)

- Presentation: for each reference, in order images, videos, audio: `<Picture i>: ` + `<|vision_start|>`
  + vision embeds + `<|vision_end|>`; `<Audio j>: ` (text only); `<Video k>: ` then per 2-frame block
  `<t seconds>` + a vision block. Then the prompt. Image spans (plus the flanking tokens) carry
  modality tag 0 (video) in the text span; everything else tag 1.
- Vision tower (Qwen3-VL-32B `visual.*`, 1.19 GB bf16 inside the text-encoder checkpoint): patch embed
  conv 3x2x16x16 -> 1152 (a 1536-input GEMM on 16x16 patches, the 2 temporal slots both the image),
  learned 48x48 position embedding bilinearly interpolated to the grid and permuted into 2x2 merge
  order, 27 blocks of LN(bias) + qkv(bias) + 2D RoPE over the full 72-dim head (36 h-freqs, 36
  w-freqs, theta 10000) + full attention per image + proj, LN + MLP 4304 GELU(tanh); the merger LN(1152)
  -> view 4608 -> fc1 -> GELU(erf) -> fc2 -> 5120; DeepStack mergers (LN(4608), fc1, GELU(erf), fc2)
  after blocks 8, 16, 24, added to the LLM hidden state at the image rows after LLM layers 0, 1, 2.
  The LLM uses interleaved mrope (t, h, w) position ids for the image rows (dim pair i takes axis i % 3).
- Video VAE encoder (`encoder.*`, `quant_conv` in minimax_h3_video_vae_fp16): causal 3D convs, ch 128,
  ch_mult (1,2,2,4,4,8), 2 res blocks per level (GroupNorm(32, eps 1e-6, per frame) + SiLU + conv),
  space_down (2,2,2,2,1,1), time_down (1,2,2,1,1,1), reflect spatial padding, causal temporal padding;
  conv_out -> 48 -> quant_conv -> moments, mean = first 24 channels, (mean - latents_mean)/latents_std.
  Images: one frame, `moments[:, :, -1:]`. Videos: 17-frame clips (repeat-padded), token_drop 3.
  Spatial tiles of 256 px, overlap >= 64, blended in latent space.
- Audio VAE encoder (`encoder.block.*`, `pre_block.*`, `mean_proj` in minimax_h3_audio_vae_fp32): mono
  per stereo channel, right-pad to a multiple of 800; conv1d 1->64 k7; five EncoderBlocks (dims 128,
  256, 512, 1024, 2048; strides 2,4,4,5,5): three ResidualUnits (Snake, conv k7 dilation 1/3/9,
  Snake, conv k1, residual) + Snake + strided conv k=2s; Snake + conv 2048->2048 k3; AttnProjection:
  norm3 -> proj(2048->32) + causal attention (qkv 2048, q/v bias, mean over heads, adaptive average
  pool 2048->32, proj) on norm1; + GeGLU MLP (norm2, w0/w1 32->64, GELU(tanh), w2); mean_proj 1x1;
  (z - latents_mean)/latents_std -> [32, 2, T].
- DiT layout: [text | keyframe cond rows | reference blocks | audio | video]. Cond/ref rows are the
  patchified reference latents through the same video/audio patch projections (visual ones mixed with
  seeded noise at 0.999, audio at 1.0 = none), positioned at the reference's own grid with the time
  cursor advancing by each block's span (image 1.0, audio ref_audio_t, video max of both), modality
  tags 0/2, timestep classes t = max(t_v, 0.999) and max(t_a, 1.0), re-set every step, their outputs
  dropped. The target streams follow with the cursor after the references.

## Work packages, each gated by a ComfyUI ground truth (tools/ref_truth_comfy.py -> build/ref_truth)

0. Truth: audio-encoder latents for the fox clip's wav; video-VAE latents for a frame and a 17-frame
   clip; the vision tower's merged + deepstack outputs, the presentation ids, tags and text states for
   a prompt with one image; one-step ref2va denoised outputs (image + audio refs, and audio-only) with
   the noise saved.
1. Tokenizer presentation (h3tok): `<Picture i>: `, vision start/pad/end ids, `<Audio j>: `, video
   blocks; per-token tags. Gate: ids identical.
2. Audio encoder in Loom (conv1d with stride, snake, layernorm, small f32 GEMM, causal attention,
   channel pool, GeGLU). Gate: latents cosine > 0.9999 against truth.
3. Layout with cond/ref segments, four timestep classes, cond rows through the patch projections,
   ABI `h3pipe_denoise` with references; audio-only ref2va. Gate: one-step denoised vs truth.
4. Vision tower in Loom (f16 GEMMs with bias/GELU epilogues from dinov3-loom, head dim padded 72 -> 128
   for the attention kernels, host cos/sin for the 2D rope), deepstack into the text encoder, mrope
   tables, image embeds in the presentation. Gate: merged/deepstack cosine, text states cosine.
5. Video VAE encoder in Loom (implicit-GEMM 3D conv, GroupNorm+SiLU, reflect/causal padding,
   tiling and latent blending on the host). Gate: image and clip latents vs truth.
6. Keyframes (fl2va) and reference images/videos in the layout; ref2va checkpoint export; CLI,
   Python, tests, docs. Gate: one-step denoised vs truth for each task.

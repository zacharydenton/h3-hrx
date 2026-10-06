# 3D upscaler reference

`input.f32` and `output.f32` are little-endian FP32, channel-first tensors
with shapes `[24,2,2,2]` and `[24,2,4,4]`. Inputs are already normalized.

Generated once with PyTorch 2.14.0 on CPU, four threads, inference mode,
and the unmodified `LatentResizer3D` class from upstream revision
`40316cf008b2fd8663263270669eb4da23f89d2c`. All weights and activations are FP16.
Input element `i` is `(i % 23 - 11) / 32`. Call:
`model(input.half(), scale=2.0, target_size=(2,4,4), enable_chunking=True)`.
Output is widened to FP32 for storage. No raw-VAE normalization is applied.

Checkpoint: `LBH-123-AI/Minimax_h3_latent_Upscaler`,
`minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors`.
SHA256: `043e5a48e161610ef6c3ea974645220354d06fa618abca15f76d084812eb55c2`.

The native regression allows FP16 rounding differences between CPU and GPU
convolutions. Tests consume these fixtures directly; Python is not required.
See the root `THIRD_PARTY_NOTICES` for upstream attribution.

`er_sde.json` records every pre-evaluation state and final output of ComfyUI's
`sample_er_sde`, revision `a7169322485d0049380fb207fa17e9fb3ec40486`, on CPU FP32.
It uses the listed sigmas and explicit noise, `s_noise=1`, `max_stage=3`,
flow SNR `-logit(sigma)`, and a synthetic clean predictor
`0.25*x + [0.1,-0.4,0.7,0.2] + 0.1*sigma`. This checks all three solver stages
independently of model weights and random-number implementations.

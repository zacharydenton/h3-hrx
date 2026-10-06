# 3D upscaler references

The reference is the complete upstream `MinimaxH3LatentUpscaler3D.execute`
node, including its input/output per-channel transforms. Those transforms are
part of the upscaler even though H3's incoming video latents are already
normalized by the VAE. Comparing only `LatentResizer3D.forward` misses this
boundary and previously allowed severely incorrect colors through the tests.

Generated with PyTorch 2.14.0 on CPU, four threads, inference mode, using
upstream revision `40316cf008b2fd8663263270669eb4da23f89d2c`. The node runs at
FP16 precision, scale 2, alignment 32, and temporal chunking enabled. Model
loading is supplied from the checkpoint below; the node's `execute` method
and network are unmodified. Outputs retain the FP32 input dtype.

All binary fixtures contain little-endian FP32 tensors, channel first:

- `input.f32`: `[24,2,2,2]`, element `i` is `(i % 23 - 11) / 32`.
- `node-output.f32`: the node's `[24,2,4,4]` output for `input.f32`.
- `node-spatial-output.f32`: `[24,2,16,16]` output for a `[24,2,8,8]`
  input whose FP32 element `i` is `(i % 103) / 51 - 1`.

`normalization.json` records Torch FP16 input and output transforms for all
24 channels, plus the restored value for a network output of 0.25. It tests
the intermediate half-precision rounding boundaries without a GPU.

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

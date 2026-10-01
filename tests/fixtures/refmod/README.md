# RefMod interoperability fixtures

Generated with PyTorch and the unmodified `core.py` / `bundle.py` from
ComfyUI-MiniMaxH3Mod commit `f9462081e28794389b5a6c5067eb327412ad8ee7`.
Upstream source: https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod/tree/f9462081e28794389b5a6c5067eb327412ad8ee7

These tiny synthetic tensors contain no source images, voices, or model weights.
No Python package is needed to run the Rust tests.

- `visual.safetensors`: `H3RefMod.save`, v4 video member named `fixture_visual`;
  `((arange(24*3*4*6) % 37 - 18) / 16).reshape(1,24,3,4,6).half()`.
- `combined.safetensors`: `save_bundle`, name `fixture`, containing that visual
  and a v4 audio member named `fixture_audio`;
  `((arange(32*2*19) % 53 - 26) / 16).reshape(1,32,2,19).float()`.
- `strength.safetensors`: upstream `ref_block(0.35)` results for those members,
  tensor names `visual_strength` and `audio_strength`.

Members use mode `encode`, concept types `identity` / `voice`, and default
metadata otherwise. The visual explicitly sets latent_t=3, latent_h=4,
latent_w=6. The bundle writer receives each member once at strength 1.

`strength.safetensors` also contains `irregular_strength`, from `ref_block(0.35)`
on a visual with the same modulo-37 input formula, shape `[1,24,2,18,20]` and
F16 dtype. This exercises spatial interpolation and nondivisible pooling bins.

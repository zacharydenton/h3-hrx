"""reference/h3_ref.py against ComfyUI's MiniMaxH3Model at toy size, random weights, f32 on CPU.

Runs where ComfyUI imports (the amd-strix-halo-comfyui image; see scripts/test.sh):
    PYTHONPATH=$HOME/code/ComfyUI python3 tests/test_ref_vs_comfy.py
The toy keeps head_dim = 128 (the rotary layout depends on it) and shrinks everything else.
"""
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reference"))
import h3_ref as R


def main() -> int:
    sys.argv = [sys.argv[0], "--cpu"]          # ComfyUI probes for a GPU at import otherwise
    import comfy.options
    comfy.options.enable_args_parsing()
    import comfy.ops
    from comfy.ldm.minimax.model import MiniMaxH3Model

    torch.manual_seed(0)
    hidden, heads, ffn, text_dim, layers, refiner = 256, 2, 512, 64, 2, 1   # K of every block linear is a multiple of the 256-group
    latent_t, lat_h, lat_w, audio_t, text_len = 2, 4, 6, 3, 5
    R.HIDDEN, R.HEADS, R.INNER, R.FFN, R.TEXT_DIM = hidden, heads, heads * 128, ffn, text_dim
    model = MiniMaxH3Model(hidden_size=hidden, num_layers=layers, token_refiner_num_layers=refiner,
                           num_attention_heads=heads, attention_head_dim=128, ffn_hidden_size=ffn,
                           text_dim=text_dim, time_embed_dim=8, adaln_curve_grid=1025,
                           dtype=torch.float32, device="cpu", operations=comfy.ops.disable_weight_init).eval()
    sd = {}
    for name, p in model.state_dict().items():
        if name == "rope.inv_freq":
            sd[name] = 10000.0 ** (-torch.arange(0, 32, 2, dtype=torch.float32) / 32)
        elif name.endswith("norm.weight") or name.endswith("norm1.weight") or name.endswith("norm2.weight"):
            sd[name] = 1.0 + 0.1 * torch.randn_like(p)
        else:
            sd[name] = torch.randn_like(p) * (0.5 / math.sqrt(p.shape[-1]) if p.ndim == 2 else 0.1)
    model.load_state_dict(sd)
    model.requires_grad_(False)

    class FakeCkpt:
        device, dtype = "cpu", torch.float32
        def tensor(self, name, dtype=None): return sd[name].to(dtype or torch.float32)
        def linear(self, name): return R.rotate_groups(sd[name + ".weight"].float(), R.hadamard(R.HADAMARD_GROUP))   # the checkpoint stores W H
    ref = R.H3Ref(FakeCkpt(), quant="none", layers=layers)
    ref.text_in = _text_in_toy(ref, refiner)

    text = torch.randn(text_len, text_dim)
    video = torch.randn(1, 24, latent_t, lat_h, lat_w)
    audio = torch.randn(1, 32, 2, audio_t)
    sigma_v = 0.7
    layout = R.Layout(text_len, latent_t, lat_h, lat_w, audio_t)
    t_v = 1.0 - sigma_v
    t_a = 1.0 - R.time_shift_sigma(sigma_v, 12.0, 3.0)
    with torch.no_grad():
        want_v, want_a = model([video, audio], torch.tensor([sigma_v * 1000.0]), text[None], transformer_options={}, minimax_payload={})
        from comfy.ldm.minimax.model import patchify_video, pack_audio
        got_v, got_a = ref.forward(text, patchify_video(video), pack_audio(audio), layout, t_v, t_a)
        got_v = R_unpatchify(got_v, latent_t, lat_h // 2, lat_w // 2)
        got_a = got_a.reshape(2, audio_t, 32).permute(2, 0, 1)[None]
    ok = True
    for name, a, b in (("video", want_v.float(), got_v.float()), ("audio", want_a.float(), got_a.float())):
        err = (a - b).abs().max().item(); rel = err / a.abs().max().item()
        print(f"  {'PASS' if rel < 1e-4 else 'FAIL'} {name}: max_abs={err:.3e} rel={rel:.3e} shape={tuple(a.shape)}")
        ok &= rel < 1e-4
    return 0 if ok else 1


def R_unpatchify(rows, t, h, w, c=24):
    x = rows.reshape(t, h, w, c, 1, 2, 2).permute(3, 0, 4, 1, 5, 2, 6).reshape(1, c, t, h * 2, w * 2)
    return x


def _text_in_toy(ref, refiner):
    """The token refiner with a configurable depth (the real model has two blocks)."""
    def text_in(text_states):
        x = ref.plain_linear("condition_proj", text_states.to(ref.dtype))
        for j in range(refiner):
            p = f"token_refiner.blocks.{j}"
            x = x + ref.refiner_attention(p, R.rms_norm(x, ref.t(f"{p}.norm1.weight"), ref.eps))
            h = R.rms_norm(x, ref.t(f"{p}.norm2.weight"), ref.eps)
            x = x + ref.plain_linear(f"{p}.mlp.fc2", ref.swiglu(ref.plain_linear(f"{p}.mlp.fc1", h, bias=False)), bias=False)
        return R.rms_norm(x, ref.t("token_refiner.final_norm.weight"), ref.eps)
    return text_in


if __name__ == "__main__":
    sys.exit(main())

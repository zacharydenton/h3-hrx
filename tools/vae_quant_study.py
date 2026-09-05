"""Does the ViT decoder survive int4? Fake-quantise its transformer-block linears (int4 per-row
weights, rotated by the group-256 Hadamard; activations rotated and int4 per token) inside
diffusers' decoder, decode the first clip of saved latents, and report PSNR against the fp16
decode. Modes: w4a4 (the kernels), w4a16, w8a8.
    python3 tools/vae_quant_study.py build/fox_480p_5s_latents.pt [w4a4 w4a16 w8a8]"""
import sys, time
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reference")); sys.path.insert(0, str(ROOT / "tools"))
import h3_ref as R
from pipeline import MODELS
from diffusers import AutoencoderKLMiniMaxH3

dev = "cuda"
fx = torch.load(sys.argv[1]); modes = sys.argv[2:] or ["w4a4", "w4a16", "w8a8"]
z = fx["video"].to(dev)[:, :, :7]                                   # one decoder clip: 7 latent frames
vae = AutoencoderKLMiniMaxH3.from_pretrained(str(MODELS / "vae"), torch_dtype=torch.float16).to(dev).eval()
vae.disable_tiling()
mean = torch.tensor(vae.config.latents_mean, device=dev).view(1, -1, 1, 1, 1); std = torch.tensor(vae.config.latents_std, device=dev).view(1, -1, 1, 1, 1)
zin = (z * std + mean).float()                                       # diffusers keeps the decoder in fp32
h = R.hadamard(R.HADAMARD_GROUP).to(dev)


class FakeQuantLinear(torch.nn.Module):
    def __init__(self, lin, mode):
        super().__init__()
        w = lin.weight.data.float()
        wr = R.rotate_groups(w, h)
        if mode.startswith("w4"):
            q, s = R.quant_int4_rows(wr); wr = q * s
        elif mode.startswith("w8"):
            q, s = R.quant_int8_rows(wr); wr = q * s
        self.w = wr.float(); self.b = None if lin.bias is None else lin.bias.data.float(); self.mode = mode
    def forward(self, x):
        xr = R.rotate_groups(x.float(), h)
        if self.mode.endswith("a4"):
            q, s = R.quant_int4_rows(xr); xr = q * s
        elif self.mode.endswith("a8"):
            q, s = R.quant_int8_rows(xr); xr = q * s
        y = xr @ self.w.t()
        return y + self.b.to(y.dtype) if self.b is not None else y


def swap(mode):
    n = 0
    for blk in vae.decoder.transformer_blocks:
        for parent, name in ((blk.attn, "to_q"), (blk.attn, "to_k"), (blk.attn, "to_v"), (blk.attn.to_out, "0"), (blk.ff.net[0], "proj"), (blk.ff.net, "2")):
            lin = getattr(parent, name) if not name.isdigit() else parent[int(name)]
            if isinstance(lin, torch.nn.Linear):
                fq = FakeQuantLinear(lin, mode)
                if name.isdigit(): parent[int(name)] = fq
                else: setattr(parent, name, fq)
                n += 1
    return n


with torch.no_grad():
    t0 = time.time(); ref = vae.decoder(vae.post_quant_conv(zin)).float(); torch.cuda.synchronize()
    print(f"fp32 decode of one clip: {time.time() - t0:.1f} s, frames {tuple(ref.shape)}", flush=True)
state = {k: v.clone() for k, v in vae.decoder.state_dict().items()}
for mode in modes:
    vae.decoder.load_state_dict(state)
    n = swap(mode)
    with torch.no_grad():
        t0 = time.time(); out = vae.decoder(vae.post_quant_conv(zin)).float(); torch.cuda.synchronize()
    imstd = torch.tensor((0.229, 0.224, 0.225), device=dev).view(1, 3, 1, 1, 1); immean = torch.tensor((0.485, 0.456, 0.406), device=dev).view(1, 3, 1, 1, 1)
    a = (ref * imstd + immean).clamp(0, 1); b = (out * imstd + immean).clamp(0, 1)
    mse = ((a - b) ** 2).mean().item(); psnr = 10 * torch.log10(torch.tensor(1.0 / max(mse, 1e-12))).item()
    print(f"  {mode}: {n} linears, PSNR vs fp32 {psnr:.2f} dB, max abs {(a - b).abs().max().item():.3f}, {time.time() - t0:.1f} s", flush=True)
    # restore the original modules for the next mode
    vae2 = AutoencoderKLMiniMaxH3.from_pretrained(str(MODELS / "vae"), torch_dtype=torch.float16).to(dev).eval(); vae.decoder = vae2.decoder; del vae2

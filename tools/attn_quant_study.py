"""Can H3's attention take low-precision Q, K (and V)? SageAttention-style study on real activations:
the fox latents (first 7 latent frames) noised to a few sigmas, the real prompt, the reference blocks in
bf16; at chosen layers the attention output with quantised operands against f32 attention.
    python3 tools/attn_quant_study.py [--layers 0,10,25,40,49] [--sigmas 0.9,0.5,0.2]"""
import argparse, math, sys
from pathlib import Path
import torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reference")); sys.path.insert(0, str(ROOT / "tools"))
import h3_ref as R
from encode_prompt import prompt_path
PROMPT = "A red fox trotting through a snowy forest at dawn, cinematic"


def hadamard128(device):
    h4 = R.hadamard(4).double(); h2 = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float64) / math.sqrt(2.0)
    h = torch.kron(torch.kron(torch.kron(h4, h4), h4), h2)            # 128 x 128 orthogonal
    return h.float().to(device)


def quant_rows(x, bits, group=None):
    """per-token (last axis) symmetric quantisation; group: scale per `group` channels instead."""
    qmax = 7 if bits == 4 else 127
    if group:
        xg = x.reshape(*x.shape[:-1], x.shape[-1] // group, group)
        s = xg.abs().amax(-1, keepdim=True).clamp_min(1e-12) / qmax
        return ((xg / s).round().clamp(-qmax, qmax) * s).reshape(x.shape)
    s = x.abs().amax(-1, keepdim=True).clamp_min(1e-12) / qmax
    return (x / s).round().clamp(-qmax, qmax) * s


def attention(q, k, v):
    """[S, H, D] f32 -> [S, H, D]"""
    return F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None])[0].transpose(0, 1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--layers", default="0,10,25,40,49"); ap.add_argument("--sigmas", default="0.9,0.5,0.2"); a = ap.parse_args()
    dev = "cuda"; layers = [int(v) for v in a.layers.split(",")]; sigmas = [float(v) for v in a.sigmas.split(",")]
    fx = torch.load(ROOT / "build/fox_480p_5s_latents.pt"); prompt = torch.load(prompt_path(PROMPT, ROOT / "build/prompts"))
    latents = fx["video"][0, :, :7].float(); T, Hh, W = latents.shape[1:]
    audio = torch.nan_to_num(fx["audio"].float())[:, :, :37]
    layout = R.Layout(prompt["embeds"].shape[0], T, Hh, W, audio.shape[-1])
    ckpt = R.Checkpoint(device=dev, dtype=torch.bfloat16); ref = R.H3Ref(ckpt, quant="none", cache_linears=False)
    H = hadamard128(dev)
    with torch.no_grad():
        text_x = ref.text_in(prompt["embeds"].to(dev)); cos, sin = R.rope_tables(layout.position_ids, ref.inv_freq, dev); rows = layout.adaln_rows.to(dev)
        gen = torch.Generator(dev).manual_seed(0)
        for sigma in sigmas:
            t_v, t_a = 1.0 - sigma, 1.0 - R.time_shift_sigma(sigma, 12.0, 3.0)
            temb = ref.t_emb(torch.tensor([t_v, t_a]))
            vid = latents.to(dev); noise = torch.randn(vid.shape, generator=gen, device=dev)
            xt = (1 - sigma) * vid + sigma * noise
            video_rows = xt[None].reshape(1, 24, T, Hh // 2, 2, W // 2, 2).permute(0, 2, 3, 5, 1, 4, 6).reshape(-1, 96)
            arows = audio.to(dev).permute(0, 2, 1).reshape(-1, 32); anoise = torch.randn(arows.shape, generator=gen, device=dev)
            sig_a = R.time_shift_sigma(sigma, 12.0, 3.0); arows = (1 - sig_a) * arows + sig_a * anoise
            x = torch.cat([text_x, ref.audio_in(arows), ref.video_in(video_rows)], dim=0)
            print(f"sigma {sigma}: {x.shape[0]} rows")
            for i in range(max(layers) + 1):
                mods = ref.block_mods(i, temb)
                if i in layers:
                    p = f"blocks.{i}"; m = mods[rows]; shift_msa, scale_msa = m[:, 0].to(x.dtype), m[:, 1].to(x.dtype)
                    h = R.rms_norm(x, ref.t(f"{p}.norm1.weight"), ref.eps) * (1.0 + scale_msa) + shift_msa
                    q, k, v = ref.qkv_heads(f"{p}.attn", ref.lin(f"{p}.attn.qkv_proj")(h), cos, sin)
                    q, k, v = q.float(), k.float(), v.float()
                    base = attention(q, k, v)
                    kbar = k.mean(0, keepdim=True)                                   # smoothing: softmax-invariant per query row
                    qr, kr, vr = q @ H, k @ H, v @ H                                 # the same rotation on q and k keeps q.k
                    variants = {
                        "K,Q int8 per token": (quant_rows(q, 8), quant_rows(k, 8), v),
                        "K,Q int4 per token": (quant_rows(q, 4), quant_rows(k, 4), v),
                        "K,Q int4 per token, rotated": (quant_rows(qr, 4) @ H.T, quant_rows(kr, 4) @ H.T, v),
                        "K smoothed, K,Q int4 rotated": (quant_rows(qr, 4) @ H.T, quant_rows((k - kbar) @ H, 4) @ H.T, v),
                        "K smoothed, K,Q int4 rotated, 32-ch groups": (quant_rows(qr, 4, 32) @ H.T, quant_rows((k - kbar) @ H, 4, 32) @ H.T, v),
                        "K smoothed, K int4 rotated, Q int8 rotated": (quant_rows(qr, 8) @ H.T, quant_rows((k - kbar) @ H, 4) @ H.T, v),
                        "+ V int8 per token rotated": (quant_rows(qr, 8) @ H.T, quant_rows((k - kbar) @ H, 4) @ H.T, quant_rows(vr, 8) @ H.T),
                        "V int8 per token rotated only": (q, k, quant_rows(vr, 8) @ H.T),
                    }
                    print(f"  layer {i:2d}  |q| {q.abs().max():.1f} |k| {k.abs().max():.1f} |v| {v.abs().max():.1f}")
                    for name, (qq, kk, vv) in variants.items():
                        out = attention(qq, kk, vv)
                        err = ((out - base).norm() / base.norm()).item()
                        per_head = ((out - base).flatten(0, 0).transpose(0, 1).flatten(1).norm(dim=1) / base.transpose(0, 1).flatten(1).norm(dim=1))
                        print(f"    {name:48s} rel err {err:.4f}  worst head {per_head.max().item():.4f}")
                x = ref.block(i, x, mods, rows, cos, sin)
    return 0


if __name__ == "__main__":
    sys.exit(main())

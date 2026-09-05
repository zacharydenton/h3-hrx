"""The Loom decoder blocks against diffusers' fp32 decoder blocks on a real clip's tokens:
cosine of the residual update per depth, and the final frames' PSNR through the shared head.
    python3 tests/test_vae_blocks.py [--curve 1,4,12,36]"""
import argparse, math, sys, time
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from pipeline import MODELS
from decode_loom import decoder_tokens, decoder_head
from h3vae_loom import H3VaeBlocks


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--curve", default="1,4,12,36"); ap.add_argument("--latents", default=str(ROOT / "build/fox_480p_5s_latents.pt")); ap.add_argument("--profile", action="store_true"); ap.add_argument("--weights", default=None); ap.add_argument("--bits", type=int, default=4)
    a = ap.parse_args(); dev = "cuda"
    from diffusers import AutoencoderKLMiniMaxH3
    vae = AutoencoderKLMiniMaxH3.from_pretrained(str(MODELS / "vae"), torch_dtype=torch.float32).to(dev).eval(); vae.disable_tiling()
    fx = torch.load(a.latents); mean = torch.tensor(vae.config.latents_mean, device=dev).view(1, -1, 1, 1, 1); std = torch.tensor(vae.config.latents_std, device=dev).view(1, -1, 1, 1, 1)
    z = (fx["video"].to(dev) * std + mean).float()[:, :, :vae.tokens_chunk_size + vae.token_overlap]
    with torch.no_grad():
        hs, cos, sin, num_patches, fhw = decoder_tokens(vae, z)
        cos48, sin48 = vae.decoder.rope(torch.zeros(1, 1, 3, device=dev))  # unused; the blocks take the full tables below
        pos_cos, pos_sin = torch.cat([cos, cos], dim=-1)[None, :, None, :], torch.cat([sin, sin], dim=-1)[None, :, None, :]
    n = hs.shape[1]; print(f"clip tokens {n} (grid {fhw})")
    ok = True
    for depth in [int(v) for v in a.curve.split(",")]:
        with torch.no_grad():
            ref = hs.clone()
            for blk in vae.decoder.transformer_blocks[:depth]:
                ref = blk(ref, (pos_cos, pos_sin))
        loom = H3VaeBlocks(n, layers=depth, weights=a.weights, bits=a.bits)
        if a.profile: loom.profile(True)
        t0 = time.time(); got = loom.forward(hs[0], cos, sin).to(dev)[None]; dt = time.time() - t0; loom.close()
        upd = lambda t: (t - hs).flatten()
        c = torch.nn.functional.cosine_similarity(upd(got), upd(ref), dim=0).item()
        with torch.no_grad():
            fr = decoder_head(vae, ref, num_patches, fhw); fg = decoder_head(vae, got, num_patches, fhw)
        imstd = torch.tensor((0.229, 0.224, 0.225), device=dev).view(1, 3, 1, 1, 1); immean = torch.tensor((0.485, 0.456, 0.406), device=dev).view(1, 3, 1, 1, 1)
        p, q = (fr * imstd + immean).clamp(0, 1), (fg * imstd + immean).clamp(0, 1)
        psnr = 10 * math.log10(1 / max(((p - q) ** 2).mean().item(), 1e-12))
        print(f"  {depth:2d} blocks {dt * 1e3:8.0f} ms: update cosine vs fp32 {c:.5f}, frame PSNR through the head {psnr:.2f} dB")
        ok &= c > 0.9
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""Independent fp32 VAE accuracy gate with one reference block resident at a time.

Default: a real 256px decoder tile. --full-clip also bounds reference attention memory.
"""
import argparse
import math
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from decode_loom import decoder_tokens, decoder_head
from h3vae_loom import H3VaeBlocks
from reference.vae_decoder import DecoderWeights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", default="1,4,12,36")
    ap.add_argument("--latents", default=str(ROOT / "build/fox_480p_5s_latents.pt"))
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--weights")
    ap.add_argument("--bits", type=int, choices=(4, 8), default=8)
    ap.add_argument("--full-clip", action="store_true")
    args = ap.parse_args()
    depths = sorted(set(int(v) for v in args.curve.split(",")))
    if not depths or depths[0] < 1 or depths[-1] > 36:
        ap.error("depths must be 1..36")
    weights = DecoderWeights(); vae = weights.heads()
    fx = torch.load(args.latents, map_location="cpu")
    z = fx["video"][:, :, :7]
    if not args.full_clip:
        z = z[:, :, :, :16, :16]
    z = z.to("cuda").float()
    mean = z.new_tensor(vae.config.latents_mean)[None, :, None, None, None]
    std = z.new_tensor(vae.config.latents_std)[None, :, None, None, None]
    with torch.no_grad():
        hs, cos, sin, patches, grid = decoder_tokens(vae, z * std + mean)
    print(f"reference tile {grid}, {hs.shape[1]} tokens; one float block at a time", flush=True)
    # Keep snapshots on the CPU, release the float block before creating a native session.
    snapshots = dict(weights.blocks(hs.clone(), cos, sin, depths))
    ok = True
    minimum_cosine, minimum_psnr = ((.999, 40.0) if args.bits == 8 else (.9, 25.0))
    for depth in depths:
        loom = H3VaeBlocks(hs.shape[1], layers=depth, weights=args.weights, bits=args.bits)
        try:
            if args.profile:
                loom.profile(True)
            start = time.time(); got = loom.forward(hs[0], cos, sin).to("cuda")[None]
        finally:
            loom.close()
        ref = snapshots.pop(depth).to("cuda")
        with torch.no_grad():
            cosine = torch.nn.functional.cosine_similarity((got - hs).flatten(), (ref - hs).flatten(), dim=0).item()
            fr, fg = decoder_head(vae, ref, patches, grid), decoder_head(vae, got, patches, grid)
            mean = fr.new_tensor((.485, .456, .406))[None, :, None, None, None]
            std = fr.new_tensor((.229, .224, .225))[None, :, None, None, None]
            p, q = (fr * std + mean).clamp(0, 1), (fg * std + mean).clamp(0, 1)
            mse = ((p - q) ** 2).mean().item()
            psnr = 10 * math.log10(1 / max(mse, 1e-12))
        passed = math.isfinite(cosine) and math.isfinite(psnr) and cosine > minimum_cosine and psnr > minimum_psnr
        print(f"{'PASS' if passed else 'FAIL'} {depth} blocks: update cosine {cosine:.6f}, frame PSNR {psnr:.2f} dB ({time.time() - start:.2f}s)", flush=True)
        ok &= passed
    print(f"required cosine > {minimum_cosine}, PSNR > {minimum_psnr} dB; peak torch allocations {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

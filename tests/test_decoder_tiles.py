"""The C tiled decoder (the checkpoint's f16 blocks) against diffusers' f32 decoder on the same latents.

Run in the torch venv: python tests/test_decoder_tiles.py [--latents saved_video.npy]
"""
import argparse
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from h3_loom import H3


def diffusers_decode(z, frames=None):
    """diffusers' AutoencoderKLMiniMaxH3 decoding the same latents in f32: the independent oracle for the C decoder."""
    import torch
    from reference.vae_decoder import DecoderWeights
    weights = DecoderWeights(); config = weights.config
    vae = weights.full() if hasattr(weights, "full") else weights.decoder()
    with torch.no_grad():
        zt = torch.from_numpy(z)[None].to("cuda")
        mean = zt.new_tensor(config["latents_mean"])[None, :, None, None, None]
        std = zt.new_tensor(config["latents_std"])[None, :, None, None, None]
        video = vae._decode(zt * std + mean)
        vmean = video.new_tensor((.485, .456, .406))[None, :, None, None, None]
        vstd = video.new_tensor((.229, .224, .225))[None, :, None, None, None]
        out = ((video * vstd + vmean).clamp(0, 1) * 255).round().to(torch.uint8)[0].permute(1, 2, 3, 0).cpu().numpy()
    del vae; torch.cuda.empty_cache()
    return out[:frames] if frames else out


def session_isolation():
    """A decoder-only session (no DiT or text checkpoint) decodes, and two sessions on the same file agree."""
    z = np.random.default_rng(3).normal(size=(24, 2, 4, 6)).astype(np.float32)
    results = []
    for _ in range(2):
        pipe = H3(dit="/unused/dit.safetensors", te="/unused/te.safetensors")
        try:
            results.append(pipe.decode_video(pipe.params(height=64, width=96, frames=5), z))
        finally:
            pipe.close()
    np.testing.assert_array_equal(results[0], results[1])
    print("PASS a decoder-only session and five-frame decoding")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", type=Path)
    ap.add_argument("--out", type=Path, help="save corrected RGB frames as .npy")
    args = ap.parse_args()
    # Cross both tile boundaries, including unequal overlaps, and a temporal boundary.
    z = np.load(args.latents) if args.latents else np.random.default_rng(0).normal(size=(24, 7, 20, 24)).astype(np.float32)
    _, t, h, w = z.shape
    frames = (t - 2) // 5 * 17 + 5
    assert t >= 7 and (t - 2) % 5 == 0 and h > 16 and w > 16
    # Missing DiT/text paths must be harmless for a decoder-only session.
    pipe = H3(dit="/unused/dit.safetensors", te="/unused/te.safetensors")
    try:
        got = pipe.decode_video(pipe.params(height=h * 16, width=w * 16, frames=frames), z)
    finally:
        pipe.close()
    want = diffusers_decode(z)
    assert got.shape == want.shape
    mse = np.mean((got.astype(np.float64) - want.astype(np.float64)) ** 2)
    psnr = 10 * math.log10(255 ** 2 / max(mse, 1e-12))
    print(f"C tiled decoder vs diffusers: {psnr:.2f} dB, {got.shape[0]} frames at {w * 16}x{h * 16}")
    if args.out:
        np.save(args.out, got)
    assert psnr > 40, psnr
    session_isolation()


if __name__ == "__main__":
    main()

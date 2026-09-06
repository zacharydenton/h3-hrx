"""C decoder vs diffusers' spatial/temporal stitching with Loom blocks, without loading full models.

Run in the torch venv: python tests/test_decoder_tiles.py [--latents saved_video.npy]
"""
import argparse
import math
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from h3pipe_loom import H3Pipe


def decoder_glue(directory):
    """Copy only decoder tables for tests with different session weights."""
    glue, empty = directory / "glue", directory / "empty"
    glue.mkdir(); empty.mkdir()
    (empty / "weights.bin").write_bytes(b"\0")
    (empty / "manifest.txt").write_text("")
    manifest = []
    source = ROOT / "build/weights_glue"
    with (source / "weights.bin").open("rb") as src, (glue / "weights.bin").open("wb") as dst:
        for line in (source / "manifest.txt").read_text().splitlines():
            if len(line.split()) != 5:
                continue
            name, offset, size, dtype, shape = line.split()
            if name == "te.embed":
                manifest.append("te.embed 0 0 torch.bfloat16 0")
                continue
            if not name.startswith("vae."):
                continue
            manifest.append(f"{name} {dst.tell()} {size} {dtype} {shape}")
            src.seek(int(offset)); remaining = int(size)
            while remaining:
                chunk = src.read(min(remaining, 1 << 20))
                if not chunk:
                    raise EOFError(name)
                dst.write(chunk); remaining -= len(chunk)
    (glue / "manifest.txt").write_text("\n".join(manifest) + "\n")
    return glue, empty


def python_decode(z):
    import torch
    from decode_loom import LoomClipDecoder
    from reference.vae_decoder import DecoderWeights

    weights = DecoderWeights()
    config = weights.config
    vae = weights.heads()
    loom = LoomClipDecoder(vae, weights=ROOT / "build/weights_vae_i8", bits=8)
    vae.decoder.forward = loom.forward
    try:
        with torch.no_grad():
            z = torch.from_numpy(z)[None].to("cuda")
            mean = z.new_tensor(config["latents_mean"])[None, :, None, None, None]
            std = z.new_tensor(config["latents_std"])[None, :, None, None, None]
            video = vae._decode(z * std + mean)
            mean = video.new_tensor((.485, .456, .406))[None, :, None, None, None]
            std = video.new_tensor((.229, .224, .225))[None, :, None, None, None]
            return ((video * std + mean).clamp(0, 1) * 255).round().to(torch.uint8)[0].permute(1, 2, 3, 0).cpu().numpy()
    finally:
        loom.close()


def session_isolation():
    """A second session must use its own post-quant weights, even after a prior decode."""
    import shutil
    z = np.random.default_rng(3).normal(size=(24, 2, 4, 6)).astype(np.float32)
    with tempfile.TemporaryDirectory(prefix="h3-decoder-sessions-") as tmp:
        directory = Path(tmp)
        first, empty = decoder_glue(directory)
        second = directory / "second"
        shutil.copytree(first, second)
        with (second / "weights.bin").open("r+b") as f:
            for line in (second / "manifest.txt").read_text().splitlines():
                name, offset, size, *_ = line.split()
                if name.startswith("vae.post_quant_conv."):
                    f.seek(int(offset)); f.write(bytes(int(size)))
        results = []
        for glue in (first, second, first):
            pipe = H3Pipe(glue=glue, blocks=empty)
            try:
                results.append(pipe.decode_video(pipe.params(height=64, width=96, frames=5), z))
            finally:
                pipe.close()
        assert not np.array_equal(results[0], results[1]), "second session reused the first session's decoder weights"
        np.testing.assert_array_equal(results[0], results[2])
        print("PASS distinct session weights and five-frame decoding")


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
    pipe = H3Pipe(blocks="/unused/dit", te="/unused/text")
    try:
        got = pipe.decode_video(pipe.params(height=h * 16, width=w * 16, frames=frames), z)
    finally:
        pipe.close()
    want = python_decode(z)
    assert got.shape == want.shape
    mse = np.mean((got.astype(np.float64) - want.astype(np.float64)) ** 2)
    psnr = 10 * math.log10(255 ** 2 / max(mse, 1e-12))
    print(f"C tiled decoder vs Python: {psnr:.2f} dB, {got.shape[0]} frames at {w * 16}x{h * 16}")
    if args.out:
        np.save(args.out, got)
    assert psnr > 40, psnr
    session_isolation()


if __name__ == "__main__":
    main()

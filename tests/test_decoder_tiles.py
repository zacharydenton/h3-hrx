"""C decoder vs diffusers' spatial/temporal stitching with Loom blocks, without loading full models.

Run in the torch venv: python tests/test_decoder_tiles.py [--latents saved_video.npy]
"""
import argparse
import json
import math
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from h3pipe_loom import H3Pipe


def decoder_glue(directory):
    """Copy only decoder/constructor tables; never load DiT or text encoder weights."""
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
            if not (name.startswith("vae.") or ".adaln." in name or name in ("h3.adaln_t_table", "h3.rope_inv_freq")):
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
    from safetensors import safe_open
    from diffusers import AutoencoderKLMiniMaxH3
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import MiniMaxH3VideoRotaryPosEmbed
    from decode_loom import LoomClipDecoder

    model = Path.home() / "h3-models/vae"
    config = json.loads((model / "config.json").read_text())
    index = json.loads((model / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    # Keep the real diffusers tiling/chunking methods, but allocate only the small heads.
    with torch.device("meta"):
        vae = AutoencoderKLMiniMaxH3.from_config(config)
    del vae.encoder, vae.quant_conv
    vae.decoder.transformer_blocks = torch.nn.ModuleList()
    state = {}
    for name, shard in index.items():
        if name.startswith("post_quant_conv.") or (name.startswith("decoder.") and not name.startswith("decoder.transformer_blocks.")):
            with safe_open(str(model / shard), framework="pt", device="cpu") as f:
                state[name] = f.get_tensor(name).to("cuda")
    vae.load_state_dict(state, strict=True, assign=True)
    vae.decoder.rope = MiniMaxH3VideoRotaryPosEmbed(48).to("cuda")
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
        for session in loom.sessions.values():
            session.close()


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
    with tempfile.TemporaryDirectory(prefix="h3-decoder-tiles-") as tmp:
        glue, empty = decoder_glue(Path(tmp))
        pipe = H3Pipe(glue=glue, blocks=empty)
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


if __name__ == "__main__":
    main()

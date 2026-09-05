"""GPTQ for the video VAE decoder's block linears (int4 per output row, the plain GEMM's
format), block by block on the tokens of a real clip, each block calibrated on the output of
the already-quantised blocks before it (int4 per-token activations, as the kernels). Writes
build/weights_vae_gptq/ and reports the frame PSNR of the quantised stack through the shared
head against fp32.
    python3 tools/gptq_export_vae.py [--latents build/fox_480p_5s_latents.pt] [--blocksize 128] [--damp 0.05]"""
import argparse, json, math, sys, time
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "reference")); sys.path.insert(0, str(ROOT / "tools"))
import h3_ref as R
from export_weights import pack_i4, interleave_gate_up
from gptq_export import gptq_rows
from pipeline import MODELS
from decode_loom import decoder_tokens, decoder_head


class QuantLinear(torch.nn.Module):
    """The kernels' arithmetic: rotated activations int4 per token, dequantised int4 rows."""
    def __init__(self, w_deq, bias, h):
        super().__init__(); self.w = w_deq; self.b = bias; self.h = h
    def forward(self, x):
        xr = R.rotate_groups(x.float(), self.h); q, s = R.quant_int4_rows(xr); xr = q * s
        y = xr @ self.w.t()
        return y + self.b if self.b is not None else y


class Recorder(torch.nn.Module):
    """Runs the original linear on the rotated input and accumulates X^T X."""
    def __init__(self, lin, h, store, key):
        super().__init__(); self.lin, self.h, self.store, self.key = lin, h, store, key
    def forward(self, x):
        xr = R.rotate_groups(x.float(), self.h)
        x2 = xr.reshape(-1, xr.shape[-1])
        self.store[self.key] = self.store.get(self.key, 0) + x2.t() @ x2
        return torch.nn.functional.linear(xr, R.rotate_groups(self.lin.weight.float(), self.h), self.lin.bias)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--latents", default=str(ROOT / "build/fox_480p_5s_latents.pt"))
    ap.add_argument("--blocksize", type=int, default=128); ap.add_argument("--damp", type=float, default=0.05); ap.add_argument("--out", default=str(ROOT / "build/weights_vae_gptq"))
    a = ap.parse_args(); dev = "cuda"
    from diffusers import AutoencoderKLMiniMaxH3
    vae = AutoencoderKLMiniMaxH3.from_pretrained(str(MODELS / "vae"), torch_dtype=torch.float32).to(dev).eval(); vae.disable_tiling()
    fx = torch.load(a.latents); mean = torch.tensor(vae.config.latents_mean, device=dev).view(1, -1, 1, 1, 1); std = torch.tensor(vae.config.latents_std, device=dev).view(1, -1, 1, 1, 1)
    z = (fx["video"].to(dev) * std + mean).float()[:, :, :vae.tokens_chunk_size + vae.token_overlap]
    h = R.hadamard(R.HADAMARD_GROUP).to(dev)
    with torch.no_grad():
        hs, cos, sin, num_patches, fhw = decoder_tokens(vae, z)
        rot = (torch.cat([cos, cos], -1)[None, :, None, :], torch.cat([sin, sin], -1)[None, :, None, :])
        ref = hs.clone()
        for blk in vae.decoder.transformer_blocks: ref = blk(ref, rot)
        frames_ref = decoder_head(vae, ref, num_patches, fhw)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); blobs = []
    def add(name, t): blobs.append((name, t.detach().contiguous().cpu()))
    x = hs.clone(); t0 = time.time()
    names = (("attn.to_q", "attn.to_k", "attn.to_v"), ("attn.to_out.0",), ("ff.net.0.proj",), ("ff.net.2",))
    tags = ("qkv", "out", "gu", "down")
    for i, blk in enumerate(vae.decoder.transformer_blocks):
        store = {}
        originals = {}
        for group in names:
            for nm in group:
                parent, leaf = blk, nm.split(".")
                for part in leaf[:-1]: parent = getattr(parent, part) if not part.isdigit() else parent[int(part)]
                lin = getattr(parent, leaf[-1]) if not leaf[-1].isdigit() else parent[int(leaf[-1])]
                originals[nm] = (parent, leaf[-1], lin)
                rec = Recorder(lin, h, store, group[0] if nm in group else nm)   # to_q/k/v share one input -> one Hessian
                if leaf[-1].isdigit(): parent[int(leaf[-1])] = rec
                else: setattr(parent, leaf[-1], rec)
        with torch.no_grad():
            blk(x, rot)
        for group, tag in zip(names, tags):
            w = torch.cat([R.rotate_groups(originals[nm][2].weight.float(), h) for nm in group], dim=0)
            bias = torch.cat([originals[nm][2].bias.float() for nm in group], dim=0)
            q, s, wq = gptq_rows(w, store[group[0]], a.blocksize, a.damp)
            packed, scale = pack_i4(q.to(torch.int16)), s.float()
            if tag == "gu":
                packed, scale = interleave_gate_up(packed), interleave_gate_up(scale.view(-1, 1)).view(-1)
                add(f"blocks.{i}.gu.b", interleave_gate_up(bias.view(-1, 1)).view(-1))
            else:
                add(f"blocks.{i}.{tag}.b", bias)
            add(f"blocks.{i}.{tag}.q", packed); add(f"blocks.{i}.{tag}.s", scale)
            # install the quantised linears for the calibration stream
            if len(group) == 3:
                rows = w.shape[0] // 3
                for j, nm in enumerate(group):
                    parent, leaf, lin = originals[nm]
                    ql = QuantLinear(wq[j * rows:(j + 1) * rows], lin.bias.float(), h)
                    if leaf.isdigit(): parent[int(leaf)] = ql
                    else: setattr(parent, leaf, ql)
            else:
                parent, leaf, lin = originals[group[0]]
                ql = QuantLinear(wq, lin.bias.float(), h)
                if leaf.isdigit(): parent[int(leaf)] = ql
                else: setattr(parent, leaf, ql)
        add(f"blocks.{i}.norm1", blk.norm1.weight.float()); add(f"blocks.{i}.norm2", blk.norm2.weight.float())
        add(f"blocks.{i}.scale1", blk.scale1.float()); add(f"blocks.{i}.scale2", blk.scale2.float())
        with torch.no_grad():
            x = blk(x, rot)
        print(f"  block {i}: {time.time() - t0:.0f} s", flush=True)
    with torch.no_grad():
        frames = decoder_head(vae, x, num_patches, fhw)
    imstd = torch.tensor((0.229, 0.224, 0.225), device=dev).view(1, 3, 1, 1, 1); immean = torch.tensor((0.485, 0.456, 0.406), device=dev).view(1, 3, 1, 1, 1)
    p, q = (frames_ref * imstd + immean).clamp(0, 1), (frames * imstd + immean).clamp(0, 1)
    print(f"GPTQ int4 decoder, int4 per-token activations: frame PSNR vs fp32 {10 * math.log10(1 / max(((p - q) ** 2).mean().item(), 1e-12)):.2f} dB (RTN: 28.31)")
    manifest, offset = [], 0
    with open(out / "weights.bin", "wb") as f:
        for name, t in blobs:
            b = t.numpy().tobytes(); manifest.append(f"{name} {offset} {len(b)} {t.dtype} {'x'.join(map(str, t.shape))}"); f.write(b); offset += len(b)
    (out / "manifest.txt").write_text("\n".join(manifest) + "\n")
    (out / "config.json").write_text(json.dumps(dict(layers=36, hidden=2048, heads=32, head_dim=64, ffn=8192, rope_dim=48, group=256, quant="gptq_int4_rows"), indent=1))
    print(f"wrote {out}/weights.bin: {offset / 1e9:.2f} GB, {len(manifest)} tensors, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
    sys.stdout.flush(); sys.stderr.flush()
    import os; os._exit(0)          # the HIP teardown after 36 GPTQ blocks hangs holding all memory; the file is written

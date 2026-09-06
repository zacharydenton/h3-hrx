"""Export the ComfyUI int8 ConvRot checkpoint as the int4 (default) or int8 (--bits 8) operands the Loom runtime loads.

The checkpoint's block linears are already rotated along K (group-256 Hadamard) and
quantised to int8 per output row. Each row is requantised to symmetric int4 (absmax / 7,
nibbles low first) with an f32 scale; nothing is rotated again. Per block:
  qkv  : attn.qkv_proj  [21504][5376]                      (q | k | v, 7168 each)
  out  : attn.out_proj  [5376][7168]
  gu   : mlp.fc1 rows interleaved in 16-row groups [gate o..o+15 | up o..o+15] so the fused
         SwiGLU GEMM epilogue holds both halves of the same outputs  (gate = rows 0..14335)
  down : mlp.fc2        [5376][14336]
plus f32 vectors norm1, norm2, q_norm, k_norm. The AdaLN tables are computed per forward by
the Python wrapper from the 8-d curve (tiny) and are not exported.
Output: build/weights/weights.bin + manifest.txt (name offset bytes dtype shape) + config.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reference"))
import h3_ref as R


def pack_i4(q: torch.Tensor) -> torch.Tensor:
    """[N][K] int codes in [-7, 7] -> [N][K/2] u8, low nibble first."""
    q = q.to(torch.int16) & 0xF
    return (q[:, 0::2] | (q[:, 1::2] << 4)).to(torch.uint8)


def requantize(ckpt: R.Checkpoint, name: str, device) -> tuple[torch.Tensor, torch.Tensor]:
    """int8 rows -> (packed int4 [N][K/2] u8, f32 scale [N]) on the CPU."""
    q8 = ckpt.raw(name + ".weight").to(device)
    s8 = ckpt.raw(name + ".weight_scale").float().view(-1).to(device)
    amax = q8.abs().amax(dim=1).clamp_min(1).float()                 # in int8 code units
    q4 = torch.round(q8.float() * (7.0 / amax[:, None])).clamp(-7, 7)
    s4 = s8 * amax / 7.0                                              # the int4 code's value in the weight's units
    return pack_i4(q4).cpu(), s4.cpu()


def interleave_gate_up(w: torch.Tensor) -> torch.Tensor:
    inter = w.shape[0] // 2
    gate, up = w[:inter], w[inter:]
    return torch.stack([gate.reshape(inter // 16, 16, -1), up.reshape(inter // 16, 16, -1)], dim=1).reshape(w.shape[0], -1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=50)
    ap.add_argument("--out", default=str(ROOT / "build/weights"))
    ap.add_argument("--source", default=str(R.CKPT))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bits", type=int, choices=(4, 8, 16), default=4, help="8: the checkpoint's int8 rows and scales verbatim (tools/quant_study.py: velocity cosine 0.9997 vs 0.98-0.99 for int4)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ckpt = R.Checkpoint(Path(a.source), device="cpu", dtype=torch.float32)
    blobs = []
    def add(name, t): blobs.append((name, t.contiguous().cpu()))
    for i in range(a.layers):
        p = f"blocks.{i}"
        for tag, name in (("qkv", f"{p}.attn.qkv_proj"), ("out", f"{p}.attn.out_proj"), ("gu", f"{p}.mlp.fc1"), ("down", f"{p}.mlp.fc2")):
            if a.bits == 16:   # f16 rows rotated along K: a bf16 (pruned_bf16) checkpoint is rotated here, int8 rows come back dequantised and already rotated
                q, s = ckpt.linear(name).to(a.device).to(torch.float16).cpu(), None
            elif a.bits == 8: q, s = ckpt.raw(name + ".weight").to(torch.int8).cpu(), ckpt.raw(name + ".weight_scale").float().view(-1).cpu()
            else: q, s = requantize(ckpt, name, a.device)
            if tag == "gu":
                q, s = interleave_gate_up(q), (interleave_gate_up(s.view(-1, 1)).view(-1) if s is not None else None)
            add(f"{p}.{tag}.q", q)
            if s is not None: add(f"{p}.{tag}.s", s)
        for vec in ("norm1", "norm2"):
            add(f"{p}.{vec}", ckpt.raw(f"{p}.{vec}.weight").float())
        add(f"{p}.qnorm", ckpt.raw(f"{p}.attn.q_norm.weight").float())
        add(f"{p}.knorm", ckpt.raw(f"{p}.attn.k_norm.weight").float())
        print(f"  block {i}: {time.time() - t0:.0f} s", flush=True)
    manifest, offset = [], 0
    with open(out / "weights.bin", "wb") as f:
        for name, t in blobs:
            b = t.numpy().tobytes()
            manifest.append(f"{name} {offset} {len(b)} {t.dtype} {'x'.join(map(str, t.shape))}")
            f.write(b); offset += len(b)
    (out / "manifest.txt").write_text("\n".join(manifest) + "\n")
    (out / "config.json").write_text(json.dumps(dict(layers=a.layers, hidden=R.HIDDEN, heads=R.HEADS, head_dim=R.HEAD_DIM,
                                                    inner=R.INNER, ffn=R.FFN, group=R.HADAMARD_GROUP, rope_dim=R.ROPE_DIM, bits=a.bits), indent=1))
    print(f"wrote {out}/weights.bin: {offset / 1e9:.2f} GB, {len(manifest)} tensors, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

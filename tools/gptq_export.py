"""GPTQ for the block linears: int4 per output row (the plain GEMM's format) with error
feedback against the calibration activations of the fixture, block by block in sequence
(each block is quantised on the activations produced by the already-quantised blocks before
it). Writes build/weights_gptq/ in the runtime's export format and reports the 50-block
final-layer velocity cosine of the result against the int8 checkpoint.

    python3 tools/gptq_export.py [--blocksize 128] [--damp 0.01] [--out build/weights_gptq]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reference")); sys.path.insert(0, str(ROOT / "tools"))
import h3_ref as R
from export_weights import pack_i4, interleave_gate_up


def gptq_rows(w: torch.Tensor, hessian: torch.Tensor, blocksize: int, damp: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """W [N][K] f32, H [K][K] = X^T X. -> (int4 codes [N][K] in [-7, 7], scale [N], dequantised W)."""
    n, k = w.shape
    w = w.clone()
    h = hessian.clone()
    dead = torch.diag(h) == 0
    h[dead, dead] = 1; w[:, dead] = 0
    # the calibration set has fewer rows than K, so H is rank-deficient: damp until it factors (f64)
    h = h.double(); base = torch.diag(h).mean()
    d = damp
    while True:
        try:
            hd = h + d * base * torch.eye(k, device=h.device, dtype=h.dtype)
            hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(hd)), upper=True).float()
            break
        except Exception:
            d *= 2
            if d > 10: raise
    scale = (w.abs().amax(dim=1, keepdim=True).clamp_min(1e-30) / 7.0)      # per-row scale from the unquantised row
    q = torch.zeros_like(w)
    for i1 in range(0, k, blocksize):
        i2 = min(i1 + blocksize, k)
        w1 = w[:, i1:i2].clone(); err1 = torch.zeros_like(w1); hinv1 = hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = w1[:, i]
            d = hinv1[i, i]
            qc = torch.round(col / scale[:, 0]).clamp(-7, 7)
            q[:, i1 + i] = qc
            e = (col - qc * scale[:, 0]) / d
            w1[:, i:] -= e[:, None] * hinv1[i, i:][None]
            err1[:, i] = e
        w[:, i2:] -= err1 @ hinv[i1:i2, i2:]
    return q, scale[:, 0], q * scale


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocksize", type=int, default=128)
    ap.add_argument("--damp", type=float, default=0.05)
    ap.add_argument("--out", default=str(ROOT / "build/weights_gptq"))
    ap.add_argument("--fixture", default=str(ROOT / "build/fixture.pt"))
    a = ap.parse_args()
    dev = "cuda"
    fx = torch.load(a.fixture)
    ckpt = R.Checkpoint(device=dev, dtype=torch.bfloat16)
    ref = R.H3Ref(ckpt, quant="none", cache_linears=False)
    layout = R.Layout(*fx["layout"])
    temb = fx["temb"].to(dev); rows = fx["rows"].to(dev); cos, sin = fx["cos"].to(dev), fx["sin"].to(dev); tclass = layout.tclass.to(dev)
    x0 = fx["x"].to(dev, torch.bfloat16)
    with torch.no_grad():
        x_ref = ref.blocks_forward(x0.clone(), temb, rows, cos, sin)
        v_ref, a_ref = ref.final(x_ref, temb, tclass, layout)
        vel_ref = torch.cat([v_ref.flatten(), a_ref.flatten()])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    blobs = []
    def add(name, t): blobs.append((name, t.contiguous().cpu()))
    x = x0.clone()
    t0 = time.time()
    names = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
    tags = {"attn.qkv_proj": "qkv", "attn.out_proj": "out", "mlp.fc1": "gu", "mlp.fc2": "down"}
    for i in range(50):
        p = f"blocks.{i}"
        # 1. calibration: run the block unquantised, recording the rotated input of each linear
        hessians = {}
        originals = {}
        for nm in names:
            lin = ref.lin(f"{p}.{nm}")
            originals[nm] = lin
            def make(nm, lin):
                def call(xin):
                    xr = R.rotate_groups(xin.float(), ref.h)
                    hessians[nm] = hessians.get(nm, 0) + xr.t() @ xr
                    return (xr.to(lin.w.dtype) @ lin.w.t()).to(xin.dtype)
                return call
            ref._lin[f"{p}.{nm}"] = type("L", (), {"__call__": staticmethod(make(nm, lin))})()
        with torch.no_grad():
            ref.block(i, x, ref.block_mods(i, temb), rows, cos, sin)
        # 2. quantise each linear with GPTQ against its Hessian; install the quantised weights
        for nm in names:
            w = originals[nm].w.float()
            q, s, wq = gptq_rows(w, hessians[nm], a.blocksize, a.damp)
            packed = pack_i4(q.to(torch.int16)); scale = s.float()
            if nm == "mlp.fc1":
                packed, scale = interleave_gate_up(packed), interleave_gate_up(scale.view(-1, 1)).view(-1)
            add(f"{p}.{tags[nm]}.q", packed); add(f"{p}.{tags[nm]}.s", scale)
            lin = R.QuantLinear.__new__(R.QuantLinear); lin.h, lin.quant, lin.aq, lin.w = ref.h, "gptq", "4", wq.to(torch.bfloat16)
            ref._lin[f"{p}.{nm}"] = lin
        for vec in ("norm1", "norm2"):
            add(f"{p}.{vec}", ckpt.raw(f"{p}.{vec}.weight").float())
        add(f"{p}.qnorm", ckpt.raw(f"{p}.attn.q_norm.weight").float()); add(f"{p}.knorm", ckpt.raw(f"{p}.attn.k_norm.weight").float())
        # 3. advance the calibration stream through the quantised block (int4 per-token activations, as the kernels)
        with torch.no_grad():
            x = ref.block(i, x, ref.block_mods(i, temb), rows, cos, sin)
        for key in [k for k in ref._lin if k.startswith(p + ".")]: del ref._lin[key]
        del hessians, originals
        print(f"  block {i}: {time.time() - t0:.0f} s", flush=True)
    # the quantised stack's velocity: re-run with the exported codes through a w4a4-style reference
    with torch.no_grad():
        v, aa = ref.final(x, temb, tclass, layout)
    vel = torch.cat([v.flatten(), aa.flatten()])
    cs = torch.nn.functional.cosine_similarity(vel, vel_ref, dim=0).item()
    err = ((vel - vel_ref).norm() / vel_ref.norm()).item()
    print(f"GPTQ int4 per row, int4 per-token activations: final velocity cosine {cs:.5f}, rel rms err {err:.4f} (RTN: 0.97860 / 0.2091)")
    manifest, offset = [], 0
    with open(out / "weights.bin", "wb") as f:
        for name, t in blobs:
            b = t.numpy().tobytes()
            manifest.append(f"{name} {offset} {len(b)} {t.dtype} {'x'.join(map(str, t.shape))}")
            f.write(b); offset += len(b)
    (out / "manifest.txt").write_text("\n".join(manifest) + "\n")
    (out / "config.json").write_text(json.dumps(dict(layers=50, hidden=R.HIDDEN, heads=R.HEADS, head_dim=R.HEAD_DIM, inner=R.INNER, ffn=R.FFN, group=R.HADAMARD_GROUP, rope_dim=R.ROPE_DIM, quant="gptq_int4_rows"), indent=1))
    print(f"wrote {out}/weights.bin: {offset / 1e9:.2f} GB, {len(manifest)} tensors, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

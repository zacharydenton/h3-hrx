"""prepare_{norm,plain}_i4 vs the reference's own quantisation (h3_ref): the same row formed
in float, rotated by the group-256 Hadamard, quantised int4 per token (absmax / 7)."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT / "reference"))
from kernel_test import compile_kernel, launch, workdir
import h3_ref as R


def unpack(q: np.ndarray) -> np.ndarray:
    lo = (q & 15).astype(np.int8); hi = (q >> 4).astype(np.int8)
    lo = np.where(lo > 7, lo - 16, lo); hi = np.where(hi > 7, hi - 16, hi)
    return np.stack([lo, hi], axis=-1).reshape(q.shape[0], -1)


def lanes_for(width):
    return next(l for l in (256, 128, 96, 64, 32) if width % (8 * l) == 0 and (width // 4) % l == 0)


def check(name, tmp, tokens, width, x_expected, args, cfg, bits=4):
    ns, sym = f"h3.prepare_{name}_i{bits}", f"h3_prepare_{name}_i{bits}"
    hs = tmp / f"{name}_{width}_{bits}.hsaco"
    lanes = lanes_for(width)
    compile_kernel(ROOT / f"kernels/prepare_{name}_i{bits}.loom", sym, {f"{ns}.width": width, f"{ns}.lanes": lanes, **cfg}, hs)
    (q, s), t = launch(hs, sym, (tokens, 1, 1), (lanes, 1, 1), [("i32", tokens)] + args +
                       [("out", ((tokens, width // (8 // bits)), np.uint8)), ("out", ((tokens,), np.float32))], tmp, repeat=3)
    h = R.hadamard(R.HADAMARD_GROUP)
    xr = R.rotate_groups(torch.from_numpy(x_expected).float(), h)
    want_q, want_s = (R.quant_int4_rows if bits == 4 else R.quant_int8_rows)(xr)
    got = (unpack(q) if bits == 4 else q.view(np.int8)).astype(np.float32); wq = want_q.numpy()
    differ = np.mean(got != wq); by_one = np.mean(np.abs(got - wq) <= 1)
    scale_err = np.max(np.abs(s - want_s.numpy()[:, 0]) / want_s.numpy()[:, 0])
    tol_codes, tol_scale = (5e-3, 2e-3) if name == "plain" else (2e-3, 1e-5)
    ok = differ < tol_codes and by_one == 1.0 and scale_err < tol_scale
    print(f"  {'PASS' if ok else 'FAIL'} prepare_{name}_i{bits}: tokens={tokens} width={width} lanes={lanes}  {t['per_launch_us'] / 1e3:.3f} ms  "
          f"codes differ {differ * 100:.3f}% ({'all by 1, ties' if by_one == 1.0 else 'NOT all by 1'}) scale rel err {scale_err:.1e}")
    return ok


def main() -> int:
    rng = np.random.default_rng(0)
    tokens, width, classes = 300, 5376, 6
    ok = True
    with workdir() as tmp:
        tmp = Path(tmp)
        h = (rng.standard_normal((tokens, width)) * 1.5e5).astype(np.float32)      # the f32 residual stream, at H3's deep-block magnitude
        w = (1.0 + rng.standard_normal(width) * 0.1).astype(np.float32)
        table = (rng.standard_normal((classes * 2, width)) * 0.2).astype(np.float32)     # rows 2c scale, 2c+1 shift
        cls = rng.integers(0, classes, tokens).astype(np.int32)
        hf = h.astype(np.float32)
        normed = hf / np.sqrt((hf * hf).mean(axis=1, keepdims=True) + 1e-5) * w
        x = (1.0 + table[2 * cls]) * normed + table[2 * cls + 1]
        ok &= check("norm", tmp, tokens, width, x.astype(np.float32),
                    [("in", h), ("in", w), ("in", table), ("in_i32", cls)],
                    {"h3.prepare_norm_i4.eps": 1e-5, "h3.prepare_norm_i4.classes": classes})
        for pw in (7168, 14336):
            x = (rng.standard_normal((tokens, pw)) * 0.5).astype(np.float16)
            ok &= check("plain", tmp, tokens, pw, x.astype(np.float32), [("in_f16", x)], {})
        # int8 (the VAE decoder's W8A8 path and the text encoder)
        x = (rng.standard_normal((tokens, 8192)) * 0.5).astype(np.float16)
        ok &= check("plain", tmp, tokens, 8192, x.astype(np.float32), [("in_f16", x)], {}, bits=8)
        h2 = (rng.standard_normal((tokens, 2048)) * 1.5e3).astype(np.float32); w2 = (1.0 + rng.standard_normal(2048) * 0.1).astype(np.float32)
        table2 = np.zeros((2, 2048), np.float32); cls2 = np.zeros(tokens, np.int32)
        h2f = h2 / np.sqrt((h2 * h2).mean(axis=1, keepdims=True) + 1e-5) * w2
        ok &= check("norm", tmp, tokens, 2048, h2f.astype(np.float32), [("in", h2), ("in", w2), ("in", table2), ("in_i32", cls2)], {"h3.prepare_norm_i8.eps": 1e-5, "h3.prepare_norm_i8.classes": 1}, bits=8)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

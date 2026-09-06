"""Small GPU regressions for GroupNorm and experimental 32-key attention (no torch/models)."""
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from kernel_test import compile_kernel, launch, report


def groupnorm(tmp):
    stem = "gn_silu_f16"; ns = "h3." + stem + "."; sym = "h3_" + stem
    cfg = {ns + "channels": 32, ns + "groups": 1, ns + "plane": 2, ns + "rows_bound": 64, ns + "eps": 1e-6}
    hs = tmp / "gn.hsaco"
    compile_kernel(ROOT / "kernels" / (stem + ".loom"), sym, cfg, hs)
    for amplitude in (0.0005, 0.0, 0.5):
        x = np.tile(np.array([-amplitude, amplitude], np.float16), 32).reshape(2, 32)
        xf = x.astype(np.float64)
        stats = np.array([xf.sum(), (xf * xf).sum()], np.float32)
        normalized = (xf - xf.mean()) / np.sqrt(xf.var() + 1e-6)
        expected = normalized / (1 + np.exp(-normalized))
        (out,), _ = launch(hs, sym, (1, 1, 1), (256, 1, 1), [
            ("i32", 1), ("in_f16", x), ("in", stats), ("in", np.ones(32)), ("in", np.zeros(32)),
            ("out_f16", (x.shape, np.float16))], tmp)
        if not report(f"GroupNorm amplitude={amplitude}", out, expected, atol=5e-4, rtol=2e-3):
            return False
    return True


def attention(tmp, stem, waves):
    ns = "h3." + stem + "."; sym = "h3_" + stem
    tokens, capacity, width = 32, 128, 128
    q = np.zeros((capacity, width), np.float16); q[:tokens, 0] = 1
    k = np.zeros_like(q); k[16, 0] = 15   # maximum in upper key tile, even lane
    rng = np.random.default_rng(4)
    v = np.zeros_like(q); v[:tokens] = rng.uniform(-1, 1, (tokens, width))
    scores = q[:tokens].astype(np.float64) @ k[:tokens].astype(np.float64).T
    p = np.exp(scores - scores.max(axis=1, keepdims=True)); p /= p.sum(axis=1, keepdims=True)
    expected = p @ v[:tokens].astype(np.float64)
    cfg = {ns + "q_stride": width, ns + "kv_stride": width, ns + "tokens": tokens,
           ns + "token_capacity": capacity, ns + "scale": 1.0, ns + "out_stride": width}
    hs = tmp / (stem + ".hsaco")
    compile_kernel(ROOT / "experiments" / (stem + ".loom"), sym, cfg, hs)
    (out,), _ = launch(hs, sym, (1, 1, 1), (32 * waves, 1, 1), [
        ("i32", tokens), ("in_f16", q), ("in_f16", k), ("in_f16", v),
        ("out_f16", ((tokens, width), np.float16))], tmp)
    return report(stem + " upper-tile maximum", out, expected, atol=2e-3, rtol=2e-3)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="h3-kernel-regression-") as directory:
        tmp = Path(directory)
        ok = groupnorm(tmp)
        for stem, waves in [("attention_mhat32_lds_f16_wmma", 4), ("attention_mha8t32_lds_f16_wmma", 8),
                            ("attention_mha8h2t32_lds_f16_wmma", 8), ("attention_mha8h0t32_lds_f16_wmma", 8)]:
            ok &= attention(tmp, stem, waves)
        sys.exit(0 if ok else 1)

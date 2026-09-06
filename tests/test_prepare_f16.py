"""prepare_{norm,plain}_f16 (tools/gen_gemm_f16.py): the row formed in float, rotated by the normalised group-256 Hadamard, in f16."""
import sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT / "tests")); sys.path.insert(0, str(ROOT / "reference"))
from kernel_test import compile_kernel, launch, report, workdir
from test_prepare import lanes_for
import h3_ref as R


def check(name, tmp, tokens, width, x_expected, args, cfg):
    ns, sym = f"h3.prepare_{name}_f16", f"h3_prepare_{name}_f16"; hs = tmp / f"{name}_{width}_f16.hsaco"; lanes = lanes_for(width)
    compile_kernel(ROOT / f"kernels/prepare_{name}_f16.loom", sym, {f"{ns}.width": width, f"{ns}.lanes": lanes, **cfg}, hs)
    (o,), t = launch(hs, sym, (tokens, 1, 1), (lanes, 1, 1), [("i32", tokens)] + args + [("out_f16", ((tokens, width), np.float16))], tmp, repeat=3)
    want = (R.rotate_groups(torch.from_numpy(x_expected).double(), R.hadamard(R.HADAMARD_GROUP).double())).numpy()   # R.hadamard carries the 1/16
    return report(f"prepare_{name}_f16 tokens={tokens} width={width}  {t['per_launch_us'] / 1e3:.3f} ms", o.astype(np.float64), want, atol=2e-3, rtol=2e-3)


def main() -> int:
    rng = np.random.default_rng(0); tokens, width, classes = 300, 5376, 6; ok = True
    with workdir() as tmp:
        tmp = Path(tmp)
        h = (rng.standard_normal((tokens, width)) * 1.5e5).astype(np.float32); w = (1.0 + rng.standard_normal(width) * 0.1).astype(np.float32)
        table = (rng.standard_normal((classes * 2, width)) * 0.2).astype(np.float32); cls = rng.integers(0, classes, tokens).astype(np.int32)
        normed = h / np.sqrt((h * h).mean(axis=1, keepdims=True) + 1e-5) * w
        x = (1.0 + table[2 * cls]) * normed + table[2 * cls + 1]
        ok &= check("norm", tmp, tokens, width, x.astype(np.float32), [("in", h), ("in", w), ("in", table), ("in_i32", cls)], {"h3.prepare_norm_f16.eps": 1e-5, "h3.prepare_norm_f16.classes": classes})
        for pw in (7168, 14336):
            x = (rng.standard_normal((tokens, pw)) * 0.5).astype(np.float16)
            ok &= check("plain", tmp, tokens, pw, x.astype(np.float32), [("in_f16", x)], {})
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

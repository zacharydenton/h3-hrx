"""prepare_{norm,lnorm,plain}_{f16,bf16} (tools/gen_prepare.py): the row formed in f32 and written narrowed, unrotated and
unscaled, at the out_stride pitch: the operand of the checkpoint's unrotated f16 / bf16 rows."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT / "tests"))
from kernel_test import compile_kernel, launch, report, workdir, from_bf16
from test_prepare import lanes_for


def check(name, out, tmp, tokens, width, want, args, cfg, out_stride=None):
    out_stride = out_stride or width
    ns, sym = f"h3.prepare_{name}_{out}", f"h3_prepare_{name}_{out}"; hs = tmp / f"{name}_{width}_{out}.hsaco"; lanes = lanes_for(width)
    compile_kernel(ROOT / f"kernels/prepare_{name}_{out}.loom", sym, {f"{ns}.width": width, f"{ns}.lanes": lanes, f"{ns}.out_stride": out_stride, **cfg}, hs)
    (o,), t = launch(hs, sym, (tokens, 1, 1), (lanes, 1, 1), [("i32", tokens)] + args + [(f"out_{out}", ((tokens, out_stride), None))], tmp, repeat=3)
    got = (o.astype(np.float64) if out == "f16" else from_bf16(o).astype(np.float64))[:, :width]
    tol = 2e-3 if out == "f16" else 8e-3   # f16 keeps 11 significant bits, bf16 8 (the values are O(1) after normalisation)
    return report(f"prepare_{name}_{out} tokens={tokens} width={width} stride={out_stride}  {t['per_launch_us'] / 1e3:.3f} ms", got, want, atol=tol, rtol=tol)


def main() -> int:
    rng = np.random.default_rng(0); tokens, classes = 300, 6; ok = True
    with workdir() as tmp:
        tmp = Path(tmp)
        for width in (2048, 5376):   # the VAE decoder and the refiner
            h = (rng.standard_normal((tokens, width)) * 1.5e5).astype(np.float32); w = (1.0 + rng.standard_normal(width) * 0.1).astype(np.float32)
            table = (rng.standard_normal((classes * 2, width)) * 0.2).astype(np.float32); cls = rng.integers(0, classes, tokens).astype(np.int32)
            hd = h.astype(np.float64)
            rms = hd / np.sqrt((hd * hd).mean(axis=1, keepdims=True) + 1e-5) * w
            mu = hd.mean(axis=1, keepdims=True); var = ((hd - mu) ** 2).mean(axis=1, keepdims=True)
            ln = (hd - mu) / np.sqrt(var + 1e-5) * w
            for out in ("f16", "bf16"):
                for name, normed in (("norm", rms), ("lnorm", ln)):
                    x = (1.0 + table[2 * cls]) * normed + table[2 * cls + 1]
                    cfg = {f"h3.prepare_{name}_{out}.eps": 1e-5, f"h3.prepare_{name}_{out}.classes": classes}
                    ok &= check(name, out, tmp, tokens, width, x, [("in", h), ("in", w), ("in", table), ("in_i32", cls)], cfg)
                ok &= check("norm", out, tmp, tokens, width, (1.0 + table[2 * cls]) * rms + table[2 * cls + 1], [("in", h), ("in", w), ("in", table), ("in_i32", cls)],
                            {f"h3.prepare_norm_{out}.eps": 1e-5, f"h3.prepare_norm_{out}.classes": classes}, out_stride=width + 64)   # the padded pitch
                x16 = (rng.standard_normal((tokens, width)) * 0.5).astype(np.float16)
                ok &= check("plain", out, tmp, tokens, width, x16.astype(np.float64), [("in_f16", x16)], {})
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""The f16 block GEMMs (tools/gen_gemm_f16.py) against a float64 reference: plain (+bias), resid (f32 residual += gate[cls] * out)
and swiglu (interleaved gate/up rows), on the 256x128 tile with f16 A [M][K] and W [N][K] rows.
    python3 tests/test_gemm_f16.py [M]"""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT / "tests"))
from kernel_test import compile_kernel, launch, report, workdir
from test_gemm import m_group, interleave, CLASSES


def run(tmp, mode, M, K, N, rng, bias=False):
    stem = {"plain": "gemm_f16_256", "resid": "gemm_f16_resid_256", "swiglu": "gemm_f16_swiglu_256"}[mode] + ("b" if bias else "") + ("_gs" if (bias and mode == "swiglu") else "")
    ns, sym = "h3." + stem, "h3_" + stem
    a = (rng.standard_normal((M, K)) * 0.5).astype(np.float16); w = (rng.standard_normal((N, K)) / np.sqrt(K)).astype(np.float16)
    full = a.astype(np.float64) @ w.astype(np.float64).T
    b_vec = (rng.standard_normal(N) * 0.1).astype(np.float32) if bias else None
    if bias: full = full + b_vec[None]
    g = m_group(M, 256); gy = ((M + 255) // 256 + g - 1) // g * g
    cfg = {f"{ns}.k_size": K, f"{ns}.n_size": N, f"{ns}.m_group": g}
    args = [("i32", M), ("in_f16", a), ("in_f16", w if mode != "swiglu" else interleave(w))]
    if mode == "plain":
        args.append(("out_f16", ((M, N), np.float16))); want = full
    elif mode == "resid":
        cfg[f"{ns}.classes"] = CLASSES
        x = (rng.standard_normal((M, N)) * 1e4).astype(np.float32); gate = (rng.standard_normal((CLASSES, N)) * 0.5).astype(np.float32)
        cls = rng.integers(0, CLASSES, M).astype(np.int32)
        args += [("inout", (x, x.shape)), ("in", gate), ("in_i32", cls)]; want = x.astype(np.float64) + gate[cls] * full
    else:
        args.append(("out_f16", ((M, N // 2), np.float16)))
        first, second = full[:, :N // 2], full[:, N // 2:]
        want = (first / (1 + np.exp(-first)) * second) if not bias else (second / (1 + np.exp(-second)) * first)
    if bias: args.append(("in", b_vec if mode != "swiglu" else interleave(b_vec[:, None])[:, 0]))
    hs = tmp / f"{stem}_{K}_{N}.hsaco"
    compile_kernel(ROOT / "kernels" / f"{stem}.loom", sym, cfg, hs)
    (out,), t = launch(hs, sym, (N // 128, gy, 1), (256, 1, 1), args, tmp, repeat=1 if mode == "resid" else 3)
    tflops = 2.0 * M * K * N / (t["per_launch_us"] * 1e-6) / 1e12
    return report(f"{stem:24s} {M}x{K}x{N} {t['per_launch_us'] / 1e3:8.3f} ms {tflops:5.1f} TFLOP/s", out.astype(np.float64), want, atol=(4e-3 if mode != "resid" else 2e-1), rtol=4e-3)


def main() -> int:
    M = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 2097
    rng = np.random.default_rng(0); ok = True
    shapes = {"plain": (5376, 21504), "resid": (7168, 5376), "swiglu": (5376, 28672)}
    with workdir() as tmp:
        tmp = Path(tmp)
        for mode in ("plain", "resid", "swiglu"):
            ok &= run(tmp, mode, M, *shapes[mode], rng)
            ok &= run(tmp, mode, M, *shapes[mode], rng, bias=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""The vision tower's bf16 WMMA matmuls (tools/gen_matmul_bf16.py) against a float64 reference on the bf16-rounded operands:
bias (f32 A in, f32 out), resid (f16 A in, f16 C += lambda * (A W + b)), gelu (tanh form) and gelu_erf, at the tower's shapes;
and matmul_f32 (the patch embedders and the final layer) past its old 32768-row bound.
    python3 tests/test_matmul_bf16.py"""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT / "tests"))
from kernel_test import compile_kernel, launch, report, workdir, to_bf16, from_bf16
from math import erf


def gelu_tanh(x): return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))
def gelu_erf(x): return 0.5 * x * (1.0 + np.vectorize(erf)(x / np.sqrt(2.0)))


def run(tmp, kind, M, K, N, rng):
    stem = f"matmul_{kind}_bf16_wmma"; ns, sym = "h3." + stem, "h3_" + stem
    w = from_bf16(to_bf16(rng.standard_normal((N, K)) / np.sqrt(K))); b = (rng.standard_normal(N) * 0.1).astype(np.float32)
    a32 = (rng.standard_normal((M, K)) * 0.5).astype(np.float32)
    args = [("i32", M)]
    if kind == "resid":
        a16 = a32.astype(np.float16); a = from_bf16(to_bf16(a16.astype(np.float32)))            # f16 in, narrowed to bf16 at staging
        c = (rng.standard_normal((M, N)) * 0.5).astype(np.float16); lam = (rng.standard_normal(N) * 0.3).astype(np.float32)
        args += [("in_f16", a16), ("in_bf16", w), ("in", b), ("inout_f16", (c, c.shape)), ("in", lam)]
        want = c.astype(np.float64) + lam * (a.astype(np.float64) @ w.astype(np.float64).T + b)
    else:
        a = from_bf16(to_bf16(a32))
        args += [("in", a32), ("in_bf16", w), ("in", b), ("out", ((M, N), np.float32))]
        full = a.astype(np.float64) @ w.astype(np.float64).T + b
        want = {"bias": full, "gelu": gelu_tanh(full), "gelu_erf": gelu_erf(full)}[kind]
    hs = tmp / f"{stem}_{K}_{N}.hsaco"; compile_kernel(ROOT / "h3/kernels" / f"{stem}.loom", sym, {f"{ns}.k_size": K, f"{ns}.n_size": N}, hs)
    (out,), t = launch(hs, sym, (N // 64, (M + 63) // 64, 1), (256, 1, 1), args, tmp, repeat=1 if kind == "resid" else 3)   # resid accumulates into C: once
    tflops = 2.0 * M * K * N / (t["per_launch_us"] * 1e-6) / 1e12
    atol = 2e-2 if kind == "resid" else 2e-3   # the resid output is f16
    return report(f"{stem:26s} {M}x{K}x{N} {t['per_launch_us'] / 1e3:8.3f} ms {tflops:5.1f} TFLOP/s", out.astype(np.float64), want, atol=atol, rtol=4e-3)


def run_f32(tmp, M, K, N, rng):
    stem = "matmul_f32"; ns, sym = "h3." + stem, "h3_" + stem
    x = (rng.standard_normal((M, K)) * 0.5).astype(np.float32); w = (rng.standard_normal((N, K)) / np.sqrt(K)).astype(np.float32); b = (rng.standard_normal(N) * 0.1).astype(np.float32)
    hs = tmp / f"{stem}_{K}_{N}.hsaco"; compile_kernel(ROOT / "h3/kernels" / f"{stem}.loom", sym, {f"{ns}.k": K, f"{ns}.n": N}, hs)
    (out,), t = launch(hs, sym, ((N + 255) // 256, M, 1), (256, 1, 1), [("i32", M), ("in", x), ("in", w), ("in", b), ("out", ((M, N), np.float32))], tmp, repeat=3)
    tflops = 2.0 * M * K * N / (t["per_launch_us"] * 1e-6) / 1e12
    return report(f"{stem:26s} {M}x{K}x{N} {t['per_launch_us'] / 1e3:8.3f} ms {tflops:5.1f} TFLOP/s", out.astype(np.float64), x.astype(np.float64) @ w.astype(np.float64).T + b, atol=1e-4, rtol=1e-4)


def main() -> int:
    rng = np.random.default_rng(0); ok = True
    with workdir() as tmp:
        tmp = Path(tmp)
        for kind, K, N in (("bias", 1152, 3456), ("bias", 1536, 1152), ("resid", 2048, 1152), ("resid", 4352, 1152), ("gelu", 1152, 4352), ("gelu_erf", 4608, 4608), ("bias", 4608, 5120)):
            ok &= run(tmp, kind, 1005, K, N, rng)
        ok &= run_f32(tmp, 40000, 96, 256, rng)     # 768p has 37.7k rows: past the old 32768 bound
        ok &= run_f32(tmp, 1005, 5376, 128, rng)    # the final layer's shape
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

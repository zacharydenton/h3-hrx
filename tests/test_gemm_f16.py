"""The f16 and bf16 block GEMMs (tools/gen_gemm_f16.py) against a float64 reference: plain (+bias), resid (f32 residual += gate[cls] * out)
and swiglu (interleaved gate/up rows), on the 256x128 tile with 16-bit float A [M][K] and W [N][K] rows (f32 accumulation, f16 out).
    python3 tests/test_gemm_f16.py [M]        GEMM_ELEM=bf16 python3 tests/test_gemm_f16.py [M]"""
import os, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT / "tests"))
from kernel_test import compile_kernel, launch, report, workdir, to_bf16, from_bf16
from test_gemm import m_group, interleave, CLASSES
ELEM = os.environ.get("GEMM_ELEM", "f16")


def narrow(x):
    """The operand as the kernel sees it: f16 values, or f32 values rounded to bf16 (the bit patterns travel as uint16)."""
    return x.astype(np.float16) if ELEM == "f16" else from_bf16(to_bf16(x))


def run(tmp, mode, M, K, N, rng, bias=False, kpad=0):
    stem = {"plain": f"gemm_{ELEM}_256", "resid": f"gemm_{ELEM}_resid_256", "swiglu": f"gemm_{ELEM}_swiglu_256"}[mode] + ("b" if bias else "") + ("_gs" if (bias and mode == "swiglu") else "")
    ns, sym = "h3." + stem, "h3_" + stem
    a = narrow(rng.standard_normal((M, K)) * 0.5); w = narrow(rng.standard_normal((N, K)) / np.sqrt(K))
    full = a.astype(np.float64) @ w.astype(np.float64).T
    b_vec = (rng.standard_normal(N) * 0.1).astype(np.float32) if bias else None
    if bias: full = full + b_vec[None]
    g = m_group(M, 256); gy = ((M + 255) // 256 + g - 1) // g * g
    cfg = {f"{ns}.k_size": K, f"{ns}.n_size": N, f"{ns}.m_group": g, f"{ns}.k_stride": K + kpad}   # kpad: extra pitch columns the kernel never reads (gemm_pitch)
    operand = "in_f16" if ELEM == "f16" else "in_bf16"
    pad = lambda x: np.concatenate([x, np.zeros((x.shape[0], kpad), x.dtype)], 1) if kpad else x
    args = [("i32", M), (operand, pad(a)), (operand, pad(w if mode != "swiglu" else interleave(w)))]
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
        ok &= run(tmp, "resid", M, 7168, 5376, rng, kpad=64)                  # the padded pitch the out projection runs at
        ok &= run(tmp, "resid", M, 64, 2048, rng, bias=True)                  # the VAE decoder's x_embedder: K 24 zero-padded to 64
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

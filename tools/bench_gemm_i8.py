"""Timing-only int8 GEMM bench (no reference; random operands): compile a stem for a shape and report TOPS.
    python3 tools/bench_gemm_i8.py STEM MxKxN [MxKxN ...] [--m-group G] [--repeat R]
STEM is looked up in kernels/ then experiments/ (e.g. gemm_i8_256, gemm_i8u_256)."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT / "tests"))
from kernel_test import compile_kernel, launch, workdir
from test_gemm import m_group


def main():
    args = sys.argv[1:]; stem = args[0]; g_override = None; repeat = 3
    if "--m-group" in args: g_override = int(args[args.index("--m-group") + 1])
    if "--repeat" in args: repeat = int(args[args.index("--repeat") + 1])
    shapes = [tuple(int(v) for v in a.split("x")) for a in args[1:] if "x" in a and a[0].isdigit()]
    src = ROOT / "kernels" / f"{stem}.loom"
    if not src.exists(): src = ROOT / "experiments" / f"{stem}.loom"
    ns, sym = "h3." + stem, "h3_" + stem
    rng = np.random.default_rng(0)
    with workdir() as tmp:
        tmp = Path(tmp)
        for (M, K, N) in shapes:
            g = g_override or m_group(M, 256); gy = ((M + 255) // 256 + g - 1) // g * g
            cfg = {f"{ns}.k_size": K, f"{ns}.n_size": N, f"{ns}.m_group": g, f"{ns}.k_stride": K}
            hs = tmp / f"{stem}_{K}_{N}_{g}.hsaco"; compile_kernel(src, sym, cfg, hs)
            a = rng.integers(-127, 128, (M, K)).astype(np.int8).view(np.uint8); w = rng.integers(-127, 128, (N, K)).astype(np.int8).view(np.uint8)
            args_ = [("i32", M), ("in_u8", a), ("in_u8", w), ("in", (rng.random(N, dtype=np.float32) / K)), ("in", rng.random(M, dtype=np.float32)), ("out_f16", ((M, N), np.float16))]
            if "_256b" in stem or stem.endswith("b"): args_.append(("in", np.zeros(N, np.float32)))
            _, t = launch(hs, sym, (N // 128, gy, 1), (256, 1, 1), args_, tmp, repeat=repeat)
            us = t["per_launch_us"]; print(f"  {stem:18s} {M}x{K}x{N} g={g}  {us / 1e3:8.3f} ms  {2.0 * M * K * N / (us * 1e-6) / 1e12:5.1f} TOPS", flush=True)


if __name__ == "__main__":
    main()

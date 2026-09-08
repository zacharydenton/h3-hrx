"""Wide decoder GEMMs: compile-only is CPU-safe; --gpu runs numerical checks.

    python3 tests/test_gemm_f16_wide.py --compile-only
    python3 tests/test_gemm_f16_wide.py --gpu

GPU checks cover ragged M, grouped rasterization tails, K substeps, garbage
operand padding, SwiGLU output guards, and all four decoder projection shapes.
"""
import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from gen_gemm_f16_wide import STEMS, generate
from kernel_test import compile_kernel, workdir, LOOM_COMPILE
from test_gemm_f16 import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--compile-only", action="store_true")
    action.add_argument("--gpu", action="store_true")
    parser.add_argument("--fast", action="store_true", help="check the 128x256 candidate instead")
    args = parser.parse_args()
    stems = STEMS
    generator = generate
    if args.fast:
        from gen_gemm_f16_fast import generate as generator
        stems = {mode: stem.replace("_wide_", "_fast_") for mode, stem in STEMS.items()}
    shapes = [("plain", 2048, 6144), ("swiglu", 2048, 16384),
              ("resid", 8192, 2048), ("resid", 2048, 2048)]
    with workdir() as directory:
        tmp = Path(directory)
        formatter = LOOM_COMPILE.parent.parent / "loom-format/loom-format"
        for mode, stem in stems.items():
            source = tmp / f"{stem}.loom"
            source.write_text(generator(mode))
            subprocess.run([str(formatter), "--in-place", str(source)], check=True, capture_output=True)
            if source.read_bytes() != (ROOT / "h3/kernels" / source.name).read_bytes():
                raise RuntimeError(f"{stem} differs from its generator")
        for mode, k, n in shapes:
            stem = stems[mode]; ns = "h3." + stem
            for group in ((1, 2, 3, 4, 15) if args.fast else (1, 2, 3, 4)):
                cfg = {f"{ns}.k_size": k, f"{ns}.n_size": n,
                       f"{ns}.k_stride": k + 128, f"{ns}.m_group": group}
                if mode == "resid": cfg[f"{ns}.classes"] = 1
                if mode == "swiglu": cfg[f"{ns}.out_stride"] = n // 2 + 128
                compile_kernel(ROOT / "h3/kernels" / f"{stem}.loom", "h3_" + stem, cfg,
                               tmp / f"{mode}_{k}_{n}_{group}.hsaco")
        print(f"PASS generated {'fast' if args.fast else 'wide'} kernels and CPU compilation of {20 if args.fast else 16} decoder configurations", flush=True)
        if args.compile_only:
            return 0
        rng = np.random.default_rng(42); ok = True
        for m in (1, 255, 256, 257, 517, 1797):
            for mode in STEMS:
                ok &= run(tmp, mode, m, 128, 512, rng, bias=True, kpad=128,
                          wide=not args.fast, fast=args.fast, outpad=128 if mode == "swiglu" else 0)
        for mode, k, n in shapes:
            ok &= run(tmp, mode, 517, k, n, rng, bias=True, kpad=128,
                      wide=not args.fast, fast=args.fast, outpad=128 if mode == "swiglu" else 0)
            if args.fast:
                ok &= run(tmp, mode, 1797, k, n, rng, bias=True, kpad=128, fast=True,
                          outpad=128 if mode == "swiglu" else 0, group=1 if k == 8192 else 15)
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

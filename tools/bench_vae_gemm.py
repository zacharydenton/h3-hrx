"""Compare current and wide decoder GEMMs with separately allocated weight copies.

--compile-only does not initialize HIP or allocate operands. --gpu is required
to measure. Run numerical gates first: tests/test_gemm_f16_wide.py --gpu.
The copies contain identical values at distinct device addresses; rotating them
removes the single-matrix cache advantage without changing the GEMM arithmetic.
This is a streaming microbenchmark, not a full-decoder performance claim.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from kernel_test import ROOT, compile_kernel, launch, workdir

SHAPES = {"gu": ("swiglu", 2048, 16384), "down": ("resid", 8192, 2048),
          "qkv": ("plain", 2048, 6144), "out": ("resid", 2048, 2048)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument("--compile-only", action="store_true")
    action.add_argument("--gpu", action="store_true")
    p.add_argument("--stage", choices=[*SHAPES, "all"], default="all")
    p.add_argument("--variants", nargs="+", choices=("base", "wide", "fast"), default=["base", "fast"])
    p.add_argument("--tokens", type=int, default=1797)
    p.add_argument("--copies", type=int, default=36)
    p.add_argument("--pad", type=int, default=128, help="extra operand elements per row (multiple of 64)")
    p.add_argument("--repeat", type=int, default=144)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--m-group", type=int, choices=(1, 2, 3, 4, 5, 8, 15), help="default: host's grouping rule; groups >4 require fast kernels")
    p.add_argument("--output", type=Path)
    opt = p.parse_args()
    if not 1 <= opt.tokens <= 65536 or not 2 <= opt.copies <= 64 or opt.repeat < opt.copies or opt.rounds < 1:
        p.error("require tokens 1..65536, copies 2..64, repeat >= copies, rounds >= 1")
    if opt.pad < 0 or opt.pad % 64 or opt.pad + 8192 > 65536:
        p.error("pad must be a nonnegative multiple of 64 with stride <= 65536")
    if opt.m_group and opt.m_group > 4 and any(v != "fast" for v in opt.variants):
        p.error("row groups greater than 4 require --variants fast")
    stages = list(SHAPES) if opt.stage == "all" else [opt.stage]
    results = []
    with workdir() as directory:
        tmp = Path(directory); compiled = {}
        # Finish compilation before allocating or measuring: an APU shares its
        # power and memory budget with the CPU compiler.
        for stage in stages:
            mode, k, n = SHAPES[stage]
            for variant in opt.variants:
                wide = variant == "wide"
                tm, tn, threads = (128, 256, 256) if variant == "fast" else (256, 256, 512) if wide else (256, 128, 256)
                tiles = (opt.tokens + tm - 1) // tm
                group = opt.m_group or (1 if tiles == 1 else min((4, 3, 2), key=lambda g: ((tiles + g - 1) // g * g, -g)))
                if opt.m_group is None and variant == "fast" and opt.tokens == 1797:
                    group = 1 if k == 8192 else 15
                stem = "gemm_f16" + ("" if variant == "base" else "_" + variant) + ("" if mode == "plain" else "_" + mode) + "_256b" + ("_gs" if mode == "swiglu" else "")
                ns = "h3." + stem
                cfg = {f"{ns}.k_size": k, f"{ns}.n_size": n, f"{ns}.k_stride": k + opt.pad,
                       f"{ns}.m_group": group}
                if mode == "resid": cfg[f"{ns}.classes"] = 1
                if variant != "base" and mode == "swiglu": cfg[f"{ns}.out_stride"] = n // 2 + 128
                hs = tmp / f"{stage}_{variant}.hsaco"
                source = ROOT / "h3/kernels" / f"{stem}.loom"
                if not source.exists(): source = ROOT / "experiments" / f"{stem}.loom"
                compile_kernel(source, "h3_" + stem, cfg, hs)
                compiled[stage, variant] = (stem, hs, tm, tn, threads, group)
        print("Compiled selected decoder GEMMs (CPU only)", flush=True)
        if opt.compile_only: return 0
        rng = np.random.default_rng(42); m = opt.tokens
        for stage in stages:
            mode, k, n = SHAPES[stage]
            a = np.full((m, k + opt.pad), 113, np.float16)
            w = np.full((n, k + opt.pad), 113, np.float16)
            a[:, :k] = rng.standard_normal((m, k), dtype=np.float32) * 0.5
            w[:, :k] = rng.standard_normal((n, k), dtype=np.float32) / np.sqrt(k)
            bias = np.zeros(n, np.float32)
            for round_id in range(opt.rounds):
                # Alternate the order to expose thermal/drift effects.
                for variant in (opt.variants if round_id % 2 == 0 else list(reversed(opt.variants))):
                    wide = variant == "wide"
                    stem, hs, tm, tn, threads, group = compiled[stage, variant]
                    gy = ((m + tm - 1) // tm + group - 1) // group * group
                    args = [("i32", m), ("in_f16", a), ("in_f16", w)]
                    if mode == "resid":
                        x = np.zeros((m, n), np.float32)
                        args += [("inout", (x, x.shape)), ("in", np.ones((1, n), np.float32)),
                                 ("in_i32", np.zeros(m, np.int32))]
                    else:
                        width = n if mode == "plain" else n // 2 + (128 if variant != "base" else 0)
                        args.append(("out_f16", ((m, width), np.float16)))
                    args.append(("in", bias))
                    _, timing = launch(hs, "h3_" + stem, (n // tn, gy, 1),
                                       (threads, 1, 1), args, tmp,
                                       repeat=opt.repeat, rotate_input=(2, opt.copies))
                    record = dict(stage=stage, variant=variant, wide=wide, round=round_id, tokens=m, k=k, n=n,
                                  m_group=group, tile_m=tm, tile_n=tn, pad=opt.pad, weight_bytes=w.nbytes * opt.copies, **timing)
                    results.append(record); print(json.dumps(record), flush=True)
                    if opt.output:
                        opt.output.parent.mkdir(parents=True, exist_ok=True)
                        opt.output.write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

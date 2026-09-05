"""Interleaved best-of-N A/B of int4-QK attention kernels (kernels/ or experiments/ stems) at one row count.
    python3 tools/ab_attention_i4.py <tokens> <stem> [stem ...] [--rounds N]"""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from kernel_test import compile_kernel, launch, workdir
HEADS, D = 56, 128


def main():
    argv = sys.argv[1:]; rounds = 4
    if "--rounds" in argv: i = argv.index("--rounds"); rounds = int(argv[i + 1]); argv = argv[:i] + argv[i + 2:]
    tokens, stems = int(argv[0]), argv[1:]
    rng = np.random.default_rng(0)
    with workdir() as tmp:
        tmp = Path(tmp); built = {}
        for stem in stems:
            waves = 16 if "mha16" in stem else (8 if "mha8" in stem else 4)
            cap = max((tokens + 16 + 31) // 32 * 32, (tokens + 16 * waves - 1) // (16 * waves) * (16 * waves))
            src = ROOT / "kernels" / f"{stem}.loom"; src = src if src.exists() else ROOT / "experiments" / f"{stem}.loom"
            ns, sym = "h3." + stem, "h3_" + stem; hs = tmp / f"{stem}.hsaco"
            compile_kernel(src, sym, {f"{ns}.q_stride": HEADS * D, f"{ns}.kv_stride": HEADS * D, f"{ns}.tokens": tokens, f"{ns}.token_capacity": cap, f"{ns}.scale": 1.0, f"{ns}.out_stride": HEADS * D, **({f"{ns}.skip_tau": float(__import__("os").environ.get("ATTN_SKIP_TAU", "1e30"))} if "skip_tau" in src.read_text() else {})}, hs)
            qi = rng.integers(-2**31, 2**31, size=(cap, HEADS * 16), dtype=np.int32); ki = rng.integers(-2**31, 2**31, size=(cap, HEADS * 16), dtype=np.int32)
            qs = (rng.standard_normal((cap, HEADS)) * 1e-3).astype(np.float32); ks = np.abs(rng.standard_normal((cap, HEADS))).astype(np.float32) * 0.05
            vT = (rng.standard_normal((HEADS * D, cap)) * 0.5).astype(np.float16)
            built[stem] = (hs, sym, waves, [("i32", tokens), ("i32", HEADS), ("in_i32", qi), ("in", qs), ("in_i32", ki), ("in", ks), ("in_f16", vT), ("out_f16", ((tokens, HEADS * D), np.float16))])
        best = {s: 1e9 for s in stems}
        for r in range(rounds):
            for stem in (stems if r % 2 == 0 else stems[::-1]):
                hs, sym, waves, a = built[stem]
                _, t = launch(hs, sym, ((tokens + 16 * waves - 1) // (16 * waves), HEADS, 1), (32 * waves, 1, 1), a, tmp, repeat=1)
                best[stem] = min(best[stem], t["per_launch_us"])
        flops = 4.0 * tokens * tokens * D * HEADS
        for stem in stems: print(f"  {stem:40s} {best[stem] / 1e3:8.1f} ms  {flops / (best[stem] * 1e-6) / 1e12:5.1f} TFLOP/s-eq  {best[stems[0]] / best[stem]:.3f}x vs {stems[0]}")


if __name__ == "__main__":
    main()

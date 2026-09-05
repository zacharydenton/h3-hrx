"""Interleaved A/B of the down-projection input path at the Krea 2 shape, best of N rounds:
  A: gemm_i4 (N = 32768, gate|up) + prepare_swiglu   (the unfused pair, experiments/)
  B: gemm_i4_swiglu (silu(g)*u in the epilogue) + prepare_plain
Both produce the int4 codes and token scales the down GEMM consumes; B's must match A's.

    python3 tools/ab_gu.py [tokens] [rounds]"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernel_test import compile_kernel, launch, workdir, ROOT

K, INTER = 6144, 16384
N = 2 * INTER


def m_group(tokens):
    tiles = (tokens + 127) // 128
    return min((4, 3, 2), key=lambda g: ((tiles + g - 1) // g * g, -g))


def interleave(w):
    """[gate | up] rows -> 16-row groups [gate o..o+15 | up o..o+15] (as export_weights.py)."""
    gate, up = w[:INTER], w[INTER:]
    return np.stack([gate.reshape(INTER // 16, 16, -1), up.reshape(INTER // 16, 16, -1)], axis=1).reshape(N, -1)


def main() -> int:
    tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 4115
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (tokens, K // 2), dtype=np.uint8)
    w = rng.integers(0, 256, (N, K // 2), dtype=np.uint8)
    w_scale = (rng.random(N, dtype=np.float32) * 0.5 + 0.5) / K * 8
    a_scale = (rng.random(tokens, dtype=np.float32) * 0.5 + 0.5) / 7
    w_b = np.ascontiguousarray(interleave(w)); w_scale_b = np.ascontiguousarray(interleave(w_scale[:, None])[:, 0])
    g = m_group(tokens); gy = ((tokens + 127) // 128 + g - 1) // g * g
    with workdir() as tmp:
        tmp = Path(tmp)
        hs = {}
        for stem, sub, cfg in (("gemm_i4", "kernels", {"k_size": K, "n_size": N, "m_group": g}),
                               ("gemm_i4_swiglu", "kernels", {"k_size": K, "n_size": N, "m_group": g}),
                               ("prepare_swiglu_i4", "experiments", {"width": INTER, "gate_stride": N}),
                               ("prepare_plain_i4", "kernels", {"width": INTER})):
            out = tmp / f"{stem}.hsaco"
            compile_kernel(ROOT / sub / f"{stem}.loom", f"krea2_{stem}", {f"krea2.{stem}.{k}": v for k, v in cfg.items()}, out)
            hs[stem] = out
        best = {"A": 1e9, "B": 1e9}; parts = {"A": (1e9, 1e9), "B": (1e9, 1e9)}; results = {}
        for r in range(rounds):
            for which in (("A", "B") if r % 2 == 0 else ("B", "A")):
                if which == "A":
                    (gu,), t1 = launch(hs["gemm_i4"], "krea2_gemm_i4", (N // 128, gy, 1), (256, 1, 1),
                                       [("i32", tokens), ("in_u8", a), ("in_u8", w), ("in", w_scale), ("in", a_scale), ("out_f16", ((tokens, N), np.float16))], tmp, repeat=3)
                    (q, s), t2 = launch(hs["prepare_swiglu_i4"], "krea2_prepare_swiglu_i4", (tokens, 1, 1), (256, 1, 1),
                                        [("i32", tokens), ("in_f16", gu), ("out", ((tokens, INTER // 2), np.uint8)), ("out", ((tokens,), np.float32))], tmp, repeat=3)
                else:
                    (gu,), t1 = launch(hs["gemm_i4_swiglu"], "krea2_gemm_i4_swiglu", (N // 128, gy, 1), (256, 1, 1),
                                       [("i32", tokens), ("in_u8", a), ("in_u8", w_b), ("in", w_scale_b), ("in", a_scale), ("out_f16", ((tokens, INTER), np.float16))], tmp, repeat=3)
                    (q, s), t2 = launch(hs["prepare_plain_i4"], "krea2_prepare_plain_i4", (tokens, 1, 1), (256, 1, 1),
                                        [("i32", tokens), ("in_f16", gu), ("out", ((tokens, INTER // 2), np.uint8)), ("out", ((tokens,), np.float32))], tmp, repeat=3)
                total = t1["per_launch_us"] + t2["per_launch_us"]
                if total < best[which]:
                    best[which] = total; parts[which] = (t1["per_launch_us"], t2["per_launch_us"])
                results[which] = (q, s)
        qa, sa = results["A"]; qb, sb = results["B"]
        lo = lambda q: (q & 15).astype(np.int8); hi = lambda q: (q >> 4).astype(np.int8)
        differ = np.mean((lo(qa) != lo(qb)) | (hi(qa) != hi(qb)))
        scale_err = np.max(np.abs(sa - sb) / np.maximum(np.abs(sa), 1e-12))
        print(f"tokens {tokens}, best of {rounds} interleaved rounds x 3 (gemm + prepare):")
        for which, label in (("A", "gemm_i4 + prepare_swiglu"), ("B", "gemm_i4_swiglu + prepare_plain")):
            print(f"  {label:<34s} {best[which] / 1e3:8.2f} ms  (gemm {parts[which][0] / 1e3:.2f}, prepare {parts[which][1] / 1e3:.2f})")
        print(f"  speedup {best['A'] / best['B']:.3f}x; codes differ {differ * 100:.3f}%, scale rel err {scale_err:.1e}")
    return 0 if differ < 0.01 and scale_err < 1e-2 else 1


if __name__ == "__main__":
    sys.exit(main())

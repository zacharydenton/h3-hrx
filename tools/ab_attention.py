"""Interleaved A/B of two attention kernels at the Krea 2 shape, best of N rounds.

    python3 tools/ab_attention.py attention_gqa_f16_wmma attention_gqa32_f16_wmma [tokens] [rounds]"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernel_test import compile_kernel, launch, report, workdir, ROOT

HEADS, KV, D = 48, 12, 128


def main() -> int:
    a_stem, b_stem = sys.argv[1], sys.argv[2]
    tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 4115
    rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    torch.manual_seed(0)
    q = (torch.randn(tokens, HEADS, D) * 0.5).half(); k = (torch.randn(tokens, KV, D) * 0.5).half(); v = (torch.randn(tokens, KV, D) * 0.5).half()
    qf, kf, vf = (t.float().cuda() for t in (q, k.repeat_interleave(4, dim=1), v.repeat_interleave(4, dim=1)))
    want = torch.nn.functional.scaled_dot_product_attention(qf.transpose(0, 1)[None], kf.transpose(0, 1)[None], vf.transpose(0, 1)[None])[0].transpose(0, 1).reshape(tokens, HEADS * D).cpu().numpy()
    capacity = max((tokens + 16 + 31) // 32 * 32, (tokens + 63) // 64 * 64)
    def pad(t):
        out = np.zeros((capacity, t.shape[1] * D), np.float16); out[:tokens] = t.reshape(tokens, -1).numpy(); return out
    vt = np.ascontiguousarray(pad(v).T)
    ok = True
    with workdir() as tmp:
        tmp = Path(tmp); built = {}
        for stem in (a_stem, b_stem):
            ns, sym = "krea2." + stem, "krea2_" + stem
            hs = tmp / f"{stem}.hsaco"
            cfg = {f"{ns}.q_stride": HEADS * D, f"{ns}.kv_stride": KV * D, f"{ns}.tokens": tokens, f"{ns}.token_capacity": capacity, f"{ns}.scale": 1.0 / math.sqrt(D), f"{ns}.out_stride": HEADS * D}
            if "lds" not in stem:
                cfg[f"{ns}.kv_groups"] = 4
            src = ROOT / "kernels" / f"{stem}.loom"
            compile_kernel(src if src.exists() else ROOT / "experiments" / f"{stem}.loom", sym, cfg, hs)
            built[stem] = (hs, sym)
        best = {a_stem: 1e9, b_stem: 1e9}
        for r in range(rounds):
            for stem in ((a_stem, b_stem) if r % 2 == 0 else (b_stem, a_stem)):
                hs, sym = built[stem]
                if "lds" in stem:     # four query heads per workgroup, V in its natural layout
                    (out,), t = launch(hs, sym, ((tokens + 15) // 16, KV, 1), (128, 1, 1),
                                       [("i32", tokens), ("i32", KV), ("in_f16", pad(q)), ("in_f16", pad(k)), ("in_f16", pad(v)), ("out_f16", ((tokens, HEADS * D), np.float16))], tmp, repeat=3)
                else:
                    (out,), t = launch(hs, sym, ((tokens + 15) // 16, HEADS, 1), (32, 1, 1),
                                       [("i32", tokens), ("i32", HEADS), ("in_f16", pad(q)), ("in_f16", pad(k)), ("in_f16", vt), ("out_f16", ((tokens, HEADS * D), np.float16))], tmp, repeat=3)
                if r == 0:
                    ok &= report(f"  {stem} correctness", out, want, atol=2e-2, rtol=2e-2)
                best[stem] = min(best[stem], t["per_launch_us"])
        flops = 4.0 * tokens * tokens * D * HEADS
        print(f"tokens {tokens}, best of {rounds} interleaved rounds x 3:")
        for stem in (a_stem, b_stem):
            print(f"  {stem:<28s} {best[stem] / 1e3:8.2f} ms  {flops / (best[stem] * 1e-6) / 1e12:5.1f} TFLOP/s")
        print(f"  speedup {best[a_stem] / best[b_stem]:.3f}x")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

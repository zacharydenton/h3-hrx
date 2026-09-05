"""attention_mha_lds_f16_wmma vs torch SDPA: 56 heads of 128, contiguous q/k/v [tokens][7168],
one workgroup of four query tiles per head."""
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from kernel_test import compile_kernel, launch, report, workdir

HEADS, D = 56, 128
STEM = "attention_mha_lds_f16_wmma"
NS, SYM = "h3." + STEM, "h3_" + STEM


def capacity_for(tokens: int) -> int:
    return max((tokens + 16 + 31) // 32 * 32, (tokens + 63) // 64 * 64)


def run(tmp: Path, tokens: int, heads=HEADS) -> bool:
    torch.manual_seed(0)
    q = (torch.randn(tokens, heads, D) * 0.5).half(); k = (torch.randn(tokens, heads, D) * 0.5).half(); v = (torch.randn(tokens, heads, D) * 0.5).half()
    qf, kf, vf = (t.float().cuda() for t in (q, k, v))
    want = torch.nn.functional.scaled_dot_product_attention(qf.transpose(0, 1)[None], kf.transpose(0, 1)[None], vf.transpose(0, 1)[None])[0].transpose(0, 1).reshape(tokens, heads * D).cpu().numpy()
    capacity = capacity_for(tokens)
    def pad(t):
        out = np.zeros((capacity, heads * D), np.float16); out[:tokens] = t.reshape(tokens, -1).numpy(); return out
    hs = tmp / f"{STEM}_{heads}.hsaco"
    cfg = {f"{NS}.q_stride": heads * D, f"{NS}.kv_stride": heads * D, f"{NS}.tokens": tokens, f"{NS}.token_capacity": capacity,
           f"{NS}.scale": 1.0 / math.sqrt(D), f"{NS}.out_stride": heads * D}
    compile_kernel(ROOT / "kernels" / f"{STEM}.loom", SYM, cfg, hs)
    (out,), t = launch(hs, SYM, ((tokens + 63) // 64, heads, 1), (128, 1, 1),
                       [("i32", tokens), ("i32", heads), ("in_f16", pad(q)), ("in_f16", pad(k)), ("in_f16", pad(v)),
                        ("out_f16", ((tokens, heads * D), np.float16))], tmp, repeat=3)
    us = t["per_launch_us"]; flops = 4.0 * tokens * tokens * D * heads
    return report(f"{STEM} tokens={tokens} heads={heads}  {us / 1e3:8.3f} ms  {flops / (us * 1e-6) / 1e12:5.1f} TFLOP/s", out, want, atol=2e-2, rtol=2e-2)


def main() -> int:
    ok = True
    with workdir() as tmp:
        tmp = Path(tmp)
        for tokens in (100, 1000, 5504):
            ok &= run(tmp, tokens)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

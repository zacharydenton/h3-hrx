"""attention_mha_lds_f16_wmma vs torch SDPA: 56 heads of 128, contiguous q/k/v [tokens][7168],
one workgroup of four query tiles per head."""
import math
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from kernel_test import compile_kernel, launch, report, workdir

import os
D = int(os.environ.get("ATTN_D", "128"))
HEADS = 32 if D == 64 else 56
WAVES = int(os.environ.get("ATTN_WAVES", "4"))
GQA = int(os.environ.get("ATTN_GQA", "1"))          # 8: text-encoder mode, 8 query heads per kv head, causal
if GQA == 8:
    WAVES, HEADS = 8, 64
STEM = os.environ.get("ATTN", "attention_gqa8c_lds_f16_wmma" if GQA == 8 else {(128, 4): "attention_mha_lds_f16_wmma", (128, 8): "attention_mha8_lds_f16_wmma", (64, 4): "attention_mha64_lds_f16_wmma", (64, 8): "attention_mha648_lds_f16_wmma"}[(D, WAVES)])
NS, SYM = "h3." + STEM, "h3_" + STEM


def capacity_for(tokens: int) -> int:
    query_block = 16 * WAVES
    return max((tokens + 16 + 31) // 32 * 32, (tokens + query_block - 1) // query_block * query_block)


def run(tmp: Path, tokens: int, heads=HEADS) -> bool:
    torch.manual_seed(0)
    kv_heads = heads // GQA
    q = (torch.randn(tokens, heads, D) * 0.5).half(); k = (torch.randn(tokens, kv_heads, D) * 0.5).half(); v = (torch.randn(tokens, kv_heads, D) * 0.5).half()
    qf, kf, vf = (t.float().cuda() for t in (q, k, v))
    want = torch.nn.functional.scaled_dot_product_attention(qf.transpose(0, 1)[None], kf.transpose(0, 1)[None], vf.transpose(0, 1)[None], is_causal=GQA == 8, enable_gqa=True)[0].transpose(0, 1).reshape(tokens, heads * D).cpu().numpy()
    capacity = capacity_for(tokens)
    def pad(t):
        out = np.zeros((capacity, t.shape[1] * D), np.float16)
        out[:tokens] = t.reshape(tokens, -1).numpy()
        if STEM == "attention_mha64hm32_lds_f16_wmma":
            out = out.reshape(capacity, t.shape[1], D).transpose(1, 0, 2).copy()
        return out
    hs = tmp / f"{STEM}_{heads}.hsaco"
    cfg = {f"{NS}.q_stride": heads * D, f"{NS}.kv_stride": kv_heads * D, f"{NS}.tokens": tokens, f"{NS}.token_capacity": capacity,
           f"{NS}.scale": 1.0 / math.sqrt(D), f"{NS}.out_stride": heads * D}
    source = ROOT / "kernels" / f"{STEM}.loom"
    if not source.exists(): source = ROOT / "experiments" / f"{STEM}.loom"
    compile_kernel(source, SYM, cfg, hs)
    query_block = 16 * WAVES
    grid = ((tokens + 15) // 16, kv_heads, 1) if GQA == 8 else ((tokens + query_block - 1) // query_block, heads, 1)
    (out,), t = launch(hs, SYM, grid, (32 * WAVES, 1, 1),
                       [("i32", tokens), ("i32", kv_heads), ("in_f16", pad(q)), ("in_f16", pad(k)), ("in_f16", pad(v)),
                        ("out_f16", ((tokens, heads * D), np.float16))], tmp, repeat=3)
    us = t["per_launch_us"]; flops = 4.0 * tokens * tokens * D * heads / (2 if GQA == 8 else 1)
    return report(f"{STEM} tokens={tokens} heads={heads}  {us / 1e3:8.3f} ms  {flops / (us * 1e-6) / 1e12:5.1f} TFLOP/s", out, want, atol=2e-2, rtol=2e-2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+")
    args = parser.parse_args()
    if args.tokens and any(n < 1 or n > 65536 for n in args.tokens): parser.error("tokens must be 1..65536")
    ok = True
    with workdir() as tmp:
        tmp = Path(tmp)
        for tokens in (args.tokens or ((13, 28, 100, 512, 1000) if GQA == 8 else (13, 100, 1000, 5504))):
            ok &= run(tmp, tokens)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

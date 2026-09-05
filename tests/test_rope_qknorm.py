"""rope_qknorm_f16 vs the reference's rms_norm + apply_rope on a fused [tokens][21504] buffer
(q at 0, k at 7168, v at 14336), writing contiguous q/k/v [tokens][7168]."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT / "reference"))
from kernel_test import compile_kernel, launch, report, workdir
import h3_ref as R

import os
D = int(os.environ.get("ROPE_D", "128"))
R_DIM = int(os.environ.get("ROPE_R", "96" if D == 128 else "48"))
STEM = {(128, 96): "rope_qknorm_f16", (64, 48): "rope64_qknorm_f16", (128, 128): "rope128_qknorm_f16"}[(D, R_DIM)]
NS, SYM = "h3." + STEM, "h3_" + STEM


def main() -> int:
    torch.manual_seed(0)
    tokens, heads, d = 200, (56 if D == 128 else 32), D
    kv_heads = int(os.environ.get("ROPE_KV", heads))       # 8 for the text encoder's GQA layout
    rope_dim = R_DIM
    stride, k_off = (heads + 2 * kv_heads) * d, heads * d
    fused = (torch.randn(tokens, stride) * 0.7).half()
    qw = (1.0 + torch.randn(d) * 0.1).float(); kw = (1.0 + torch.randn(d) * 0.1).float()
    layout = R.Layout(8, 3, 8, 8, 4)                     # 8 text + 8 audio + 48 video rows = 64; the rest repeats
    pos = layout.position_ids.repeat((tokens + layout.seq_len - 1) // layout.seq_len, 1)[:tokens]
    inv_freq = 10000.0 ** (-torch.arange(0, 32, 2, dtype=torch.float32) / 32)
    cos, sin = R.rope_tables(pos, inv_freq, "cpu")                 # [T, 48]
    if rope_dim == 48:                            # the decoder rotates 48 channels: 8 frequencies per axis
        cos, sin = cos[:, :24].contiguous(), sin[:, :24].contiguous()
    elif rope_dim == 128:                         # the text encoder rotates all 128: 64 angles
        ang = torch.arange(tokens, dtype=torch.float32)[:, None] * (10000.0 ** (-torch.arange(0, 128, 2, dtype=torch.float32) / 128))[None]
        cos, sin = torch.cos(ang).contiguous(), torch.sin(ang).contiguous()
    q = fused[:, :heads * d].view(tokens, heads, d)
    k = fused[:, k_off:k_off + kv_heads * d].view(tokens, kv_heads, d)
    want_q = R.apply_rope(R.rms_norm(q, qw), cos, sin, rope_dim).reshape(tokens, -1)
    want_k = R.apply_rope(R.rms_norm(k, kw), cos, sin, rope_dim).reshape(tokens, -1)
    with workdir() as tmp:
        tmp = Path(tmp); hs = tmp / "rope.hsaco"
        compile_kernel(ROOT / "kernels" / f"{STEM}.loom", SYM,
                       {f"{NS}.row_stride": stride, f"{NS}.heads": heads, f"{NS}.kv_heads": kv_heads, f"{NS}.k_offset": k_off, f"{NS}.eps": 1e-5}, hs)
        (qo, ko, vo), t = launch(hs, SYM, (tokens, 1, 1), (256, 1, 1),
                                 [("i32", tokens), ("in_f16", fused.numpy()), ("in", qw.numpy()), ("in", kw.numpy()),
                                  ("in", np.ascontiguousarray(cos.numpy())), ("in", np.ascontiguousarray(sin.numpy())),
                                  ("out_f16", ((tokens, heads * d), np.float16)), ("out_f16", ((tokens, kv_heads * d), np.float16)), ("out_f16", ((tokens, kv_heads * d), np.float16))], tmp, repeat=1)
        ok = report(f"rope_qknorm q tokens={tokens}  {t['per_launch_us'] / 1e3:.3f} ms", qo, want_q.float().numpy(), atol=2e-2, rtol=2e-2)
        ok &= report("rope_qknorm k", ko, want_k.float().numpy(), atol=2e-2, rtol=2e-2)
        ok &= np.array_equal(vo, fused.numpy()[:, k_off + kv_heads * d:k_off + 2 * kv_heads * d])
        print("  PASS v copied" if ok else "  FAIL v copy")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

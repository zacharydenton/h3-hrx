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

NS, SYM = "h3.rope_qknorm_f16", "h3_rope_qknorm_f16"


def main() -> int:
    torch.manual_seed(0)
    tokens, heads, d = 200, 56, 128
    stride, k_off = 3 * heads * d, heads * d
    fused = (torch.randn(tokens, stride) * 0.7).half()
    qw = (1.0 + torch.randn(d) * 0.1).float(); kw = (1.0 + torch.randn(d) * 0.1).float()
    layout = R.Layout(8, 3, 8, 8, 4)                     # 8 text + 8 audio + 48 video rows = 64; the rest repeats
    pos = layout.position_ids.repeat((tokens + layout.seq_len - 1) // layout.seq_len, 1)[:tokens]
    inv_freq = 10000.0 ** (-torch.arange(0, 32, 2, dtype=torch.float32) / 32)
    cos, sin = R.rope_tables(pos, inv_freq, "cpu")
    q = fused[:, :heads * d].view(tokens, heads, d)
    k = fused[:, k_off:k_off + heads * d].view(tokens, heads, d)
    want_q = R.apply_rope(R.rms_norm(q, qw), cos, sin).reshape(tokens, -1)
    want_k = R.apply_rope(R.rms_norm(k, kw), cos, sin).reshape(tokens, -1)
    with workdir() as tmp:
        tmp = Path(tmp); hs = tmp / "rope.hsaco"
        compile_kernel(ROOT / "kernels/rope_qknorm_f16.loom", SYM,
                       {f"{NS}.row_stride": stride, f"{NS}.heads": heads, f"{NS}.k_offset": k_off, f"{NS}.eps": 1e-5}, hs)
        (qo, ko, vo), t = launch(hs, SYM, (tokens, 1, 1), (256, 1, 1),
                                 [("i32", tokens), ("in_f16", fused.numpy()), ("in", qw.numpy()), ("in", kw.numpy()),
                                  ("in", np.ascontiguousarray(cos.numpy())), ("in", np.ascontiguousarray(sin.numpy())),
                                  ("out_f16", ((tokens, heads * d), np.float16)), ("out_f16", ((tokens, heads * d), np.float16)), ("out_f16", ((tokens, heads * d), np.float16))], tmp, repeat=1)
        ok = report(f"rope_qknorm q tokens={tokens}  {t['per_launch_us'] / 1e3:.3f} ms", qo, want_q.float().numpy(), atol=2e-2, rtol=2e-2)
        ok &= report("rope_qknorm k", ko, want_k.float().numpy(), atol=2e-2, rtol=2e-2)
        ok &= np.array_equal(vo, fused.numpy()[:, 2 * k_off:3 * k_off])
        print("  PASS v copied" if ok else "  FAIL v copy")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

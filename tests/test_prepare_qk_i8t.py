"""prepare_qk_i8t: the same codes as prepare_qk_i8, scales written transposed and parity-split ([heads][capacity]).
    python3 tests/test_prepare_qk_i8t.py"""
import math, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT / "tests"))
from kernel_test import compile_kernel, launch, report, workdir
from test_prepare_qk_i8 import replica, HEADS, D
from test_attention_i8t import parity_split


def main():
    torch.manual_seed(0); T = 300; cap = 320
    x = (torch.randn(T, HEADS * D) * 0.7).half(); mean = (torch.randn(HEADS * D) * 0.1).float()
    extra = 1.0 / math.sqrt(D) / 128.0
    want_codes, want_scales, _ = replica(x.float(), mean, extra)
    want = parity_split(np.concatenate([want_scales.numpy(), np.zeros((cap - T, HEADS), np.float32)]))
    with workdir() as tmp:
        tmp = Path(tmp); hs = tmp / "pqkt.hsaco"; ns, sym = "h3.prepare_qk_i8t", "h3_prepare_qk_i8t"
        compile_kernel(ROOT / "h3/kernels/prepare_qk_i8t.loom", sym, {f"{ns}.row_stride": HEADS * D, f"{ns}.head_offset": 0, f"{ns}.heads": HEADS, f"{ns}.extra_scale": extra, f"{ns}.token_capacity": cap}, hs)
        (codes, scales), t = launch(hs, sym, (T, 1, 1), (256, 1, 1),
                                    [("i32", T), ("in_f16", x.numpy()), ("in", mean.numpy()), ("out", ((T, HEADS * 32), np.int32)), ("out", ((HEADS, cap), np.float32))], tmp, repeat=1)
    differ = (codes != want_codes.numpy()).mean()
    print(f"  {'PASS' if differ < 2e-3 else 'FAIL'} prepare_qk_i8t codes: {differ * 100:.4f}% words differ (rounding ties)  {t['per_launch_us'] / 1e3:.3f} ms")
    ok = report("prepare_qk_i8t scales (transposed, parity-split)", scales, want, atol=1e-7, rtol=1e-4) and differ < 2e-3
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

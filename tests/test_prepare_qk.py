"""prepare_qk_i4 against a torch replica: per (token, head) mean-subtract, Sylvester H_128 (unnormalised),
int4 per (token, head), packed codes and scales; then the dequantised q.k against the exact q.k.
    python3 tests/test_prepare_qk.py"""
import math, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from kernel_test import compile_kernel, launch, report, workdir

HEADS, D = 56, 128
H2 = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float64)
H128 = H2
for _ in range(6): H128 = torch.kron(H128, H2)          # Sylvester order: bit 0 innermost, unnormalised


def replica(x, mean, extra):
    """x [T][HEADS*D] f32, mean [HEADS*D] -> codes [T][HEADS*16] i32, scales [T][HEADS], dequantised rotated [T][HEADS][D]."""
    T = x.shape[0]
    y = (x.double() - mean.double()).reshape(T, HEADS, D) @ H128            # index bit 0 innermost: element e = lane*4 + j
    amax = y.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    q = (y / (amax / 7)).round().clamp(-7, 7)
    codes = (q.long() & 15).reshape(T, HEADS, 16, 8)
    words = torch.zeros(T, HEADS, 16, dtype=torch.int64)
    for j in range(8): words |= codes[..., j] << (4 * j)
    words = torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32).reshape(T, HEADS * 16)
    return words, (amax[..., 0] / 7 * extra).float(), q * (amax / 7)


def main():
    torch.manual_seed(0); T = 300
    x = (torch.randn(T, HEADS * D) * 0.7).half(); mean = (torch.randn(HEADS * D) * 0.1).float()
    extra = 1.0 / math.sqrt(D) / 128.0
    want_codes, want_scales, deq = replica(x.float(), mean, extra)
    with workdir() as tmp:
        tmp = Path(tmp); hs = tmp / "pqk.hsaco"; ns, sym = "h3.prepare_qk_i4", "h3_prepare_qk_i4"
        compile_kernel(ROOT / "h3/kernels/prepare_qk_i4.loom", sym, {f"{ns}.row_stride": HEADS * D, f"{ns}.head_offset": 0, f"{ns}.heads": HEADS, f"{ns}.extra_scale": extra}, hs)
        (codes, scales), t = launch(hs, sym, (T, 1, 1), (256, 1, 1),
                                    [("i32", T), ("in_f16", x.numpy()), ("in", mean.numpy()), ("out", ((T, HEADS * 16), np.int32)), ("out", ((T, HEADS), np.float32))], tmp, repeat=1)
    same = np.array_equal(codes, want_codes.numpy()); differ = (codes != want_codes.numpy()).mean()
    print(f"  {'PASS' if differ < 2e-3 else 'FAIL'} prepare_qk_i4 codes: {differ * 100:.4f}% words differ (rounding ties)  {t['per_launch_us'] / 1e3:.3f} ms")
    ok = report("prepare_qk_i4 scales", scales, want_scales.numpy(), atol=1e-7, rtol=1e-4) and differ < 2e-3
    # the operand pairing: (q_i4 . k_i4) * s_q * s_k against the exact q . k, on two independent draws
    k = (torch.randn(T, HEADS * D) * 0.7).half(); kc, ks, kdeq = replica(k.float(), torch.zeros(HEADS * D), 1.0)
    exact = torch.einsum("thd,shd->hts", x.float().reshape(T, HEADS, D).double(), k.float().reshape(T, HEADS, D).double())
    approx = torch.einsum("thd,shd->hts", deq, kdeq) / 128.0
    err = ((approx - exact).norm() / exact.norm()).item(); print(f"  int4 (rotated) q.k vs exact: rel err {err:.4f} (random gaussian operands)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

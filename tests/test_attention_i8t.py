"""attention_i8qkt_mha8 (the transposed formulation: S^T = K.Q^T, in-lane softmax, P^T repacked to an rhs fragment
through v_permlanex16, O^T += V^T.P^T) against the torch replica of the same quantised operands and exact f16 attention.
The K scales arrive transposed and parity-split per 16-key block: [heads][capacity], index blk*16 + (t&1)*8 + ((t&15)>>1).
    python3 tests/test_attention_i8t.py [tokens ...] [--time-only]"""
import math, os, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT / "tests"))
from kernel_test import compile_kernel, launch, report, workdir
from test_prepare_qk_i8 import replica, HEADS, D
WAVES = 8; STEM = os.environ.get("ATTN_STEM", "attention_i8qkt_mha8_lds_f16_wmma"); NS, SYM = "h3." + STEM, "h3_" + STEM
KERNEL = ROOT / "kernels" / f"{STEM}.loom"
if not KERNEL.exists(): KERNEL = ROOT / "experiments" / f"{STEM}.loom"
TIME_ONLY = "--time-only" in sys.argv


def capacity_for(tokens):
    return max((tokens + 16 + 31) // 32 * 32, (tokens + 16 * WAVES - 1) // (16 * WAVES) * (16 * WAVES))


def parity_split(ks):
    """[capacity][heads] -> [heads][capacity] with each 16-key block reordered [even keys | odd keys]."""
    cap, heads = ks.shape; t = np.arange(cap)
    idx = (t // 16) * 16 + (t & 1) * 8 + ((t & 15) >> 1)
    out = np.zeros((heads, cap), ks.dtype); out[:, idx] = ks.T
    return np.ascontiguousarray(out)


def run(tmp, tokens, heads=HEADS):
    torch.manual_seed(0)
    q = (torch.randn(tokens, heads, D) * 0.5).half(); k = (torch.randn(tokens, heads, D) * 0.5).half(); v = (torch.randn(tokens, heads, D) * 0.5).half()
    kmean = k.float().reshape(tokens, heads * D).mean(0)
    extra = 1.0 / math.sqrt(D) / 128.0
    qc, qs, qdeq = replica(q.float().reshape(tokens, -1), torch.zeros(heads * D), extra)
    kc, ks, kdeq = replica(k.float().reshape(tokens, -1), kmean, 1.0)
    qcodes = (qdeq / (qdeq.abs().amax(-1, keepdim=True).clamp_min(1e-30)) * 127).float(); kcodes = (kdeq / (kdeq.abs().amax(-1, keepdim=True).clamp_min(1e-30)) * 127).float()
    want_rep = np.zeros((tokens, heads * D), np.float32)
    for h in (range(heads) if not TIME_ONLY else ()):
        s = (qcodes[:, h].cuda() @ kcodes[:, h].cuda().T) * qs[:, h].cuda()[:, None] * ks[:, h].cuda()[None, :]
        want_rep[:, h * D:(h + 1) * D] = (torch.softmax(s, -1) @ v[:, h].float().cuda()).cpu().numpy(); del s
    if not TIME_ONLY:
        qf, kf, vf = (t.float().cuda() for t in (q, k, v))
        want_exact = torch.nn.functional.scaled_dot_product_attention(qf.transpose(0, 1)[None], kf.transpose(0, 1)[None], vf.transpose(0, 1)[None])[0].transpose(0, 1).reshape(tokens, heads * D).cpu().numpy()
    cap = capacity_for(tokens)
    def pad(t, w, dt): out = np.zeros((cap, w), dt); out[:tokens] = t; return out
    vT = np.ascontiguousarray(pad(v.reshape(tokens, -1).numpy(), heads * D, np.float16).T)
    kst = parity_split(pad(ks.numpy(), heads, np.float32))
    hs = tmp / f"{STEM}.hsaco"
    compile_kernel(KERNEL, SYM, {f"{NS}.q_stride": heads * D, f"{NS}.kv_stride": heads * D, f"{NS}.tokens": tokens, f"{NS}.token_capacity": cap, f"{NS}.scale": 1.0, f"{NS}.out_stride": heads * D}, hs)
    args = [("i32", tokens), ("i32", heads), ("in_i32", pad(qc.numpy(), heads * 32, np.int32)), ("in", pad(qs.numpy(), heads, np.float32)),
            ("in_i32", pad(kc.numpy(), heads * 32, np.int32)), ("in", kst), ("in_f16", vT), ("out_f16", ((tokens, heads * D), np.float16))]
    grid, block = ((tokens + 16 * WAVES - 1) // (16 * WAVES), heads, 1), (32 * WAVES, 1, 1)
    (out,), t = launch(hs, SYM, grid, block, args, tmp, repeat=3)
    (out2,), _ = launch(hs, SYM, grid, block, args, tmp, repeat=1)
    print(f"    deterministic across launches: {np.array_equal(out, out2)} (max |diff| {np.abs(out.astype(np.float32) - out2.astype(np.float32)).max():.3g})")
    us = t["per_launch_us"]; flops = 4.0 * tokens * tokens * D * heads
    if TIME_ONLY:
        print(f"  {STEM} tokens={tokens}  {us / 1e3:8.3f} ms  {flops / (us * 1e-6) / 1e12:5.1f} TFLOP/s (timing only)"); return True
    ok = report(f"{STEM} vs replica tokens={tokens}  {us / 1e3:8.3f} ms  {flops / (us * 1e-6) / 1e12:5.1f} TFLOP/s", out, want_rep, atol=2e-2, rtol=2e-2)
    err = np.linalg.norm(out.astype(np.float32) - want_exact) / np.linalg.norm(want_exact); print(f"    vs exact f16 attention: rel err {err:.4f}")
    return ok


def main():
    ok = True
    with workdir() as tmp:
        tmp = Path(tmp)
        for tokens in [int(v) for v in sys.argv[1:] if v.isdigit()] or (100, 1000, 5504):
            ok &= run(tmp, tokens)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

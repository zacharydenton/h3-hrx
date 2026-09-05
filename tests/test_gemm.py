"""The int4 GEMM family against a float64 reference: plain (f16 out with the token scale),
resid (f32 residual stream += gate[cls] * out) and swiglu (interleaved gate/up rows, f16
silu(g)*u out), on the 128x128 and the 256x128 tiles.

    python3 tests/test_gemm.py [M] [tile ...]
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from kernel_test import compile_kernel, launch, report, workdir

CLASSES = 12


def m_group(tokens, tile_m):
    tiles = (tokens + tile_m - 1) // tile_m
    return min((4, 3, 2), key=lambda g: ((tiles + g - 1) // g * g, -g))


def pack_i4(q):
    q = q.astype(np.int16) & 0xF
    return (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)


def interleave(w):
    inter = w.shape[0] // 2
    return np.stack([w[:inter].reshape(inter // 16, 16, -1), w[inter:].reshape(inter // 16, 16, -1)], axis=1).reshape(w.shape[0], -1)


def run(tmp, mode, tile_m, M, K, N, rng, bits=4, bias=False):
    stem = {"plain": "gemm_i4", "resid": "gemm_i4_resid", "swiglu": "gemm_i4_swiglu"}[mode] + ("_256" if tile_m == 256 else "")
    stem = stem.replace("i4", f"i{bits}") + ("b" if bias else "") + ("_gs" if (bias and mode == "swiglu") else "")
    ns, sym = "h3." + stem, "h3_" + stem
    lim = 7 if bits == 4 else 127
    a = rng.integers(-lim, lim + 1, (M, K)).astype(np.int8); w = rng.integers(-lim, lim + 1, (N, K)).astype(np.int8)
    w_scale = (rng.random(N, dtype=np.float32) * 0.5 + 0.5) / K * 8 * (7.0 / lim)
    a_scale = (rng.random(M, dtype=np.float32) * 0.5 + 0.5) / lim
    full = (a.astype(np.float64) @ w.astype(np.float64).T) * w_scale[None] * a_scale[:, None]
    b_vec = (rng.standard_normal(N) * 0.1).astype(np.float32) if bias else None
    if bias: full = full + b_vec[None]
    g = m_group(M, tile_m); gy = ((M + tile_m - 1) // tile_m + g - 1) // g * g
    cfg = {f"{ns}.k_size": K, f"{ns}.n_size": N, f"{ns}.m_group": g}
    pack = pack_i4 if bits == 4 else (lambda q: q.view(np.uint8))
    args = [("i32", M), ("in_u8", pack(a)), ("in_u8", pack(w if mode != "swiglu" else interleave(w))), ("in", w_scale if mode != "swiglu" else interleave(w_scale[:, None])[:, 0]), ("in", a_scale)]
    if mode == "plain":
        args.append(("out_f16", ((M, N), np.float16))); want = full
    elif mode == "resid":
        cfg[f"{ns}.classes"] = CLASSES
        x = (rng.standard_normal((M, N)) * 1e4).astype(np.float32); gate = (rng.standard_normal((CLASSES, N)) * 0.5).astype(np.float32)
        cls = rng.integers(0, CLASSES, M).astype(np.int32)
        args += [("inout", (x, x.shape)), ("in", gate), ("in_i32", cls)]; want = x.astype(np.float64) + gate[cls] * full
    else:
        args.append(("out_f16", ((M, N // 2), np.float16)))
        first, second = full[:, :N // 2], full[:, N // 2:]
        want = (first / (1 + np.exp(-first)) * second) if not bias else (second / (1 + np.exp(-second)) * first)   # _gs: silu on the second half
    if bias:
        args.append(("in", b_vec if mode != "swiglu" else interleave(b_vec[:, None])[:, 0]))
    hs = tmp / f"{stem}_{K}_{N}.hsaco"
    compile_kernel(ROOT / "kernels" / f"{stem}.loom", sym, cfg, hs)
    (out,), t = launch(hs, sym, (N // 128, gy, 1), (256, 1, 1), args, tmp, repeat=1 if mode == "resid" else 3)   # in place: once
    tops = 2.0 * M * K * N / (t["per_launch_us"] * 1e-6) / 1e12
    return report(f"{stem:20s} {M}x{K}x{N} {t['per_launch_us'] / 1e3:8.3f} ms {tops:5.1f} TOPS", out.astype(np.float64), want, atol=(2e-3 if mode != "resid" else 2e-1), rtol=2e-3)


def main() -> int:
    M = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 2097
    tiles = [int(v) for v in sys.argv[2:] if v in ("128", "256")] or [128, 256]
    modes = [v for v in sys.argv[2:] if v in ("plain", "resid", "swiglu")] or ["plain", "resid", "swiglu"]
    family = next((v for v in sys.argv[2:] if v in ("i4", "i4b", "i8b")), "i4")     # i4: the H3 blocks; i4b / i8b: the VAE decoder's (biases, diffusers' SwiGLU order)
    rng = np.random.default_rng(0)
    ok = True
    with workdir() as tmp:
        tmp = Path(tmp)
        shapes = {"plain": (5376, 21504), "resid": (7168, 5376), "swiglu": (5376, 28672)} if family == "i4" else {"plain": (2048, 6144), "resid": (8192, 2048), "swiglu": (2048, 16384)}
        for tile_m in (tiles if family == "i4" else [256]):
            for mode in modes:
                ok &= run(tmp, mode, tile_m, M, *shapes[mode], rng, bits=(8 if family == "i8b" else 4), bias=(family != "i4"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

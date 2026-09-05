"""Compile every kernel for one packed-sequence length into build/kernels/T<tokens>/.

    python3 scripts/build_kernels.py <tokens>
Shapes are MiniMax H3's: hidden 5376, 56 heads of 128 (q, k, v each 7168, fused 21504),
SwiGLU 14336 (fused gate|up 28672), AdaLN classes = (timestep, modality) rows.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from kernel_test import compile_kernel

HIDDEN, HEADS, D, FFN = 5376, 56, 128, 14336
INNER = HEADS * D
CLASSES = 12                 # up to four timestep classes x three modalities


GEMM_TILE = int(os.environ.get("H3_GEMM_TILE", "256"))    # m rows per GEMM workgroup tile: 128 (32x64 waves) or 256 (64x64 waves)


def gemm_m_group(tokens):
    """m-tiles per raster group: of 4, 3, 2 the one that pads the tile rows least (ties to the larger).
    The host (host/h3.cpp gemm_m_group) applies the same rule, reading the tile from gemm_tile.txt."""
    if os.environ.get("H3_M_GROUP"):
        return int(os.environ["H3_M_GROUP"])
    tiles = (tokens + GEMM_TILE - 1) // GEMM_TILE
    return min((4, 3, 2), key=lambda g: ((tiles + g - 1) // g * g, -g))


ATTN_QK = os.environ.get("H3_ATTN_QK", "f16")        # "i4": QK^T in int4 WMMA on prepare_qk_i4 operands (SageAttention-style), PV f16


def build(tokens: int) -> Path:
    out = ROOT / ("build/kernels" if ATTN_QK == "f16" else "build/kernels_i4qk") / f"T{tokens}"
    out.mkdir(parents=True, exist_ok=True)
    m_group = gemm_m_group(tokens)
    waves = int(os.environ.get("H3_ATTN_WAVES", "8" if tokens >= 4096 else "4"))   # query tiles per attention workgroup
    capacity = max((tokens + 16 + 31) // 32 * 32, (tokens + 16 * waves - 1) // (16 * waves) * (16 * waves))   # tokens+16 headroom, whole query blocks
    attn_stem = ("attention_mha8_lds_f16_wmma" if waves == 8 else "attention_mha_lds_f16_wmma") if ATTN_QK == "f16" else ("attention_i4qk_mha8_lds_f16_wmma" if waves == 8 else "attention_i4qk_mha_lds_f16_wmma")
    attn = "h3." + attn_stem
    sfx = "_256" if GEMM_TILE == 256 else ""
    g4 = lambda stem: stem + sfx
    specs = [
        ("prepare_norm_i4", "h3_prepare_norm_i4", "prepare_norm_i4", {"h3.prepare_norm_i4.width": HIDDEN, "h3.prepare_norm_i4.lanes": 96, "h3.prepare_norm_i4.eps": 1e-5, "h3.prepare_norm_i4.classes": CLASSES}),
        ("prepare_plain_i4", "h3_prepare_plain_i4", "prepare_attn_i4", {"h3.prepare_plain_i4.width": INNER, "h3.prepare_plain_i4.lanes": 128}),
        ("prepare_plain_i4", "h3_prepare_plain_i4", "prepare_down_i4", {"h3.prepare_plain_i4.width": FFN, "h3.prepare_plain_i4.lanes": 256}),
        (g4("gemm_i4"), "h3_" + g4("gemm_i4"), "gemm_qkv", {"h3." + g4("gemm_i4") + ".k_size": HIDDEN, "h3." + g4("gemm_i4") + ".n_size": 3 * INNER, "h3." + g4("gemm_i4") + ".m_group": m_group}),
        (g4("gemm_i4_swiglu"), "h3_" + g4("gemm_i4_swiglu"), "gemm_gu", {"h3." + g4("gemm_i4_swiglu") + ".k_size": HIDDEN, "h3." + g4("gemm_i4_swiglu") + ".n_size": 2 * FFN, "h3." + g4("gemm_i4_swiglu") + ".m_group": m_group}),
        (g4("gemm_i4_resid"), "h3_" + g4("gemm_i4_resid"), "gemm_out", {"h3." + g4("gemm_i4_resid") + ".k_size": INNER, "h3." + g4("gemm_i4_resid") + ".n_size": HIDDEN, "h3." + g4("gemm_i4_resid") + ".m_group": m_group, "h3." + g4("gemm_i4_resid") + ".classes": CLASSES}),
        (g4("gemm_i4_resid"), "h3_" + g4("gemm_i4_resid"), "gemm_down", {"h3." + g4("gemm_i4_resid") + ".k_size": FFN, "h3." + g4("gemm_i4_resid") + ".n_size": HIDDEN, "h3." + g4("gemm_i4_resid") + ".m_group": m_group, "h3." + g4("gemm_i4_resid") + ".classes": CLASSES}),
        ("rope_qknorm_f16", "h3_rope_qknorm_f16", "rope_qknorm", {"h3.rope_qknorm_f16.row_stride": 3 * INNER, "h3.rope_qknorm_f16.heads": HEADS, "h3.rope_qknorm_f16.kv_heads": HEADS, "h3.rope_qknorm_f16.k_offset": INNER, "h3.rope_qknorm_f16.eps": 1e-5}),
        (attn_stem, "h3_" + attn_stem, "attention", {f"{attn}.q_stride": INNER, f"{attn}.kv_stride": INNER, f"{attn}.tokens": tokens, f"{attn}.token_capacity": capacity, f"{attn}.scale": D ** -0.5, f"{attn}.out_stride": INNER}),
    ]
    if ATTN_QK == "i4":
        pq = "h3.prepare_qk_i4."
        specs += [("colmean_f32", "h3_colmean_f32", "colmean", {"h3.colmean_f32.width": INNER}),
                  ("prepare_qk_i4", "h3_prepare_qk_i4", "prepare_q_i4", {pq + "row_stride": INNER, pq + "head_offset": 0, pq + "heads": HEADS, pq + "extra_scale": D ** -0.5 / 128.0}),
                  ("prepare_qk_i4", "h3_prepare_qk_i4", "prepare_k_i4", {pq + "row_stride": INNER, pq + "head_offset": 0, pq + "heads": HEADS, pq + "extra_scale": 1.0})]
    for stem, sym, name, cfg in specs:
        hs = out / f"{name}.hsaco"
        if not hs.exists():
            compile_kernel(ROOT / "kernels" / f"{stem}.loom", sym, cfg, hs)
    (out / "attention_qk.txt").write_text(f"{ATTN_QK}\n")
    (out / "capacity.txt").write_text(f"{capacity}\n")
    (out / "gemm_tile.txt").write_text(f"{GEMM_TILE}\n")
    (out / "attention_waves.txt").write_text(f"{waves}\n")
    return out


if __name__ == "__main__":
    print(build(int(sys.argv[1]) if len(sys.argv) > 1 else 5504))

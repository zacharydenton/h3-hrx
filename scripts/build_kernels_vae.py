"""Compile the video VAE decoder's kernel set for one clip's token count into build/kernels_vae/T<tokens>/.
    python3 scripts/build_kernels_vae.py <tokens>
Shapes: hidden 2048, 32 heads of 64 (qkv 6144), SwiGLU 8192 (fused 16384), one AdaLN class (the norm
tables are zero; the layer scales are the residual gate tables), biases on every linear."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT / "scripts"))
from kernel_cache import compile_cached
from build_kernels import gemm_m_group

HIDDEN, HEADS, D, FFN = 2048, 32, 64, 8192
INNER = HEADS * D


BITS = int(os.environ.get("H3VAE_BITS", "4"))


def build(tokens: int) -> Path:
    out = ROOT / ("build/kernels_vae" if BITS == 4 else "build/kernels_vae_i8") / f"T{tokens}"
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("H3_GEMM_TILE", "256")
    tiles = (tokens + 255) // 256
    m_group = min((4, 3, 2), key=lambda g: ((tiles + g - 1) // g * g, -g))
    waves = 8 if tokens >= 4096 else 4
    capacity = max((tokens + 16 + 31) // 32 * 32, (tokens + 16 * waves - 1) // (16 * waves) * (16 * waves))
    attn_stem = "attention_mha648_lds_f16_wmma" if waves == 8 else "attention_mha64_lds_f16_wmma"
    attn = "h3." + attn_stem
    g8 = lambda stem: stem.replace("i4", "i8") if BITS == 8 else stem
    specs = [
        (g8("prepare_norm_i4"), "h3_" + g8("prepare_norm_i4"), "prepare_norm_i4", {"h3." + g8("prepare_norm_i4") + ".width": HIDDEN, "h3." + g8("prepare_norm_i4") + ".lanes": 256, "h3." + g8("prepare_norm_i4") + ".eps": 1e-5, "h3." + g8("prepare_norm_i4") + ".classes": 1}),
        (g8("prepare_plain_i4"), "h3_" + g8("prepare_plain_i4"), "prepare_attn_i4", {"h3." + g8("prepare_plain_i4") + ".width": INNER, "h3." + g8("prepare_plain_i4") + ".lanes": 256}),
        (g8("prepare_plain_i4"), "h3_" + g8("prepare_plain_i4"), "prepare_down_i4", {"h3." + g8("prepare_plain_i4") + ".width": FFN, "h3." + g8("prepare_plain_i4") + ".lanes": 256}),
        (g8("gemm_i4_256b"), "h3_" + g8("gemm_i4_256b"), "gemm_qkv", {"h3." + g8("gemm_i4_256b") + ".k_size": HIDDEN, "h3." + g8("gemm_i4_256b") + ".n_size": 3 * INNER, "h3." + g8("gemm_i4_256b") + ".m_group": m_group}),
        (g8("gemm_i4_swiglu_256b_gs"), "h3_" + g8("gemm_i4_swiglu_256b_gs"), "gemm_gu", {"h3." + g8("gemm_i4_swiglu_256b_gs") + ".k_size": HIDDEN, "h3." + g8("gemm_i4_swiglu_256b_gs") + ".n_size": 2 * FFN, "h3." + g8("gemm_i4_swiglu_256b_gs") + ".m_group": m_group}),
        (g8("gemm_i4_resid_256b"), "h3_" + g8("gemm_i4_resid_256b"), "gemm_out", {"h3." + g8("gemm_i4_resid_256b") + ".k_size": INNER, "h3." + g8("gemm_i4_resid_256b") + ".n_size": HIDDEN, "h3." + g8("gemm_i4_resid_256b") + ".m_group": m_group, "h3." + g8("gemm_i4_resid_256b") + ".classes": 1}),
        (g8("gemm_i4_resid_256b"), "h3_" + g8("gemm_i4_resid_256b"), "gemm_down", {"h3." + g8("gemm_i4_resid_256b") + ".k_size": FFN, "h3." + g8("gemm_i4_resid_256b") + ".n_size": HIDDEN, "h3." + g8("gemm_i4_resid_256b") + ".m_group": m_group, "h3." + g8("gemm_i4_resid_256b") + ".classes": 1}),
        ("rope64_qknorm_f16", "h3_rope64_qknorm_f16", "rope_qknorm", {"h3.rope64_qknorm_f16.row_stride": 3 * INNER, "h3.rope64_qknorm_f16.heads": HEADS, "h3.rope64_qknorm_f16.kv_heads": HEADS, "h3.rope64_qknorm_f16.k_offset": INNER, "h3.rope64_qknorm_f16.eps": 1e-5}),
        (attn_stem, "h3_" + attn_stem, "attention", {f"{attn}.q_stride": INNER, f"{attn}.kv_stride": INNER, f"{attn}.tokens": tokens, f"{attn}.token_capacity": capacity, f"{attn}.scale": D ** -0.5, f"{attn}.out_stride": INNER}),
    ]
    for stem, sym, name, cfg in specs:   # pitch configs default to K / width here; the host pads them (gemm_pitch in host/h3pipe.cpp)
        ns = "h3." + stem + "."
        if stem.startswith("gemm_i") and ns + "k_size" in cfg: cfg.setdefault(ns + "k_stride", cfg[ns + "k_size"])
        if stem.startswith("prepare_") and stem[-3:] in ("_i4", "_i8") and ns + "width" in cfg: cfg.setdefault(ns + "out_stride", cfg[ns + "width"])
        hs = out / f"{name}.hsaco"
        compile_cached(ROOT / "kernels" / f"{stem}.loom", sym, cfg, hs)
    (out / "capacity.txt").write_text(f"{capacity}\n"); (out / "gemm_tile.txt").write_text("256\n"); (out / "attention_waves.txt").write_text(f"{waves}\n"); (out / "bits.txt").write_text(f"{BITS}\n")
    return out


if __name__ == "__main__":
    print(build(int(sys.argv[1]) if len(sys.argv) > 1 else 11345))

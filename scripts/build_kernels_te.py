"""Compile the text encoder's kernel set for one prompt's token count into build/kernels_te/T<tokens>/.
    python3 scripts/build_kernels_te.py <tokens>
Shapes: Qwen3-VL-32B's language model, hidden 5120, 64 query heads and 8 key/value heads of 128
(fused qkv 10240), SwiGLU 25600 (fused 51200), no biases, RMSNorm eps 1e-6, W8A8 throughout,
causal attention with eight query heads per key/value head."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT / "scripts"))
from kernel_test import compile_kernel

HIDDEN, HEADS, KV_HEADS, D, FFN = 5120, 64, 8, 128, 25600
INNER, KV_INNER = HEADS * D, KV_HEADS * D
QKV = INNER + 2 * KV_INNER
EPS = 1e-6


def build(tokens: int) -> Path:
    out = ROOT / "build/kernels_te" / f"T{tokens}"
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("H3_GEMM_TILE", "256")
    tiles = (tokens + 255) // 256
    m_group = 1 if tiles == 1 else min((4, 3, 2), key=lambda g: ((tiles + g - 1) // g * g, -g))   # a prompt is one row tile: no padded group streaming the weights twice
    capacity = max((tokens + 16 + 31) // 32 * 32, (tokens + 127) // 128 * 128)
    attn = "h3.attention_gqa8c_lds_f16_wmma"
    specs = [
        ("prepare_norm_i8", "h3_prepare_norm_i8", "prepare_norm", {"h3.prepare_norm_i8.width": HIDDEN, "h3.prepare_norm_i8.lanes": 160, "h3.prepare_norm_i8.eps": EPS, "h3.prepare_norm_i8.classes": 1}),
        ("prepare_plain_i8", "h3_prepare_plain_i8", "prepare_attn", {"h3.prepare_plain_i8.width": INNER, "h3.prepare_plain_i8.lanes": 256}),
        ("prepare_plain16_i8", "h3_prepare_plain16_i8", "prepare_down", {"h3.prepare_plain16_i8.width": FFN, "h3.prepare_plain16_i8.lanes": 320}),   # f16 LDS: 25600 f32 would not fit
        ("gemm_i8_256", "h3_gemm_i8_256", "gemm_qkv", {"h3.gemm_i8_256.k_size": HIDDEN, "h3.gemm_i8_256.n_size": QKV, "h3.gemm_i8_256.m_group": m_group}),
        ("gemm_i8_swiglu_256", "h3_gemm_i8_swiglu_256", "gemm_gu", {"h3.gemm_i8_swiglu_256.k_size": HIDDEN, "h3.gemm_i8_swiglu_256.n_size": 2 * FFN, "h3.gemm_i8_swiglu_256.m_group": m_group}),
        ("gemm_i8_resid_256", "h3_gemm_i8_resid_256", "gemm_out", {"h3.gemm_i8_resid_256.k_size": INNER, "h3.gemm_i8_resid_256.n_size": HIDDEN, "h3.gemm_i8_resid_256.m_group": m_group, "h3.gemm_i8_resid_256.classes": 1}),
        ("gemm_i8_resid_256", "h3_gemm_i8_resid_256", "gemm_down", {"h3.gemm_i8_resid_256.k_size": FFN, "h3.gemm_i8_resid_256.n_size": HIDDEN, "h3.gemm_i8_resid_256.m_group": m_group, "h3.gemm_i8_resid_256.classes": 1}),
        ("rope128_qknorm_f16", "h3_rope128_qknorm_f16", "rope_qknorm", {"h3.rope128_qknorm_f16.row_stride": QKV, "h3.rope128_qknorm_f16.heads": HEADS, "h3.rope128_qknorm_f16.kv_heads": KV_HEADS, "h3.rope128_qknorm_f16.k_offset": INNER, "h3.rope128_qknorm_f16.eps": EPS}),
        ("attention_gqa8c_lds_f16_wmma", "h3_attention_gqa8c_lds_f16_wmma", "attention", {f"{attn}.q_stride": INNER, f"{attn}.kv_stride": KV_INNER, f"{attn}.tokens": tokens, f"{attn}.token_capacity": capacity, f"{attn}.scale": D ** -0.5, f"{attn}.out_stride": INNER}),
    ]
    for stem, sym, name, cfg in specs:
        hs = out / f"{name}.hsaco"
        if not hs.exists():
            compile_kernel(ROOT / "kernels" / f"{stem}.loom", sym, cfg, hs)
    (out / "capacity.txt").write_text(f"{capacity}\n"); (out / "gemm_tile.txt").write_text("256\n")
    return out


if __name__ == "__main__":
    print(build(int(sys.argv[1]) if len(sys.argv) > 1 else 28))

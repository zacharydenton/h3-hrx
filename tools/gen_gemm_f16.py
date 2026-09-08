"""f16 and bf16 twins of the int8 block GEMMs (kernels/gemm_i8_*_256*.loom -> gemm_f16_*_256*.loom and
gemm_bf16_*_256*.loom): the same 256x128 workgroup tile, 64x64 wave tiles and one-step-ahead register packets,
with 16-bit float operands (vector<16xf16> / vector<16xbf16> fragments, f32 accumulation, no scales, f16 output).
The stage rows grow from 80 to 144 bytes (64 elements + 8 pad): 55296 B of LDS, one workgroup per CU.
A rows come from prepare_*_{f16,bf16} (the formed row narrowed, unrotated); W rows are the checkpoint's f16 / bf16
rows as stored (the video VAE decoder, the token refiner).
    python3 tools/gen_gemm_f16.py"""
import re
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
GEMMS = ["gemm_i8_256", "gemm_i8_256b", "gemm_i8_resid_256", "gemm_i8_resid_256b", "gemm_i8_swiglu_256", "gemm_i8_swiglu_256b_gs"]


def gemm(K: str, elem: str = "f16") -> str:
    """The int8 kernel text -> the f16 twin; elem "bf16" then swaps the operand side (views, loads, stage, fragments, MMA)
    to bf16, leaving the f16 output and its saturation as they are."""
    assert elem in ("f16", "bf16")
    def sub(old, new, count=None):
        nonlocal K
        n = K.count(old); assert n > 0 and (count is None or n == count), (old[:80], n)
        K = K.replace(old, new)
    K = K.replace("gemm_i8_", "gemm_f16_")
    K = K.replace("// int4 GEMM on the gfx11 WMMA with 64x64 wave tiles: C[m, n] = f16(scale[n] * sum_k A[m, k] * W[n, k])", "// f16 GEMM on the gfx11 WMMA with 64x64 wave tiles: C[m, n] = f16(sum_k A[m, k] * W[n, k])").replace("with A [M][K] and W [N][K] signed int4 nibbles, low nibble first, i32 accumulation, f32 scale.", "with A [M][K] and W [N][K] f16 rows, f32 accumulation (tools/gen_gemm_f16.py from the int8 kernel).")
    # launch: no scales
    sub("%a: buffer, %w: buffer, %scale: buffer, %a_scale: buffer, %c: buffer", "%a: buffer, %w: buffer, %c: buffer")
    for line in ("  %scale_global = buffer.assume.memory_space<global> %scale : buffer\n", "  %scale_view = buffer.view %scale_global[%c0_offset] : buffer -> view<[%n_size]xf32>\n",
                 "  %a_scale_global = buffer.assume.memory_space<global> %a_scale : buffer\n", "  %a_scale_view = buffer.view %a_scale_global[%c0_offset] : buffer -> view<[%m_bounded]xf32>\n",
                 "  %i4_schema = encoding.define #encoding.operand<element_format=i8, payload_elements=16, payload_registers=4> : encoding<schema>\n"):
        sub(line, "")
    # LDS geometry: 144-byte rows
    for name in ("lds_bytes", "w_stage_offset"):
        match = re.search(rf"%{name} = index.constant (\d+) : offset", K)
        assert match and int(match[1]) % 80 == 0, name
        sub(match[0], f"%{name} = index.constant {int(match[1]) // 80 * 144} : offset", 1)
    # global operand views and loads: f16 elements, 16 per lane per row
    # the operand views carry the padded row pitch (k_stride, as the int8 kernel's k_quads = k_stride / 4); the loads stay within k_size
    sub("  %a_view = buffer.view %a_global[%c0_offset] : buffer -> view<[%m_bounded]x[%k_quads]xi32>\n", "  %a_view = buffer.view %a_global[%c0_offset] : buffer -> view<[%m_bounded]x[%k_stride]xf16>\n  %k_last16 = index.sub %k_size, %c16 : index\n")
    sub("  %w_view = buffer.view %w_global[%c0_offset] : buffer -> view<[%n_size]x[%k_quads]xi32>\n", "  %w_view = buffer.view %w_global[%c0_offset] : buffer -> view<[%n_size]x[%k_stride]xf16>\n")
    sub("  %kq_first = index.assume %st_quad [le(%st_quad, %k_quad_limit), mul(%st_quad, 4)] : index\n",
        "  %kq_first = index.assume %st_quad [le(%st_quad, %k_quad_limit), mul(%st_quad, 4)] : index\n  %ke_first0 = index.mul %kq_first, %c4 : index\n  %ke_first = index.assume %ke_first0 [le(%ke_first0, %k_last16), mul(%ke_first0, 16)] : index\n")
    sub("    %kq_next = index.assume %kq_safe [le(%kq_safe, %k_quad_limit), mul(%kq_safe, 4)] : index\n",
        "    %kq_next = index.assume %kq_safe [le(%kq_safe, %k_quad_limit), mul(%kq_safe, 4)] : index\n    %ke_next0 = index.mul %kq_next, %c4 : index\n    %ke_next = index.assume %ke_next0 [le(%ke_next0, %k_last16), mul(%ke_next0, 16)] : index\n")
    K = re.sub(r"vector\.load %a_view\[(%a_row\d), %kq_(first|next)\] : view<\[%m_bounded\]x\[%k_quads\]xi32> -> vector<4xi32>", r"vector.load %a_view[\1, %ke_\2] : view<[%m_bounded]x[%k_stride]xf16> -> vector<16xf16>", K)
    K = re.sub(r"vector\.load %w_view\[(%w_row\d), %kq_(first|next)\] : view<\[%n_size\]x\[%k_quads\]xi32> -> vector<4xi32>", r"vector.load %w_view[\1, %ke_\2] : view<[%n_size]x[%k_stride]xf16> -> vector<16xf16>", K)
    # stage: element columns
    sub("  %st_quad = index.mul %st_sub, %c4 : index\n", "  %st_quad = index.mul %st_sub, %c4 : index\n  %st_col0 = index.mul %st_sub, %c16 : index\n  %st_col = index.assume %st_col0 [lt(%st_col0, %c64), mul(%st_col0, 16)] : index\n")
    K = re.sub(r"vector\.store (%c[aw]\d), (%[aw]_stage)\[(%st_[aw]row\d), %st_quad\] : vector<4xi32>, view<(\d+)x20xi32>", r"vector.store \1, \2[\3, %st_col] : vector<16xf16>, view<\4x72xf16>", K)
    K = re.sub(r"vector\.load (%[aw]_stage)\[(%f[ab]\d), %c(\d+)\] : view<(\d+)x20xi32> -> vector<4xi32>", lambda m: f"vector.load {m.group(1)}[{m.group(2)}, %c{4 * int(m.group(3))}] : view<{m.group(4)}x72xf16> -> vector<16xf16>", K)
    K = K.replace("view<256x20xi32>", "view<256x72xf16>").replace("view<128x20xi32>", "view<128x72xf16>")
    # fragments and MMAs
    K = re.sub(r"(vector\.fragment<(?:lhs|rhs)> %\w+ shape \[%\w, %\w\]) using \{schema = %i4_schema : encoding<schema>\} : vector<4xi32>", r"\1 : vector<16xf16>", K)
    K = K.replace(": vector<4xi32>, vector<4xi32>, vector<8xi32>", ": vector<16xf16>, vector<16xf16>, vector<8xf32>")
    sub("  %zero_i32x8 = vector.constant 0 : vector<8xi32>\n", "  %zero_i32x8 = vector.constant 0.0 : vector<8xf32>\n")
    sub("  %zero_i32x4 = vector.constant 0 : vector<4xi32>\n", "  %zero_i32x4 = vector.constant 0.0 : vector<16xf16>\n")
    # epilogue: f32 result stage, no scales
    K = K.replace("view<16x16xi32>", "view<16x16xf32>")
    if "%gate_f32 = vector.sitofp" in K:   # swiglu
        for old, new in (("%gate_values = vector.load %result_view[%publish_row, %publish_col] : view<16x16xf32> -> vector<4xi32>", "%gate_values = vector.load %result_view[%publish_row, %publish_col] : view<16x16xf32> -> vector<4xf32>"),
                         ("%up_values = vector.load %up_view[%publish_row, %publish_col] : view<16x16xf32> -> vector<4xi32>", "%up_values = vector.load %up_view[%publish_row, %publish_col] : view<16x16xf32> -> vector<4xf32>"),
                         ("%gate_f32 = vector.sitofp %gate_values : vector<4xi32> to vector<4xf32>", "%gate_f32 = vector.addf %gate_values, %zero4 : vector<4xf32>"),
                         ("%up_f32 = vector.sitofp %up_values : vector<4xi32> to vector<4xf32>", "%up_f32 = vector.addf %up_values, %zero4 : vector<4xf32>"),
                         ("%gate_scaled0 = vector.mulf %gate_f32, %gate_scale : vector<4xf32>", "%gate_scaled0 = vector.addf %gate_f32, %zero4 : vector<4xf32>"),
                         ("%up_scaled0 = vector.mulf %up_f32, %up_scale : vector<4xf32>", "%up_scaled0 = vector.addf %up_f32, %zero4 : vector<4xf32>"),
                         ("%g0 = vector.mulf %gate_scaled0, %a_scale_vector : vector<4xf32>", "%g0 = vector.addf %gate_scaled0, %zero4 : vector<4xf32>"),
                         ("%u0 = vector.mulf %up_scaled0, %a_scale_vector : vector<4xf32>", "%u0 = vector.addf %up_scaled0, %zero4 : vector<4xf32>")):
            sub(old, new, 1)
        K = re.sub(r" *%(gate|up)_scale = vector\.load %scale_view\[%\w+\] : view<\[%n_size\]xf32> -> vector<4xf32>\n", "", K)
    else:
        sub("%values = vector.load %result_view[%publish_row, %publish_col] : view<16x16xf32> -> vector<4xi32>\n", "%values = vector.load %result_view[%publish_row, %publish_col] : view<16x16xf32> -> vector<4xf32>\n")
        sub("%values_f32 = vector.sitofp %values : vector<4xi32> to vector<4xf32>", "%values_f32 = vector.addf %values, %zero4 : vector<4xf32>")
        sub("%scaled0 = vector.mulf %values_f32, %scale_values : vector<4xf32>", "%scaled0 = vector.addf %values_f32, %zero4 : vector<4xf32>")
        sub("%scaled1 = vector.mulf %scaled0, %a_scale_vector : vector<4xf32>", "%scaled1 = vector.addf %scaled0, %zero4 : vector<4xf32>")
        K = re.sub(r" *%scale_values = vector\.load %scale_view\[%out_col\] : view<\[%n_size\]xf32> -> vector<4xf32>\n", "", K)
    K = re.sub(r" *%a_scale_value = view\.load %a_scale_view\[%bounded\] : view<\[%m_bounded\]xf32> -> f32\n *%a_scale_vector = vector\.splat %a_scale_value : vector<4xf32>\n", "", K)
    K = K.replace("vector<8xi32>", "vector<8xf32>").replace("vector<4xi32>", "vector<16xf16>")
    assert "xi32" not in K.replace("view<[%m_bounded]xi32>", "").replace("-> i32", "").replace(": i32", ""), [l for l in K.splitlines() if "xi32" in l][:5]
    left = [l for l in K.splitlines() if "scale" in l and "scaled" not in l and "index.scale" not in l and not l.strip().startswith("//")]
    assert not left, left[:5]
    if elem == "bf16":
        for old in ("view<[%m_bounded]x[%k_stride]xf16>", "view<[%n_size]x[%k_stride]xf16>", "view<256x72xf16>", "view<128x72xf16>", "vector<16xf16>"):
            sub(old, old.replace("f16", "bf16"))
        sub("gemm_f16_", "gemm_bf16_")
        sub("// f16 GEMM on the gfx11 WMMA", "// bf16 GEMM on the gfx11 WMMA"); sub("W [N][K] f16 rows", "W [N][K] bf16 rows")
        assert K.count("xbf16") >= 20 and "vector<16xf16>" not in K, "every operand-side f16 became bf16"
    return K


def main():
    for stem in GEMMS:
        for elem in ("f16", "bf16"):
            out = ROOT / "h3/kernels" / f"{stem.replace('gemm_i8_', f'gemm_{elem}_')}.loom"; out.write_text(gemm((ROOT / "h3/kernels" / f"{stem}.loom").read_text(), elem)); print("wrote", out.name)


if __name__ == "__main__":
    main()

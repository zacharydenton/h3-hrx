"""Generate the gfx1151 head-major INT8 attention and matching operand preparation.

The transposed 32-key template supplies ordinary Loom addressing and softmax.
Small public low.invoke fragments express probability packing and WMMA order.
Run: python3 tools/gen_attention_i8_head_major.py
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEM = "attention_i8qkhm_mha8_lds_f16_wmma"


def attention_source():
    s = (
        (ROOT / "experiments/attention_i8qkt32_mha8_lds_f16_wmma.loom")
        .read_text()
        .replace("attention_i8qkt32_mha8_lds_f16_wmma", STEM)
    )
    vg = "reg<amdgpu.vgpr>"
    helper = f"low.func.def target<amdgpu.gfx11.generic.core>(@h3_{STEM}_gfx11)\n    @h3_pack_probability(%even: reg<amdgpu.vgpr x4>, %odd: reg<amdgpu.vgpr x4>) -> (reg<amdgpu.vgpr x8>) asm {{\n  %lo = s_mov_b32 0x05040100\n  %hi = s_mov_b32 0x07060302\n"
    for i in range(4):
        helper += f"  %e{i} = slice %even[{i}] : reg<amdgpu.vgpr x4> -> {vg}\n  %o{i} = slice %odd[{i}] : reg<amdgpu.vgpr x4> -> {vg}\n  %p{2 * i} = v_perm_b32 %o{i}, %e{i}, %lo\n  %p{2 * i + 1} = v_perm_b32 %o{i}, %e{i}, %hi\n"
    helper += (
        "  %p = concat("
        + ", ".join(f"%p{i}" for i in range(8))
        + ") : ("
        + ", ".join([vg] * 8)
        + ") -> reg<amdgpu.vgpr x8>\n  return %p\n}\n"
    )
    helper += (
        """low.func.def schedule(locked) target<amdgpu.gfx11.generic.core>(@h3_"""
        + STEM
        + """_gfx11)
    @h3_pv32(%v: reg<amdgpu.vgpr x8>, %p: reg<amdgpu.vgpr x8>, %acc: reg<amdgpu.vgpr x8>) -> (reg<amdgpu.vgpr x8>) asm {
  %owned = copy %acc : reg<amdgpu.vgpr x8> -> reg<amdgpu.vgpr x8>
  %result = v_wmma_f32_16x16x16_f16 %v, %p, %owned
  return %result
}
"""
    )
    a = s.index("kernel.def")
    s = s[:a] + helper + s[a:]
    s = s.replace("  %qs_one = view.load", "  %qs_unscaled = view.load")
    a = s.index("  %qs_vec =")
    s = (
        s[:a]
        + "  %log2e = scalar.constant 1.4426950408889634 : f32\n  %qs_one = scalar.mulf %qs_unscaled, %log2e : f32\n"
        + s[a:]
    )
    s = s.replace("vector.expf<afn>", "vector.exp2f<afn>").replace(
        "scalar.expf<afn>", "scalar.exp2f<afn>"
    )
    a = s.index("    %other_weight,")
    b = s.index("    %vrow0", a)
    pack = ""
    for suffix in ("", "_b"):
        pack += f"""    %weight_half{suffix} = vector.fptrunc %weight{suffix} : vector<8xf32> to vector<8xf16>
    %weight_words{suffix} = vector.bitcast %weight_half{suffix} : vector<8xf16> to vector<4xi32>
    %other_words{suffix}, %weight_ok{suffix} = kernel.subgroup.shuffle<xor> %weight_words{suffix}, %i32_16, %i32_32 : vector<4xi32>, i32, i32
    %even_words{suffix} = scf.select %lane_group_even, %weight_words{suffix}, %other_words{suffix} : vector<4xi32>
    %odd_words{suffix} = scf.select %lane_group_even, %other_words{suffix}, %weight_words{suffix} : vector<4xi32>
    %probability{suffix} = low.invoke @h3_pack_probability(%even_words{suffix}, %odd_words{suffix}) : (vector<4xi32>, vector<4xi32>) -> (vector<16xf16>)
"""
    s = s[:a] + pack + s[b:]
    for c in range(8):
        s = re.sub(rf"^    %v{c}(?:_b)? = .*\n", "", s, flags=re.MULTILINE)
        s = re.sub(
            rf"^    %partial{c} = .*\n",
            f"    %partial{c} = low.invoke @h3_pv32(%v_data{c}, %probability, %rescaled{c}) : (vector<16xf16>, vector<16xf16>, vector<8xf32>) -> (vector<8xf32>)\n",
            s,
            flags=re.MULTILINE,
        )
        s = re.sub(
            rf"^    %next{c} = .*\n",
            f"    %next{c} = low.invoke @h3_pv32(%v_data{c}_b, %probability_b, %partial{c}) : (vector<16xf16>, vector<16xf16>, vector<8xf32>) -> (vector<8xf32>)\n",
            s,
            flags=re.MULTILINE,
        )
    return s


def transform(s):
    helper = f"""low.func.def schedule(locked) target<amdgpu.gfx11.generic.core>(@h3_{STEM}_gfx11)
        @h3_qk_pair(%ka: reg<amdgpu.vgpr x4>, %kb: reg<amdgpu.vgpr x4>, %q: reg<amdgpu.vgpr x4>, %aa: reg<amdgpu.vgpr x8>, %ab: reg<amdgpu.vgpr x8>) -> (reg<amdgpu.vgpr x8>, reg<amdgpu.vgpr x8>) asm {{
      %oa = copy %aa : reg<amdgpu.vgpr x8> -> reg<amdgpu.vgpr x8>
  %ob = copy %ab : reg<amdgpu.vgpr x8> -> reg<amdgpu.vgpr x8>
  %ra = v_wmma_i32_16x16x16_iu8 %ka, %q, %oa {{neg_lo = 3}}
      %rb = v_wmma_i32_16x16x16_iu8 %kb, %q, %ob {{neg_lo = 3}}
      return %ra, %rb
    }}
    """
    a = s.index("kernel.def")
    s = s[:a] + helper + s[a:]
    s = re.sub(
        r"^    %(?:k_data\d(?:_b)?|rhs\d(?:_b)?|qk\d(?:_b)?|raw_scores_i(?:_b)?) = .*\n",
        "",
        s,
        flags=re.MULTILINE,
    )
    block = ""
    for i in range(8):
        destA = f"%pair{i}"
        destB = f"%pair{i}_b"
        if i == 7:
            destA = "%raw_scores_i"
            destB = "%raw_scores_i_b"
        ca = "%zero_i32x8" if i == 0 else f"%pair{i - 1}"
        cb = "%zero_i32x8" if i == 0 else f"%pair{i - 1}_b"
        block += f"""    %ka{i} = vector.load %k_tile_b[%lane_column, %c{4 * i}] : view<32x36xi32> -> vector<4xi32>
        %kb{i} = vector.load %k_tile_b[%key_lane_b, %c{4 * i}] : view<32x36xi32> -> vector<4xi32>
        {destA}, {destB} = low.invoke @h3_qk_pair(%ka{i}, %kb{i}, %qd{i}, {ca}, {cb}) : (vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<8xi32>, vector<8xi32>) -> (vector<8xi32>, vector<8xi32>)
    """
    a = s.index("    %raw_scores =")
    s = s[:a] + block + s[a:]
    a = s.index("  %qi_view =")
    s = (
        s[:a]
        + """  %q_rows = index.mul %head_limit, %padded_tokens : index
      %k_rows = index.mul %kv_head_limit, %padded_tokens : index
      %q_hbase = index.mul %head, %padded_tokens : index
      %k_hbase = index.mul %kv_head, %padded_tokens : index
    """
        + s[a:]
    )
    s = s.replace(
        "view<[%padded_tokens]x[%qi_words]xi32>", "view<[%q_rows]x32xi32>"
    ).replace("view<[%padded_tokens]x[%ki_words]xi32>", "view<[%k_rows]x32xi32>")
    s = s.replace(
        "view<[%padded_tokens]x[%head_limit]xf32>",
        "view<[%head_limit]x[%padded_tokens]xf32>",
    ).replace(
        "view<[%padded_tokens]x[%kv_head_limit]xf32>",
        "view<[%kv_head_limit]x[%padded_tokens]xf32>",
    )
    s = s.replace("%qs_view[%q_row, %head]", "%qs_view[%head, %q_row]")
    s = re.sub(r"%ks_view\[(%\w+), %kv_head\]", r"%ks_view[%kv_head, \1]", s)
    a = s.index("  %q_wbase =")
    s = (
        s[:a]
        + """  %q_flat0 = index.add %q_hbase, %q_row : index
      %q_flat = index.assume %q_flat0 [lt(%q_flat0, %q_rows)] : index
    """
        + s[a:]
    )
    for c in range(8):
        s = s.replace(f"%qi_view[%q_row, %q_word{c}]", f"%qi_view[%q_flat, %c{4 * c}]")
    for suffix in ("", "_b"):
        anchor = (
            f"      %k_chunk{suffix} = vector.load %ki_view[%st_row{suffix}, %st_col]"
        )
        replacement = f"""      %k_flat0{suffix} = index.add %k_hbase, %st_row{suffix} : index
          %k_flat{suffix} = index.assume %k_flat0{suffix} [lt(%k_flat0{suffix}, %k_rows)] : index
          %k_chunk{suffix} = vector.load %ki_view[%k_flat{suffix}, %st_chunk]"""
        assert anchor in s
        s = s.replace(anchor, replacement)
    # Full key tiles need no per-element mask after specialization.
    a = s.index("  %q_row0 =")
    s = (
        s[:a]
        + "  %key_tail = index.rem %tokens0, %c32 : index\n  %full_keys = index.cmp eq, %key_tail, %c0 : index\n"
        + s[a:]
    )
    s = re.sub(
        r"^(    %valid\d(?:_b)?) = (index.cmp ult, .* : index)\n",
        r"\1_partial = \2\n\1 = scalar.ori \1_partial, %full_keys : i1\n",
        s,
        flags=re.MULTILINE,
    )
    # Inactive query waves may read a clamped row, but must not race its writer.
    s = s.replace(
        "  %writes = scalar.andi %row_present, %row_in_range : i1",
        "  %writes0 = scalar.andi %row_present, %row_in_range : i1\n  %tile_present = index.cmp ult, %tile_in_image0, %tiles_per_image : index\n  %writes = scalar.andi %writes0, %tile_present : i1",
    )
    s = (
        "// Generated by tools/gen_attention_i8_head_major.py. Dense INT8 QK, FP32\n// online softmax/accumulation, FP16 P and V. Q/K/scales are head-major.\n// Eight wave32 query tiles share 32 keys; launch ceil(N/128) x heads.\n"
        + s[s.index("amdgpu.target") :]
    )
    s = s.replace(
        "// int4 operands: [tokens][heads * 16] i32 words (8 codes each), scales [tokens][heads] f32",
        "// INT8 codes [heads][capacity][32] i32 words; scales [heads][capacity] f32",
    )
    return s


def prepare_source():
    s = (
        (ROOT / "kernels/prepare_qk_i8.loom")
        .read_text()
        .replace("prepare_qk_i8", "prepare_qk_i8hm")
    )
    a = s.index("kernel.def")
    s = (
        s[:a]
        + "config.decl @h3.prepare_qk_i8hm.token_capacity : %value: index where [range(%value, 32, 1048576), mul(%value, 32)]\n\n"
        + s[a:]
    )
    a = s.index("  %row_stride =")
    s = (
        s[:a]
        + "  %capacity = config.get @h3.prepare_qk_i8hm.token_capacity : index\n"
        + s[a:]
    )
    s = s.replace(
        "  %words = index.mul %heads, %c32 : index",
        "  %words = index.mul %heads, %c32 : index\n  %rows = index.mul %heads, %capacity : index",
    )
    s = s.replace("view<[%tokens_b]x[%words]xi32>", "view<[%rows]x32xi32>").replace(
        "view<[%tokens_b]x[%heads]xf32>", "view<[%heads]x[%capacity]xf32>"
    )
    s = s.replace(
        "      view.store %word, %c_view[%token, %wi]",
        "      %flat0 = index.mul %head, %capacity : index\n      %flat1 = index.add %flat0, %token : index\n      %flat = index.assume %flat1 [lt(%flat1, %rows)] : index\n      view.store %word, %c_view[%flat, %lane]",
    )
    s = s.replace("%s_view[%token, %head]", "%s_view[%head, %token]")
    s = (
        "// Generated by tools/gen_attention_i8_head_major.py. The same Hadamard INT8\n// quantization as prepare_qk_i8, written directly to codes [heads][capacity][32]\n// i32 and scales [heads][capacity] f32. The caller zeroes padded capacity once.\n"
        + s[s.index("amdgpu.target") :]
    )
    return s


def main():
    (ROOT / "kernels" / f"{STEM}.loom").write_text(transform(attention_source()))
    (ROOT / "kernels/prepare_qk_i8hm.loom").write_text(prepare_source())


if __name__ == "__main__":
    main()

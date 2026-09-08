"""32-key tiles for the int4-QK attention kernels: a post-pass on a shipped 16-key kernel text that stages two key
sub-tiles per barrier and runs both through the loop body (16 int4 + 16 f16 MMAs per barrier; every per-tile cost
amortised over twice the MMAs). Usage: python3 tools/gen_attention_i4_32.py <src stem> <out stem> [lds_pad]"""
import re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent


def doubled(K: str, src_stem: str, out_stem: str, lds_pad: int = 0) -> str:
    def sub(old, new, count=1):
        nonlocal K
        assert K.count(old) == count, (old[:90], K.count(old)); K = K.replace(old, new)
    # constants and LDS layout: K slot 32x20 i32 (2560 B), V slot 128x40 f16 (10240 B), 1 KB scratch per wave
    if "%c31 = index.constant" not in K: sub("  %c32 = index.constant 32 : index\n", "  %c32 = index.constant 32 : index\n  %c31 = index.constant 31 : index\n")
    sub("  %v_tile_offset = index.constant 2560 : offset\n", "  %v_tile_offset = index.constant 5120 : offset\n")
    sub("  %scratch_offset = index.constant 14848 : offset\n  %kbuf_bytes = index.constant 1280 : offset\n  %vbuf_bytes = index.constant 6144 : offset\n",
        "  %scratch_offset = index.constant 25600 : offset\n  %kbuf_bytes = index.constant 2560 : offset\n  %vbuf_bytes = index.constant 10240 : offset\n")
    K = re.sub(r"  %q_tile_offset = index.constant \d+ : offset\n", f"  %q_tile_offset = index.constant {25600 + 8 * 1024} : offset\n", K)
    K = re.sub(r"  %lds_bytes = index.constant \d+ : offset\n", f"  %lds_bytes = index.constant {25600 + 8 * 1024 + lds_pad} : offset\n", K)
    K = K.replace("view<16x20xi32>", "view<32x20xi32>").replace("view<128x24xf16>", "view<128x40xf16>")
    sub("  %wave_scratch_bytes = index.constant 512 : offset\n", "  %wave_scratch_bytes = index.constant 1024 : offset\n  %scratch_hi_offset = index.constant 512 : offset\n")
    sub("  %scratch_view = buffer.view %lds[%wave_scratch_offset] : buffer -> view<16x16xf16>\n",
        "  %scratch_view = buffer.view %lds[%wave_scratch_offset] : buffer -> view<16x16xf16>\n  %wave_scratch_offset_hi = index.add %wave_scratch_offset, %scratch_hi_offset : offset\n  %scratch_view_hi = buffer.view %lds[%wave_scratch_offset_hi] : buffer -> view<16x16xf16>\n")
    # key tiles of 32
    sub("  %key_tile_count = index.add %tiles_per_image, %c0 : index\n", "  %key_rounded = index.add %tokens0, %c31 : index\n  %key_tile_count = index.div %key_rounded, %c32 : index\n  %st_key_hi = index.add %st_key, %c16 : index\n  %lane_column_hi = index.add %lane_column, %c16 : index\n")
    sub("  %tile_origin_limit = index.sub %padded_tokens, %c16 : index\n", "  %tile_origin_limit = index.sub %padded_tokens, %c32 : index\n  %tile_origin_limit_hi = index.sub %padded_tokens, %c16 : index\n")
    sub("    %key_origin1 = index.mul %key_tile, %c16 : index\n    %key_origin0 = index.assume %key_origin1 [le(%key_origin1, %tile_origin_limit), mul(%key_origin1, 16)] : index\n",
        "    %key_origin1 = index.mul %key_tile, %c32 : index\n    %key_origin0 = index.assume %key_origin1 [le(%key_origin1, %tile_origin_limit), mul(%key_origin1, 32)] : index\n    %key_origin_hi0 = index.add %key_origin0, %c16 : index\n    %key_origin_hi = index.assume %key_origin_hi0 [le(%key_origin_hi0, %tile_origin_limit_hi), mul(%key_origin_hi0, 16)] : index\n")
    # the second ks carry
    sub("  %ks_last = index.sub %padded_tokens, %c1 : index\n", "  %ks_last = index.sub %padded_tokens, %c1 : index\n  %ks_key0_b0 = index.add %lane_column, %c16 : index\n  %ks_key0_b = index.assume %ks_key0_b0 [lt(%ks_key0_b0, %padded_tokens)] : index\n  %ks_first_b = view.load %ks_view[%ks_key0_b, %kv_head] : view<[%padded_tokens]x[%kv_head_limit]xf32> -> f32\n")
    hdr = re.search(r"  (%final_max, %final_sum, .*?), %ks_end = scf\.for %key_tile = \[%c0 to %key_tile_count step %c1\]\((.*?), %ks_carry = %ks_first : f32\) -> \((.*?), f32\) \{\n", K)
    assert hdr, "loop header"
    K = K.replace(hdr.group(0), f"  {hdr.group(1)}, %ks_end, %ks_end_b = scf.for %key_tile = [%c0 to %key_tile_count step %c1]({hdr.group(2)}, %ks_carry = %ks_first : f32, %ks_carry_b = %ks_first_b : f32) -> ({hdr.group(3)}, f32, f32) {{\n")
    # staging: two K rows and two V chunks per lane
    sub("      vector.store %k_chunk, %k_tile_b[%st_key, %st_chunk] : vector<2xi32>, view<32x20xi32>\n",
        "      vector.store %k_chunk, %k_tile_b[%st_key, %st_chunk] : vector<2xi32>, view<32x20xi32>\n      %st_row_hi0 = index.add %st_row0, %c16 : index\n      %st_row_hi = index.assume %st_row_hi0 [lt(%st_row_hi0, %padded_tokens)] : index\n      %k_chunk_hi = vector.load %ki_view[%st_row_hi, %st_col] : view<[%padded_tokens]x[%ki_words]xi32> -> vector<2xi32>\n      vector.store %k_chunk_hi, %k_tile_b[%st_key_hi, %st_chunk] : vector<2xi32>, view<32x20xi32>\n")
    sub("      vector.store %v_chunk, %v_tile_b[%st_lane, %c0] : vector<16xf16>, view<128x40xf16>\n",
        "      vector.store %v_chunk, %v_tile_b[%st_lane, %c0] : vector<16xf16>, view<128x40xf16>\n      %v_chunk_hi = vector.load %v_view[%st_chan, %key_origin_hi] : view<[%kv_stride0]x[%padded_tokens]xf16> -> vector<16xf16>\n      vector.store %v_chunk_hi, %v_tile_b[%st_lane, %c16] : vector<16xf16>, view<128x40xf16>\n")
    # the second score sub-tile: duplicate from the K loads through the masked scores
    a0 = K.index("    %local_key = index.add %key_origin0, %lane_column : index\n")
    a1 = K.index("    // Tile row max across the 16 lanes")
    block = K[a0:a1]
    names = [n for line in re.findall(r"^\s+((?:%[A-Za-z_][A-Za-z_0-9]*)(?:, %[A-Za-z_][A-Za-z_0-9]*)*) =", block, re.M) for n in line.split(", ")]
    mapping = {n: n + "_b" for n in names}
    blk_b = re.sub(r"%[A-Za-z_][A-Za-z_0-9]*", lambda m: mapping.get(m[0], m[0]), block)
    blk_b = blk_b.replace("%k_tile_b[%lane_column,", "%k_tile_b[%lane_column_hi,").replace("%ks_carry ", "%ks_carry_b ")
    assert "%local_key_b = index.add %key_origin0, %lane_column : index" in blk_b
    blk_b = blk_b.replace("%local_key_b = index.add %key_origin0, %lane_column : index", "%local_key_b = index.add %key_origin0, %lane_column_hi : index")
    K = K[:a1] + blk_b + K[a1:]
    # one max over both sub-tiles, two weight vectors, both in the sum
    sub("    %sh1, %ok1 = kernel.subgroup.shuffle<xor> %scaled, %i32_1, %i32_32 : vector<8xf32>, i32, i32\n    %tm1 = vector.maxnumf %scaled, %sh1 : vector<8xf32>\n",
        "    %tm0 = vector.maxnumf %scaled, %scaled_b : vector<8xf32>\n    %sh1, %ok1 = kernel.subgroup.shuffle<xor> %tm0, %i32_1, %i32_32 : vector<8xf32>, i32, i32\n    %tm1 = vector.maxnumf %tm0, %sh1 : vector<8xf32>\n")
    sub("    %weight = vector.expf<afn> %delta : vector<8xf32>\n", "    %weight = vector.expf<afn> %delta : vector<8xf32>\n    %delta_b = vector.subf<reassoc|nnan|ninf|nsz> %scaled_b, %next_max : vector<8xf32>\n    %weight_b = vector.expf<afn> %delta_b : vector<8xf32>\n")
    sub("    %next_sum = vector.addf<reassoc|nnan|ninf|nsz> %scaled_sum, %weight : vector<8xf32>\n", "    %next_sum_a = vector.addf<reassoc|nnan|ninf|nsz> %scaled_sum, %weight : vector<8xf32>\n    %next_sum = vector.addf<reassoc|nnan|ninf|nsz> %next_sum_a, %weight_b : vector<8xf32>\n")
    # P.V for sub-tile A into pva{c}, then sub-tile B on top
    p0 = K.index("    %vrow0 = index.add %lane_column, %c0 : index\n")
    p1 = K.index("    %next7 = vector.mma %probability, %v7, %rescaled7 : vector<16xf16>, vector<16xf16>, vector<8xf32>\n") + len("    %next7 = vector.mma %probability, %v7, %rescaled7 : vector<16xf16>, vector<16xf16>, vector<8xf32>\n")
    pv = K[p0:p1]
    pv_a = re.sub(r"%next(\d)", r"%pva\1", pv)
    pv_b = pv.replace("%vrow", "%vrowb").replace("%v_data", "%v_datab").replace("%probability,", "%probability_b,")
    pv_b = re.sub(r"(%v_datab\d = vector.load %v_tile_b\[%vrowb\d), %c0\]", r"\1, %c16]", pv_b)
    pv_b = re.sub(r"    %v(\d) = vector.fragment<rhs> %v_datab", r"    %vb\1 = vector.fragment<rhs> %v_datab", pv_b)
    pv_b = re.sub(r"    %rescaled(\d) = vector.mulf<[^>]*> %acc\d, %selected_old_scale : vector<8xf32>\n", "", pv_b)
    pv_b = re.sub(r"    %next(\d) = vector.mma %probability_b, %v(\d), %rescaled\d", r"    %next\1 = vector.mma %probability_b, %vb\2, %pva\1", pv_b)
    pb = ("    vector.fragment.store<result> %weight_b, %scratch_view_hi[%c0, %c0] shape [%m, %n] : vector<8xf32>, view<16x16xf16>\n"
          "    kernel.barrier<workgroup> scope(subgroup) ordering(acq_rel)\n"
          "    %probability_b = vector.fragment.load<lhs> %scratch_view_hi[%c0, %c0] shape [%m, %k_frag] : view<16x16xf16> -> vector<16xf16>\n")
    K = K[:p0] + pv_a + pb + pv_b + K[p1:]
    # next tile's key scales for both halves
    sub("    %ks_key_n0 = index.add %local_key, %c16 : index\n", "    %ks_key_n0 = index.add %local_key, %c32 : index\n    %ks_key_nb0 = index.add %local_key, %c48 : index\n    %ks_key_nb1 = index.min %ks_key_nb0, %ks_last : index\n    %ks_key_nb = index.assume %ks_key_nb1 [lt(%ks_key_nb1, %padded_tokens)] : index\n    %ks_next_b = view.load %ks_view[%ks_key_nb, %kv_head] : view<[%padded_tokens]x[%kv_head_limit]xf32> -> f32\n")
    sub(", %ks_next : " + ", ".join(["vector<8xf32>"] * 10) + ", f32\n", ", %ks_next, %ks_next_b : " + ", ".join(["vector<8xf32>"] * 10) + ", f32, f32\n")
    K = K.replace("h3_" + src_stem, "h3_" + out_stem).replace("h3." + src_stem, "h3." + out_stem)
    K = K.replace("// MHA attention with int4 QK^T", "// MHA attention with int4 QK^T, 32-key tiles (tools/gen_attention_i4_32.py),")
    return K


def unrolled(K: str, src_stem: str, out_stem: str, lds_pad: int = 0) -> str:
    """32-key staging, the 16-key loop body run twice in sequence (two online-softmax updates, one barrier)."""
    def sub(old, new, count=1):
        nonlocal K
        assert K.count(old) == count, (old[:90], K.count(old)); K = K.replace(old, new)
    if "%c31 = index.constant" not in K: sub("  %c32 = index.constant 32 : index\n", "  %c32 = index.constant 32 : index\n  %c31 = index.constant 31 : index\n")
    sub("  %v_tile_offset = index.constant 2560 : offset\n", "  %v_tile_offset = index.constant 5120 : offset\n")
    sub("  %scratch_offset = index.constant 14848 : offset\n  %kbuf_bytes = index.constant 1280 : offset\n  %vbuf_bytes = index.constant 6144 : offset\n",
        "  %scratch_offset = index.constant 25600 : offset\n  %kbuf_bytes = index.constant 2560 : offset\n  %vbuf_bytes = index.constant 10240 : offset\n")
    K = re.sub(r"  %q_tile_offset = index.constant \d+ : offset\n", f"  %q_tile_offset = index.constant {25600 + 8 * 512} : offset\n", K)
    K = re.sub(r"  %lds_bytes = index.constant \d+ : offset\n", f"  %lds_bytes = index.constant {25600 + 8 * 512 + lds_pad} : offset\n", K)
    K = K.replace("view<16x20xi32>", "view<32x20xi32>").replace("view<128x24xf16>", "view<128x40xf16>")
    sub("  %key_tile_count = index.add %tiles_per_image, %c0 : index\n", "  %key_rounded = index.add %tokens0, %c31 : index\n  %key_tile_count = index.div %key_rounded, %c32 : index\n  %st_key_hi = index.add %st_key, %c16 : index\n  %lane_column_hi = index.add %lane_column, %c16 : index\n")
    sub("  %tile_origin_limit = index.sub %padded_tokens, %c16 : index\n", "  %tile_origin_limit = index.sub %padded_tokens, %c32 : index\n  %tile_origin_limit_hi = index.sub %padded_tokens, %c16 : index\n")
    sub("    %key_origin1 = index.mul %key_tile, %c16 : index\n    %key_origin0 = index.assume %key_origin1 [le(%key_origin1, %tile_origin_limit), mul(%key_origin1, 16)] : index\n",
        "    %key_origin1 = index.mul %key_tile, %c32 : index\n    %key_origin0 = index.assume %key_origin1 [le(%key_origin1, %tile_origin_limit), mul(%key_origin1, 32)] : index\n    %key_origin_hi0 = index.add %key_origin0, %c16 : index\n    %key_origin_hi = index.assume %key_origin_hi0 [le(%key_origin_hi0, %tile_origin_limit_hi), mul(%key_origin_hi0, 16)] : index\n")
    sub("  %ks_last = index.sub %padded_tokens, %c1 : index\n", "  %ks_last = index.sub %padded_tokens, %c1 : index\n  %ks_key0_b0 = index.add %lane_column, %c16 : index\n  %ks_key0_b = index.assume %ks_key0_b0 [lt(%ks_key0_b0, %padded_tokens)] : index\n  %ks_first_b = view.load %ks_view[%ks_key0_b, %kv_head] : view<[%padded_tokens]x[%kv_head_limit]xf32> -> f32\n")
    hdr = re.search(r"  (%final_max, %final_sum, .*?), %ks_end = scf\.for %key_tile = \[%c0 to %key_tile_count step %c1\]\((.*?), %ks_carry = %ks_first : f32\) -> \((.*?), f32\) \{\n", K)
    assert hdr, "loop header"
    K = K.replace(hdr.group(0), f"  {hdr.group(1)}, %ks_end, %ks_end_b = scf.for %key_tile = [%c0 to %key_tile_count step %c1]({hdr.group(2)}, %ks_carry = %ks_first : f32, %ks_carry_b = %ks_first_b : f32) -> ({hdr.group(3)}, f32, f32) {{\n")
    sub("      vector.store %k_chunk, %k_tile_b[%st_key, %st_chunk] : vector<2xi32>, view<32x20xi32>\n",
        "      vector.store %k_chunk, %k_tile_b[%st_key, %st_chunk] : vector<2xi32>, view<32x20xi32>\n      %st_row_hi0 = index.add %st_row0, %c16 : index\n      %st_row_hi = index.assume %st_row_hi0 [lt(%st_row_hi0, %padded_tokens)] : index\n      %k_chunk_hi = vector.load %ki_view[%st_row_hi, %st_col] : view<[%padded_tokens]x[%ki_words]xi32> -> vector<2xi32>\n      vector.store %k_chunk_hi, %k_tile_b[%st_key_hi, %st_chunk] : vector<2xi32>, view<32x20xi32>\n")
    sub("      vector.store %v_chunk, %v_tile_b[%st_lane, %c0] : vector<16xf16>, view<128x40xf16>\n",
        "      vector.store %v_chunk, %v_tile_b[%st_lane, %c0] : vector<16xf16>, view<128x40xf16>\n      %v_chunk_hi = vector.load %v_view[%st_chan, %key_origin_hi] : view<[%kv_stride0]x[%padded_tokens]xf16> -> vector<16xf16>\n      vector.store %v_chunk_hi, %v_tile_b[%st_lane, %c16] : vector<16xf16>, view<128x40xf16>\n")
    # the compute block, run again for keys 16..31 on the first pass's state
    a0 = K.index("    %local_key = index.add %key_origin0, %lane_column : index\n")
    end_line = "    %next7 = vector.mma %probability, %v7, %rescaled7 : vector<16xf16>, vector<16xf16>, vector<8xf32>\n"
    p1 = K.index(end_line) + len(end_line)
    block = K[a0:p1]
    names = [n for line in re.findall(r"^\s+((?:%[A-Za-z_][A-Za-z_0-9]*)(?:, %[A-Za-z_][A-Za-z_0-9]*)*) =", block, re.M) for n in line.split(", ")]
    mapping = {n: n + "_b" for n in names}
    mapping.update({"%row_max": "%next_max", "%row_sum": "%next_sum", "%ks_carry": "%ks_carry_b"})
    mapping.update({f"%acc{c}": f"%next{c}" for c in range(8)})
    blk_b = re.sub(r"%[A-Za-z_][A-Za-z_0-9]*", lambda m: mapping.get(m[0], m[0]), block)
    blk_b = blk_b.replace("%k_tile_b[%lane_column,", "%k_tile_b[%lane_column_hi,")
    blk_b = blk_b.replace("%local_key_b = index.add %key_origin0, %lane_column : index", "%local_key_b = index.add %key_origin0, %lane_column_hi : index")
    blk_b = re.sub(r"(%v_data\d_b = vector.load %v_tile_b\[%vrow\d_b), %c0\]", r"\1, %c16]", blk_b)
    K = K[:p1] + blk_b + K[p1:]
    sub("    %ks_key_n0 = index.add %local_key, %c16 : index\n", "    %ks_key_n0 = index.add %local_key, %c32 : index\n    %ks_key_nb0 = index.add %local_key, %c48 : index\n    %ks_key_nb1 = index.min %ks_key_nb0, %ks_last : index\n    %ks_key_nb = index.assume %ks_key_nb1 [lt(%ks_key_nb1, %padded_tokens)] : index\n    %ks_next_b = view.load %ks_view[%ks_key_nb, %kv_head] : view<[%padded_tokens]x[%kv_head_limit]xf32> -> f32\n")
    sub("    scf.yield %next_max, %next_sum, %next0, %next1, %next2, %next3, %next4, %next5, %next6, %next7, %ks_next : " + ", ".join(["vector<8xf32>"] * 10) + ", f32\n",
        "    scf.yield %next_max_b, %next_sum_b, %next0_b, %next1_b, %next2_b, %next3_b, %next4_b, %next5_b, %next6_b, %next7_b, %ks_next, %ks_next_b : " + ", ".join(["vector<8xf32>"] * 10) + ", f32, f32\n")
    K = K.replace("h3_" + src_stem, "h3_" + out_stem).replace("h3." + src_stem, "h3." + out_stem)
    return K


if __name__ == "__main__":
    argv = [a for a in sys.argv[1:] if a != "--unroll"]; src_stem, out_stem = argv[0], argv[1]; pad = int(argv[2]) if len(argv) > 2 else 0
    fn = unrolled if "--unroll" in sys.argv else doubled
    src = ROOT / "h3/kernels" / f"{src_stem}.loom"
    out = ROOT / ("h3/kernels" if out_stem in ("attention_i4qk32_mha8_lds_f16_wmma",) else "experiments") / f"{out_stem}.loom"
    out.write_text(fn(src.read_text(), src_stem, out_stem, pad)); print("wrote", out)

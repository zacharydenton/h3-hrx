"""Experimental gfx11 attention: KQ^T and V^T P^T keep each query in a lane pair.

This removes the probability LDS round trip and reduces softmax communication.
Generate with python3 tools/gen_attention_i8_transposed.py.
"""
from pathlib import Path
import re
import argparse

ROOT = Path(__file__).resolve().parents[1]


def keys32(src):
    src = src.replace("attention_i8qkt_", "attention_i8qkt32_")
    src = src.replace("view<16x36xi32>", "view<32x36xi32>")
    src = src.replace("view<128x24xf16>", "view<128x40xf16>")
    src = src.replace("%kbuf_bytes = index.constant 2304", "%kbuf_bytes = index.constant 4608")
    src = src.replace("%vbuf_bytes = index.constant 6144", "%vbuf_bytes = index.constant 10240")
    src = src.replace("%v_tile_offset = index.constant 4608", "%v_tile_offset = index.constant 9216")
    src = src.replace("%lds_bytes = index.constant 16896", "%lds_bytes = index.constant 29696")
    src = src.replace("  %key_tile_count = index.add %tiles_per_image, %c0 : index", "  %c31 = index.constant 31 : index\n  %key_round = index.add %tokens0, %c31 : index\n  %key_tile_count = index.div %key_round, %c32 : index")
    src = src.replace("%key_origin1 = index.mul %key_tile, %c16", "%key_origin1 = index.mul %key_tile, %c32")
    anchor = "    %buf = index.rem"
    pos = src.index(anchor)
    src = src[:pos] + "    %key_origin_b = index.add %key_origin0, %c16 : index\n    %key_lane_b = index.add %lane_column, %c16 : index\n" + src[pos:]
    # Each staging thread loads two 16-key halves into the same wider tile.
    anchor = "      vector.store %k_chunk, %k_tile_b[%st_key, %st_chunk] : vector<4xi32>, view<32x36xi32>"
    src = src.replace(anchor, anchor + """
      %st_row_b0 = index.add %st_row, %c16 : index
      %st_row_b = index.assume %st_row_b0 [lt(%st_row_b0, %padded_tokens)] : index
      %st_key_b = index.add %st_key, %c16 : index
      %k_chunk_b = vector.load %ki_view[%st_row_b, %st_col] : view<[%padded_tokens]x[%ki_words]xi32> -> vector<4xi32>
      vector.store %k_chunk_b, %k_tile_b[%st_key_b, %st_chunk] : vector<4xi32>, view<32x36xi32>""")
    anchor = "      vector.store %v_chunk, %v_tile_b[%st_lane, %c0] : vector<16xf16>, view<128x40xf16>"
    src = src.replace(anchor, anchor + """
      %v_key_b = index.assume %key_origin_b [le(%key_origin_b, %tile_origin_limit), mul(%key_origin_b, 16)] : index
      %v_chunk_b = vector.load %v_view[%st_chan, %v_key_b] : view<[%kv_stride0]x[%padded_tokens]xf16> -> vector<16xf16>
      vector.store %v_chunk_b, %v_tile_b[%st_lane, %c16] : vector<16xf16>, view<128x40xf16>""")
    # Duplicate the score fragment, with fresh SSA names but shared Q operands.
    a = src.index("    %k_data0 =")
    b = src.index("    %local_max =", a)
    fragment = src[a:b]
    names = set(re.findall(r"^    %([\w]+) =", fragment, re.M))
    extra = re.sub(r"%([\w]+)", lambda m: "%" + m[1] + ("_b" if m[1] in names else ""), fragment)
    extra = extra.replace("%k_tile_b[%lane_column,", "%k_tile_b[%key_lane_b,")
    extra = extra.replace("index.add %key_origin0,", "index.add %key_origin_b,")
    src = src[:b] + extra + src[b:]
    src = src.replace("%local_max = vector.reduce", "%local_max_a = vector.reduce")
    anchor = "    %other_max,"
    b = src.index(anchor)
    src = src[:b] + """    %local_max_b = vector.reduce<maxnumf> %scaled_b, %negative_large : vector<8xf32>, f32
    %local_max = scalar.maxnumf %local_max_a, %local_max_b : f32
""" + src[b:]
    anchor = "    %old_delta ="
    b = src.index(anchor)
    src = src[:b] + """    %delta_b = vector.subf %scaled_b, %max_vec : vector<8xf32>
    %weight_b = vector.expf<afn> %delta_b : vector<8xf32>
""" + src[b:]
    src = src.replace("%local_sum = vector.reduce", "%local_sum_a = vector.reduce")
    b = src.index("    %scaled_sum =")
    src = src[:b] + """    %local_sum_b = vector.reduce<addf> %weight_b, %zero_f32 : vector<8xf32>, f32
    %local_sum = scalar.addf %local_sum_a, %local_sum_b : f32
""" + src[b:]
    a = src.index("    %other_weight,")
    b = src.index("    %vrow0", a)
    probability = src[a:b]
    names = set(re.findall(r"%([\w]+)", probability)) - {"weight", "i32_16", "i32_32", "lane_group_even", "k_frag", "n"}
    probability_b = re.sub(r"%([\w]+)", lambda m: "%" + m[1] + ("_b" if m[1] in names or m[1] == "weight" else ""), probability)
    src = src[:b] + probability_b + src[b:]
    for c in range(8):
        anchor = f"    %next{c} = vector.mma %v{c}, %probability, %rescaled{c} : vector<16xf16>, vector<16xf16>, vector<8xf32>"
        replacement = anchor.replace(f"%next{c} =", f"%partial{c} =") + f"""
    %v_data{c}_b = vector.load %v_tile_b[%vrow{c}, %c16] : view<128x40xf16> -> vector<16xf16>
    %v{c}_b = vector.fragment<lhs> %v_data{c}_b shape [%m, %k_frag] : vector<16xf16>
    %next{c} = vector.mma %v{c}_b, %probability_b, %partial{c} : vector<16xf16>, vector<16xf16>, vector<8xf32>"""
        assert anchor in src
        src = src.replace(anchor, replacement)
    return src


def generate(key_count=16):
    old = "attention_i8qk_mha8_lds_f16_wmma"
    stem = "attention_i8qkt_mha8_lds_f16_wmma"
    src = (ROOT / "h3/kernels" / (old + ".loom")).read_text()
    src = "// Generated by tools/gen_attention_i8_transposed.py: INT8 KQ^T, FP32 softmax, FP16 V^T P^T.\n" + src[src.index("amdgpu.target"):]
    src = src.replace(old, stem)
    src = src.replace("  %i32_32 =", "  %i32_16 = scalar.constant 16 : i32\n  %i32_32 =")
    # With transposed scores, Q is rhs and K is lhs; physical operand layouts match.
    src = re.sub(r"vector.fragment<lhs> (%qd\d) shape \[%m, %k_frag\]", r"vector.fragment<rhs> \1 shape [%k_frag, %n]", src)
    src = re.sub(r"vector.fragment<rhs> (%k_data\d) shape \[%k_frag, %n\]", r"vector.fragment<lhs> \1 shape [%m, %k_frag]", src)
    src = re.sub(r"vector.mma %lhs(\d), %rhs\1,", r"vector.mma %rhs\1, %lhs\1,", src)
    start = src.index("  // this lane's query rows")
    end = src.index("  %final_max", start)
    src = src[:start] + """  %qs_one = view.load %qs_view[%q_row, %head] : view<[%padded_tokens]x[%head_limit]xf32> -> f32
  %qs_vec = vector.splat %qs_one : vector<8xf32>
""" + src[end:]
    src = src.replace(", %ks_end = scf.for", " = scf.for")
    src = src.replace("%row_max = %negative_vector : vector<8xf32>, %row_sum = %zero_vector : vector<8xf32>", "%row_max = %negative_large : f32, %row_sum = %zero_f32 : f32")
    src = src.replace(", %ks_carry = %ks_first : f32", "")
    src = src.replace("-> (" + ", ".join(["vector<8xf32>"] * 10 + ["f32"]) + ")", "-> (" + ", ".join(["f32", "f32"] + ["vector<8xf32>"] * 8) + ")")
    start = src.index("    %ks_key =")
    end = src.index("    %vrow0", start)
    body = []
    def emit(s): body.append(s)
    for i in range(8):
        emit(f"""    %sk{i}a = index.add %key_origin0, %c{2*i} : index
    %sk{i}b = index.add %sk{i}a, %lane_group : index
    %sk{i} = index.assume %sk{i}b [lt(%sk{i}b, %padded_tokens)] : index
    %ks{i} = view.load %ks_view[%sk{i}, %kv_head] : view<[%padded_tokens]x[%kv_head_limit]xf32> -> f32
    %valid{i} = index.cmp ult, %sk{i}, %tokens0 : index
    %rs{i} = vector.extract %raw_scores[{i}] : vector<8xf32> -> f32
    %sq{i} = scalar.mulf %rs{i}, %qs_one : f32
    %ss{i} = scalar.mulf %sq{i}, %ks{i} : f32
    %ms{i} = scf.select %valid{i}, %ss{i}, %negative_large : f32""")
    emit("    %scaled = vector.from_elements " + ", ".join(f"%ms{i}" for i in range(8)) + " : vector<8xf32>")
    emit("""    %local_max = vector.reduce<maxnumf> %scaled, %negative_large : vector<8xf32>, f32
    %other_max, %max_ok = kernel.subgroup.shuffle<xor> %local_max, %i32_16, %i32_32 : f32, i32, i32
    %tile_max = scalar.maxnumf %local_max, %other_max : f32
    %next_max = scalar.maxnumf %row_max, %tile_max : f32
    %max_vec = vector.splat %next_max : vector<8xf32>
    %delta = vector.subf %scaled, %max_vec : vector<8xf32>
    %weight = vector.expf<afn> %delta : vector<8xf32>
    %old_delta = scalar.subf %row_max, %next_max : f32
    %old_scale = scalar.expf<afn> %old_delta : f32
    %selected_old_scale = vector.splat %old_scale : vector<8xf32>
    %local_sum = vector.reduce<addf> %weight, %zero_f32 : vector<8xf32>, f32
    %scaled_sum = scalar.mulf %row_sum, %old_scale : f32
    %next_sum = scalar.addf %scaled_sum, %local_sum : f32
    %other_weight, %weight_ok = kernel.subgroup.shuffle<xor> %weight, %i32_16, %i32_32 : vector<8xf32>, i32, i32
    %evens = scf.select %lane_group_even, %weight, %other_weight : vector<8xf32>
    %odds = scf.select %lane_group_even, %other_weight, %weight : vector<8xf32>
    %even16 = vector.fptrunc %evens : vector<8xf32> to vector<8xf16>
    %odd16 = vector.fptrunc %odds : vector<8xf32> to vector<8xf16>""")
    for i in range(8):
        emit(f"    %pe{i} = vector.extract %even16[{i}] : vector<8xf16> -> f16\n    %po{i} = vector.extract %odd16[{i}] : vector<8xf16> -> f16")
    emit("    %packed_p = vector.from_elements " + ", ".join(f"%pe{i}, %po{i}" for i in range(8)) + " : vector<16xf16>")
    emit("    %probability = vector.fragment<rhs> %packed_p shape [%k_frag, %n] : vector<16xf16>")
    src = src[:start] + "\n".join(body) + "\n" + src[end:]
    src = re.sub(r"vector.fragment<rhs> (%v_data\d) shape \[%k_frag, %n\]", r"vector.fragment<lhs> \1 shape [%m, %k_frag]", src)
    src = re.sub(r"vector.mma %probability, (%v\d),", r"vector.mma \1, %probability,", src)
    # Consume one V fragment at a time: keeping all eight loads and all eight
    # rescaled accumulators live simultaneously costs an extra 112 registers.
    start = src.index("    %vrow0")
    end = src.index("    %ks_key_n0", start)
    lines = src[start:end].splitlines()
    ordered = []
    for c in range(8):
        for name in (f"vrow{c}", f"v_data{c}", f"v{c}", f"rescaled{c}", f"next{c}"):
            ordered += [line for line in lines if line.startswith(f"    %{name} =")]
    src = src[:start] + "\n".join(ordered) + "\n" + src[end:]
    start = src.index("    %ks_key_n0")
    end = src.index("  %out0 =", start)
    src = src[:start] + "    scf.yield %next_max, %next_sum, " + ", ".join(f"%next{i}" for i in range(8)) + " : f32, f32, " + ", ".join(["vector<8xf32>"]*8) + "\n  }\n" + """  %sum_other, %sum_ok = kernel.subgroup.shuffle<xor> %final_sum, %i32_16, %i32_32 : f32, i32, i32
  %sum_all = scalar.addf %final_sum, %sum_other : f32
  %selected_sum = vector.splat %sum_all : vector<8xf32>
""" + src[end:]
    # Transposed accumulator columns are queries, rows are output channels.
    start = src.index("  // Publish")
    body = ["""  %row_present = index.cmp ult, %q_row, %tokens0 : index
  %row_in_range = index.cmp ult, %q_row, %token_count : index
  %writes = scalar.andi %row_present, %row_in_range : i1
  scf.if %writes {
    %global_row = index.assume %q_row [lt(%q_row, %token_count)] : index
    %out_limit = index.sub %out_stride0, %c1 : index"""]
    for c in range(8):
        for i in range(8):
            n = 16*c+2*i
            body.append(f"""    %oc{n} = index.constant {n} : index
    %col{n}a = index.add %head_base0, %oc{n} : index
    %col{n}b = index.add %col{n}a, %lane_group : index
    %col{n} = index.assume %col{n}b [le(%col{n}b, %out_limit)] : index
    %outv{n} = vector.extract %out{c}[{i}] : vector<8xf32> -> f32
    %outh{n} = scalar.fptrunc %outv{n} : f32 to f16
    view.store %outh{n}, %out_view[%global_row, %col{n}] : f16, view<[%token_count]x[%out_stride0]xf16>""")
    src = src[:start] + "\n".join(body) + "\n  }\n  kernel.return\n}\n"
    # No per-wave P scratch remains; double-buffered K and V use 16,896 bytes.
    src = src.replace("%lds_bytes = index.constant 20992", "%lds_bytes = index.constant 16896")
    # Drop unused scratch views that would lie outside the smaller allocation.
    src = re.sub(r"^  %(?:scratch_view|wave_scratch\w*|result_view|wave_result\w*) = .*\n", "", src, flags=re.M)
    if key_count == 32:
        src = keys32(src)
        stem = stem.replace("i8qkt_", "i8qkt32_")
    out = ROOT / "experiments" / (stem + ".loom")
    out.write_text(src)
    print(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keys", type=int, choices=[16, 32], default=16)
    generate(parser.parse_args().keys)

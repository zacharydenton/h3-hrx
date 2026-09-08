"""The transposed int8-QK attention kernel (attention_i8qkt_mha8): S^T = K Q^T puts one query's keys in a lane pair, so the
softmax is an in-lane reduction plus one lane-16 exchange, P^T is repacked in registers to the PV rhs (Loom's
result_f32_to_rhs_b16_permlane strategy: v_permlanex16 + cvt + pack + v_perm), and O^T accumulates V^T (lhs) times P^T.
No LDS round trip and no subgroup barrier per tile. Key scales come parity-split per 16-key block, transposed
([heads][tokens]: even keys 0..7 then odd keys 8..15 of each block), written by prepare_qk_i8t.
Derived from the shipped attention_i8qkf_mha8 (next-tile K prefetch, PV loads interleaved).
    python3 tools/gen_attention_transposed.py"""
import re
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent


def convert(K: str) -> str:
    def sub(old, new, count=1):
        nonlocal K
        n = K.count(old); assert n == count, (old[:90], n); K = K.replace(old, new)
    K = K.replace("i8qkf_mha8", "i8qkt_mha8")
    # --- Q as the rhs operand, K as the lhs operand ---
    for c in range(8):
        sub(f"  %lhs{c} = vector.fragment<lhs> %qd{c} shape [%m, %k_frag] using {{schema = %i8_schema : encoding<schema>}} : vector<4xi32>\n",
            f"  %qrhs{c} = vector.fragment<rhs> %qd{c} shape [%k_frag, %n] using {{schema = %i8_schema : encoding<schema>}} : vector<4xi32>\n")
        sub(f"    %rhs{c} = vector.fragment<rhs> %k_data{c} shape [%k_frag, %n] using {{schema = %i8_schema : encoding<schema>}} : vector<4xi32>\n",
            f"    %klhs{c} = vector.fragment<lhs> %k_data{c} shape [%m, %k_frag] using {{schema = %i8_schema : encoding<schema>}} : vector<4xi32>\n")
        out = "%raw_scores_i" if c == 7 else f"%qk{c}"; prev = "%init_i" if c == 0 else f"%qk{c - 1}"
        sub(f"    {out} = vector.mma %lhs{c}, %rhs{c}, {prev} : vector<4xi32>, vector<4xi32>, vector<8xi32>\n",
            f"    {out} = vector.mma %klhs{c}, %qrhs{c}, {prev} : vector<4xi32>, vector<4xi32>, vector<8xi32>\n")
    # --- scales: one query scale per lane, eight key scales per lane from the parity-split transposed table ---
    i = K.index("  // this lane's query rows are origin + 2e + lane_group"); j = K.index("  %qs_vec = vector.from_elements"); j = K.index("\n", j) + 1
    K = K[:i] + """  // this lane's query: origin + lane_column; its scale splat over the eight keys the lane holds
  %qs_rowa = index.add %query_origin0, %lane_column : index
  %qs_row = index.assume %qs_rowa [lt(%qs_rowa, %padded_tokens)] : index
  %qs_lane = view.load %qs_view[%qs_row, %head] : view<[%padded_tokens]x[%head_limit]xf32> -> f32
  %qs_vec = vector.splat %qs_lane : vector<8xf32>
  // key scales, transposed and parity-split: block of 16 keys = [even keys 0..7 | odd keys 8..15]
  %kst_view = buffer.view %ks_global[%c0_offset] : buffer -> view<[%kv_head_limit]x[%padded_tokens]xf32>
  %ks_par = index.mul %lane_group, %c8 : index
""" + K[j:]
    sub("  %ks_key0 = index.assume %lane_column [lt(%lane_column, %padded_tokens)] : index\n  %ks_first = view.load %ks_view[%ks_key0, %kv_head] : view<[%padded_tokens]x[%kv_head_limit]xf32> -> f32\n", "")
    sub("  %i32_32 = scalar.constant 32 : i32\n", "  %i32_32 = scalar.constant 32 : i32\n  %i32_16 = scalar.constant 16 : i32\n")
    sub("%ks_end, %k_end, %v_end = scf.for %key_tile", "%k_end, %v_end = scf.for %key_tile")
    sub("%ks_carry = %ks_first : f32, %k_pre = %p_first_k : vector<4xi32>, %v_pre = %p_first_v : vector<16xf16>) -> (", "%k_pre = %p_first_k : vector<4xi32>, %v_pre = %p_first_v : vector<16xf16>) -> (")
    K = re.sub(r"\(%row_max = %negative_vector : vector<8xf32>, %row_sum = %zero_vector : vector<8xf32>,", "(%row_max = %negative_large : f32, %row_sum = %zero_f32 : f32,", K)
    hdr_i = K.index("%k_end, %v_end = scf.for %key_tile"); hdr_j = K.index("{\n", hdr_i)
    hdr = K[hdr_i:hdr_j]
    hdr2 = hdr.replace("-> (vector<8xf32>, vector<8xf32>,", "-> (f32, f32,").replace(", f32, vector<4xi32>, vector<16xf16>)", ", vector<4xi32>, vector<16xf16>)")
    assert hdr2 != hdr; K = K[:hdr_i] + hdr2 + K[hdr_j:]
    sub("""    %ks_key = index.assume %local_key_safe [lt(%local_key_safe, %padded_tokens)] : index
    %ks_vec = vector.splat %ks_carry : vector<8xf32>
""", """    %ks_offa = index.add %key_origin0, %ks_par : index
    %ks_off = index.assume %ks_offa [le(%ks_offa, %ks_off_limit), mul(%ks_offa, 8)] : index
    %ks_vec = vector.load %kst_view[%kv_head, %ks_off] : view<[%kv_head_limit]x[%padded_tokens]xf32> -> vector<8xf32>
""")
    sub("  %ks_last = index.sub %padded_tokens, %c1 : index\n", "  %ks_off_limit = index.sub %padded_tokens, %c8 : index\n")
    K = re.sub(r"    %ks_key_n0 = index\.add %local_key, %c16 : index\n    %ks_key_n1 = index\.min %ks_key_n0, %ks_last : index\n    %ks_key_n = index\.assume %ks_key_n1 \[lt\(%ks_key_n1, %padded_tokens\)\] : index\n    %ks_next = view\.load %ks_view\[%ks_key_n, %kv_head\] : view<\[%padded_tokens\]x\[%kv_head_limit\]xf32> -> f32\n", "", K)
    assert "%ks_next" not in K.split("scf.yield %next_max")[0].split("scf.for %key_tile")[1]
    # --- mask: keys origin + 2i + lane_group past the sequence get -large ---
    sub("""    %scaled = scf.if %key_valid -> (vector<8xf32>) {
      scf.yield %scaled0 : vector<8xf32>
    } else {
      scf.yield %negative_vector : vector<8xf32>
    }
""", "".join(f"    %mk{i}a = index.add %key_origin0, %c{2 * i} : index\n    %mk{i} = index.add %mk{i}a, %lane_group : index\n    %mv{i} = index.cmp ult, %mk{i}, %tokens0 : index\n    %mm{i} = scf.select %mv{i}, %zero_f32, %negative_large : f32\n" for i in range(8)) + "    %kmask = vector.from_elements " + ", ".join(f"%mm{i}" for i in range(8)) + " : vector<8xf32>\n    %scaled = vector.addf<reassoc|nnan|ninf|nsz> %scaled0, %kmask : vector<8xf32>\n")
    # --- softmax: in-lane over the eight keys, one exchange with the partner lane ---
    i = K.index("    // Tile row max across the 16 lanes of this lane's half"); j = K.index("    // the f32 weights become an f16 lhs through this wave's 512-byte scratch")
    K = K[:i] + """    // this lane's query: the tile max over its eight keys, joined with the partner lane's (lane ^ 16)
    %m8 = vector.reduce<maxnumf> %scaled, %negative_large : vector<8xf32>, f32
    %m8s, %mok = kernel.subgroup.shuffle<xor> %m8, %i32_16, %i32_32 : f32, i32, i32
    %tile_max = scalar.maxnumf %m8, %m8s : f32
    %next_max = scalar.maxnumf %row_max, %tile_max : f32
    %next_max_v = vector.splat %next_max : vector<8xf32>
    %delta = vector.subf<reassoc|nnan|ninf|nsz> %scaled, %next_max_v : vector<8xf32>
    // an invalid key's score is -large, so its weight is exp(-large) = 0 without a mask
    %weight = vector.expf<afn> %delta : vector<8xf32>
    %old_delta = scalar.subf %row_max, %next_max : f32
    %old_scale = scalar.expf<afn> %old_delta : f32
    %selected_old_scale = vector.splat %old_scale : vector<8xf32>
    %w8 = vector.reduce<addf> %weight, %zero_f32 : vector<8xf32>, f32
    %w8s, %sok = kernel.subgroup.shuffle<xor> %w8, %i32_16, %i32_32 : f32, i32, i32
    %tile_sum = scalar.addf %w8, %w8s : f32
    %scaled_sum = scalar.mulf %row_sum, %old_scale : f32
    %next_sum = scalar.addf %scaled_sum, %tile_sum : f32
""" + K[j:]
    sub("""    // the f32 weights become an f16 lhs through this wave's 512-byte scratch
    vector.fragment.store<result> %weight, %scratch_view[%c0, %c0] shape [%m, %n] : vector<8xf32>, view<16x16xf16>
    kernel.barrier<workgroup> scope(subgroup) ordering(acq_rel)
    %probability = vector.fragment.load<lhs> %scratch_view[%c0, %c0] shape [%m, %k_frag] : view<16x16xf16> -> vector<16xf16>
""", """    // P^T as the PV rhs, repacked in registers (v_permlanex16 + cvt + pack + v_perm)
    %probability = vector.fragment.repack<rhs> %weight shape [%m, %n] : vector<8xf32> -> vector<16xf16>
""")
    # --- PV: V^T rows as the lhs, O^T accumulators ---
    for n in range(8):
        sub(f"    %v{n} = vector.fragment<rhs> %v_data{n} shape [%k_frag, %n] : vector<16xf16>\n", f"    %v{n} = vector.fragment<lhs> %v_data{n} shape [%m, %k_frag] : vector<16xf16>\n")
        sub(f"    %next{n} = vector.mma %probability, %v{n}, %rescaled{n} : vector<16xf16>, vector<16xf16>, vector<8xf32>\n", f"    %next{n} = vector.mma %v{n}, %probability, %rescaled{n} : vector<16xf16>, vector<16xf16>, vector<8xf32>\n")
    # yield: scalar max/sum, no key-scale carry
    i = K.index("    scf.yield %next_max, %next_sum,"); j = K.index("\n", i); line = K[i:j]
    line2 = line.replace(", %ks_next, %k_next, %v_next : vector<8xf32>, vector<8xf32>,", ", %k_next, %v_next : f32, f32,").replace(", f32, vector<4xi32>, vector<16xf16>", ", vector<4xi32>, vector<16xf16>")
    assert line2 != line, line; K = K[:i] + line2 + K[j:]
    # --- after the loop: the row sum is already this lane's query's sum ---
    i = K.index("  // the row sum: the per-lane partials of the 16 lanes of this half"); j = K.index("  %selected_sum = vector.addf<reassoc|nnan|ninf|nsz> %fs8, %zero_vector : vector<8xf32>\n")
    K = K[:i] + "  %selected_sum = vector.splat %final_sum : vector<8xf32>\n" + K[j + len("  %selected_sum = vector.addf<reassoc|nnan|ninf|nsz> %fs8, %zero_vector : vector<8xf32>\n"):]
    # --- epilogue: the staged tile is [channel][query]; each lane publishes four queries of one channel ---
    sub("""      scf.if %writes {
        %global_row = index.assume %local_row [lt(%local_row, %token_count)] : index
        %values = vector.load %result_view[%publish_row, %publish_col] : view<16x16xf32> -> vector<4xf32>
        %halves = vector.fptrunc %values : vector<4xf32> to vector<4xf16>
        vector.store %halves, %out_view[%global_row, %out_column] : vector<4xf16>, view<[%token_count]x[%out_stride0]xf16>
      }
""", """      %values = vector.load %result_view[%publish_row, %publish_col] : view<16x16xf32> -> vector<4xf32>
      %halves = vector.fptrunc %values : vector<4xf32> to vector<4xf16>
""" + "".join(f"""      %pe{jj} = vector.extract %halves[{jj}] : vector<4xf16> -> f16
      %pq{jj}a = index.add %query_origin0, %publish_col : index
      %pq{jj} = index.add %pq{jj}a, %c{jj} : index
      %pp{jj} = index.cmp ult, %pq{jj}, %tokens0 : index
      %pr{jj} = index.cmp ult, %pq{jj}, %token_count : index
      %pw{jj} = scalar.andi %pp{jj}, %pr{jj} : i1
      scf.if %pw{jj} {{
        %pg{jj} = index.assume %pq{jj} [lt(%pq{jj}, %token_count)] : index
        view.store %pe{jj}, %out_view[%pg{jj}, %out_column_c] : f16, view<[%token_count]x[%out_stride0]xf16>
      }}
""" for jj in range(4)))
    # the channel column: fragment base + this lane's stage row; the query validity moves into the per-element guards
    sub("""      %row_offset = index.mul %half, %c8 : index
      %publish_row = index.add %publish_row0, %row_offset : index
      %local_row = index.add %query_origin0, %publish_row : index
      %present = index.cmp ult, %local_row, %tokens0 : index
      %in_range = index.cmp ult, %local_row, %token_count : index
      %writes = scalar.andi %present, %in_range : i1
""", """      %row_offset = index.mul %half, %c8 : index
      %publish_row = index.add %publish_row0, %row_offset : index
      %out_column_ca = index.add %out_column_base, %publish_row : index
      %out_column_c = index.assume %out_column_ca [lt(%out_column_ca, %out_stride0)] : index
""")
    sub("""    %frag_col = index.mul %fragment, %c16 : index
    %out_col_local = index.add %frag_col, %publish_col : index
    %out_column0 = index.add %head_base0, %out_col_local : index
    %out_column = index.assume %out_column0 [le(%out_column0, %publish_limit)] : index
""", """    %frag_col = index.mul %fragment, %c16 : index
    %out_column_base = index.add %head_base0, %frag_col : index
""")
    K = K.replace("  %publish_limit = index.sub %out_stride0, %c4 : index\n", "")
    return K


def drop_v_placeholder(K: str) -> str:
    """The K-only prefetch form still carries a zero vector<16xf16> V packet through the loop; yield only the K packet."""
    def sub(a, b, n=None):
        nonlocal K
        assert K.count(a) >= 1, (a, K.count(a))
        if n is not None: assert K.count(a) == n, (a, K.count(a))
        K = K.replace(a, b)
    sub("%p_first_k, %p_first_v = scf.if %st_is_k -> (vector<4xi32>, vector<16xf16>) {", "%p_first_k = scf.if %st_is_k -> (vector<4xi32>) {", 1)
    sub("%k_next, %v_next = scf.if %st_is_k -> (vector<4xi32>, vector<16xf16>) {", "%k_next = scf.if %st_is_k -> (vector<4xi32>) {", 1)
    sub("scf.yield %r, %zero_v16 : vector<4xi32>, vector<16xf16>", "scf.yield %r : vector<4xi32>", 2)
    sub("scf.yield %zero_k4, %zero_v16 : vector<4xi32>, vector<16xf16>", "scf.yield %zero_k4 : vector<4xi32>", 1)
    sub("    %r = vector.load %v_view[%st_chan, %c0] : view<[%kv_stride0]x[%padded_tokens]xf16> -> vector<16xf16>\n    scf.yield %zero_k4, %r : vector<4xi32>, vector<16xf16>\n", "    scf.yield %zero_k4 : vector<4xi32>\n", 1)
    sub("%k_end, %v_end = scf.for %key_tile", "%k_end = scf.for %key_tile", 1)
    sub(", %v_pre = %p_first_v : vector<16xf16>) -> (", ") -> (", 1)
    sub(", vector<4xi32>, vector<16xf16>) {", ", vector<4xi32>) {", 1)
    sub(", %k_next, %v_next : ", ", %k_next : ", 1)
    i = K.index("scf.yield %next_max"); j = K.index("\n", i)
    line = K[i:j]; assert line.endswith(", vector<4xi32>, vector<16xf16>"), line[-60:]
    K = K[:i] + line[:-len(", vector<16xf16>")] + K[j:]
    K = K.replace("  %zero_v16 = vector.constant 0.0 : vector<16xf16>\n", "")
    assert "%v_pre" not in K and "%zero_v16" not in K and "%p_first_v" not in K
    return K


def prepare(K: str) -> str:
    K = K.replace("prepare_qk_i8", "prepare_qk_i8t")
    # the scale plane is [heads][token_capacity]: the pitch is the attention kernel's capacity, not the token count
    K = K.replace("config.decl @h3.prepare_qk_i8t.extra_scale : f32\n", "config.decl @h3.prepare_qk_i8t.extra_scale : f32\n\nconfig.decl @h3.prepare_qk_i8t.token_capacity : %value: index where [range(%value, 16, 1048576), mul(%value, 16)]\n")
    K = K.replace("  %extra = config.get @h3.prepare_qk_i8t.extra_scale : f32\n", "  %extra = config.get @h3.prepare_qk_i8t.extra_scale : f32\n  %scap = config.get @h3.prepare_qk_i8t.token_capacity : index\n")
    old = "  %s_view = buffer.view %s_global[%c0_offset] : buffer -> view<[%tokens_b]x[%heads]xf32>\n"
    assert K.count(old) == 1
    K = K.replace(old, "  %s_view = buffer.view %s_global[%c0_offset] : buffer -> view<[%heads]x[%scap]xf32>\n  // transposed, parity-split per 16-token block: [even tokens 0..7 | odd tokens 8..15]\n  %blk = index.div %token, %c16 : index\n  %blk16 = index.mul %blk, %c16 : index\n  %in_blk = index.rem %token, %c16 : index\n  %par = index.rem %in_blk, %c2 : index\n  %par8 = index.mul %par, %c8 : index\n  %pos = index.div %in_blk, %c2 : index\n  %sidx0 = index.add %blk16, %par8 : index\n  %sidx1 = index.add %sidx0, %pos : index\n  %sidx = index.assume %sidx1 [lt(%sidx1, %scap)] : index\n")
    old2 = "        view.store %scale, %s_view[%token, %head] : f32, view<[%tokens_b]x[%heads]xf32>\n"
    assert K.count(old2) == 1
    return K.replace(old2, "        view.store %scale, %s_view[%head, %sidx] : f32, view<[%heads]x[%scap]xf32>\n")


def main():
    src = (ROOT / "h3/kernels/attention_i8qkf_mha8_lds_f16_wmma.loom").read_text()
    (ROOT / "h3/kernels/attention_i8qkt_mha8_lds_f16_wmma.loom").write_text(drop_v_placeholder(convert(src))); print("wrote attention_i8qkt_mha8")
    # the shipped form minus the zero V packet it still carries: experiments/, timed against the shipped kernel
    (ROOT / "experiments/attention_i8qkfz_mha8_lds_f16_wmma.loom").write_text(drop_v_placeholder(src).replace("attention_i8qkf_mha8", "attention_i8qkfz_mha8")); print("wrote attention_i8qkfz_mha8")
    (ROOT / "h3/kernels/prepare_qk_i8t.loom").write_text(prepare((ROOT / "h3/kernels/prepare_qk_i8.loom").read_text())); print("wrote prepare_qk_i8t")


if __name__ == "__main__":
    main()

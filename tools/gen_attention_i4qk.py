"""kernels/attention_i4qk_*.loom: the MHA attention kernel with QK^T in int4 WMMA (SageAttention-style
operands from prepare_qk_i4: per-token, per-head int4 codes and scales, the head rotated, K mean-smoothed)
and PV in f16 as before. Derived from gen_attention_lds.py by substitution: Q lives in registers as eight
int4 fragments (16 VGPRs instead of 64), a K tile is 16 keys x 16 words staged into LDS, the i32 scores
become f32 through the row's Q scale (attention scale and the Hadamard's 1/128 folded in) and the key's K scale.
    ATTN_WAVES=4|8 python3 tools/gen_attention_i4qk.py"""
import os, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
WAVES = int(os.environ.get("ATTN_WAVES", "8"))
STEM = os.environ.get("ATTN_STEM", {8: "attention_i4qk_mha8_lds_f16_wmma", 4: "attention_i4qk_mha_lds_f16_wmma", 16: "attention_i4qk_mha16_lds_f16_wmma"}[WAVES])
OUT = ROOT / ("kernels" if STEM in ("attention_i4qk_mha8_lds_f16_wmma", "attention_i4qk_mha_lds_f16_wmma") else "experiments") / f"{STEM}.loom"

# run the f16 generator in-process with every Q fragment hoisted (no Q LDS) and capture its text
env = {"ATTN_WAVES": str(WAVES), "ATTN_HOIST": "8", "ATTN_QLDS": "0", "ATTN_D": "128", "ATTN_GQA": "1", "ATTN_TILE": "16", "ATTN_CAUSAL": "0",
       "ATTN_STEM": "attention_i4qk_scratch"}
src = (ROOT / "tools/gen_attention_lds.py").read_text()
os.environ.update(env)
ns = {"__name__": "gen", "__file__": str(ROOT / "tools/gen_attention_lds.py")}
src = src.replace("OUT.write_text(K)\nprint(\"wrote\", OUT)", "")
exec(compile(src, "gen_attention_lds.py", "exec"), ns)
K = ns["K"]; SYM = "h3_" + STEM; NS = "h3." + STEM
K = K.replace("h3_attention_i4qk_scratch", SYM).replace("h3.attention_i4qk_scratch", NS)

def sub(old, new, count=1):
    global K
    assert K.count(old) == count, (old[:80], K.count(old))
    K = K.replace(old, new)

# --- header, launch, views ---
sub("} launch(%token_count: index, %q: buffer, %k: buffer, %v: buffer, %out: buffer) {",
    "} launch(%token_count: index, %qi: buffer, %qs: buffer, %ki: buffer, %ks: buffer, %v: buffer, %out: buffer) {")
sub("""  %q_global = buffer.assume.memory_space<global> %q : buffer
  %k_global = buffer.assume.memory_space<global> %k : buffer
  %v_global = buffer.assume.memory_space<global> %v : buffer
  %out_global = buffer.assume.memory_space<global> %out : buffer
  %q_view = buffer.view %q_global[%c0_offset] : buffer -> view<[%padded_tokens]x[%q_stride0]xf16>
  %k_view = buffer.view %k_global[%c0_offset] : buffer -> view<[%padded_tokens]x[%kv_stride0]xf16>
""", """  %qi_global = buffer.assume.memory_space<global> %qi : buffer
  %qs_global = buffer.assume.memory_space<global> %qs : buffer
  %ki_global = buffer.assume.memory_space<global> %ki : buffer
  %ks_global = buffer.assume.memory_space<global> %ks : buffer
  %v_global = buffer.assume.memory_space<global> %v : buffer
  %out_global = buffer.assume.memory_space<global> %out : buffer
  // int4 operands: [tokens][heads * 16] i32 words (8 codes each), scales [tokens][heads] f32
  %qi_words = index.div %q_stride0, %c8 : index
  %ki_words = index.div %kv_stride0, %c8 : index
  %qi_view = buffer.view %qi_global[%c0_offset] : buffer -> view<[%padded_tokens]x[%qi_words]xi32>
  %ki_view = buffer.view %ki_global[%c0_offset] : buffer -> view<[%padded_tokens]x[%ki_words]xi32>
  %qs_view = buffer.view %qs_global[%c0_offset] : buffer -> view<[%padded_tokens]x[%head_limit]xf32>
  %ks_view = buffer.view %ks_global[%c0_offset] : buffer -> view<[%padded_tokens]x[%kv_head_limit]xf32>
  %i4_schema = encoding.define #encoding.operand<element_format=i4, payload_elements=16, payload_registers=2> : encoding<schema>
  %zero_i32x8 = vector.constant 0 : vector<8xi32>
  %init_i = vector.fragment<init> %zero_i32x8 shape [%m, %n] : vector<8xi32>
""")
# LDS: the K tile becomes 16 x 20 i32 words (1280 B); the result stage (1 KB per wave) needs at least WAVES KB
K = re.sub(r"  %v_tile_offset = index.constant \d+ : offset\n", "  %v_tile_offset = index.constant 1280 : offset\n", K)
m = re.search(r"  %scratch_offset = index.constant (\d+) : offset\n", K); old_scratch = int(m.group(1))
v_bytes = 128 * (16 + 8) * 2
K = re.sub(r"  %scratch_offset = index.constant \d+ : offset\n", f"  %scratch_offset = index.constant {1280 + v_bytes} : offset\n", K)
m = re.search(r"  %q_tile_offset = index.constant (\d+) : offset\n", K)
K = re.sub(r"  %q_tile_offset = index.constant \d+ : offset\n", f"  %q_tile_offset = index.constant {1280 + v_bytes + WAVES * 512} : offset\n", K)
K = re.sub(r"  %lds_bytes = index.constant \d+ : offset\n", f"  %lds_bytes = index.constant {max(1280 + v_bytes + WAVES * 512, WAVES * 1024)} : offset\n", K)
K = re.sub(r"  %k_tile = buffer.view %lds\[%k_tile_offset\] : buffer -> view<16x\d+xf16>\n", "  %k_tile = buffer.view %lds[%k_tile_offset] : buffer -> view<16x20xi32>\n", K)
# --- Q: eight int4 fragments from this lane's row, hoisted; the row's Q scales as an 8-vector ---
q_hoist = ""
for c in range(8):
    q_hoist += (f"  %q_channel{c} = index.add %head_base0, %c{16*c} : index\n" if c else "  %q_channel0 = index.add %head_base0, %c0 : index\n")
    q_hoist += f"  %lhs{c} = vector.fragment.load<lhs> %q_view[%query_origin0, %q_channel{c}] shape [%m, %k_frag] : view<[%padded_tokens]x[%q_stride0]xf16> -> vector<16xf16>\n"
assert q_hoist in K
new_hoist = """  %q_row0 = index.add %query_origin0, %lane_column : index
  %q_row = index.assume %q_row0 [lt(%q_row0, %padded_tokens)] : index
  %q_wbase = index.mul %head, %c16 : index
  %q_word_limit = index.sub %qi_words, %c2 : index
"""
for c in range(8):
    new_hoist += f"  %q_word{c}a = index.add %q_wbase, %c{2*c} : index\n  %q_word{c} = index.assume %q_word{c}a [le(%q_word{c}a, %q_word_limit), mul(%q_word{c}a, 2)] : index\n"
    new_hoist += f"  %qd{c} = vector.load %qi_view[%q_row, %q_word{c}] : view<[%padded_tokens]x[%qi_words]xi32> -> vector<2xi32>\n"
    new_hoist += f"  %lhs{c} = vector.fragment<lhs> %qd{c} shape [%m, %k_frag] using {{schema = %i4_schema : encoding<schema>}} : vector<2xi32>\n"
new_hoist += "  // this lane's query rows are origin + 2e + lane_group: their Q scales (attention scale and 1/128 folded in)\n"
for e in range(8):
    new_hoist += f"  %qs_row{e}a = index.add %query_origin0, %c{2*e} : index\n  %qs_row{e}b = index.add %qs_row{e}a, %lane_group : index\n  %qs_row{e} = index.assume %qs_row{e}b [lt(%qs_row{e}b, %padded_tokens)] : index\n"
    new_hoist += f"  %qs{e} = view.load %qs_view[%qs_row{e}, %head] : view<[%padded_tokens]x[%head_limit]xf32> -> f32\n"
new_hoist += "  %qs_vec = vector.from_elements " + ", ".join(f"%qs{e}" for e in range(8)) + " : vector<8xf32>\n"
K = K.replace(q_hoist, new_hoist)
# --- K staging: 16 keys x 16 words; lane -> key = st_lane / 8, word pair = (st_lane % 8) * 2 ---
sub("""  %st_lane = index.rem %workitem, %c_klanes : index
  %st_key = index.div %st_lane, %c_nf : index
  %st_chunk0 = index.rem %st_lane, %c_nf : index
  %st_chunk = index.mul %st_chunk0, %c16 : index
  %st_col0 = index.add %kv_base0, %st_chunk : index
  %kv_col_limit = index.sub %kv_stride0, %c16 : index
  %q_col_limit = index.sub %q_stride0, %c16 : index
  %st_col = index.assume %st_col0 [le(%st_col0, %kv_col_limit), mul(%st_col0, 16)] : index
""", """  %st_lane = index.rem %workitem, %c_klanes : index
  %st_key = index.div %st_lane, %c_nf : index
  %st_chunk0 = index.rem %st_lane, %c_nf : index
  %st_chunk = index.mul %st_chunk0, %c2 : index
  %kv_word0 = index.mul %kv_head, %c16 : index
  %st_col0 = index.add %kv_word0, %st_chunk : index
  %kv_col_limit = index.sub %kv_stride0, %c16 : index
  %ki_col_limit = index.sub %ki_words, %c2 : index
  %st_col = index.assume %st_col0 [le(%st_col0, %ki_col_limit), mul(%st_col0, 2)] : index
""")
sub("""    %k_chunk = vector.load %k_view[%st_row, %st_col] : view<[%padded_tokens]x[%kv_stride0]xf16> -> vector<16xf16>
""", """    %k_chunk = vector.load %ki_view[%st_row, %st_col] : view<[%padded_tokens]x[%ki_words]xi32> -> vector<2xi32>
""")
K = re.sub(r"    vector.store %k_chunk, %k_tile\[%st_key, %st_chunk\] : vector<16xf16>, view<16x\d+xf16>\n",
           "    vector.store %k_chunk, %k_tile[%st_key, %st_chunk] : vector<2xi32>, view<16x20xi32>\n", K)
# --- scores: int4 MMAs into i32, then f32 through the two scales ---
prev = "%init"; old_scores = ""
for c in range(8):
    old_scores += f"    %k_data{c} = vector.load %k_tile[%lane_column, %c{16*c}] : view<16x136xf16> -> vector<16xf16>\n" if c else "    %k_data0 = vector.load %k_tile[%lane_column, %c0] : view<16x136xf16> -> vector<16xf16>\n"
    old_scores += f"    %rhs{c} = vector.fragment<rhs> %k_data{c} shape [%k_frag, %n] : vector<16xf16>\n"
    out = "%raw_scores" if c == 7 else f"%qk{c}"
    old_scores += f"    {out} = vector.mma %lhs{c}, %rhs{c}, {prev} : vector<16xf16>, vector<16xf16>, vector<8xf32>\n"
    prev = out
if old_scores not in K:
    m = re.search(r"view<16x(\d+)xf16> -> vector<16xf16>\n    %rhs0", K); row = m.group(1)
    old_scores = old_scores.replace("view<16x136xf16>", f"view<16x{row}xf16>")
assert old_scores in K, "score block"
prev = "%init_i"; new_scores = ""
for c in range(8):
    new_scores += f"    %k_data{c} = vector.load %k_tile[%lane_column, %c{2*c}] : view<16x20xi32> -> vector<2xi32>\n"
    new_scores += f"    %rhs{c} = vector.fragment<rhs> %k_data{c} shape [%k_frag, %n] using {{schema = %i4_schema : encoding<schema>}} : vector<2xi32>\n"
    out = "%raw_scores_i" if c == 7 else f"%qk{c}"
    new_scores += f"    {out} = vector.mma %lhs{c}, %rhs{c}, {prev} : vector<2xi32>, vector<2xi32>, vector<8xi32>\n"
    prev = out
new_scores += """    %raw_scores = vector.sitofp %raw_scores_i : vector<8xi32> to vector<8xf32>
    %ks_key = index.assume %local_key_safe [lt(%local_key_safe, %padded_tokens)] : index
    %ks_col = view.load %ks_view[%ks_key, %kv_head] : view<[%padded_tokens]x[%kv_head_limit]xf32> -> f32
    %ks_vec = vector.splat %ks_col : vector<8xf32>
    %qk_scaled = vector.mulf<reassoc|nnan|ninf|nsz|contract> %raw_scores, %qs_vec : vector<8xf32>
"""
K = K.replace(old_scores, new_scores)
# --- V transposed in global memory ([kv_stride][padded_tokens]): staging is one 32-byte load + one 32-byte store per V lane
sub("  %v_view = buffer.view %v_global[%c0_offset] : buffer -> view<[%padded_tokens]x[%kv_stride0]xf16>\n",
    "  %v_view = buffer.view %v_global[%c0_offset] : buffer -> view<[%kv_stride0]x[%padded_tokens]xf16>\n")
sub("""  %st_key_v = index.rem %st_lane, %c16 : index
""", """  %st_key_v = index.rem %st_lane, %c16 : index
  %st_chan_v = index.add %kv_base0, %st_lane : index
  %st_chan_limit = index.sub %kv_stride0, %c1 : index
  %st_chan = index.assume %st_chan_v [le(%st_chan_v, %st_chan_limit)] : index
""")
K = re.sub(r"( *)%st_row_v0 = index.add %key_origin0, %st_key_v : index\n *%st_row_v = index.assume %st_row_v0 \[lt\(%st_row_v0, %padded_tokens\)\] : index\n *%v_chunk = vector.load %v_view\[%st_row_v, %st_col_v\] : view<\[%padded_tokens\]x\[%kv_stride0\]xf16> -> vector<16xf16>\n",
           r"\1%v_chunk = vector.load %v_view[%st_chan, %key_origin0] : view<[%kv_stride0]x[%padded_tokens]xf16> -> vector<16xf16>\n", K)
n_stores = len(re.findall(r" *%ve\d+ = vector.extract %v_chunk\[\d+\] : vector<16xf16> -> f16\n *%vr\d+ = index.add %st_chunk_v, %cj\d+ : index\n *view.store %ve\d+, %v_tile\[%vr\d+, %st_key_v\] : f16, view<128x24xf16>\n", K))
assert n_stores == 16, n_stores
K = re.sub(r"( *)%ve15 = vector.extract %v_chunk\[15\] : vector<16xf16> -> f16\n *%vr15 = index.add %st_chunk_v, %cj15 : index\n *view.store %ve15, %v_tile\[%vr15, %st_key_v\] : f16, view<128x24xf16>\n",
           r"\1vector.store %v_chunk, %v_tile[%st_lane, %c0] : vector<16xf16>, view<128x24xf16>\n", K)
K = re.sub(r" *%ve\d+ = vector.extract %v_chunk\[\d+\] : vector<16xf16> -> f16\n *%vr\d+ = index.add %st_chunk_v, %cj\d+ : index\n *view.store %ve\d+, %v_tile\[%vr\d+, %st_key_v\] : f16, view<128x24xf16>\n", "", K)
sub("    %scaled0 = vector.mulf<reassoc|nnan|ninf|nsz|contract> %raw_scores, %scale_vector : vector<8xf32>\n",
    "    %scaled0 = vector.mulf<reassoc|nnan|ninf|nsz|contract> %qk_scaled, %ks_vec : vector<8xf32>\n")
sub("""    %local_key = index.add %key_origin0, %lane_column : index
    %key_valid = index.cmp ult, %local_key, %tokens0 : index
""", """    %local_key = index.add %key_origin0, %lane_column : index
    %key_valid = index.cmp ult, %local_key, %tokens0 : index
    %local_key_safe = index.min %local_key, %tile_origin_limit : index
""")
K = K.replace("// GENERATED by tools/gen_attention_lds.py (h3); edit the generator.", "// GENERATED by tools/gen_attention_i4qk.py (h3): QK^T in int4 WMMA on prepare_qk_i4 operands, PV in f16; edit the generator.")
K = K.replace("// GQA attention for Krea 2 with K/V tiles staged in LDS", "// MHA attention with int4 QK^T (SageAttention-style operands) and K/V tiles staged in LDS")
if not os.environ.get("ATTN_DBUF_STEM"):          # a named variant leaves the shipped stem alone
    OUT.write_text(K); print("wrote", OUT)
else:
    OUT = ROOT / "experiments" / "_variant_base.loom"; OUT.write_text(K)


# --- ATTN_DBUF=1: double-buffered K/V tiles (no extra registers): the trailing barrier goes; the buffer alternates per tile
if os.environ.get("ATTN_DBUF", "1") == "1":
    K = OUT.read_text()
    KBUF, VBUF = 1280, 128 * 24 * 2
    K = K.replace("  %v_tile_offset = index.constant 1280 : offset\n", f"  %v_tile_offset = index.constant {2 * KBUF} : offset\n")
    K = re.sub(r"  %scratch_offset = index.constant \d+ : offset\n", f"  %scratch_offset = index.constant {2 * KBUF + 2 * VBUF} : offset\n  %kbuf_bytes = index.constant {KBUF} : offset\n  %vbuf_bytes = index.constant {VBUF} : offset\n", K)
    K = re.sub(r"  %q_tile_offset = index.constant \d+ : offset\n", f"  %q_tile_offset = index.constant {2 * KBUF + 2 * VBUF + WAVES * 512} : offset\n", K)
    LDS_PAD = int(os.environ.get("ATTN_LDS_PAD", "0"))       # extra bytes requested, to cap workgroups per CU (measured: fewer concurrent K/V streams help)
    K = re.sub(r"  %lds_bytes = index.constant \d+ : offset\n", f"  %lds_bytes = index.constant {max(2 * KBUF + 2 * VBUF + WAVES * 512, WAVES * 1024) + LDS_PAD} : offset\n", K)
    body_start = K.index("    // stage K and V tiles")
    K = K[:body_start] + """    %buf = index.rem %key_tile, %c2 : index
    %kb_off = index.scale %buf, %kbuf_bytes : index, offset -> offset
    %k_tile_b = buffer.view %lds[%kb_off] : buffer -> view<16x20xi32>
    %vb_off0 = index.scale %buf, %vbuf_bytes : index, offset -> offset
    %vb_off = index.add %v_tile_offset, %vb_off0 : offset
    %v_tile_b = buffer.view %lds[%vb_off] : buffer -> view<128x24xf16>
""" + K[body_start:]
    loop_end = K.index("  }\n  // the row sum", body_start)
    body = K[body_start:loop_end].replace("%k_tile[", "%k_tile_b[").replace("%v_tile[", "%v_tile_b[")
    body = body.replace("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)\n    scf.yield %next_max, %next_sum, ", "    scf.yield %next_max, %next_sum, ")
    K = K[:body_start] + body + "  }\n  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)\n" + K[loop_end + len("  }\n"):]
    STEM3 = os.environ.get("ATTN_DBUF_STEM", STEM)
    if STEM3 != STEM: K = K.replace(SYM, "h3_" + STEM3).replace(NS, "h3." + STEM3)
    OUT3 = (ROOT / "kernels" / f"{STEM3}.loom") if STEM3 == STEM or STEM3 == "attention_i4qkl_mha8_lds_f16_wmma" else (ROOT / "experiments" / f"{STEM3}.loom")
    OUT3.write_text(K); print("wrote", OUT3, "(double-buffered)")

# --- ATTN_PREFETCH=1: double-buffered K/V tiles, the next tile's global loads issued before this tile's
# compute (their latency overlaps the MMAs and softmax), one workgroup barrier per tile instead of two.
if os.environ.get("ATTN_PREFETCH", "0") == "1" and WAVES == 8:
    K = OUT.read_text()
    def sub2(old, new, count=1):
        global K
        assert K.count(old) == count, (old[:80], K.count(old))
        K = K.replace(old, new)
    KBUF, VBUF = 1280, 128 * 24 * 2
    sub2("  %v_tile_offset = index.constant 1280 : offset\n", "  %v_tile_offset = index.constant 2560 : offset\n  %kbuf_bytes = index.constant 1280 : offset\n  %vbuf_bytes = index.constant 6144 : offset\n")
    K = re.sub(r"  %scratch_offset = index.constant \d+ : offset\n", f"  %scratch_offset = index.constant {2 * KBUF + 2 * VBUF} : offset\n", K)
    K = re.sub(r"  %q_tile_offset = index.constant \d+ : offset\n", f"  %q_tile_offset = index.constant {2 * KBUF + 2 * VBUF + 8 * 512} : offset\n", K)
    K = re.sub(r"  %lds_bytes = index.constant \d+ : offset\n", f"  %lds_bytes = index.constant {2 * KBUF + 2 * VBUF + 8 * 512} : offset\n", K)
    # the staging block of the loop body: split loads from stores
    body_start = K.index("    scf.if %st_is_k {\n")
    body_end = K.index("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)\n", body_start) + len("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)\n")
    staging = K[body_start:body_end]
    k_branch = staging[staging.index("    scf.if %st_is_k {\n") + len("    scf.if %st_is_k {\n"):staging.index("    } else {\n")]
    v_branch = staging[staging.index("    } else {\n") + len("    } else {\n"):staging.rindex("    }\n")]
    k_load = "".join(l + "\n" for l in k_branch.splitlines() if "vector.store" not in l)
    k_store = "".join(l + "\n" for l in k_branch.splitlines() if "vector.store" in l)
    v_load = "".join(l + "\n" for l in v_branch.splitlines() if "%v_chunk = " in l or "%st_row_v" in l)
    v_store = "".join(l + "\n" for l in v_branch.splitlines() if not ("%v_chunk = " in l or "%st_row_v" in l))
    def loads_for(origin_expr: str, tag: str) -> str:
        kl = k_load.replace("%key_origin0", origin_expr).replace("%k_chunk", f"%k_{tag}").replace("%st_row0", f"%st_row0_{tag}").replace("%st_row ", f"%st_row_{tag} ").replace("%st_row]", f"%st_row_{tag}]").replace("[%st_row,", f"[%st_row_{tag},")
        vl = v_load.replace("%key_origin0", origin_expr).replace("%v_chunk", f"%v_{tag}").replace("%st_row_v0", f"%st_row_v0_{tag}").replace("%st_row_v ", f"%st_row_v_{tag} ").replace("[%st_row_v,", f"[%st_row_v_{tag},")
        return (f"    %k_{tag} = scf.if %st_is_k -> (vector<2xi32>) {{\n" + "".join("  " + l + "\n" for l in kl.splitlines()) + f"      scf.yield %k_{tag} : vector<2xi32>\n    }} else {{\n      scf.yield %zero_k : vector<2xi32>\n    }}\n"
                f"    %v_{tag} = scf.if %st_is_k -> (vector<16xf16>) {{\n      scf.yield %zero_v : vector<16xf16>\n    }} else {{\n" + "".join("  " + l + "\n" for l in vl.splitlines()) + f"      scf.yield %v_{tag} : vector<16xf16>\n    }}\n")
    # inside the scf.if branches the loaded value must have a distinct inner name: rename the inner definitions
    def fix_inner(text: str, tag: str) -> str:
        return text.replace(f"      %k_{tag} = vector.load", f"      %k_{tag}_in = vector.load").replace(f"      scf.yield %k_{tag} :", f"      scf.yield %k_{tag}_in :") \
                   .replace(f"      %v_{tag} = vector.load", f"      %v_{tag}_in = vector.load").replace(f"      scf.yield %v_{tag} :", f"      scf.yield %v_{tag}_in :")
    prologue = fix_inner(loads_for("%c0", "first"), "first").replace("    ", "  ", 1)
    prologue = "".join(l[2:] + "\n" if l.startswith("    ") else l + "\n" for l in prologue.splitlines())
    next_loads = fix_inner(loads_for("%key_next", "next"), "next")
    stores = ("    %buf = index.rem %key_tile, %c2 : index\n    %kb_off = index.scale %buf, %kbuf_bytes : index, offset -> offset\n    %k_tile_b = buffer.view %lds[%kb_off] : buffer -> view<16x20xi32>\n"
              "    %vb_off0 = index.scale %buf, %vbuf_bytes : index, offset -> offset\n    %vb_off = index.add %v_tile_offset, %vb_off0 : offset\n    %v_tile_b = buffer.view %lds[%vb_off] : buffer -> view<128x24xf16>\n"
              "    scf.if %st_is_k {\n" + "".join("  " + l + "\n" for l in k_store.replace("%k_chunk", "%k_pre").replace("%k_tile[", "%k_tile_b[").splitlines())
              + "    } else {\n" + "".join("  " + l + "\n" for l in v_store.replace("%v_chunk", "%v_pre").replace("%v_tile[", "%v_tile_b[").splitlines()) + "    }\n"
              "    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)\n"
              "    %key_next0 = index.add %key_origin0, %c16 : index\n    %key_next1 = index.min %key_next0, %tile_origin_limit : index\n    %key_next = index.assume %key_next1 [le(%key_next1, %tile_origin_limit), mul(%key_next1, 16)] : index\n" + next_loads)
    K = K[:body_start] + stores + K[body_end:]
    # the compute reads this iteration's buffers
    compute_start = body_start + len(stores); loop_end = K.index("  }\n  // the row sum", compute_start)
    compute = K[compute_start:loop_end].replace("%k_tile[", "%k_tile_b[").replace("%v_tile[", "%v_tile_b[")
    compute = compute.replace("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)\n    scf.yield %next_max, %next_sum, ", "    scf.yield %next_max, %next_sum, ")
    compute = compute.replace("%next7 : " + ", ".join(["vector<8xf32>"] * 10), "%next7, %k_next, %v_next : " + ", ".join(["vector<8xf32>"] * 10) + ", vector<2xi32>, vector<16xf16>")
    K = K[:compute_start] + compute + "  }\n  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)\n" + K[loop_end + len("  }\n"):]
    # loop header: carried prefetch registers; prologue loads before the loop; zero constants
    hdr = re.search(r"  (%final_max, %final_sum, .*?) = scf\.for %key_tile = \[%c0 to %key_tile_count step %c1\]\((.*?)\) -> \((.*?)\) \{\n", K)
    assert hdr, "loop header"
    new_hdr = f"  {hdr.group(1)}, %k_last, %v_last = scf.for %key_tile = [%c0 to %key_tile_count step %c1]({hdr.group(2)}, %k_pre = %k_first : vector<2xi32>, %v_pre = %v_first : vector<16xf16>) -> ({hdr.group(3)}, vector<2xi32>, vector<16xf16>) {{\n"
    K = K.replace(hdr.group(0), prologue + new_hdr)
    sub2("  %zero_acc = vector.constant 0.0 : vector<8xf32>\n", "  %zero_acc = vector.constant 0.0 : vector<8xf32>\n  %zero_k = vector.constant 0 : vector<2xi32>\n  %zero_v = vector.constant 0.0 : vector<16xf16>\n")
    STEM2 = STEM.replace("i4qk", "i4qkp")
    K = K.replace(SYM, "h3_" + STEM2).replace(NS, "h3." + STEM2)
    OUT2 = ROOT / "experiments" / f"{STEM2}.loom"
    OUT2.write_text(K); print("wrote", OUT2)

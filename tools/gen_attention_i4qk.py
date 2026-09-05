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
STEM = os.environ.get("ATTN_STEM", "attention_i4qk_mha8_lds_f16_wmma" if WAVES == 8 else "attention_i4qk_mha_lds_f16_wmma")
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
OUT.write_text(K)
print("wrote", OUT)

"""int8 QK^T attention kernels (f16 PV) from the shipped int4-QK ones: the same schedule with 8-bit operands
(payload_registers 4, 32 i32 words per head, K tiles 16 x 36 words). tools/prepare_qk_i8.loom makes the operands
(absmax/127 per token and head, rotated as the int4 ones). Writes kernels/attention_i8qk_*.loom.
    python3 tools/gen_attention_i8qk.py"""
import re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
STEMS = ["attention_i4qk_mha_lds_f16_wmma", "attention_i4qk_mha8_lds_f16_wmma", "attention_i4qkl_mha8_lds_f16_wmma"]


def convert(K: str) -> str:
    def sub(old, new, count=None):
        nonlocal K
        n = K.count(old); assert n > 0 and (count is None or n == count), (old[:70], n)
        K = K.replace(old, new)
    K = K.replace("i4qk", "i8qk")
    sub("element_format=i4, payload_elements=16, payload_registers=2", "element_format=i8, payload_elements=16, payload_registers=4")
    K = K.replace("%i4_schema", "%i8_schema")
    sub("  %qi_words = index.div %q_stride0, %c8 : index\n  %ki_words = index.div %kv_stride0, %c8 : index\n", "  %qi_words = index.div %q_stride0, %c4 : index\n  %ki_words = index.div %kv_stride0, %c4 : index\n")
    sub("  %q_wbase = index.mul %head, %c16 : index\n", "  %q_wbase = index.mul %head, %c32 : index\n")
    sub("  %q_word_limit = index.sub %qi_words, %c2 : index\n", "  %q_word_limit = index.sub %qi_words, %c4 : index\n")
    for c in range(8):
        sub(f"  %q_word{c}a = index.add %q_wbase, %c{2*c} : index\n  %q_word{c} = index.assume %q_word{c}a [le(%q_word{c}a, %q_word_limit), mul(%q_word{c}a, 2)] : index\n",
            f"  %q_word{c}a = index.add %q_wbase, %c{4*c} : index\n  %q_word{c} = index.assume %q_word{c}a [le(%q_word{c}a, %q_word_limit), mul(%q_word{c}a, 4)] : index\n")
    sub("  %kv_word0 = index.mul %kv_head, %c16 : index\n", "  %kv_word0 = index.mul %kv_head, %c32 : index\n")
    sub("  %st_chunk = index.mul %st_chunk0, %c2 : index\n", "  %st_chunk = index.mul %st_chunk0, %c4 : index\n")
    sub("  %ki_col_limit = index.sub %ki_words, %c2 : index\n", "  %ki_col_limit = index.sub %ki_words, %c4 : index\n")
    sub("mul(%st_col0, 2)] : index", "mul(%st_col0, 4)] : index")
    K = re.sub(r"%k_tile(_b|_o)?\[%lane_column, %c(\d+)\] : view<16x20xi32>", lambda m: f"%k_tile{m.group(1) or ''}[%lane_column, %c{2 * int(m.group(2))}] : view<16x36xi32>", K)
    K = K.replace("view<16x20xi32>", "view<16x36xi32>").replace("vector<2xi32>", "vector<4xi32>")
    K = K.replace("%zero_k = vector.constant 0 : vector<4xi32>", "%zero_k = vector.constant 0 : vector<4xi32>")
    # LDS: every offset at or past the K tile grows by the tile difference (2304 - 1280 per buffered tile)
    grow = 2304 - 1280
    def offsets(m):
        v = int(m.group(2)); name = m.group(1)
        if name in ("kbuf_bytes",): return f"  %{name} = index.constant 2304 : offset\n"
        if v >= 1280: v += grow * (2 if "kbuf_bytes" in K and name != "kbuf_bytes" and v >= 2560 else 1)
        return f"  %{name} = index.constant {v} : offset\n"
    K = re.sub(r"  %(v_tile_offset|scratch_offset|q_tile_offset|lds_bytes|kbuf_bytes) = index.constant (\d+) : offset\n", offsets, K)
    # constants the wider word offsets need
    have = set(re.findall(r"%c(\d+) = index.constant \d+ : index", K))
    need = [n for n in (20, 24, 28, 32) if str(n) not in have]
    anchor = "  %c0 = index.constant 0 : index\n"
    at = K.index(anchor, K.index("launch("))
    K = K[:at] + anchor + "".join(f"  %c{n} = index.constant {n} : index\n" for n in need) + K[at + len(anchor):]
    return K


def main():
    for stem in STEMS:
        src = ROOT / "h3/kernels" / f"{stem}.loom"; K = convert(src.read_text())
        out = ROOT / "h3/kernels" / f"{stem.replace('i4qk', 'i8qk')}.loom"; out.write_text(K); print("wrote", out.name, len(K))


if __name__ == "__main__":
    main()

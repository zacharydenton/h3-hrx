"""Direct-load twins of the int8-QK attention kernels: no K/V staging through LDS and no workgroup barrier per tile.
Every lane loads its K fragment (16 int8 channels of its key) and its V^T fragments (16 keys of its channel rows)
straight from global memory, the way aotriton's flash kernel does on gfx115x (four waves, zero LDS, 30.8 TFLOP/s).
The per-wave 512-byte P scratch and the epilogue stage stay. Writes kernels/attention_i8qkd_*.loom.
    python3 tools/gen_attention_direct.py"""
import re
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
import os
PRELOAD = os.environ.get("ATTN_PRELOAD_V", "1") == "1"


def convert(K: str, waves: int) -> str:
    def sub(old, new, count=1):
        nonlocal K
        n = K.count(old); assert n == count, (old[:80], n); K = K.replace(old, new)
    K = K.replace("i8qk", "i8qkd")
    # the loop's staging block: from the double-buffer selection to the workgroup barrier
    i = K.index("    %buf = index.rem %key_tile, %c2 : index\n")
    j = K.index("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)\n\n", i) + len("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)\n\n")
    K = K[:i] + "    %krow0 = index.add %key_origin0, %lane_column : index\n    %krow = index.assume %krow0 [lt(%krow0, %padded_tokens)] : index\n" + K[j:]
    # K fragments straight from the operand rows
    for c in range(8):
        sub(f"    %k_data{c} = vector.load %k_tile_b[%lane_column, %c{4 * c}] : view<16x36xi32> -> vector<4xi32>\n",
            f"    %k_data{c} = vector.load %ki_view[%krow, %kw{c}] : view<[%padded_tokens]x[%ki_words]xi32> -> vector<4xi32>\n")
    # V^T fragments straight from the transposed V rows (16 keys per 32-byte load)
    for n in range(8):
        sub(f"    %v_data{n} = vector.load %v_tile_b[%vrow{n}, %c0] : view<128x24xf16> -> vector<16xf16>\n",
            f"    %vch{n}a = index.add %kv_base0, %vrow{n} : index\n    %vch{n} = index.assume %vch{n}a [le(%vch{n}a, %st_chan_limit)] : index\n    %v_data{n} = vector.load %v_view[%vch{n}, %key_origin0] : view<[%kv_stride0]x[%padded_tokens]xf16> -> vector<16xf16>\n")
    # the K word columns of this lane's head, hoisted
    anchor = "  %ks_last = index.sub %padded_tokens, %c1 : index\n"
    hoist = "".join(f"  %kw{c}a = index.add %kv_word0, %c{4 * c} : index\n  %kw{c} = index.assume %kw{c}a [le(%kw{c}a, %ki_col_limit), mul(%kw{c}a, 4)] : index\n" for c in range(8))
    sub(anchor, anchor + hoist)
    # LDS: only the P scratch and the publish stage remain
    K = re.sub(r"  %scratch_offset = index.constant \d+ : offset\n", "  %scratch_offset = index.constant 0 : offset\n", K)
    K = re.sub(r"  %lds_bytes = index.constant \d+ : offset\n", f"  %lds_bytes = index.constant {waves * 1024} : offset\n", K)
    assert "%k_tile_b" not in K and "%v_tile_b" not in K, [l for l in K.splitlines() if "tile_b" in l][:3]
    if PRELOAD:   # issue this tile's V^T loads before the QK^T chain so their latency hides behind it (Triton's PRE_LOAD_V)
        vloads = re.findall(r"    %vrow\d+ = .*\n    %vch\d+a = .*\n    %vch\d+ = .*\n    %v_data\d+ = .*\n", K); assert len(vloads) == 8, len(vloads)
        for v in vloads: K = K.replace(v, "", 1)
        anchor = "    %krow = index.assume %krow0 [lt(%krow0, %padded_tokens)] : index\n"; assert K.count(anchor) == 1
        K = K.replace(anchor, anchor + "".join(vloads))
    return K


def main():
    for stem, waves in (("attention_i8qk_mha_lds_f16_wmma", 4), ("attention_i8qk_mha8_lds_f16_wmma", 8)):
        src = (ROOT / "h3/kernels" / f"{stem}.loom").read_text()
        try:
            out = convert(src, waves)
        except AssertionError as e:
            print("skipped", stem, e); continue
        (ROOT / "experiments" / f"{stem.replace('i8qk', 'i8qkd')}.loom").write_text(out); print("wrote", stem.replace("i8qk", "i8qkd"))


if __name__ == "__main__":
    main()

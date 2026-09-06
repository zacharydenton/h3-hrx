"""Next-tile K prefetch for the int8-QK attention kernels: each K-staging lane loads its 16-byte K packet for tile i+1
right after storing tile i's into the other LDS buffer, so that global latency hides behind tile i's compute (four
loop-carried registers). The V^T packet stays loaded at the top of its tile: carrying it too (the `fv` form in
experiments/) spills 72 bytes and runs at 0.4x. The eight-wave form also interleaves each PV fragment load with its MMA.
Measured at 37723 tokens: 22.0 -> 24.3 TFLOP/s (docs/notes.md). Writes kernels/attention_i8qkf_mha8_* and *_mha_*.
    python3 tools/gen_attention_prefetch.py"""
import re
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent


def convert(K: str) -> str:
    def sub(old, new):
        nonlocal K
        assert K.count(old) == 1, old[:90]; K = K.replace(old, new)
    K = K.replace("i8qk_mha8", "i8qkf_mha8")
    sub("  %ks_last = index.sub %padded_tokens, %c1 : index\n",
        """  %ks_last = index.sub %padded_tokens, %c1 : index
  %zero_k4 = vector.constant 0 : vector<4xi32>
  %zero_v16 = vector.constant 0.0 : vector<16xf16>
  %tiles_last = index.sub %key_tile_count, %c1 : index
  %p_first_k, %p_first_v = scf.if %st_is_k -> (vector<4xi32>, vector<16xf16>) {
    %st_row_f = index.assume %st_key [lt(%st_key, %padded_tokens)] : index
    %r = vector.load %ki_view[%st_row_f, %st_col] : view<[%padded_tokens]x[%ki_words]xi32> -> vector<4xi32>
    scf.yield %r, %zero_v16 : vector<4xi32>, vector<16xf16>
  } else {
    %r = vector.load %v_view[%st_chan, %c0] : view<[%kv_stride0]x[%padded_tokens]xf16> -> vector<16xf16>
    scf.yield %zero_k4, %r : vector<4xi32>, vector<16xf16>
  }
""")
    sub("%ks_end = scf.for %key_tile", "%ks_end, %k_end, %v_end = scf.for %key_tile")
    sub("%ks_carry = %ks_first : f32) -> (", "%ks_carry = %ks_first : f32, %k_pre = %p_first_k : vector<4xi32>, %v_pre = %p_first_v : vector<16xf16>) -> (")
    # the loop's result type list ends with 'f32) {'
    sub(", f32) {\n    %key_origin1 = index.mul %key_tile, %c16 : index\n", ", f32, vector<4xi32>, vector<16xf16>) {\n    %key_origin1 = index.mul %key_tile, %c16 : index\n")
    sub("""    scf.if %st_is_k {
      %st_row0 = index.add %key_origin0, %st_key : index
      %st_row = index.assume %st_row0 [lt(%st_row0, %padded_tokens)] : index
      %k_chunk = vector.load %ki_view[%st_row, %st_col] : view<[%padded_tokens]x[%ki_words]xi32> -> vector<4xi32>
      vector.store %k_chunk, %k_tile_b[%st_key, %st_chunk] : vector<4xi32>, view<16x36xi32>
    } else {
      %v_chunk = vector.load %v_view[%st_chan, %key_origin0] : view<[%kv_stride0]x[%padded_tokens]xf16> -> vector<16xf16>
      vector.store %v_chunk, %v_tile_b[%st_lane, %c0] : vector<16xf16>, view<128x24xf16>
    }
""", """    scf.if %st_is_k {
      vector.store %k_pre, %k_tile_b[%st_key, %st_chunk] : vector<4xi32>, view<16x36xi32>
    } else {
      vector.store %v_pre, %v_tile_b[%st_lane, %c0] : vector<16xf16>, view<128x24xf16>
    }
    // the next tile's packets, issued now and consumed after this tile's compute
    %nt0 = index.add %key_tile, %c1 : index
    %nt = index.min %nt0, %tiles_last : index
    %ko_n1 = index.mul %nt, %c16 : index
    %ko_n = index.assume %ko_n1 [le(%ko_n1, %tile_origin_limit), mul(%ko_n1, 16)] : index
    %k_next, %v_next = scf.if %st_is_k -> (vector<4xi32>, vector<16xf16>) {
      %st_row_n0 = index.add %ko_n, %st_key : index
      %st_row_n = index.assume %st_row_n0 [lt(%st_row_n0, %padded_tokens)] : index
      %r = vector.load %ki_view[%st_row_n, %st_col] : view<[%padded_tokens]x[%ki_words]xi32> -> vector<4xi32>
      scf.yield %r, %zero_v16 : vector<4xi32>, vector<16xf16>
    } else {
      %r = vector.load %v_view[%st_chan, %ko_n] : view<[%kv_stride0]x[%padded_tokens]xf16> -> vector<16xf16>
      scf.yield %zero_k4, %r : vector<4xi32>, vector<16xf16>
    }
""")
    i = K.index("    scf.yield %next_max, %next_sum,"); j = K.index("\n", i); line = K[i:j]
    assert line.endswith(", f32"), line[-40:]
    K = K[:i] + line.replace("%ks_next : ", "%ks_next, %k_next, %v_next : ") + ", vector<4xi32>, vector<16xf16>" + K[j:]
    return K


def konly(K: str) -> str:
    K = K.replace("i8qkf_mha8", "i8qkfk_mha8")
    K = K.replace("""    } else {
      vector.store %v_pre, %v_tile_b[%st_lane, %c0] : vector<16xf16>, view<128x24xf16>
    }""", """    } else {
      %v_chunk = vector.load %v_view[%st_chan, %key_origin0] : view<[%kv_stride0]x[%padded_tokens]xf16> -> vector<16xf16>
      vector.store %v_chunk, %v_tile_b[%st_lane, %c0] : vector<16xf16>, view<128x24xf16>
    }""")
    K = K.replace("""    } else {
      %r = vector.load %v_view[%st_chan, %ko_n] : view<[%kv_stride0]x[%padded_tokens]xf16> -> vector<16xf16>
      scf.yield %zero_k4, %r : vector<4xi32>, vector<16xf16>
    }""", """    } else {
      scf.yield %zero_k4, %zero_v16 : vector<4xi32>, vector<16xf16>
    }""")
    return K


def four_wave(K: str) -> str:
    def sub(old, new):
        nonlocal K
        assert K.count(old) == 1, old[:90]; K = K.replace(old, new)
    K = K.replace("i8qk_mha_", "i8qkf_mha_")
    sub("  %ks_last = index.sub %padded_tokens, %c1 : index\n",
        """  %ks_last = index.sub %padded_tokens, %c1 : index
  %tiles_last = index.sub %key_tile_count, %c1 : index
  %st_row_f = index.assume %st_key [lt(%st_key, %padded_tokens)] : index
  %p_first_k = vector.load %ki_view[%st_row_f, %st_col] : view<[%padded_tokens]x[%ki_words]xi32> -> vector<4xi32>
""")
    sub("%ks_end = scf.for %key_tile", "%ks_end, %k_end = scf.for %key_tile")
    sub("%ks_carry = %ks_first : f32) -> (", "%ks_carry = %ks_first : f32, %k_pre = %p_first_k : vector<4xi32>) -> (")
    sub(", f32) {\n    %key_origin1 = index.mul %key_tile, %c16 : index\n", ", f32, vector<4xi32>) {\n    %key_origin1 = index.mul %key_tile, %c16 : index\n")
    sub("""    %st_row0 = index.add %key_origin0, %st_key : index
    %st_row = index.assume %st_row0 [lt(%st_row0, %padded_tokens)] : index
    %k_chunk = vector.load %ki_view[%st_row, %st_col] : view<[%padded_tokens]x[%ki_words]xi32> -> vector<4xi32>
    %v_chunk = vector.load %v_view[%st_chan, %key_origin0] : view<[%kv_stride0]x[%padded_tokens]xf16> -> vector<16xf16>
    vector.store %k_chunk, %k_tile_b[%st_key, %st_chunk] : vector<4xi32>, view<16x36xi32>
""", """    %v_chunk = vector.load %v_view[%st_chan, %key_origin0] : view<[%kv_stride0]x[%padded_tokens]xf16> -> vector<16xf16>
    vector.store %k_pre, %k_tile_b[%st_key, %st_chunk] : vector<4xi32>, view<16x36xi32>
    %nt0 = index.add %key_tile, %c1 : index
    %nt = index.min %nt0, %tiles_last : index
    %ko_n1 = index.mul %nt, %c16 : index
    %ko_n = index.assume %ko_n1 [le(%ko_n1, %tile_origin_limit), mul(%ko_n1, 16)] : index
    %st_row_n0 = index.add %ko_n, %st_key : index
    %st_row_n = index.assume %st_row_n0 [lt(%st_row_n0, %padded_tokens)] : index
    %k_next = vector.load %ki_view[%st_row_n, %st_col] : view<[%padded_tokens]x[%ki_words]xi32> -> vector<4xi32>
""")
    i = K.index("    scf.yield %next_max, %next_sum,"); j = K.index("\n", i); line = K[i:j]
    assert line.endswith(", f32"), line[-40:]
    K = K[:i] + line.replace("%ks_next : ", "%ks_next, %k_next : ") + ", vector<4xi32>" + K[j:]
    return K


def interleave_pv(K: str) -> str:
    vloads = [re.search(rf"    %vrow{n} = .*\n    %v_data{n} = .*\n    %v{n} = .*\n", K).group(0) for n in range(8)]
    mmas = [re.search(rf"    %next{n} = vector.mma .*\n", K).group(0) for n in range(8)]
    for x in vloads + mmas: K = K.replace(x, "", 1)
    anchor = "    %rescaled7 = vector.mulf<reassoc|nnan|ninf|nsz|contract> %acc7, %selected_old_scale : vector<8xf32>\n"; assert K.count(anchor) == 1
    return K.replace(anchor, anchor + "".join(vloads[n] + mmas[n] for n in range(8)))


def main():
    src = (ROOT / "kernels/attention_i8qk_mha8_lds_f16_wmma.loom").read_text()
    full = convert(src)
    (ROOT / "experiments/attention_i8qkfv_mha8_lds_f16_wmma.loom").write_text(full.replace("i8qkf_mha8", "i8qkfv_mha8"))   # the V-carrying form: spills
    (ROOT / "kernels/attention_i8qkf_mha8_lds_f16_wmma.loom").write_text(interleave_pv(konly(full).replace("i8qkfk_mha8", "i8qkf_mha8")))
    (ROOT / "experiments/attention_i8qkf_mha_lds_f16_wmma.loom").write_text(four_wave((ROOT / "kernels/attention_i8qk_mha_lds_f16_wmma.loom").read_text()))
    print("wrote kernels/attention_i8qkf_mha8_*; experiments: the four-wave form (0.78x at 2922 rows) and the V-carrying form (spills)")


if __name__ == "__main__":
    main()

"""Where does the int4-QK attention kernel's time go? Ablations of the shipped 8-wave kernel, each
removing one component (wrong results, right timing), timed interleaved at one row count.
    python3 tools/ablate_attention_i4.py [tokens]"""
import re, subprocess, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from kernel_test import compile_kernel, launch, workdir
HEADS, D, WAVES = 56, 128, 8
BASE = ROOT / "kernels/attention_i4qk_mha8_lds_f16_wmma.loom"


def variants(src: str):
    out = {"base": src}
    # no exp: the weights are the scaled scores (softmax reductions and the rescale stay)
    s = src.replace("%weight = vector.expf<afn> %delta", "%weight = vector.addf %delta, %zero_vector").replace("%selected_old_scale = vector.expf<afn> %old_delta", "%selected_old_scale = vector.addf %old_delta, %zero_vector")
    out["no exp"] = s
    # no reductions: the row max is this lane's own score (the four xor butterflies and the final sum butterflies go)
    s = re.sub(r"    %next_max = vector.maxnumf %row_max, %tm8 : vector<8xf32>\n", "    %next_max = vector.maxnumf %row_max, %scaled : vector<8xf32>\n", src)
    out["no max butterflies"] = s
    # no accumulator rescale (8 multiplies of 8-vectors)
    s = re.sub(r"    %rescaled(\d) = vector.mulf<[^>]*> %acc(\d), %selected_old_scale : vector<8xf32>\n", r"    %rescaled\1 = vector.addf %acc\2, %zero_vector : vector<8xf32>\n", src)
    out["no rescale"] = s
    # no PV: the PV MMAs (and, by dead-code, the V fragment loads and V staging) go
    s = re.sub(r"    %next(\d) = vector.mma %probability, %v\d, %rescaled\d : vector<16xf16>, vector<16xf16>, vector<8xf32>\n", r"    %next\1 = vector.addf %rescaled\1, %zero_vector : vector<8xf32>\n", src)
    out["no PV (MMAs, V reads)"] = s
    # no QK: the int4 MMAs (and by dead-code the K fragment loads) go; scores from the scales alone
    s = re.sub(r"    %raw_scores = vector.sitofp %raw_scores_i : vector<8xi32> to vector<8xf32>\n", "    %raw_scores = vector.addf %qs_vec, %zero_vector : vector<8xf32>\n", src)
    out["no QK (MMAs, K reads)"] = s
    # no V transpose staging: the 16 scalar stores per lane become one 32-byte vector store of the chunk (wrong layout)
    s = re.sub(r"      %ve(\d+) = vector.extract %v_chunk\[\d+\] : vector<16xf16> -> f16\n      %vr\d+ = index.add %st_chunk_v, %cj\d+ : index\n      view.store %ve\d+, %v_tile\[%vr\d+, %st_key_v\] : f16, view<128x24xf16>\n", "", src)
    s = s.replace("    } else {\n    }\n", "    } else {\n      vector.store %v_chunk, %v_tile[%st_chunk_v, %c0] : vector<16xf16>, view<128x24xf16>\n    }\n")
    out["V staged untransposed"] = s
    # no P round trip through LDS: a V fragment's data stands in for the probability lhs (wrong values)
    s = src.replace("    %probability = vector.fragment.load<lhs> %scratch_view[%c0, %c0] shape [%m, %k_frag] : view<16x16xf16> -> vector<16xf16>\n", "    %probability = vector.fragment<lhs> %v_data0 shape [%m, %k_frag] : vector<16xf16>\n")
    s = re.sub(r"    vector.fragment.store<result> %weight, %scratch_view\[%c0, %c0\] shape \[%m, %n\] : vector<8xf32>, view<16x16xf16>\n    kernel.barrier<workgroup> scope\(subgroup\) ordering\(acq_rel\)\n", "", s)
    out["no P LDS round trip"] = s
    # no trailing barrier (a race; timing only)
    idx = src.rindex("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)\n    scf.yield %next_max")
    out["no trailing barrier"] = src[:idx] + src[idx + len("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)\n"):]
    # no K/V staging: the tiles are never loaded or stored (stale LDS; timing only)
    s = re.sub(r"      %k_chunk = vector.load %ki_view\[%st_row, %st_col\] : view<\[%padded_tokens\]x\[%ki_words\]xi32> -> vector<2xi32>\n      vector.store %k_chunk, %k_tile\[%st_key, %st_chunk\] : vector<2xi32>, view<16x20xi32>\n", "", src)
    s = re.sub(r"      %v_chunk = vector.load %v_view\[%st_chan, %key_origin0\] : view<\[%kv_stride0\]x\[%padded_tokens\]xf16> -> vector<16xf16>\n      vector.store %v_chunk, %v_tile\[%st_lane, %c0\] : vector<16xf16>, view<128x24xf16>\n", "", s)
    out["no K/V staging"] = s
    # no epilogue publish loop
    i0 = src.index("  scf.for %fragment = [%c0 to %c_nf step %c1] {"); i1 = src.index("  kernel.return")
    out["no epilogue"] = src[:i0] + src[i1:]
    # LDS padded to 40 KB: at most one workgroup per CU (occupancy sensitivity)
    s = re.sub(r"  %lds_bytes = index.constant \d+ : offset\n", "  %lds_bytes = index.constant 40960 : offset\n", src)
    out["LDS padded (1 WG/CU)"] = s
    return out


def main():
    tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 15427
    cap = max((tokens + 16 + 31) // 32 * 32, (tokens + 16 * WAVES - 1) // (16 * WAVES) * (16 * WAVES))
    rng = np.random.default_rng(0)
    qi = rng.integers(-2**31, 2**31, size=(cap, HEADS * 16), dtype=np.int32); ki = rng.integers(-2**31, 2**31, size=(cap, HEADS * 16), dtype=np.int32)
    qs = (rng.standard_normal((cap, HEADS)) * 1e-3).astype(np.float32); ks = np.abs(rng.standard_normal((cap, HEADS))).astype(np.float32) * 0.05
    v = (rng.standard_normal((cap, HEADS * D)) * 0.5).astype(np.float16)
    src = BASE.read_text(); vs = variants(src)
    with workdir() as tmp:
        tmp = Path(tmp); built = {}
        for name, text in vs.items():
            stem = f"ablate_{abs(hash(name)) % 100000}"; path = tmp / f"{stem}.loom"
            text = text.replace("h3_attention_i4qk_mha8_lds_f16_wmma", "h3_" + stem).replace("h3.attention_i4qk_mha8_lds_f16_wmma", "h3." + stem)
            path.write_text(text); ns, sym = "h3." + stem, "h3_" + stem
            try:
                compile_kernel(path, sym, {f"{ns}.q_stride": HEADS * D, f"{ns}.kv_stride": HEADS * D, f"{ns}.tokens": tokens, f"{ns}.token_capacity": cap, f"{ns}.scale": 1.0, f"{ns}.out_stride": HEADS * D}, tmp / f"{stem}.hsaco")
                built[name] = (tmp / f"{stem}.hsaco", sym)
            except subprocess.CalledProcessError as e:
                print(f"  {name}: does not compile ({(e.stderr or '')[-200:].strip().splitlines()[-1] if e.stderr else ''})")
        best = {k: 1e9 for k in built}
        for r in range(3):
            order = list(built) if r % 2 == 0 else list(built)[::-1]
            for name in order:
                hs, sym = built[name]
                _, t = launch(hs, sym, ((tokens + 16 * WAVES - 1) // (16 * WAVES), HEADS, 1), (32 * WAVES, 1, 1), [("i32", tokens), ("i32", HEADS), ("in_i32", qi), ("in", qs), ("in_i32", ki), ("in", ks), ("in_f16", v), ("out_f16", ((tokens, HEADS * D), np.float16))], tmp, repeat=1)
                best[name] = min(best[name], t["per_launch_us"])
        base = best["base"]
        print(f"int4-QK attention ablations at {tokens} rows (best of 3 interleaved; time removed = base - variant):")
        for name in built:
            print(f"  {name:28s} {best[name] / 1e3:8.1f} ms   {100 * (base - best[name]) / base:6.1f}% of base removed")


if __name__ == "__main__":
    main()

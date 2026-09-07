"""Generate the 64-key head-major attention kernel from the 32-key template.

The same INT8 QK / FP16 PV arithmetic uses two 32-register output groups,
one shared-memory buffer, and explicit lookahead for K/V fragment loads.
Run after tools/gen_attention_i8_head_major.py; the 32-key kernel is the template.
"""

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "attention_i8qkhm_mha8_lds_f16_wmma"
STEM = "attention_i8qkhm_mha8_k64_lds_f16_wmma"
HRX_BUILD = Path(os.environ.get("HRX_BUILD", Path.home() / "code/hrx-system/build-cuda"))
FORMAT = HRX_BUILD / "loom/src/loom/tools/loom-format/loom-format"


def keys64(src):
    s = src.replace("view<32x36xi32>", "view<64x36xi32>").replace(
        "view<128x40xf16>", "view<128x72xf16>"
    )
    s = (
        s.replace("%c31 = index.constant 31", "%c31 = index.constant 63")
        .replace(
            "%key_tile_count = index.div %key_round, %c32",
            "%key_tile_count = index.div %key_round, %c64",
        )
        .replace(
            "%key_tail = index.rem %tokens0, %c32",
            "%key_tail = index.rem %tokens0, %c64",
        )
        .replace(
            "%key_origin1 = index.mul %key_tile, %c32",
            "%key_origin1 = index.mul %key_tile, %c64",
        )
    )
    s = s.replace(
        "%buf = index.rem %key_tile, %c2 : index", "%buf = index.constant 0 : index"
    ).replace("29696 : offset", "27648 : offset")
    s = s.replace(
        "%key_lane_b = index.add %lane_column, %c16 : index",
        "%key_lane_b = index.add %lane_column, %c16 : index\n    %key_lane_c = index.add %lane_column, %c32 : index\n    %key_lane_d = index.add %lane_column, %c48 : index\n    %key_origin_c = index.add %key_origin0, %c32 : index\n    %key_origin_d = index.add %key_origin0, %c48 : index",
    )
    a = s.index("      %st_row_b0 =")
    b = s.index("    } else {", a)
    chunk = s[a:b]
    for suffix, offset in (("_c", 32), ("_d", 48)):
        s = s[:b] + chunk.replace("_b", suffix).replace("%c16", f"%c{offset}") + s[b:]
        b += len(chunk.replace("_b", suffix).replace("%c16", f"%c{offset}"))
    a = s.index("      %v_key_b =")
    b = s.index("    }\n    kernel.barrier", a)
    chunk = s[a:b]
    for suffix, offset in (("_c", 32), ("_d", 48)):
        new = (
            chunk.replace("_b", suffix)
            .replace("%v_tile" + suffix, "%v_tile_b")
            .replace("%c16", f"%c{offset}")
        )
        s = s[:b] + new + s[b:]
        b += len(new)
    # Generate four QK halves, using paired invocations.
    a = s.index("    %ka0 =")
    b = s.index("    %raw_scores =", a)
    block = ""
    for i in range(8):
        for pair, halves in enumerate((("", "_b"), ("_c", "_d"))):
            x, y = halves
            for suffix, lane in (
                (x, "%lane_column" if x == "" else "%key_lane" + x),
                (y, "%key_lane" + y),
            ):
                block += f"    %kk{i}{suffix} = vector.load %k_tile_b[{lane}, %c{4 * i}] : view<64x36xi32> -> vector<4xi32>\n"
            dest = lambda t, i=i: "%raw_scores_i" + t if i == 7 else f"%pair{i}" + t
            prev = lambda t, i=i: "%zero_i32x8" if i == 0 else f"%pair{i - 1}" + t
            block += f"    {dest(x)}, {dest(y)} = low.invoke @h3_qk_pair(%kk{i}{x}, %kk{i}{y}, %qd{i}, {prev(x)}, {prev(y)}) : (vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<8xi32>, vector<8xi32>) -> (vector<8xi32>, vector<8xi32>)\n"
    s = s[:a] + block + s[b:]
    a = s.index("    %raw_scores_b =")
    b = s.index("    %local_max_a =", a)
    chunk = s[a:b]
    s = s[:b] + chunk.replace("_b", "_c") + chunk.replace("_b", "_d") + s[b:]
    s = s.replace(
        "    %local_max = scalar.maxnumf %local_max_a, %local_max_b : f32",
        """    %local_max_c = vector.reduce<maxnumf> %scaled_c, %negative_large : vector<8xf32>, f32
        %local_max_d = vector.reduce<maxnumf> %scaled_d, %negative_large : vector<8xf32>, f32
        %local_max_ab = scalar.maxnumf %local_max_a, %local_max_b : f32
        %local_max_cd = scalar.maxnumf %local_max_c, %local_max_d : f32
        %local_max = scalar.maxnumf %local_max_ab, %local_max_cd : f32""",
    )
    a = s.index("    %delta_b =")
    b = s.index("    %old_delta =", a)
    chunk = s[a:b]
    s = s[:b] + chunk.replace("_b", "_c") + chunk.replace("_b", "_d") + s[b:]
    s = s.replace(
        "    %local_sum = scalar.addf %local_sum_a, %local_sum_b : f32",
        """    %local_sum_c = vector.reduce<addf> %weight_c, %zero_f32 : vector<8xf32>, f32
        %local_sum_d = vector.reduce<addf> %weight_d, %zero_f32 : vector<8xf32>, f32
        %local_sum_ab = scalar.addf %local_sum_a, %local_sum_b : f32
        %local_sum_cd = scalar.addf %local_sum_c, %local_sum_d : f32
        %local_sum = scalar.addf %local_sum_ab, %local_sum_cd : f32""",
    )
    a = s.index("    %weight_half_b =")
    b = s.index("    %vrow0 =", a)
    chunk = s[a:b]
    s = s[:b] + chunk.replace("_b", "_c") + chunk.replace("_b", "_d") + s[b:]
    for i in range(8):
        old = f"    %next{i} = low.invoke @h3_pv32(%v_data{i}_b, %probability_b, %partial{i}) : (vector<16xf16>, vector<16xf16>, vector<8xf32>) -> (vector<8xf32>)"
        new = old.replace(f"%next{i}", f"%partial{i}_b")
        for suffix, prev, offset in (("_c", "_b", 32), ("_d", "_c", 48)):
            dest = f"%next{i}" if suffix == "_d" else f"%partial{i}_c"
            new += f"\n    %v_data{i}{suffix} = vector.load %v_tile_b[%vrow{i}, %c{offset}] : view<128x72xf16> -> vector<16xf16>\n    {dest} = low.invoke @h3_pv32(%v_data{i}{suffix}, %probability{suffix}, %partial{i}{prev}) : (vector<16xf16>, vector<16xf16>, vector<8xf32>) -> (vector<8xf32>)"
        assert old in s
        s = s.replace(old, new)
    s = s.replace(
        "    scf.yield %next_max",
        "    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)\n    scf.yield %next_max",
    )
    # K stores must still use the same buffer name after suffix substitution.
    s = s.replace("%k_tile_c[", "%k_tile_b[").replace("%k_tile_d[", "%k_tile_b[")
    return s


def group_accumulators(s):
    group, width, stem = 4, 32, STEM
    helper = (
        f"low.func.def target<amdgpu.gfx11.generic.core>(@h3_{stem}_gfx11) @h3_accsplit(%acc: reg<amdgpu.vgpr x{width}>) -> ("
        + ", ".join(["reg<amdgpu.vgpr x8>"] * group)
        + ") asm {\n"
    )
    for i in range(group):
        helper += f"  %a{i} = slice %acc[{i * 8}] : reg<amdgpu.vgpr x{width}> -> reg<amdgpu.vgpr x8>\n"
    helper += "  return " + ", ".join(f"%a{i}" for i in range(group)) + "\n}\n"
    helper += (
        f"low.func.def target<amdgpu.gfx11.generic.core>(@h3_{stem}_gfx11) @h3_accjoin("
        + ", ".join(f"%a{i}: reg<amdgpu.vgpr x8>" for i in range(group))
        + f") -> (reg<amdgpu.vgpr x{width}>) asm {{\n  %acc = concat("
        + ", ".join(f"%a{i}" for i in range(group))
        + ") : ("
        + ", ".join(["reg<amdgpu.vgpr x8>"] * group)
        + f") -> reg<amdgpu.vgpr x{width}>\n  return %acc\n}}\n"
    )
    a = s.index("kernel.def")
    s = s[:a] + helper + s[a:]
    a = s.index("  %final_max,")
    b = s.index("\n", a)
    head = (
        f"  %init_all = vector.constant 0.0 : vector<{width}xf32>\n  %final_max, %final_sum, "
        + ", ".join(f"%final_all{i}" for i in range(8 // group))
        + " = scf.for %key_tile = [%c0 to %key_tile_count step %c1](%row_max = %negative_large : f32, %row_sum = %zero_f32 : f32, "
        + ", ".join(
            f"%acc_all{i} = %init_all : vector<{width}xf32>" for i in range(8 // group)
        )
        + ") -> (f32, f32, "
        + ", ".join([f"vector<{width}xf32>"] * (8 // group))
        + ") {\n"
    )
    for start in range(0, 8, group):
        head += (
            "    "
            + ", ".join(f"%acc{i}" for i in range(start, start + group))
            + f" = low.invoke @h3_accsplit(%acc_all{start // group}) : (vector<{width}xf32>) -> ("
            + ", ".join(["vector<8xf32>"] * group)
            + ")\n"
        )
    s = s[:a] + head + s[b:]
    a = s.index("    scf.yield %next_max")
    b = s.index("\n", a)
    tail = ""
    for start in range(0, 8, group):
        tail += (
            f"    %next_all{start // group} = low.invoke @h3_accjoin("
            + ", ".join(f"%next{i}" for i in range(start, start + group))
            + ") : ("
            + ", ".join(["vector<8xf32>"] * group)
            + f") -> (vector<{width}xf32>)\n"
        )
    tail += (
        "    scf.yield %next_max, %next_sum, "
        + ", ".join(f"%next_all{i}" for i in range(8 // group))
        + " : f32, f32, "
        + ", ".join([f"vector<{width}xf32>"] * (8 // group))
    )
    s = s[:a] + tail + s[b:]
    a = s.index("  %sum_other,")
    end = ""
    for start in range(0, 8, group):
        end += (
            "  "
            + ", ".join(f"%final{i}" for i in range(start, start + group))
            + f" = low.invoke @h3_accsplit(%final_all{start // group}) : (vector<{width}xf32>) -> ("
            + ", ".join(["vector<8xf32>"] * group)
            + ")\n"
        )
    s = s[:a] + end + s[a:]
    return s


def pipeline_fragments(s):
    # Load one complete K fragment ahead of its paired WMMA invocations.
    a, b = s.index("    %kk0 ="), s.index("    %raw_scores =")
    lines = s[a:b].splitlines(True)
    assert len(lines) == 48
    loads, calls = [], []
    for i in range(8):
        chunk = lines[6 * i : 6 * i + 6]
        loads.append("".join(l for l in chunk if " = vector.load" in l))
        calls.append("".join(l for l in chunk if " = low.invoke" in l))
    block = loads[0]
    for i in range(8):
        if i + 1 < 8:
            block += loads[i + 1]
        block += calls[i]
    s = s[:a] + block + s[b:]
    # Keep two V fragments in flight ahead of PV computation.
    a, b = s.index("    %vrow0 ="), s.index("    scf.yield %next_max")
    lines = s[a:b].splitlines(True)
    addr = [l for l in lines if " = index.add" in l]
    loads = [l for l in lines if " = vector.load" in l]
    scales = [l for l in lines if " = vector.mulf" in l]
    calls = [l for l in lines if " = low.invoke @h3_pv32" in l]
    assert len(loads) == len(calls) == 32 and len(scales) == 8
    tail = [l for l in lines if l not in addr + loads + scales + calls]
    block = "".join(addr + loads[:2])
    for i in range(32):
        if i + 2 < 32:
            block += loads[i + 2]
        if i % 4 == 0:
            block += scales[i // 4]
        block += calls[i]
    return s[:a] + block + "".join(tail) + s[b:]


def main():
    # Canonical formatting makes template anchors independent of generated spacing.
    template = subprocess.check_output(
        [str(FORMAT), str(ROOT / "kernels" / f"{BASE}.loom")], text=True
    )
    s = template.replace(BASE, STEM)
    s = pipeline_fragments(group_accumulators(keys64(s)))
    s = (
        "// Generated by tools/gen_attention_i8_head_major_64.py. Dense INT8 QK, FP32\n// online softmax/accumulation, FP16 P and V. Q/K/scales are head-major.\n// Eight wave32 query tiles share 64 keys; launch ceil(N/128) x heads.\n"
        + s[s.index("amdgpu.target") :]
    )
    s = s.replace(
        "// K and V tiles: 16 keys x 128 channels, rows padded to 136 halves",
        "// Shared K/V tiles contain 64 keys and 128 channels.",
    )
    path = ROOT / "kernels" / f"{STEM}.loom"
    path.write_text(s)
    subprocess.run([str(FORMAT), "--in-place", str(path)], check=True)


if __name__ == "__main__":
    main()

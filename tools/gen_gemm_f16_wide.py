"""Experimental decoder GEMMs: 256x256, sixteen 64x64 waves.

K-step 64 uses a bank-swizzled 64 KiB operand stage; K-step 32 uses 40 KiB.
Both alias the 32 KiB epilogue stage after the K loop. Accumulation retains
the production kernel's ascending sequence of 16-wide WMMAs. SwiGLU can publish
directly at the down projection's padded operand pitch.

Generate with python3 tools/gen_gemm_f16_wide.py, then loom-format the outputs.
The host selects these kernels only with H3_VAE_WIDE=1. Both K-stage variants
passed numerical checks but lost to the 128x256 decoder tile on the measured GPU.
"""
from pathlib import Path
import re

import gen_gemm
from gen_gemm_f16 import gemm

ROOT = Path(__file__).resolve().parent.parent
STEMS = {"plain": "gemm_f16_wide_256b", "resid": "gemm_f16_wide_resid_256b",
         "swiglu": "gemm_f16_wide_swiglu_256b_gs"}


def swizzle_stage(source):
    """Rotate each LDS row by (row % 8)*8 halves. Split packets at 8 halves
    so a rotated 16-half packet never crosses the physical row boundary."""
    def addresses(tag, row, col):
        s = f"  %{tag}_r = index.rem {row}, %c8 : index\n"
        s += f"  %{tag}_shift = index.mul %{tag}_r, %c8 : index\n"
        s += f"  %{tag}_sum = index.add {col}, %{tag}_shift : index\n"
        s += f"  %{tag}_sum8 = index.add %{tag}_sum, %c8 : index\n"
        for part, suffix in ((0, 'sum'), (1, 'sum8')):
            s += f"  %{tag}_p{part} = index.rem %{tag}_{suffix}, %c64 : index\n"
            s += f"  %{tag}_c{part} = index.assume %{tag}_p{part} [lt(%{tag}_p{part}, %c64), mul(%{tag}_p{part}, 8)] : index\n"
        return s
    def store(m):
        val, stage, row, col = m.groups(); tag = val[1:] + '_st'
        s = addresses(tag, row, col)
        for part in (0, 1):
            s += f"  %{tag}_v{part} = vector.slice {val}[{part * 8}] : vector<16xf16> -> vector<8xf16>\n"
            s += f"  vector.store %{tag}_v{part}, {stage}[{row}, %{tag}_c{part}] : vector<8xf16>, view<256x64xf16>\n"
        return s
    def read(m):
        val, stage, row, col = m.groups(); tag = val[1:] + '_ld'
        s = addresses(tag, row, col)
        for part in (0, 1):
            s += f"  %{tag}_v{part} = vector.load {stage}[{row}, %{tag}_c{part}] : view<256x64xf16> -> vector<8xf16>\n"
        return s + f"  {val} = vector.concat<0> %{tag}_v0, %{tag}_v1 : vector<8xf16>, vector<8xf16> -> vector<16xf16>\n"
    source = re.sub(r"vector.store (%c[aw]\d), (%[aw]_stage)\[(%\w+), (%\w+)\] : vector<16xf16>, view<256x64xf16>\n", store, source)
    return re.sub(r"(%d[ab]\d_\d) = vector.load (%[aw]_stage)\[(%\w+), (%\w+)\] : view<256x64xf16> -> vector<16xf16>\n", read, source)


def generate(mode, kstep=64):
    assert kstep in (32, 64)
    previous = gen_gemm.TM, gen_gemm.TN, gen_gemm.A_PACKETS, gen_gemm.PACKETS
    try:
        # 512 lanes carry one (K32) or two (K64) 16-half packets per operand.
        gen_gemm.TM, gen_gemm.TN = 256, 256
        gen_gemm.A_PACKETS, gen_gemm.PACKETS = kstep // 32, kstep // 16
        source = gemm(gen_gemm.generate(mode, bias=True, gate_first=mode != "swiglu", bits=8))
    finally:
        gen_gemm.TM, gen_gemm.TN, gen_gemm.A_PACKETS, gen_gemm.PACKETS = previous

    def replace(old, new, count=1):
        nonlocal source
        if source.count(old) != count:
            raise ValueError(f"Expected {count} occurrences of {old!r}, got {source.count(old)}")
        source = source.replace(old, new)

    source = source[source.index("amdgpu.target"):]
    source = source.replace("gemm_f16_", "gemm_f16_wide_")
    replace("%c255 = index.constant 255 : index", "%c255 = index.constant 255 : index\n  %c512 = index.constant 512 : index")
    replace("workgroup_size(%c256, %c1, %c1)", "workgroup_size(%c512, %c1, %c1)")
    replace("range(%subgroup0, 0, 7)", "range(%subgroup0, 0, 15)")
    replace("%wave_m = index.div %subgroup, %c2", "%wave_m = index.div %subgroup, %c4")
    replace("%wave_n = index.rem %subgroup, %c2", "%wave_n = index.rem %subgroup, %c4")
    replace("%n_tiles = index.div %n_size0, %c128", "%n_tiles = index.div %n_size0, %c256")
    replace("%c_n_tiles = index.div %n_size, %c128", "%c_n_tiles = index.div %n_size, %c256")
    replace("%base_n = index.mul %tile_n_id, %c128", "%base_n = index.mul %tile_n_id, %c256")
    replace("range(%value, 128, 65536), mul(%value, 128)", "range(%value, 256, 65536), mul(%value, 256)")
    replace("range(%n_size0, 128, 65536), mul(%n_size0, 128)", "range(%n_size0, 256, 65536), mul(%n_size0, 256)")

    if kstep == 64:
        replace("%lds_bytes = index.constant 73728", "%lds_bytes = index.constant 65536")
        replace("%w_stage_offset = index.constant 36864", "%w_stage_offset = index.constant 32768")
        source = source.replace("view<256x72xf16>", "view<256x64xf16>")
        for operand in ('a', 'w'):
            replace(f"%st_{operand}row1 = index.add %st_row_base, %c64", f"%st_{operand}row1 = index.add %st_row_base, %c128")
        source = swizzle_stage(source)
    else:
        replace("%lds_bytes = index.constant 73728", "%lds_bytes = index.constant 40960")
        replace("%w_stage_offset = index.constant 36864", "%w_stage_offset = index.constant 20480")
        source = source.replace("view<256x72xf16>", "view<256x40xf16>")
        replace("%st_sub = index.rem %workitem, %c4", "%st_sub = index.rem %workitem, %c2")
        replace("%st_row_base = index.div %workitem, %c4", "%st_row_base = index.div %workitem, %c2")
        replace("lt(%st_col0, %c64)", "lt(%st_col0, %c32)")
        replace("to %k_size step %c64", "to %k_size step %c32")
        replace("%k_next = index.add %k_base, %c64", "%k_next = index.add %k_base, %c32")
    # Remove the latter two substeps; their WMMAs become the next iteration's
    # first two. The last prefetch remains a safe, unused read of the first step.
        source = re.sub(r"^\s*%(?:da|db|la|rb|o)\d+_[23] = [^\n]*\n", "", source, flags=re.M)
        source = re.sub(r"(%o\d+)_3\b", r"\1_1", source)

    if mode == "swiglu":
        ns = "h3." + STEMS[mode]
        replace(f"config.decl @{ns}.m_group", f"config.decl @{ns}.out_stride : %value: index where [range(%value, 128, 65536), mul(%value, 128)]\n\nconfig.decl @{ns}.m_group")
        replace("%n_half = index.div %n_size, %c2 : index", f"%n_half = index.div %n_size, %c2 : index\n  %out_stride = config.get @{ns}.out_stride : index")
        source = source.replace("view<[%m_bounded]x[%n_half]xf16>", "view<[%m_bounded]x[%out_stride]xf16>")

    # The inherited comments describe the old geometry and int8 pitch policy.
    source = re.sub(r"^\s*//[^\n]*\n", "", source, flags=re.M)
    return (f"// Experimental f16 GEMM: 256x256, sixteen 64x64 waves, K-step {kstep}.\n"
            f"// {65536 if kstep == 64 else 40960} bytes LDS; generated by tools/gen_gemm_f16_wide.py.\n" + source)


if __name__ == "__main__":
    for mode, stem in STEMS.items():
        (ROOT / "h3/kernels" / f"{stem}.loom").write_text(generate(mode))
        print("wrote", stem)

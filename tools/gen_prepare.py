"""kernels/prepare_*_i4.loom: the GEMM-input preparation kernels, one workgroup of `lanes`
lanes per token (96 for 5376, 128 for 7168, 256 for 14336), 8 elements per lane per step (16-byte loads, 4-byte packed stores). Every variant ends the same way -- group-256 Hadamard (Kronecker
power of H4, normalised by 1/16), per-token absmax, symmetric int4 (q = round(x / s),
s = absmax/7), nibbles packed low first, f32 scale beside -- and differs in how the
row is formed first:
  norm   : rmsnorm(h) * norm_weight * (1 + scale[class]) + shift[class]  (block inputs; class = (timestep, modality) row)
  swiglu : silu(g) * u                                                     (the down input)
"""
import re
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "kernels"

def stage(d: int, lds: str = "f32") -> str:
    return f"""  scf.for %t0 = [%lane to %quads step %lanes] {{
    %t = index.assume %t0 [lt(%t0, %quads)] : index
    %g = index.div %t, %c64 : index
    %within = index.rem %t, %c64 : index
    %lo = index.rem %within, %c{d} : index
    %hi = index.div %within, %c{d} : index
    %hi4 = index.mul %hi, %c4 : index
    %hi4d = index.mul %hi4, %c{d} : index
    %g256 = index.mul %g, %c256 : index
    %base0 = index.add %g256, %hi4d : index
    %base = index.add %base0, %lo : index
    %i1 = index.add %base, %c{d} : index
    %i2 = index.add %i1, %c{d} : index
    %i3 = index.add %i2, %c{d} : index
    %e0 = view.load %x_view[%base] : view<[%width]xf32> -> f32
    %e1 = view.load %x_view[%i1] : view<[%width]xf32> -> f32
    %e2 = view.load %x_view[%i2] : view<[%width]xf32> -> f32
    %e3 = view.load %x_view[%i3] : view<[%width]xf32> -> f32
    %s01 = scalar.addf %e0, %e1 : f32
    %s23 = scalar.addf %e2, %e3 : f32
    %d01 = scalar.subf %e0, %e1 : f32
    %d23 = scalar.subf %e2, %e3 : f32
    %o0 = scalar.addf %s01, %d23 : f32
    %o1 = scalar.subf %s01, %d23 : f32
    %o2 = scalar.addf %d01, %s23 : f32
    %o3 = scalar.subf %s23, %d01 : f32
    view.store %o0, %x_view[%base] : f32, view<[%width]xf32>
    view.store %o1, %x_view[%i1] : f32, view<[%width]xf32>
    view.store %o2, %x_view[%i2] : f32, view<[%width]xf32>
    view.store %o3, %x_view[%i3] : f32, view<[%width]xf32>
  }}
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
"""

CHUNK = """    %{p}_lane_step = index.mul %{p}_j, %lanes : index
    %{p}_chunk = index.add %{p}_lane_step, %lane : index
    %{p}_i0 = index.mul %{p}_chunk, %c8 : index
    %{p}_i = index.assume %{p}_i0 [le(%{p}_i0, %width_last), mul(%{p}_i0, 8)] : index
"""

def chunk(prefix: str) -> str:
    """Index math for one 8-element chunk per lane per iteration: chunk = j*256 + lane."""
    return CHUNK.replace("{p}", prefix)

FORM = {
    "norm": dict(
        args="%h: buffer, %norm_weight: buffer, %table: buffer, %cls: buffer",
        views="""  %h_global = buffer.assume.memory_space<global> %h : buffer
  %nw_global = buffer.assume.memory_space<global> %norm_weight : buffer
  %tab_global = buffer.assume.memory_space<global> %table : buffer
  %cls_global = buffer.assume.memory_space<global> %cls : buffer
  %h_view = buffer.view %h_global[%c0_offset] : buffer -> view<[%tokens_b]x[%width]xf16>
  %nw_view = buffer.view %nw_global[%c0_offset] : buffer -> view<[%width]xf32>
  // the AdaLN table: rows 2*class (scale) and 2*class + 1 (shift), classes = (timestep, modality) pairs
  %table_rows = index.mul %classes, %c2 : index
  %tab_view = buffer.view %tab_global[%c0_offset] : buffer -> view<[%table_rows]x[%width]xf32>
  %cls_view = buffer.view %cls_global[%c0_offset] : buffer -> view<[%tokens_b]xi32>
  %cls_i32 = view.load %cls_view[%row] : view<[%tokens_b]xi32> -> i32
  %cls_idx0 = index.cast %cls_i32 : i32 to index
  %cls_idx = index.assume %cls_idx0 [range(%cls_idx0, 0, 63), lt(%cls_idx0, %classes)] : index
  %scale_row0 = index.mul %cls_idx, %c2 : index
  %scale_row = index.assume %scale_row0 [range(%scale_row0, 0, 126), lt(%scale_row0, %table_rows)] : index
  %shift_row0 = index.add %scale_row, %c1 : index
  %shift_row = index.assume %shift_row0 [range(%shift_row0, 1, 127), lt(%shift_row0, %table_rows)] : index
""",
        form="""  // sum of squares, then normalise, scale and modulate into LDS
  %ss_v = scf.for %ss_j = [%c0 to %chunks_per_lane step %c1](%ss_acc = %zero8 : vector<8xf32>) -> (vector<8xf32>) {
""" + chunk("ss") + """    %ss_v16 = vector.load %h_view[%row, %ss_i] : view<[%tokens_b]x[%width]xf16> -> vector<8xf16>
    %ss_x = vector.extf %ss_v16 : vector<8xf16> to vector<8xf32>
    %ss_sq = vector.mulf %ss_x, %ss_x : vector<8xf32>
    %ss_next = vector.addf %ss_acc, %ss_sq : vector<8xf32>
    scf.yield %ss_next : vector<8xf32>
  }
  %ss = vector.reduce<addf> %ss_v, %zero : vector<8xf32>, f32
  %total = kernel.workgroup.reduce<addf> %ss : f32
  %width_i = index.cast %width : index to i32
  %width_f = scalar.sitofp %width_i : i32 to f32
  %mean = scalar.divf %total, %width_f : f32
  %mean_eps = scalar.addf %mean, %eps : f32
  %rms_inv = scalar.rsqrtf %mean_eps : f32
  %rms_inv8 = vector.splat %rms_inv : vector<8xf32>
  scf.for %m_j = [%c0 to %chunks_per_lane step %c1] {
""" + chunk("m") + """    %m_v16 = vector.load %h_view[%row, %m_i] : view<[%tokens_b]x[%width]xf16> -> vector<8xf16>
    %m_x = vector.extf %m_v16 : vector<8xf16> to vector<8xf32>
    %m_n = vector.mulf %m_x, %rms_inv8 : vector<8xf32>
    %m_nw = vector.load %nw_view[%m_i] : view<[%width]xf32> -> vector<8xf32>
    %m_normed = vector.mulf %m_n, %m_nw : vector<8xf32>
    %m_ms = vector.load %tab_view[%scale_row, %m_i] : view<[%table_rows]x[%width]xf32> -> vector<8xf32>
    %m_ms1 = vector.addf %m_ms, %one8 : vector<8xf32>
    %m_sh = vector.load %tab_view[%shift_row, %m_i] : view<[%table_rows]x[%width]xf32> -> vector<8xf32>
    %m_scaled = vector.mulf %m_normed, %m_ms1 : vector<8xf32>
    %m_out = vector.addf %m_scaled, %m_sh : vector<8xf32>
    vector.store %m_out, %x_view[%m_i] : vector<8xf32>, view<[%width]xf32>
  }
"""),
    "plain": dict(
        args="%h: buffer",
        views="""  %h_global = buffer.assume.memory_space<global> %h : buffer
  %h_view = buffer.view %h_global[%c0_offset] : buffer -> view<[%tokens_b]x[%width]xf16>
""",
        form="""  // the row as produced (the gate|up GEMM's fused silu(g)*u output)
  scf.for %m_j = [%c0 to %chunks_per_lane step %c1] {
""" + chunk("m") + """    %m_v16 = vector.load %h_view[%row, %m_i] : view<[%tokens_b]x[%width]xf16> -> vector<8xf16>
    %m_out = vector.extf %m_v16 : vector<8xf16> to vector<8xf32>
    vector.store %m_out, %x_view[%m_i] : vector<8xf32>, view<[%width]xf32>
  }
"""),
}


def kernel(name: str) -> str:
    f = FORM[name]
    lds = "f16" if name == "plain" else "f32"
    lds_bytes = 2 if lds == "f16" else 4
    ns, sym = f"h3.prepare_{name}_i4", f"h3_prepare_{name}_i4"
    extra_cfg = "" if name in ("norm", "plain") else f"\nconfig.decl @{ns}.gate_stride : %value: index where [range(%value, 256, 65536), mul(%value, 256)]\n"
    extra_get = "" if name in ("norm", "plain") else f"  %gate_stride = config.get @{ns}.gate_stride : index\n"
    gate_last = "" if name in ("norm", "plain") else "  %gate_last = index.sub %gate_stride, %c8 : index\n"
    eps_cfg = f"\nconfig.decl @{ns}.eps : f32\n\nconfig.decl @{ns}.classes : %value: index where [range(%value, 1, 64)]\n" if name == "norm" else ""
    eps_get = f"  %eps = config.get @{ns}.eps : f32\n  %classes = config.get @{ns}.classes : index\n" if name == "norm" else ""
    return f"""// GEMM input preparation ({name}), one workgroup of `lanes` lanes per token: form the
// row in f32 in LDS, rotate it by the group-256 Hadamard (H4 (x) H4 (x) H4 (x) H4 as four
// radix-4 stages of strides 1, 4, 16, 64; the 1/16 folds into the scale), take the
// token's absmax, and write symmetric int4 (q = round(x / s), s = absmax / 7, nibbles
// low first) with the f32 scale beside it -- the operand the int4 GEMM consumes.
//
// GENERATED by tools/gen_prepare.py; edit the generator.
amdgpu.target<gfx11-generic> @{sym}_gfx11 {{subgroup_size = 32}}

config.decl @{ns}.width : %value: index where [range(%value, 256, 32768), mul(%value, 256)]

config.decl @{ns}.lanes : %value: index where [range(%value, 32, 1024), mul(%value, 32)]
{eps_cfg}{extra_cfg}
kernel.def target(@{sym}_gfx11) export("{sym}") @{sym}(%tokens: index) {{
  %c1 = index.constant 1 : index
  %lanes = config.get @{ns}.lanes : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%lanes, %c1, %c1) : index
}} launch(%tokens: index, {f["args"]}, %q: buffer, %q_scale: buffer) {{
  %width = config.get @{ns}.width : index
  %lanes = config.get @{ns}.lanes : index
{eps_get}{extra_get}  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c4 = index.constant 4 : index
  %c8 = index.constant 8 : index
  %c16 = index.constant 16 : index
  %c64 = index.constant 64 : index
  %c256 = index.constant 256 : index
  %c0_offset = index.constant 0 : offset
  %one = scalar.constant 1.0 : f32
  %zero = scalar.constant 0.0 : f32
  %seven = scalar.constant 7.0 : f32
  %neg_seven = scalar.constant -7.0 : f32
  %sixteenth = scalar.constant 0.0625 : f32
  %tiny = scalar.constant 1e-30 : f32
  %fifteen = scalar.constant 15 : i32
  %sh4 = scalar.constant 4 : i32
  %sh8 = scalar.constant 8 : i32
  %sh12 = scalar.constant 12 : i32
  %sh16 = scalar.constant 16 : i32
  %sh20 = scalar.constant 20 : i32
  %sh24 = scalar.constant 24 : i32
  %sh28 = scalar.constant 28 : i32
  %zero8 = vector.splat %zero : vector<8xf32>
  %one8 = vector.splat %one : vector<8xf32>
  %seven8 = vector.splat %seven : vector<8xf32>
  %neg_seven8 = vector.splat %neg_seven : vector<8xf32>
  %fifteen8 = vector.splat %fifteen : vector<8xi32>
  %tokens_b = index.assume %tokens [range(%tokens, 1, 1048576)] : index
  %token = kernel.workgroup.id<x> : index
  %row = index.assume %token [lt(%token, %tokens_b)] : index
  %lane = kernel.workitem.id<x> : index
  %half_width = index.div %width, %c2 : index
  %word_width = index.div %width, %c8 : index
  %width_last = index.sub %width, %c8 : index
{gate_last}  %quads = index.div %width, %c4 : index
{f["views"]}  %q_global = buffer.assume.memory_space<global> %q : buffer
  %qs_global = buffer.assume.memory_space<global> %q_scale : buffer
  %qw_view = buffer.view %q_global[%c0_offset] : buffer -> view<[%tokens_b]x[%word_width]xi32>
  %qs_view = buffer.view %qs_global[%c0_offset] : buffer -> view<[%tokens_b]xf32>
  %row_bytes0 = index.mul %width, %c{lds_bytes} : index
  %row_bytes = index.cast %row_bytes0 : index to offset
  %lds = buffer.alloca<workgroup> align(16) %row_bytes : buffer
  %x_view = buffer.view %lds[%c0_offset] : buffer -> view<[%width]x{lds}>

{f["form"]}  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)

  // Hadamard: four radix-4 stages; each lane owns whole quads, so no barrier inside a stage.
{stage(1, lds)}{stage(4, lds)}{stage(16, lds)}{stage(64, lds)}
  // absmax (the 1/16 normalisation folded into the scale), quantise, pack 8 nibbles per lane per step
  %amax_v = scf.for %a_j = [%c0 to %chunks_per_lane step %c1](%a_acc = %zero8 : vector<8xf32>) -> (vector<8xf32>) {{
    %a_lane_step = index.mul %a_j, %lanes : index
    %a_chunk = index.add %a_lane_step, %lane : index
    %a_i0 = index.mul %a_chunk, %c8 : index
    %a_i = index.assume %a_i0 [le(%a_i0, %width_last), mul(%a_i0, 8)] : index
    %a_x = vector.load %x_view[%a_i] : view<[%width]xf32> -> vector<8xf32>
    %a_abs = vector.absf %a_x : vector<8xf32>
    %a_next = vector.maxnumf %a_acc, %a_abs : vector<8xf32>
    scf.yield %a_next : vector<8xf32>
  }}
  %amax = vector.reduce<maxnumf> %amax_v, %zero : vector<8xf32>, f32
  %row_max0 = kernel.workgroup.reduce<maxnumf> %amax : f32
  %row_max = scalar.mulf %row_max0, %sixteenth : f32
  %row_max_safe = scalar.maxnumf %row_max, %tiny : f32
  %s = scalar.divf %row_max_safe, %seven : f32
  %inv_s = scalar.divf %sixteenth, %s : f32
  %inv_s8 = vector.splat %inv_s : vector<8xf32>
  scf.for %q_j = [%c0 to %chunks_per_lane step %c1] {{
    %q_lane_step = index.mul %q_j, %lanes : index
    %q_chunk = index.add %q_lane_step, %lane : index
    %q_i0 = index.mul %q_chunk, %c8 : index
    %q_i = index.assume %q_i0 [le(%q_i0, %width_last), mul(%q_i0, 8)] : index
    %q_w = index.assume %q_chunk [lt(%q_chunk, %word_width)] : index
    %q_x = vector.load %x_view[%q_i] : view<[%width]xf32> -> vector<8xf32>
    %q_r = vector.mulf %q_x, %inv_s8 : vector<8xf32>
    %q_f = vector.roundevenf %q_r : vector<8xf32>
    %q_c = vector.maxnumf %q_f, %neg_seven8 : vector<8xf32>
    %q_d = vector.minnumf %q_c, %seven8 : vector<8xf32>
    %q_q = vector.fptosi %q_d : vector<8xf32> to vector<8xi32>
    %q_n = vector.andi %q_q, %fifteen8 : vector<8xi32>
    %q_e0 = vector.extract %q_n[0] : vector<8xi32> -> i32
    %q_e1 = vector.extract %q_n[1] : vector<8xi32> -> i32
    %q_e2 = vector.extract %q_n[2] : vector<8xi32> -> i32
    %q_e3 = vector.extract %q_n[3] : vector<8xi32> -> i32
    %q_e4 = vector.extract %q_n[4] : vector<8xi32> -> i32
    %q_e5 = vector.extract %q_n[5] : vector<8xi32> -> i32
    %q_e6 = vector.extract %q_n[6] : vector<8xi32> -> i32
    %q_e7 = vector.extract %q_n[7] : vector<8xi32> -> i32
    %q_s1 = scalar.shli %q_e1, %sh4 : i32
    %q_s2 = scalar.shli %q_e2, %sh8 : i32
    %q_s3 = scalar.shli %q_e3, %sh12 : i32
    %q_s4 = scalar.shli %q_e4, %sh16 : i32
    %q_s5 = scalar.shli %q_e5, %sh20 : i32
    %q_s6 = scalar.shli %q_e6, %sh24 : i32
    %q_s7 = scalar.shli %q_e7, %sh28 : i32
    %q_o1 = scalar.ori %q_e0, %q_s1 : i32
    %q_o2 = scalar.ori %q_o1, %q_s2 : i32
    %q_o3 = scalar.ori %q_o2, %q_s3 : i32
    %q_o4 = scalar.ori %q_o3, %q_s4 : i32
    %q_o5 = scalar.ori %q_o4, %q_s5 : i32
    %q_o6 = scalar.ori %q_o5, %q_s6 : i32
    %q_o7 = scalar.ori %q_o6, %q_s7 : i32
    view.store %q_o7, %qw_view[%row, %q_w] : i32, view<[%tokens_b]x[%word_width]xi32>
  }}
  // the token scale, by one lane, after the last loop (a divergent region before a
  // loop is rejected by the branch lowering)
  %is_first = index.cmp eq, %lane, %c0 : index
  scf.if %is_first {{
    view.store %s, %qs_view[%row] : f32, view<[%tokens_b]xf32>
  }}
  kernel.return
}}
"""


def lds_type(text: str, lds: str) -> str:
    """Route every LDS row access through the chosen element type: f16 storage with
    f32 arithmetic (extf on load, fptrunc on store)."""
    if lds == "f32":
        return text
    text = text.replace("view<[%width]xf32>", "view<[%width]xf16>")
    out = []
    for line in text.split("\n"):
        m = re.match(r"^(\s*)(%\w+) = view\.load %x_view\[(%\w+)\] : view<\[%width\]xf16> -> f32$", line)
        if m:
            ind, name, idx = m.groups()
            out.append(f"{ind}{name}_h = view.load %x_view[{idx}] : view<[%width]xf16> -> f16")
            out.append(f"{ind}{name} = scalar.extf {name}_h : f16 to f32")
            continue
        m = re.match(r"^(\s*)view\.store (%\w+), %x_view\[(%\w+)\] : f32, view<\[%width\]xf16>$", line)
        if m:
            ind, name, idx = m.groups()
            out.append(f"{ind}{name}_h = scalar.fptrunc {name} : f32 to f16")
            out.append(f"{ind}view.store {name}_h, %x_view[{idx}] : f16, view<[%width]xf16>")
            continue
        m = re.match(r"^(\s*)(%\w+) = vector\.load %x_view\[(%\w+)\] : view<\[%width\]xf16> -> vector<(\d+)xf32>$", line)
        if m:
            ind, name, idx, n = m.groups()
            out.append(f"{ind}{name}_h = vector.load %x_view[{idx}] : view<[%width]xf16> -> vector<{n}xf16>")
            out.append(f"{ind}{name} = vector.extf {name}_h : vector<{n}xf16> to vector<{n}xf32>")
            continue
        m = re.match(r"^(\s*)vector\.store (%\w+), %x_view\[(%\w+)\] : vector<(\d+)xf32>, view<\[%width\]xf16>$", line)
        if m:
            ind, name, idx, n = m.groups()
            out.append(f"{ind}{name}_h = vector.fptrunc {name} : vector<{n}xf32> to vector<{n}xf16>")
            out.append(f"{ind}vector.store {name}_h, %x_view[{idx}] : vector<{n}xf16>, view<[%width]xf16>")
            continue
        out.append(line)
    return "\n".join(out)


def uniform_loops(text: str) -> str:
    """`scf.for %x0 = [%lane to %bound step %c256]` has a lane-dependent entry, which the
    branch lowering rejects; every lane runs bound/256 iterations, so loop over that
    count and add the lane inside."""
    counts = {"%width": "%width_per_lane", "%quads": "%quads_per_lane", "%half_width": "%half_per_lane"}
    def repl(m):
        var, bound, carried = m.group(1), m.group(2), m.group(3) or ""
        return (f"scf.for %{var}_j = [%c0 to {counts[bound]} step %c1]{carried} {{\n"
                f"    %{var}_lane_step = index.mul %{var}_j, %lanes : index\n"
                f"    %{var}0 = index.add %{var}_lane_step, %lane : index\n")
    text = re.sub(r"scf\.for %(\w+)0 = \[%lane to (%\w+) step %lanes\]((?:\(.*?\) -> \(.*?\))?) \{\n", repl, text)
    return text.replace("  %quads = index.div %width, %c4 : index\n",
                        "  %quads = index.div %width, %c4 : index\n  %width_per_lane = index.div %width, %lanes : index\n"
                        "  %quads_per_lane = index.div %quads, %lanes : index\n  %chunks_per_lane = index.div %word_width, %lanes : index\n")

for name in FORM:
    (OUT / f"prepare_{name}_i4.loom").write_text(uniform_loops(lds_type(kernel(name), "f16" if name == "plain" else "f32")))
    print("wrote", f"prepare_{name}_i4.loom")

"""Generate the audio decoder convolution with four independent f32 accumulators."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def generate():
    v = 4
    original = (ROOT / 'h3/kernels/conv1d_f32.loom').read_text()
    stem = 'conv1d4_f32'
    s = original[:original.index('  %active = index.cmp')]
    s = s.replace('conv1d_f32', stem)
    s = s.replace('%wg = kernel.workgroup.id<x> : index', '%wg_raw = kernel.workgroup.id<x> : index\n  %group_bound = index.div %len_bound, %c256 : index\n  %wg = index.assume %wg_raw [lt(%wg_raw, %group_bound)] : index')
    s = s.replace('  %cout = config.get', f'  %threads = index.constant {256//v} : index\n  %cout = config.get', 1)
    s = s.replace('workgroup_size(%c256,', 'workgroup_size(%threads,')
    lines = [s]
    def put(x): lines.append(x+'\n')
    put('  %bias = view.load %b_view[%o] : view<[%cout]xf32> -> f32')
    put('  %is_acc = index.cmp eq, %accumulate, %c1 : index')
    for j in range(v):
        put(f'  %offset{j} = index.constant {j*(256//v)} : index')
        put(f'  %n{j} = index.add %n, %offset{j} : index')
        put(f'  %active{j} = index.cmp ult, %n{j}, %len_b : index')
        put(f'  %read{j} = scalar.andi %is_acc, %active{j} : i1')
        put(f'  %prev{j} = scf.if %read{j} -> (f32) {{')
        put(f'    %nn = index.assume %n{j} [lt(%n{j}, %len_b)] : index')
        put('    %old = view.load %out_view[%o, %nn] : view<[%cout]x[%len_b]xf32> -> f32')
        put('    scf.yield %old : f32\n  } else {\n    scf.yield %zero : f32\n  }')
        put(f'  %start{j} = scalar.addf %prev{j}, %bias : f32')
    put('  %km1 = index.sub %ksize, %c1 : index')
    put('  %reach_max = index.mul %km1, %dilation : index')
    put('  %c255 = index.constant 255 : index')
    put('  %block_last = index.add %base, %c255 : index')
    put('  %last_plus = index.add %block_last, %reach_max : index')
    put('  %limit = index.add %len_b, %pad : index')
    put('  %ge_first = index.cmp uge, %base, %pad : index')
    put('  %lt_last = index.cmp ult, %last_plus, %limit : index')
    put('  %interior0 = scalar.andi %ge_first, %lt_last : i1')
    put('  %block_full = index.cmp ult, %block_last, %len_b : index')
    put('  %interior = scalar.andi %interior0, %block_full : i1')
    types = ', '.join(['f32']*v)
    names = lambda x: ', '.join(f'%{x}{j}' for j in range(v))
    put(f'  {names("result")} = scf.if %interior -> ({types}) {{')
    for checked in [False, True]:
        init = ', '.join(f'%ai{j} = %start{j} : f32' for j in range(v))
        put(f'    {names("acc")} = scf.for %i = [%c0 to %cin step %c1]({init}) -> ({types}) {{')
        put('      %ii = index.assume %i [lt(%i, %cin)] : index')
        put('      %row = index.mul %ii, %ksize : index')
        init = ', '.join(f'%ak{j} = %ai{j} : f32' for j in range(v))
        put(f'      {names("acc_k")} = scf.for %k = [%c0 to %ksize step %c1]({init}) -> ({types}) {{')
        put('        %kk = index.assume %k [lt(%k, %ksize)] : index')
        put('        %tap0 = index.add %row, %kk : index')
        put('        %tap = index.assume %tap0 [lt(%tap0, %taps)] : index')
        put('        %reach = index.mul %kk, %dilation : index')
        put('        %wv = view.load %w_view[%o, %tap] : view<[%cout]x[%taps]xf32> -> f32')
        for j in range(v):
            put(f'        %sp{j} = index.add %n{j}, %reach : index')
            put(f'        %raw{j} = index.sub %sp{j}, %pad : index')
            if checked:
                put(f'        %ge{j} = index.cmp uge, %sp{j}, %pad : index')
                put(f'        %lt{j} = index.cmp ult, %raw{j}, %len_b : index')
                put(f'        %valid0_{j} = scalar.andi %ge{j}, %lt{j} : i1')
                put(f'        %valid{j} = scalar.andi %valid0_{j}, %active{j} : i1')
                put(f'        %next{j} = scf.if %valid{j} -> (f32) {{')
            put(f'          %src{j} = index.assume %raw{j} [range(%raw{j}, 0, 4194304), lt(%raw{j}, %len_b)] : index')
            put(f'          %xv{j} = view.load %x_view[%ii, %src{j}] : view<[%cin]x[%len_b]xf32> -> f32')
            put(f'          %{("fma" if checked else "next")}{j} = scalar.fmaf %wv, %xv{j}, %ak{j} : f32')
            if checked:
                put(f'          scf.yield %fma{j} : f32\n        }} else {{\n          scf.yield %ak{j} : f32\n        }}')
        put(f'        scf.yield {names("next")} : {types}\n      }}')
        put(f'      scf.yield {names("acc_k")} : {types}\n    }}')
        put(f'    scf.yield {names("acc")} : {types}')
        put('  } else {' if not checked else '  }')
    for j in range(v):
        put(f'  scf.if %active{j} {{')
        put(f'    %nn = index.assume %n{j} [lt(%n{j}, %len_b)] : index')
        put(f'    view.store %result{j}, %out_view[%o, %nn] : f32, view<[%cout]x[%len_b]xf32>\n  }}')
    put('  kernel.return\n}')
    source = ''.join(lines)
    source = re.sub(r'(scf.for %k = .*? -> \([^\n]+\)) \{', r'\1 unroll {', source)
    source = source[source.index('amdgpu.target'):]
    return ('// Four f32 output samples per lane; 64 lanes cover 256 samples per workgroup.\n'
            '// Interior workgroups skip padding checks. Boundary workgroups mask each tap.\n'
            '// Ascending input-channel/tap FMA order and residual/bias rounding are preserved.\n'
            '// Generated by tools/gen_conv1d4_f32.py.\n' + source)

if __name__ == '__main__':
    (ROOT / 'h3/kernels/conv1d4_f32.loom').write_text(generate())

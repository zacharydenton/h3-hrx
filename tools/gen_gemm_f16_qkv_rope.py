"""Fuse the decoder QKV projection with normalization and rotary embedding.

The GEMM retains f32 accumulation and the original saturated f16 projection
boundary. Each wave owns one 64-channel head; its four column fragments are
reduced across 16 lanes independently for the eight rows held by each lane.
RMSNorm uses f32 sums. Q/K rotate channels 0..47 in pairs separated by 24;
V retains the rounded projection. Output is [3][32][capacity][64] f16.
"""
from pathlib import Path
import re
from gen_gemm_f16_fast import generate as generate_fast

ROOT = Path(__file__).resolve().parent.parent
STEM = "gemm_f16_qkvropehm_256b"


def generate():
    stem='gemm_f16_qkvropehm_256b';ns='h3.'+stem
    s=generate_fast('plain').replace('gemm_f16_fast_256b',stem)
    p=s.index('kernel.def')
    s=s[:p]+f'config.decl @{ns}.token_capacity : %value: index where [range(%value, 1, 16777216)]\nconfig.decl @{ns}.eps : f32\n\n'+s[p:]
    s=s.replace('%bias: buffer) {','%bias: buffer, %qw: buffer, %kw: buffer, %cos: buffer, %sin: buffer) {')
    assert '%qw: buffer' in s
    s=s[:s.index('  %publish_row0 =')]
    s+=f'''  scf.schedule.fence
      %fr_capacity = config.get @{ns}.token_capacity : index
      %fr_eps_f = config.get @{ns}.eps : f32
      %fr_eps = vector.splat %fr_eps_f : vector<8xf32>
      %fr_c3 = index.constant 3 : index
      %fr_width = index.div %n_size, %fr_c3 : index
      %fr_view = buffer.view %c_global[%c0_offset] : buffer -> view<3x[%fr_capacity]x[%fr_width]xf16>
      %fr_qwg = buffer.assume.memory_space<global> %qw : buffer
      %fr_kwg = buffer.assume.memory_space<global> %kw : buffer
      %fr_cg = buffer.assume.memory_space<global> %cos : buffer
      %fr_sg = buffer.assume.memory_space<global> %sin : buffer
      %fr_qw = buffer.view %fr_qwg[%c0_offset] : buffer -> view<64xf32>
      %fr_kw = buffer.view %fr_kwg[%c0_offset] : buffer -> view<64xf32>
      %fr_cv = buffer.view %fr_cg[%c0_offset] : buffer -> view<[%m_bounded]x24xf32>
      %fr_sv = buffer.view %fr_sg[%c0_offset] : buffer -> view<[%m_bounded]x24xf32>
      %fr_group = index.div %lane, %c16 : index
      %fr_mbase = index.add %base_m, %wave_row : index
      %fr_nbase = index.add %base_n, %wave_col : index
      %fr_type0 = index.div %fr_nbase, %fr_width : index
      %fr_type = index.assume %fr_type0 [range(%fr_type0, 0, 2)] : index
      %fr_headcol = index.rem %fr_nbase, %fr_width : index
      %fr_isq = index.cmp eq, %fr_type, %c0 : index
      %fr_isv = index.cmp eq, %fr_type, %c2 : index
      %fr_low = index.cmp ult, %lane16, %c8 : index
      %fr_zero = vector.constant 0.0 : vector<8xf32>
      %fr_inv = vector.constant 0.015625 : vector<8xf32>
      %fr_max = vector.constant 65472.0 : vector<8xf32>
      %fr_min = vector.constant -65472.0 : vector<8xf32>
      %fr_i32_32 = scalar.constant 32 : i32
      %fr_i32_8 = scalar.constant 8 : i32
      %fr_i32_4 = scalar.constant 4 : i32
      %fr_i32_2 = scalar.constant 2 : i32
      %fr_i32_1 = scalar.constant 1 : i32
      %fr_lastrow = index.sub %m_bounded, %c1 : index
      %fr_pair1a = index.add %lane16, %c16 : index
      %fr_pair1b = index.sub %lane16, %c8 : index
      %fr_pair10 = scf.select %fr_low, %fr_pair1a, %fr_pair1b : index
      %fr_pair1 = index.assume %fr_pair10 [range(%fr_pair10, 0, 23)] : index
      %fr_pair2 = index.add %lane16, %c8 : index
    '''
    for j in range(4):
     s+=f'''  %fr_off{j} = index.constant {16*j} : index
      %fr_channel{j} = index.add %lane16, %fr_off{j} : index
      %fr_col{j} = index.add %fr_nbase, %fr_channel{j} : index
      %fr_oc0{j} = index.add %fr_headcol, %fr_channel{j} : index
      %fr_oc{j} = index.assume %fr_oc0{j} [lt(%fr_oc0{j}, %fr_width)] : index
      %fr_b{j} = view.load %bias_view[%fr_col{j}] : view<[%n_size]xf32> -> f32
      %fr_bv{j} = vector.splat %fr_b{j} : vector<8xf32>
      %fr_wq{j} = view.load %fr_qw[%fr_channel{j}] : view<64xf32> -> f32
      %fr_wk{j} = view.load %fr_kw[%fr_channel{j}] : view<64xf32> -> f32
      %fr_w{j} = scf.select %fr_isq, %fr_wq{j}, %fr_wk{j} : f32
      %fr_wv{j} = vector.splat %fr_w{j} : vector<8xf32>
    '''
    for i in range(4):
     for r in range(8):
      t=f'{i}_{r}'
      s+=f'''  %fr_roff{t} = index.constant {16*i+2*r} : index
      %fr_ra{t} = index.add %fr_mbase, %fr_roff{t} : index
      %fr_row{t} = index.add %fr_ra{t}, %fr_group : index
      %fr_ok{t} = index.cmp ult, %fr_row{t}, %m_bounded : index
      %fr_safe0{t} = scf.select %fr_ok{t}, %fr_row{t}, %fr_lastrow : index
      %fr_safe{t} = index.assume %fr_safe0{t} [lt(%fr_safe0{t}, %m_bounded)] : index
    '''
     for j in range(4):
      t=f'{i}{j}'
      s+=f'''  %fr_z{t} = vector.addf %acc{t}, %fr_zero : vector<8xf32>
      %fr_biased{t} = vector.addf %fr_z{t}, %fr_bv{j} : vector<8xf32>
      %fr_sat0{t} = vector.minnumf %fr_biased{t}, %fr_max : vector<8xf32>
      %fr_sat{t} = vector.maxnumf %fr_sat0{t}, %fr_min : vector<8xf32>
      %fr_half{t} = vector.fptrunc %fr_sat{t} : vector<8xf32> to vector<8xf16>
      %fr_float{t} = vector.extf %fr_half{t} : vector<8xf16> to vector<8xf32>
      %fr_sq{t} = vector.mulf %fr_float{t}, %fr_float{t} : vector<8xf32>
    '''
     s+=f'''  %fr_sum01{i} = vector.addf %fr_sq{i}0, %fr_sq{i}1 : vector<8xf32>
      %fr_sum23{i} = vector.addf %fr_sq{i}2, %fr_sq{i}3 : vector<8xf32>
      %fr_sum{i}_0 = vector.addf %fr_sum01{i}, %fr_sum23{i} : vector<8xf32>
    '''
     prev=f'%fr_sum{i}_0'
     for step,off in enumerate((8,4,2,1),1):
      s+=f'''  %fr_shuffle{i}_{step}, %fr_valid{i}_{step} = kernel.subgroup.shuffle<xor> {prev}, %fr_i32_{off}, %fr_i32_32 : vector<8xf32>, i32, i32
      %fr_sum{i}_{step} = vector.addf {prev}, %fr_shuffle{i}_{step} : vector<8xf32>
    '''
      prev=f'%fr_sum{i}_{step}'
     s+=f'''  %fr_mean{i} = vector.mulf {prev}, %fr_inv : vector<8xf32>
      %fr_me{i} = vector.addf %fr_mean{i}, %fr_eps : vector<8xf32>
      %fr_rinv{i} = vector.rsqrtf %fr_me{i} : vector<8xf32>
    '''
     for j in range(4):
      t=f'{i}{j}'
      s+=f'''  %fr_norm{t} = vector.mulf %fr_float{t}, %fr_rinv{i} : vector<8xf32>
      %fr_x{t} = vector.mulf %fr_norm{t}, %fr_wv{j} : vector<8xf32>
    '''
      if j<3:s+=f'  %fr_sx{t}, %fr_xvalid{t} = kernel.subgroup.shuffle<xor> %fr_x{t}, %fr_i32_8, %fr_i32_32 : vector<8xf32>, i32, i32\n'
     for j in range(4):
      t=f'{i}{j}';result=f'%fr_x{t}'
      if j<3:
       lo,hi=((1,2),(2,0),(0,1))[j]
       s+=f'  %fr_px{t} = scf.select %fr_low, %fr_sx{i}{lo}, %fr_sx{i}{hi} : vector<8xf32>\n'
       for r in range(8):
        for letter,view in (('c','cv'),('s','sv')):
         col='%lane16' if j==0 else f'%fr_pair{j}'
         s+=f'  %fr_{letter}{t}_{r} = view.load %fr_{view}[%fr_safe{i}_{r}, {col}] : view<[%m_bounded]x24xf32> -> f32\n'
       for letter in ('c','s'):s+=f'  %fr_{letter}v{t} = vector.from_elements '+', '.join(f'%fr_{letter}{t}_{r}' for r in range(8))+' : vector<8xf32>\n'
       s+=f'''  %fr_xc{t} = vector.mulf %fr_x{t}, %fr_cv{t} : vector<8xf32>
      %fr_ps{t} = vector.mulf %fr_px{t}, %fr_sv{t} : vector<8xf32>
      %fr_minus{t} = vector.subf %fr_xc{t}, %fr_ps{t} : vector<8xf32>
      %fr_plus{t} = vector.addf %fr_xc{t}, %fr_ps{t} : vector<8xf32>
    '''
       if j==1:s+=f'  %fr_rot{t} = scf.select %fr_low, %fr_minus{t}, %fr_plus{t} : vector<8xf32>\n';result=f'%fr_rot{t}'
       else:result=f'%fr_minus{t}' if j==0 else f'%fr_plus{t}'
      s+=f'''  %fr_final{t} = scf.select %fr_isv, %fr_float{t}, {result} : vector<8xf32>
      %fr_out{t} = vector.fptrunc %fr_final{t} : vector<8xf32> to vector<8xf16>
    '''
      for r in range(8):
       tag=f'{t}_{r}'
       s+=f'''  scf.if %fr_ok{i}_{r} {{
        %fr_bound{tag} = index.assume %fr_row{i}_{r} [lt(%fr_row{i}_{r}, %fr_capacity)] : index
        %fr_scalar{tag} = vector.extract %fr_out{t}[{r}] : vector<8xf16> -> f16
        view.store %fr_scalar{tag}, %fr_view[%fr_type, %fr_bound{tag}, %fr_oc{j}] : f16, view<3x[%fr_capacity]x[%fr_width]xf16>
      }}
    '''
    s+='  kernel.return\n}\n'

    # A wave's four column fragments form one 64-channel head. V bypasses
    # normalization and rotation, retaining the projection's rounded half values.
    for i in reversed(range(4)):
     a=s.index(f'  %fr_sum01{i} =');b=s.index(f'  %fr_roff{i+1}_0 =') if i<3 else s.index('  kernel.return',a)
     body=s[a:b];v='  scf.if %fr_isv {\n'
     for j in range(4):
      for r in range(8):
       tag=f'{i}{j}_{r}'
       v+=f'''    scf.if %fr_ok{i}_{r} {{
          %vs_bound{tag} = index.assume %fr_row{i}_{r} [lt(%fr_row{i}_{r}, %fr_capacity)] : index
          %vs_scalar{tag} = vector.extract %fr_half{i}{j}[{r}] : vector<8xf16> -> f16
          view.store %vs_scalar{tag}, %fr_view[%fr_type, %vs_bound{tag}, %fr_oc{j}] : f16, view<3x[%fr_capacity]x[%fr_width]xf16>
        }}
    '''
     v+='  } else {\n'+body+'  }\n';s=s[:a]+v+s[b:]
    # Publish [Q/K/V][head][capacity][64], matching head-major attention.
    s=s.replace('  %fr_view =','  %fr_heads = index.div %fr_width, %c64 : index\n  %fr_view =')
    s=s.replace('view<3x[%fr_capacity]x[%fr_width]xf16>','view<3x[%fr_heads]x[%fr_capacity]x64xf16>')
    s=s.replace('  %fr_isq =','  %fr_head0 = index.div %fr_headcol, %c64 : index\n  %fr_head = index.assume %fr_head0 [lt(%fr_head0, %fr_heads)] : index\n  %fr_isq =')
    s,n=re.subn(r'%fr_view\[%fr_type, (%(?:vs|fr)_bound\w+), %fr_oc([0-3])\]',r'%fr_view[%fr_type, %fr_head, \1, %fr_channel\2]',s)
    assert n==256,n
    # The decoder's Q/K/V widths are equal, with 32 heads of 64 channels.
    s=s.replace('range(%value, 256, 65536), mul(%value, 256)', 'range(%value, 6144, 6144), mul(%value, 256)')
    s=re.sub(r'^\s*//[^\n]*\n','',s,flags=re.M)
    return ('// Decoder QKV projection, f16 rounding, head-64 RMSNorm and 48-channel RoPE.\n'
            '// 128x256 workgroups; ascending WMMA accumulation; head-major Q/K/V output.\n'
            '// Generated by tools/gen_gemm_f16_qkv_rope.py.\n'+s)


if __name__ == "__main__":
    (ROOT / "h3/kernels" / f"{STEM}.loom").write_text(generate())

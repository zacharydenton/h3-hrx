"""Head-64 f16 attention in transposed WMMA form, used by gen_attention_lds.

K Q^T makes each lane own one query and eight keys, so max/sum statistics
are scalar per lane. P^T is narrowed before exchanging packed half words,
then reused as the rhs of V^T P^T. The same register packing publishes
contiguous output channels without an LDS round trip. Model precision and
ascending 16-key PV accumulation are unchanged; online softmax uses base 2.
"""
import re


def _transposed(s):
    """Transpose both WMMAs and keep one query per lane pair."""
    def sub(a,b,n=1):
        nonlocal s
        assert s.count(a)==n,(a,s.count(a))
        s=s.replace(a,b)
    sub('  %i32_32 = scalar.constant 32 : i32','  %i32_32 = scalar.constant 32 : i32\n  %i32_16 = scalar.constant 16 : i32')
    sub('  %q_channel0 =','  %query_lane_row0 = index.add %query_origin0, %lane_column : index\n  %query_lane_row = index.assume %query_lane_row0 [lt(%query_lane_row0, %padded_tokens)] : index\n  %q_channel0 =')
    for c in range(4):
        sub(f'  %lhs{c} = vector.fragment.load<lhs> %q_view[%query_origin0, %q_channel{c}] shape [%m, %k_frag] : view<[%padded_tokens]x[%q_stride0]xf16> -> vector<16xf16>',f'  %query_packet{c} = vector.load %q_view[%query_lane_row, %q_channel{c}] : view<[%padded_tokens]x[%q_stride0]xf16> -> vector<16xf16>\n  %qrhs{c} = vector.fragment<rhs> %query_packet{c} shape [%k_frag, %n] : vector<16xf16>')
        for hi in ('','_hi'):
            sub(f'%rhs{hi}{c} = vector.fragment<rhs> %k_data{hi}{c} shape [%k_frag, %n]',f'%klhs{hi}{c} = vector.fragment<lhs> %k_data{hi}{c} shape [%m, %k_frag]')
            sub(f'vector.mma %lhs{c}, %rhs{hi}{c},',f'vector.mma %klhs{hi}{c}, %qrhs{c},')
    # Carry only this lane's query's scalar max and partial row sum.
    sub('%row_max = %negative_vector : vector<8xf32>, %row_sum = %zero_vector : vector<8xf32>','%row_max = %negative_large : f32, %row_sum = %zero_f32 : f32')
    a=s.index('  %final_max,');b=s.index('\n',a)
    line=s[a:b].replace('-> (vector<8xf32>, vector<8xf32>,','-> (f32, f32,');s=s[:a]+line+s[b:]
    # Per-element key masking in transposed scores.
    for hi,off in (('',0),('_hi',16)):
        a=s.index(f'    %scaled{hi} = scf.if');b=s.index('\n    }',s.index('else {',a))+6
        mask=''
        for i in range(8):
            tag=f'mask{hi}{i}'
            mask+=f'    %{tag}a = index.add %key_origin0, %c{2*i} : index\n'
            if off:
                mask+=f'    %{tag}b = index.add %{tag}a, %c16 : index\n'
            mask+=f'    %{tag}key = index.add %{tag}{"b" if off else "a"}, %lane_group : index\n    %{tag}ok = index.cmp ult, %{tag}key, %tokens0 : index\n    %{tag}value = scf.select %{tag}ok, %zero_f32, %negative_large : f32\n'
        mask+=f'    %mask{hi} = vector.from_elements '+', '.join(f'%mask{hi}{i}value' for i in range(8))+' : vector<8xf32>\n'
        mask+=f'    %scaled{hi} = vector.addf<reassoc|nnan|ninf|nsz> %scaled{hi}0, %mask{hi} : vector<8xf32>'
        s=s[:a]+mask+s[b:]
    a=s.index('    %sh1,');b=s.index('    %vrow0 =',a)
    s=s[:a]+'''    %half_max = vector.reduce<maxnumf> %pair_max, %negative_large : vector<8xf32>, f32
    %partner_max, %max_valid = kernel.subgroup.shuffle<xor> %half_max, %i32_16, %i32_32 : f32, i32, i32
    %tile_max = scalar.maxnumf %half_max, %partner_max : f32
    %next_max = scalar.maxnumf %row_max, %tile_max : f32
    %next_max_v = vector.splat %next_max : vector<8xf32>
    %delta = vector.subf<reassoc|nnan|ninf|nsz> %scaled, %next_max_v : vector<8xf32>
    %delta_hi = vector.subf<reassoc|nnan|ninf|nsz> %scaled_hi, %next_max_v : vector<8xf32>
    %weight = vector.expf<afn> %delta : vector<8xf32>
    %weight_hi = vector.expf<afn> %delta_hi : vector<8xf32>
    %old_delta = scalar.subf %row_max, %next_max : f32
    %old_scale = scalar.expf<afn> %old_delta : f32
    %selected_old_scale = vector.splat %old_scale : vector<8xf32>
    %tile_sum_lo = vector.reduce<addf> %weight, %zero_f32 : vector<8xf32>, f32
    %tile_sum_hi = vector.reduce<addf> %weight_hi, %zero_f32 : vector<8xf32>, f32
    %scaled_sum = scalar.mulf %row_sum, %old_scale : f32
    %next_sum_lo = scalar.addf %scaled_sum, %tile_sum_lo : f32
    %next_sum = scalar.addf %next_sum_lo, %tile_sum_hi : f32
    %probability = vector.fragment.repack<rhs> %weight shape [%m, %n] : vector<8xf32> -> vector<16xf16>
    %probability_hi = vector.fragment.repack<rhs> %weight_hi shape [%m, %n] : vector<8xf32> -> vector<16xf16>
'''+s[b:]
    for c in range(4):
        for hi in ('','_hi'):
            sub(f'%v{hi}{c} = vector.fragment<rhs> %v_data{hi}{c} shape [%k_frag, %n]',f'%v{hi}{c} = vector.fragment<lhs> %v_data{hi}{c} shape [%m, %k_frag]')
            sub(f'vector.mma %probability{hi}, %v{hi}{c},',f'vector.mma %v{hi}{c}, %probability{hi},')
    a=s.index('    scf.yield %next_max,');b=s.index('\n',a)
    line=s[a:b].replace(': vector<8xf32>, vector<8xf32>,',': f32, f32,');s=s[:a]+line+s[b:]
    a=s.index('  %fsh1,');s=s[:a]+'''  %partner_sum, %sum_valid = kernel.subgroup.shuffle<xor> %final_sum, %i32_16, %i32_32 : f32, i32, i32
  %total_sum = scalar.addf %final_sum, %partner_sum : f32
  %selected_sum = vector.splat %total_sum : vector<8xf32>
  %query_present = index.cmp ult, %query_lane_row, %tokens0 : index
  %query_in_range = index.cmp ult, %query_lane_row, %token_count : index
  %query_live = scalar.andi %query_present, %query_in_range : i1
  %writes = scalar.andi %query_live, %lane_group_even : i1
'''
    # Replicated rhs layout is exactly sixteen contiguous output channels per query lane.
    for c in range(4):
        s+=f'''  %out{c} = vector.divf<nnan|ninf|nsz|arcp> %final{c}, %selected_sum : vector<8xf32>
            %out_packet{c} = vector.fragment.repack<rhs> %out{c} shape [%m, %n] : vector<8xf32> -> vector<16xf16>
            scf.if %writes {{
                    %out_row{c} = index.assume %query_lane_row [lt(%query_lane_row, %token_count)] : index
                    vector.store %out_packet{c}, %out_view[%out_row{c}, %q_channel{c}] : vector<16xf16>, view<[%token_count]x[%out_stride0]xf16>
            }}
    '''
    s+='  kernel.return\n}\n'
    s=s.replace('  %writes = scalar.andi %query_live, %lane_group_even : i1', '  %tile_present = index.cmp ult, %tile_in_image0, %tiles_per_image : index\n  %writes0 = scalar.andi %query_live, %lane_group_even : i1\n  %writes = scalar.andi %writes0, %tile_present : i1')
    s=s.replace('%lds_bytes = index.constant 13824','%lds_bytes = index.constant 9728')
    # Remove unused scratch views now outside the smaller allocation.
    for name in ('scratch_view','scratch_view_hi'):
        s=re.sub(rf'^  %{name} = [^\n]*\n','',s,flags=re.M)
    return s


def _base2(s):
    """Use base-2 softmax and mask only the partial final key tile."""
    # Most key tiles are complete. Mask only the final partial tile.
    pos=s.index('    %mask0a =')
    s=s[:pos]+'    %key_end = index.add %key_origin0, %c32 : index\n    %keys_full = index.cmp ule, %key_end, %tokens0 : index\n'+s[pos:]
    for hi in ('','_hi'):
        a=s.index(f'    %mask{hi}0a =');b=s.index('\n',s.index(f'    %scaled{hi} = vector.addf',a))
        block=s[a:b].replace(f'%scaled{hi} = vector.addf',f'%masked{hi} = vector.addf')
        block='\n'.join('  '+l for l in block.splitlines())
        s=s[:a]+f'    %scaled{hi} = scf.if %keys_full -> (vector<8xf32>) {{\n      scf.yield %scaled{hi}0 : vector<8xf32>\n    }} else {{\n'+block+f'\n      scf.yield %masked{hi} : vector<8xf32>\n    }}'+s[b:]
    # Base-2 online softmax: fold log2(e) into the constant score scale.
    s=s.replace('  %scale_vector = vector.splat %scale : vector<8xf32>','  %log2e = scalar.constant 1.4426950408889634 : f32\n  %scale_log2 = scalar.mulf %scale, %log2e : f32\n  %scale_vector = vector.splat %scale_log2 : vector<8xf32>')
    s=s.replace('vector.expf<afn>','vector.exp2f<afn>').replace('scalar.expf<afn>','scalar.exp2f<afn>')
    return s


def _pack(s, stem):
    """Exchange narrowed half words, then interleave even/odd key rows."""
    vg='reg<amdgpu.vgpr>'
    helper=f'low.func.def target<amdgpu.gfx11.generic.core>(@h3_{stem}_gfx11)\n    @h3_vae_pack(%even: reg<amdgpu.vgpr x4>, %odd: reg<amdgpu.vgpr x4>) -> (reg<amdgpu.vgpr x8>) asm {{\n  %lo = s_mov_b32 0x05040100\n  %hi = s_mov_b32 0x07060302\n'
    for i in range(4):
        helper+=f'  %e{i} = slice %even[{i}] : reg<amdgpu.vgpr x4> -> {vg}\n  %o{i} = slice %odd[{i}] : reg<amdgpu.vgpr x4> -> {vg}\n  %p{2*i} = v_perm_b32 %o{i}, %e{i}, %lo\n  %p{2*i+1} = v_perm_b32 %o{i}, %e{i}, %hi\n'
    helper+='  %p = concat('+', '.join(f'%p{i}' for i in range(8))+') : ('+', '.join([vg]*8)+') -> reg<amdgpu.vgpr x8>\n  return %p\n}\n'
    pos=s.index('kernel.def');s=s[:pos]+helper+s[pos:]
    def pack(match):
        indent,name,operand=match.groups()
        result=f'''{indent}%{name}_half = vector.fptrunc %{operand} : vector<8xf32> to vector<8xf16>
    {indent}%{name}_words = vector.bitcast %{name}_half : vector<8xf16> to vector<4xi32>
    {indent}%{name}_partner, %{name}_valid = kernel.subgroup.shuffle<xor> %{name}_words, %i32_16, %i32_32 : vector<4xi32>, i32, i32
    {indent}%{name}_even = scf.select %lane_group_even, %{name}_words, %{name}_partner : vector<4xi32>
    {indent}%{name}_odd = scf.select %lane_group_even, %{name}_partner, %{name}_words : vector<4xi32>
    {indent}%{name}_raw = low.invoke @h3_vae_pack(%{name}_even, %{name}_odd) : (vector<4xi32>, vector<4xi32>) -> (vector<16xf16>)
    '''
        if name.startswith('probability'):
            result+=f'{indent}%{name} = vector.fragment<rhs> %{name}_raw shape [%k_frag, %n] : vector<16xf16>'
        else:
            result+=f'{indent}%{name} = vector.bitcast %{name}_raw : vector<16xf16> to vector<16xf16>'
        return result
    s,n=re.subn(r'^( +)%(\w+) = vector.fragment.repack<rhs> %(\w+) shape \[%m, %n\] : vector<8xf32> -> vector<16xf16>',pack,s,flags=re.M);assert n==6,n
    return s


def convert(source, stem):
    return _pack(_base2(_transposed(source)), stem)


def head_major(source):
    """Read [heads][capacity][64] Q/K/V, retaining row-major padded output."""
    pos = source.index("  %q_view =")
    offsets = """  %hm_qrow = index.mul %head, %padded_tokens : index
  %hm_kvrow = index.mul %kv_head, %padded_tokens : index
  %hm_row_bytes = index.constant 128 : offset
  %hm_qoffset = index.scale %hm_qrow, %hm_row_bytes : index, offset -> offset
  %hm_kvoffset = index.scale %hm_kvrow, %hm_row_bytes : index, offset -> offset
"""
    source = source[:pos] + offsets + source[pos:]
    for key, offset in (("q", "q"), ("k", "kv"), ("v", "kv")):
        source = source.replace(f"%{key}_global[%c0_offset]", f"%{key}_global[%hm_{offset}offset]")
    source = source.replace("view<[%padded_tokens]x[%q_stride0]xf16>", "view<[%padded_tokens]x64xf16>")
    source = source.replace("view<[%padded_tokens]x[%kv_stride0]xf16>", "view<[%padded_tokens]x64xf16>")
    for channel in range(4):
        source = source.replace(f"%q_view[%query_lane_row, %q_channel{channel}]",
                                f"%q_view[%query_lane_row, %c{16 * channel}]")
    source = source.replace(", %st_col] : view<[%padded_tokens]x64xf16>",
                            ", %st_chunk] : view<[%padded_tokens]x64xf16>")
    return source.replace(", %st_col_v] : view<[%padded_tokens]x64xf16>",
                          ", %st_chunk_v] : view<[%padded_tokens]x64xf16>")

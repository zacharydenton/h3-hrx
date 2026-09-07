"""Fused decoder QKV, f16 rounding, RMSNorm and RoPE numerical gates.

--compile-only verifies generated-source identity and decoder configurations
without initializing HIP. --gpu checks ragged tokens, zero heads, saturation,
V identity and output guards against separate GEMM/rotary kernels, plus an
independent float64 normalization/rotation oracle.
"""
from pathlib import Path
import sys,json,subprocess,argparse
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))
from kernel_test import compile_kernel as _compile,launch,workdir,LOOM_COMPILE
from gen_gemm_f16_qkv_rope import generate,STEM
def compile_kernel(*args):
 try: return _compile(*args)
 except subprocess.CalledProcessError as e: print(e.stderr,flush=True); raise

def run(tmp,m,k,n=6144):
 rng=np.random.default_rng(121+m);width=n//3
 cap=max((m+16+31)//32*32,(m+63)//64*64,(m+255)//256*256)
 tiles=(m+127)//128;group=15 if m==1797 else 1 if tiles==1 else min((4,3,2),key=lambda g:((tiles+g-1)//g*g,-g))
 a=np.full((m,k+128),113,np.float16);w=np.full((n,k+128),113,np.float16)
 a[:,:k]=rng.standard_normal((m,k),dtype=np.float32)*.8
 w[:,:k]=rng.standard_normal((n,k),dtype=np.float32)/np.sqrt(k)
 bias=rng.standard_normal(n,dtype=np.float32)*.3
 a[0,:k]=0; bias[:64]=0
 bias[64:128]=np.linspace(-70000,70000,64,dtype=np.float32)
 bias[-64:]=np.linspace(-70000,70000,64,dtype=np.float32)
 qw=rng.uniform(.6,1.4,64).astype(np.float32);kw=rng.uniform(.6,1.4,64).astype(np.float32)
 phase=rng.uniform(-3,3,(m,24)).astype(np.float32);cos=np.cos(phase);sin=np.sin(phase);eps=1e-5
 compiled={}
 for variant in ('fast','qkvropehm'):
  stem=f'gemm_f16_{variant}_256b';ns='h3.'+stem
  cfg={ns+'.k_size':k,ns+'.n_size':n,ns+'.k_stride':k+128,ns+'.m_group':group}
  if variant=='qkvropehm':cfg.update({ns+'.token_capacity':cap,ns+'.eps':eps})
  hs=tmp/(variant+'.hsaco');compile_kernel(ROOT / 'kernels' / (stem+'.loom'),'h3_'+stem,cfg,hs)
  compiled[variant]=(stem,hs)
 stem='rope64_qknorm_f16';ns='h3.'+stem;rhs=tmp/'rope.hsaco'
 compile_kernel(ROOT / 'kernels' / (stem+'.loom'),'h3_'+stem,{ns+'.row_stride':n,ns+'.heads':width//64,ns+'.kv_heads':width//64,ns+'.k_offset':width,ns+'.eps':eps},rhs)
 gy=(tiles+group-1)//group*group
 outs,_=launch(compiled['fast'][1],'h3_'+compiled['fast'][0],(n//256,gy,1),(256,1,1),[('i32',m),('in_f16',a),('in_f16',w),('out_f16',((m,n),np.float16)),('in',bias)],tmp)
 projected=outs[0]
 guard=np.full((cap,width),73,np.float16)
 rargs=[('i32',m),('in_f16',projected),('in',qw),('in',kw),('in',cos),('in',sin)]+[('inout_f16',(guard,guard.shape)) for _ in range(3)]
 ref,_=launch(rhs,'h3_'+stem,(m,1,1),(256,1,1),rargs,tmp)
 guard3=np.full((3,cap,width),73,np.float16)
 fargs=[('i32',m),('in_f16',a),('in_f16',w),('inout_f16',(guard3,guard3.shape)),('in',bias),('in',qw),('in',kw),('in',cos),('in',sin)]
 actual,timing=launch(compiled['qkvropehm'][1],'h3_'+compiled['qkvropehm'][0],(n//256,gy,1),(256,1,1),fargs,tmp)
 out=actual[0].reshape(3,width//64,cap,64).transpose(0,2,1,3).reshape(3,cap,width);native=np.stack(ref)
 assert np.all(out[:,m:]==73),'fused output padding'
 assert np.all(native[:,m:]==73),'reference output padding'
 assert np.array_equal(out[2,:m],native[2,:m]),'V must remain identical'
 delta=np.abs(out[:2,:m].astype(np.float32)-native[:2,:m])
 assert np.allclose(out[:2,:m],native[:2,:m],atol=.002,rtol=.002),(delta.max(),np.unravel_index(delta.argmax(),delta.shape))
 # Independent RMSNorm and rotate-half oracle, using the baseline GEMM's
 # explicitly rounded half values as the contract between the two operations.
 oracle=[]
 for component,weight in ((0,qw),(1,kw)):
  x=projected[:,component*width:(component+1)*width].reshape(m,width//64,64).astype(np.float64)
  x=x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+eps)*weight
  rot=x.copy();c=cos[:,None,:];sn=sin[:,None,:]
  rot[:,:,:24]=x[:,:,:24]*c-x[:,:,24:48]*sn
  rot[:,:,24:48]=x[:,:,24:48]*c+x[:,:,:24]*sn
  oracle.append(rot.reshape(m,width).astype(np.float16))
 assert np.allclose(out[:2,:m],np.stack(oracle),atol=.003,rtol=.003),'independent normalization/rotation oracle'
 print(json.dumps(dict(m=m,k=k,n=n,max_native_difference=float(delta.max()),**timing)),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--compile-only", action="store_true")
    action.add_argument("--gpu", action="store_true")
    opt = parser.parse_args()
    with workdir() as directory:
        tmp = Path(directory)
        source = tmp / f"{STEM}.loom"
        source.write_text(generate())
        formatter = LOOM_COMPILE.parent.parent / "loom-format/loom-format"
        subprocess.run([str(formatter), "--in-place", str(source)], check=True, capture_output=True)
        assert source.read_bytes() == (ROOT / "kernels" / source.name).read_bytes(), "generated source differs"
        ns = "h3." + STEM
        for k in (128, 2048):
            for group, capacity in ((1, 256), (2, 256), (2, 288), (3, 768), (3, 2048), (4, 2048), (15, 2048)):
                compile_kernel(source, "h3_" + STEM, {ns + ".k_size": k, ns + ".n_size": 6144,
                    ns + ".k_stride": k + 128, ns + ".m_group": group,
                    ns + ".token_capacity": capacity, ns + ".eps": 1e-5}, tmp / "fused.hsaco")
        print("PASS fused QKV source identity and 14 CPU compilation configurations", flush=True)
        if opt.gpu:
            for m in (1, 13, 127, 128, 129, 250, 517, 1797):
                run(tmp, m, 128)
            for m in (13, 517, 1797):
                run(tmp, m, 2048)
    return 0


if __name__ == "__main__":
    sys.exit(main())

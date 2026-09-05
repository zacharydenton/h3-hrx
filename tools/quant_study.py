"""Which quantisation survives H3's 50 blocks, on the final-layer velocity against the int8
checkpoint with float activations. Reference only; decides the kernel design.
    python3 tools/quant_study.py [mode ...]      modes: w4a4 w4g128a8 ... or mixed:<base>:<suffix>=<mode>,...
"""
import sys, time
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reference"))
import h3_ref as R

fx = torch.load(ROOT / "build/fixture.pt"); dev = "cuda"
ckpt = R.Checkpoint(device=dev, dtype=torch.bfloat16)
layout = R.Layout(*fx["layout"])
temb = fx["temb"].to(dev); rows = fx["rows"].to(dev); cos, sin = fx["cos"].to(dev), fx["sin"].to(dev); tclass = layout.tclass.to(dev)
x0 = fx["x"].to(dev, torch.bfloat16)
modes = sys.argv[1:] or ["w4a4", "w4g256a4", "w4g128a4", "w4g128a8", "w4g64a8", "w8a8",
                         "mixed:w4a4:mlp.fc2=w8a8", "mixed:w4a4:mlp.fc2=w8a8,attn.out_proj=w8a8", "mixed:w4g128a8:mlp.fc2=w8a8"]
vel = {}
def run(mode):
    if mode.startswith("attn:"):                        # attn:<gemm mode>:<attention mode>[:exact_from]
        parts = mode.split(":"); ref = R.H3Ref(ckpt, quant=parts[1], cache_linears=False); ref.attn_quant = parts[2]
        if len(parts) > 3: ref.attn_exact_from = int(parts[3])
    elif mode.startswith("mixed:"):
        _, base, ov = mode.split(":", 2)
        ref = R.H3Ref(ckpt, quant=base, cache_linears=False)
        ref.quant_overrides = dict(kv.split("=") for kv in ov.split(","))
    else:
        ref = R.H3Ref(ckpt, quant=mode, cache_linears=False)
    with torch.no_grad():
        x = ref.blocks_forward(x0.clone(), temb, rows, cos, sin)
        v, a = ref.final(x, temb, tclass, layout)
    return torch.cat([v.flatten(), a.flatten()])
t0 = time.time(); vel["none"] = run("none"); print(f"none: {time.time() - t0:.0f} s", flush=True)
cs = lambda p, q: torch.nn.functional.cosine_similarity(p, q, dim=0).item()
print("50 blocks, final-layer velocity against the int8 checkpoint with float activations:")
print("  mode                                              cosine    rel rms err")
for mode in modes:
    v = run(mode)
    print(f"  {mode:48s}  {cs(v, vel['none']):.5f}   {((v - vel['none']).norm() / vel['none'].norm()).item():.4f}", flush=True)

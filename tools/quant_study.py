"""Which quantisation survives H3's 50 blocks, measured where it matters: the cosine of the
final-layer video velocity (after norm_out, which removes the residual stream's outlier
channels) against the int8-weight / float-activation path, plus the raw-stream update cosine.
Modes: int4 per-row weights with activations int4 per token (w4a4, the kernels), int4 per
256-group (w4a4g), int8 per token (w4a8); and w8a8 = the checkpoint's int8 weights with int8
per-token activations. Reference only; decides the kernel design."""
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
outs, vel = {}, {}
for q in ("none", "w4a4", "w4a4g", "w4a8", "w8a8"):
    ref = R.H3Ref(ckpt, quant=q, cache_linears=False)
    t0 = time.time()
    with torch.no_grad():
        x = ref.blocks_forward(x0.clone(), temb, rows, cos, sin)
        v, a = ref.final(x, temb, tclass, layout)
    outs[q] = (x.float() - x0.float()).flatten(); vel[q] = torch.cat([v.flatten(), a.flatten()])
    print(f"{q}: {time.time() - t0:.0f} s", flush=True)
cs = lambda p, q: torch.nn.functional.cosine_similarity(p, q, dim=0).item()
print("50 blocks, against int8 weights + float activations:")
print("  mode    velocity cosine   velocity rel rms err   raw-stream update cosine")
for q in ("w4a4", "w4a4g", "w4a8", "w8a8"):
    err = ((vel[q] - vel["none"]).norm() / vel["none"].norm()).item()
    print(f"  {q:6s}  {cs(vel[q], vel['none']):.5f}           {err:.4f}                 {cs(outs[q], outs['none']):.5f}")

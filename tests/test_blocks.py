"""The native blocks (h3_loom) against the reference on a fixture built from the real checkpoint:
random text states through the real token refiner, noise video and audio rows at a mid-schedule
timestep, the packed layout of a 480p-class clip. Reports, per depth, the cosine of the update
the blocks make to the residual stream, native vs the W4A4 reference and vs the unquantised
(int8-dequantised) reference, and the reference's own W4A4-vs-none cosine for scale.

    python3 tests/test_blocks.py [--curve 1,2,4,8,16,50] [--profile] [--fixture build/fixture.pt]
"""
import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "reference"))
import h3_ref as R
from h3_loom import H3Blocks, mods_table


def build_fixture(path: Path, device="cuda", dims=(32, 5, 30, 54, 20)):
    torch.manual_seed(0)
    ckpt = R.Checkpoint(device=device, dtype=torch.bfloat16)
    ref = R.H3Ref(ckpt, quant="none", cache_linears=False)
    text_len, latent_t, lat_h, lat_w, audio_t = dims
    layout = R.Layout(text_len, latent_t, lat_h, lat_w, audio_t)
    sigma_v = 0.9
    t_v, t_a = 1.0 - sigma_v, 1.0 - R.time_shift_sigma(sigma_v, 12.0, 3.0)
    temb = ref.t_emb(torch.tensor([t_v, t_a]))
    text = torch.randn(text_len, R.TEXT_DIM, device=device) * 1.0
    video = torch.randn(layout.video_rows, R.VIDEO_PATCH, device=device)
    audio = torch.randn(layout.audio_rows, R.AUDIO_CH, device=device)
    with torch.no_grad():
        x = torch.cat([ref.text_in(text), ref.audio_in(audio), ref.video_in(video)], dim=0)
    cos, sin = R.rope_tables(layout.position_ids, ref.inv_freq, device)
    fx = dict(x=x.cpu(), rows=layout.adaln_rows, temb=temb.cpu(), cos=cos.cpu(), sin=sin.cpu(),
              layout=(text_len, latent_t, lat_h, lat_w, audio_t), t=(t_v, t_a))
    torch.save(fx, path)
    return fx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", default="1,2,4,8,16,50")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--fixture", default=str(ROOT / "build/fixture.pt"))
    ap.add_argument("--weights", default=None, help="weights directory (default build/weights; build/weights_gptq for the GPTQ export)")
    ap.add_argument("--layout", default=None, help="text,latent_t,lat_h,lat_w,audio_t for a new fixture (e.g. 32,25,30,54,80 = 480p 4 s, ~10.4k rows)")
    ap.add_argument("--profile-only", action="store_true", help="native forward and stage profile only, no reference")
    a = ap.parse_args()
    device = "cuda"
    fpath = Path(a.fixture)
    if a.layout:
        dims = tuple(int(v) for v in a.layout.split(","))
        fpath = fpath.with_name(fpath.stem + "_" + "x".join(map(str, dims)) + ".pt")
        fx = torch.load(fpath) if fpath.exists() else build_fixture(fpath, device, dims)
    else:
        fx = torch.load(fpath) if fpath.exists() else build_fixture(fpath, device)
    tokens = fx["x"].shape[0]
    print(f"fixture: {tokens} tokens (layout {fx['layout']}), t_video {fx['t'][0]:.4f}, t_audio {fx['t'][1]:.4f}")
    ckpt = R.Checkpoint(device=device, dtype=torch.bfloat16)
    refs = {q: R.H3Ref(ckpt, quant=q, cache_linears=False) for q in ("w4a4", "none")}
    temb = fx["temb"].to(device)
    x0 = fx["x"].to(device, torch.bfloat16); rows = fx["rows"].to(device); cos, sin = fx["cos"].to(device), fx["sin"].to(device)
    mods = mods_table(refs["none"], temb, 50)
    if a.profile_only:
        for n in [int(v) for v in a.curve.split(",")]:
            loom = H3Blocks(tokens, layers=n, weights=a.weights); loom.profile(True)
            t0 = time.time(); loom.forward(x0, rows, mods, cos, sin); dt = time.time() - t0
            print(f"  {n:3d} blocks {dt * 1e3:8.0f} ms (profile only)"); loom.close()
        return 0
    layout = R.Layout(*fx["layout"]); tclass = layout.tclass.to(device)
    def velocity(x):
        v, a = refs["none"].final(x.to(torch.bfloat16), temb, tclass, layout)
        return torch.cat([v.flatten(), a.flatten()])
    ok = True
    for n in [int(v) for v in a.curve.split(",")]:
        loom = H3Blocks(tokens, layers=n, weights=a.weights)
        if a.profile:
            loom.profile(True)
        t0 = time.time()
        y = loom.forward(x0, rows, mods, cos, sin).to(device).float()
        dt = time.time() - t0
        with torch.no_grad():
            outs = {q: refs[q].blocks_forward(x0.clone(), temb, rows, cos, sin, layers=n).float() for q in refs}
        upd = lambda t: (t - x0.float()).flatten()
        cos_sim = lambda p, q: torch.nn.functional.cosine_similarity(p, q, dim=0).item()
        c_w, c_n, c_ref = cos_sim(upd(y), upd(outs["w4a4"])), cos_sim(upd(y), upd(outs["none"])), cos_sim(upd(outs["w4a4"]), upd(outs["none"]))
        with torch.no_grad():
            vy, vw, vn = velocity(y), velocity(outs["w4a4"]), velocity(outs["none"])
        v_w, v_n, v_ref = cos_sim(vy, vw), cos_sim(vy, vn), cos_sim(vw, vn)
        print(f"  {n:3d} blocks {dt * 1e3:8.0f} ms:  stream update cosine vs w4a4 {c_w:.5f} vs none {c_n:.5f} (ref {c_ref:.5f});  "
              f"final velocity cosine vs w4a4 {v_w:.5f} vs none {v_n:.5f} (ref {v_ref:.5f})")
        ok &= v_w > 0.99
        loom.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

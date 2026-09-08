"""The stacks against their references, block by block, through the library's own dumps (H3_DUMP_BLOCKS): the host runs,
writes x before the first block and after each one, and the reference continues from the same x.

    python3 tests/test_stack_parity.py --stack dit    the 50 DiT blocks vs reference/h3_ref.py on the same checkpoint
    python3 tests/test_stack_parity.py --stack te     the text encoder's 50 layers vs transformers' bf16 hidden states

The ComfyUI comparison (tests/test_comfy_parity.py) is the quality gate; this locates a regression to a block."""
import argparse, os, sys, tempfile, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "reference"), str(ROOT / "tools")]


def cosine(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def dumped(dump, tag, name, width):
    return np.fromfile(dump / f"{tag}_{name}.f32", dtype=np.float32).reshape(-1, width)


def dit(depths, tokens_hw):
    """The DiT stack: the host's packed rows through reference/h3_ref.py's blocks (the same int8 checkpoint, dequantised)."""
    import torch
    import h3_ref as R
    from h3_loom import H3
    from h3tok_ids import encode_presentation
    height, width, frames = tokens_hw
    with tempfile.TemporaryDirectory(prefix="h3-dit-parity-") as tmp:
        os.environ["H3_DUMP_BLOCKS"] = tmp; os.environ["H3_DUMP_CALL"] = "0"
        pipe = H3()
        ids = np.asarray(encode_presentation("A red fox trotting through a snowy forest at dawn, cinematic"), np.int32)
        p = H3.params(height=height, width=width, frames=frames, steps=2, seed=1); s = pipe.shape(p)
        rng = np.random.default_rng(1)
        nv = rng.standard_normal((24, s.latent_t, s.lat_h, s.lat_w)).astype(np.float32); na = rng.standard_normal((2, 32, s.audio_t)).astype(np.float32)
        t0 = time.time(); pipe.denoise(ids, p, noise_video=nv, noise_audio=na); print(f"C evaluation in {time.time() - t0:.1f} s")
        pipe.close()
        x0 = dumped(Path(tmp), "dit", "h_in", R.HIDDEN)
        want_last = {d: dumped(Path(tmp), "dit", f"blk_{d - 1:02d}", R.HIDDEN) for d in depths}
    layout = R.Layout(ids.size, s.latent_t, s.lat_h, s.lat_w, s.audio_t)
    ckpt = R.Checkpoint(device="cuda", dtype=torch.bfloat16); ref = R.H3Ref(ckpt, quant="none")
    cos, sin = R.rope_tables(layout.position_ids, ref.inv_freq, "cuda"); rows = layout.adaln_rows.to("cuda")
    from diffusers import MiniMaxH3Scheduler
    sv = MiniMaxH3Scheduler(shift=12.0); sa = MiniMaxH3Scheduler(shift=3.0); sv.set_timesteps(2, device="cuda"); sa.set_timesteps(2, device="cuda")
    ok = True
    with torch.no_grad():
        temb = ref.t_emb(torch.tensor([sv.timesteps[0].item(), sa.timesteps[0].item()]))
        x = torch.from_numpy(x0).to("cuda").to(torch.bfloat16)
        for d in sorted(depths):
            y = ref.blocks_forward(x.clone(), temb, rows, cos, sin, layers=d).float().cpu().numpy()
            got = want_last[d]
            L, Na = layout.text_len, layout.audio_rows
            segs = {"text": (0, L), "audio": (L, L + Na), "video": (L + Na, got.shape[0])}
            parts = "  ".join(f"{k} {cosine(got[a:b], y[a:b]):.4f}" for k, (a, b) in segs.items())
            c = cosine(got, y); good = c > 0.99; ok &= good
            print(f"  {'PASS' if good else 'FAIL'} after {d:2d} blocks: cosine {c:.5f}  [{parts}]")
    del ref, ckpt; torch.cuda.empty_cache()
    return ok


def te(depths):
    """The text encoder: the host's embedding rows through transformers' bf16 layers on the same checkpoint's weights."""
    import torch
    from h3_loom import H3, TE
    from h3tok_ids import encode_presentation
    cache = ROOT / "build/te_hidden.pt"
    ids = np.asarray(encode_presentation("A red fox trotting through a snowy forest at dawn, cinematic"), np.int32)
    with tempfile.TemporaryDirectory(prefix="h3-te-parity-") as tmp:
        os.environ["H3_DUMP_BLOCKS"] = tmp; os.environ["H3_DUMP_CALL"] = "0"
        pipe = H3(); t0 = time.time(); pipe.text_in(ids); print(f"C text_in in {time.time() - t0:.1f} s"); pipe.close()
        got = {d: dumped(Path(tmp), "te", f"blk_{d - 1:02d}", 5120) for d in depths}
        x0 = dumped(Path(tmp), "te", "h_in", 5120)
    if not cache.exists():
        print(f"SKIP: no {cache} (the transformers reference: see docs/archive/notes.md, the text encoder)"); return True
    want = torch.load(cache)   # [layers + 1][tokens][5120] bf16 hidden states from transformers on the same ids
    ok = True
    for d in sorted(depths):
        y = want[d].float().numpy(); c = cosine(got[d], y); good = c > 0.99; ok &= good
        print(f"  {'PASS' if good else 'FAIL'} after {d:2d} layers: cosine {c:.5f}")
    print(f"  (x_in cosine {cosine(x0, want[0].float().numpy()):.5f})")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", choices=["dit", "te"], required=True)
    ap.add_argument("--depths", default="1,10,25,50", help="block counts to compare")
    ap.add_argument("--height", type=int, default=64); ap.add_argument("--width", type=int, default=96); ap.add_argument("--frames", type=int, default=22)
    a = ap.parse_args()
    depths = [int(x) for x in a.depths.split(",") if x]
    ok = dit(depths, (a.height, a.width, a.frames)) if a.stack == "dit" else te(depths)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

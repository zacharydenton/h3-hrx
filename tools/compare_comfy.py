"""The C host against ComfyUI from ComfyUI's own noise (tools/comfy_clip.py --dump-steps [--dump-blocks ...]):
  trajectory: the video latent after every evaluation of the 20-evaluation res_multistep schedule, cosine per step
  blocks:     one evaluation, the refined text rows and the residual stream after chosen blocks, cosine per row segment
    python tools/compare_comfy.py --case t2va --mode blocks --truth build/comfy_t2va_blocks --blocks build/weights_i8 --attn f16
    python tools/compare_comfy.py --case fl2va --mode trajectory --truth build/comfy_fl2va
The keyframe case resizes build/refs/fox_clean.png to the canvas as pipeline_c does. Runs outside the container (the C host)."""
import argparse, os, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
ap = argparse.ArgumentParser()
ap.add_argument("--case", choices=["t2va", "fl2va"], default="t2va"); ap.add_argument("--mode", choices=["trajectory", "blocks"], default="trajectory")
ap.add_argument("--truth", required=True, help="tools/comfy_clip.py --out directory"); ap.add_argument("--blocks", default=None); ap.add_argument("--attn", choices=["i4", "f16"], default="i4")
ap.add_argument("--prompt", default="A red fox trotting through a snowy forest at dawn, cinematic"); ap.add_argument("--first-frame", default=str(ROOT / "build/refs/fox_clean.png"))
ap.add_argument("--width", type=int, default=864); ap.add_argument("--height", type=int, default=480); ap.add_argument("--frames", type=int, default=22); ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--dump", default=None, help="directory for the per-step / per-block dumps (default: <truth>/mine_<blocks>_<attn>)")
ap.add_argument("--block-ids", default="0,1,2,5,10,20,30,40,49")
a = ap.parse_args()
from PIL import Image
from h3pipe_loom import H3Pipe
from h3tok_ids import encode_presentation
D = Path(a.truth); blocks = a.blocks or str(ROOT / "build/weights_gptq")
dump = Path(a.dump or D / f"mine_{Path(blocks).name}_{a.attn}"); dump.mkdir(parents=True, exist_ok=True)
os.environ["H3_DUMP_DIR"] = str(dump)
if a.mode == "blocks": os.environ["H3_DUMP_BLOCKS"] = str(dump)
pipe = H3Pipe(blocks=blocks, attn=a.attn)
p = H3Pipe.params(height=a.height, width=a.width, frames=a.frames, steps=2 if a.mode == "blocks" else 21, seed=a.seed, sampler="res_multistep"); sh = pipe.shape(p)
nv = np.load(D / "noise_video.npy").astype(np.float32); na = np.load(D / "noise_audio.npy").astype(np.float32)   # ComfyUI's pack: video [24,T,H,W], audio [32, 2, A]
kfs, toks = [], []
if a.case == "fl2va":
    img = np.asarray(Image.open(a.first_frame).convert("RGB").resize((a.width, a.height), Image.BILINEAR), dtype=np.float32) / 255.0
    z = pipe.encode_video(img); kfs.append({"frame_index": 0, "video": z, "pixels": img}); toks.append((a.height // 32) * (a.width // 32))
ids = np.asarray(encode_presentation(a.prompt, images=toks, audios=0), np.int32)
t0 = time.time(); v, au = pipe.denoise(ids, p, noise_video=nv, noise_audio=np.ascontiguousarray(na.transpose(1, 0, 2)), keyframes=kfs); print(f"C denoised in {time.time() - t0:.1f} s ({Path(blocks).name}, {a.attn} attention)", flush=True)
cos = lambda x, y: float((x * y).sum() / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-30))
T, Hh, Ww = sh.latent_t, sh.lat_h, sh.lat_w
if a.mode == "trajectory":
    for k in range(0, 20):
        c = np.fromfile(dump / f"x_{k:02d}.f32", dtype=np.float32).reshape(24, T, Hh, Ww).astype(np.float64); y = np.load(D / f"x_{k:02d}.npy").astype(np.float64)
        print(f"x_{k:02d}: cosine {cos(c, y):.4f}  rel err {np.linalg.norm(c - y) / np.linalg.norm(y):.4f}", flush=True)
    c = np.fromfile(dump / "x_20.f32", dtype=np.float32).reshape(24, T, Hh, Ww).astype(np.float64); y = np.load(D / "video_latent.npy").astype(np.float64)
    print(f"final: cosine {cos(c, y):.4f}  rel err {np.linalg.norm(c - y) / np.linalg.norm(y):.4f}", flush=True)
else:
    L = len(ids); Na = sh.audio_t * 2; Nv = sh.latent_t * (sh.lat_h // 2) * (sh.lat_w // 2)
    txt = pipe.text_in(ids).astype(np.float64); ref = np.load(D / "blocks/refined_text.npy").astype(np.float64).reshape(-1, txt.shape[-1])
    print(f"refined text: cosine {cos(txt, ref):.5f} rel err {np.linalg.norm(txt - ref) / np.linalg.norm(ref):.4f}", flush=True)
    segs = {"text": (0, L), "audio": (L, L + Na), "video": (L + Na, L + Na + Nv)}
    for i in [int(x) for x in a.block_ids.split(",")]:
        name = f"blk_{i:02d}"; m = np.fromfile(dump / f"{name}.f32", np.float32).reshape(-1, 5376).astype(np.float64); y = np.load(D / f"blocks/{name}.npy").astype(np.float64).reshape(-1, 5376)
        parts = "  ".join(f"{k} {cos(m[s0:s1], y[s0:s1]):.4f}" for k, (s0, s1) in segs.items())
        print(f"{name}: cosine {cos(m, y):.4f} rel err {np.linalg.norm(m - y) / np.linalg.norm(y):.4f}  [{parts}]", flush=True)

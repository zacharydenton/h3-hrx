"""ref2va with references, one step, against ComfyUI's denoised outputs (build/ref_truth from tools/ref_truth_comfy.py):
the presentation ids from our tokenizer must match, the noise is ComfyUI's, and with sigmas [1, 0] our output latents
are the denoised prediction. --case audio (an audio reference only) or both (image + audio; needs the vision tower).
    python3 tests/test_ref2va.py [--case audio] [--own-encoders]"""
import argparse, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from h3pipe_loom import H3Pipe
ap = argparse.ArgumentParser(); ap.add_argument("--case", default="audio"); ap.add_argument("--own-encoders", action="store_true")
ap.add_argument("--blocks", default=None); ap.add_argument("--glue", default=None)
ap.add_argument("--prompt", default="A red fox trotting through a snowy forest at dawn, cinematic, with the sound of <Audio 1>")
a = ap.parse_args()
T = ROOT / "build/ref_truth"; L = lambda n: np.load(T / f"{n}.npy")
ids_ref = L(f"{a.case}_pres_ids")
from h3tok_ids import encode_presentation
ids = encode_presentation(a.prompt, images=[], audios=1 if "Audio" in a.prompt else 0)
ok_ids = np.array_equal(np.asarray(ids), ids_ref); print(f"presentation ids: ours {len(ids)} vs comfy {len(ids_ref)}: {'match' if ok_ids else 'DIFFER'}")
if not ok_ids: print("  ours:", ids[:24], "\n  ref: ", ids_ref[:24].tolist())
pipe = H3Pipe(blocks=a.blocks, glue=a.glue)
nv, na = L(f"{a.case}_noise_video"), L(f"{a.case}_noise_audio")               # [24][T][H][W], [32][2][t] -> ours [2][32][t]
na = np.transpose(na, (1, 0, 2))
refs = []
if a.case == "both":
    z = L("both_ref_latent"); refs.append({"kind": "image", "video": z})
za = L(f"{a.case}_ref_audio_latent"); za = np.transpose(za, (1, 0, 2))
if a.own_encoders: za = pipe.encode_audio(L("audio_wav").astype(np.float32))
refs.append({"kind": "audio", "audio": za})
frames = 22; p = H3Pipe.params(height=480, width=864, frames=frames, steps=2, seed=0)
t0 = time.time(); video, audio = pipe.denoise(np.asarray(ids, np.int32), p, noise_video=nv, noise_audio=na, refs=refs); dt = time.time() - t0
dv, da = L(f"{a.case}_denoised_video"), np.transpose(L(f"{a.case}_denoised_audio"), (1, 0, 2))
def cmp(name, x, y):
    x, y = x.astype(np.float64).ravel(), y.astype(np.float64).ravel(); cos = float(x @ y / np.sqrt((x @ x) * (y @ y))); rel = float(np.linalg.norm(x - y) / np.linalg.norm(y))
    print(f"  {name}: cosine {cos:.5f}  rel rms {rel:.4f}"); return cos
cv = cmp("video denoised", video, dv); ca = cmp("audio denoised", audio, da)
print(f"one evaluation in {dt:.1f} s"); ok = ok_ids and cv > 0.98 and ca > 0.98
print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)

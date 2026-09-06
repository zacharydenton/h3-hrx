"""ref2va with references, one step, against ComfyUI's denoised outputs (build/ref_truth from tools/ref_truth_comfy.py):
the presentation ids from our tokenizer must match, the noise is ComfyUI's, and with sigmas [1, 0] our output latents
are the denoised prediction. --case audio (an audio reference only) or both (image + audio; needs the vision tower).
    python3 tests/test_ref2va.py [--case audio] [--own-encoders]"""
import argparse, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from h3pipe_loom import H3Pipe
ap = argparse.ArgumentParser(); ap.add_argument("--case", default="audio", help="t2va | audio | both | fl2va"); ap.add_argument("--own-encoders", action="store_true")
ap.add_argument("--blocks", default=None); ap.add_argument("--glue", default=None)
ap.add_argument("--prompt", default="A red fox trotting through a snowy forest at dawn, cinematic, with the sound of <Audio 1>")
a = ap.parse_args()
T = ROOT / "build/ref_truth"; L = lambda n: np.load(T / f"{n}.npy")
ids_ref = L(f"{a.case}_pres_ids")
from h3tok_ids import encode_presentation
img = None
if a.case == "both": img = L("image_resized").astype(np.float32); a.prompt = "<Picture 1> is the fox. " + a.prompt
if a.case in ("fl2va", "t2va"): a.prompt = a.prompt.replace(", with the sound of <Audio 1>", "")
if a.case == "fl2va": img = L("fl2va_image_resized").astype(np.float32)
n_img = (img.shape[0] // 32) * (img.shape[1] // 32) if img is not None else 0
ids = encode_presentation(a.prompt, images=[n_img] if img is not None else [], audios=1 if "Audio" in a.prompt else 0)
ids_ref = np.concatenate([np.full(n_img, -1, np.int64) if v == -1 else np.array([v], np.int64) for v in ids_ref.tolist()])   # comfy keeps one entry per vision span
ok_ids = np.array_equal(np.asarray(ids), ids_ref); print(f"presentation ids: ours {len(ids)} vs comfy {len(ids_ref)}: {'match' if ok_ids else 'DIFFER'}")
if not ok_ids: print("  ours:", ids[:24], "\n  ref: ", ids_ref[:24].tolist())
pipe = H3Pipe(blocks=a.blocks, glue=a.glue)
nv, na = L(f"{a.case}_noise_video"), L(f"{a.case}_noise_audio")               # [24][T][H][W], [32][2][t] -> ours [2][32][t]
na = np.transpose(na, (1, 0, 2))
refs, kfs = [], []
if a.case == "both":
    z = L("both_ref_latent") if not a.own_encoders else pipe.encode_video(img)
    refs.append({"kind": "image", "video": z, "pixels": img})
if a.case in ("both", "audio"):
    za = np.transpose(L(f"{a.case}_ref_audio_latent"), (1, 0, 2))
    if a.own_encoders: za = pipe.encode_audio(L("audio_wav").astype(np.float32))
    refs.append({"kind": "audio", "audio": za})
if a.case == "fl2va":
    z = L("fl2va_keyframe_latent") if not a.own_encoders else pipe.encode_video(img)
    kfs.append({"frame_index": 0, "video": z, "pixels": img})
frames = 22; p = H3Pipe.params(height=480, width=864, frames=frames, steps=2, seed=0)
t0 = time.time(); video, audio = pipe.denoise(np.asarray(ids, np.int32), p, noise_video=nv, noise_audio=na, refs=refs, keyframes=kfs); dt = time.time() - t0
dv, da = L(f"{a.case}_denoised_video"), np.transpose(L(f"{a.case}_denoised_audio"), (1, 0, 2))
raw = (T / f"{a.case}_out_video.npy").exists()
def cmp(name, x, y):
    x, y = x.astype(np.float64).ravel(), y.astype(np.float64).ravel(); cos = float(x @ y / np.sqrt((x @ x) * (y @ y))); rel = float(np.linalg.norm(x - y) / np.linalg.norm(y))
    print(f"  {name}: cosine {cos:.5f}  rel rms {rel:.4f}"); return cos
cv = cmp("video denoised", video, dv); ca = cmp("audio denoised", audio, da)
if raw:   # the network's raw outputs (ComfyUI returns [-video_out, -audio_out]; our velocity v gives x0 = x + sigma v) and its inputs
    iv, ia = L(f"{a.case}_in_video"), np.transpose(L(f"{a.case}_in_audio"), (1, 0, 2))
    ov = -L(f"{a.case}_out_video")
    # the wrapper returns, for the sampler's carried audio variable, (1 - scale) * x_a + (1 + (scale - 1) * sigma_a) * net_out with scale = 12 / 3
    # and sigma_a = 1 at the first step; the network's own output is (returned + 3 x_a) / 4, negated like the video
    oa = -((np.transpose(L(f"{a.case}_out_audio"), (1, 0, 2)) + 3.0 * ia) / 4.0)
    print(f"  comfy inputs vs our noise: video max|diff| {float(np.abs(iv - nv).max()):.3g}, audio max|diff| {float(np.abs(ia - na).max()):.3g} (audio carried by {float(np.abs(ia).mean() / max(np.abs(na).mean(), 1e-9)):.3f}); timestep {L(f'{a.case}_timestep').ravel()[:2]}")
    vv, va = video - nv, audio - na   # sigma = 1 at step 0
    cv = cmp("video velocity", vv, ov); ca = cmp("audio velocity", va, oa)
print(f"one evaluation in {dt:.1f} s"); ok = ok_ids and cv > 0.93 and ca > 0.97   # baselines against ComfyUI (int4 blocks vs bf16): t2va 0.959 / 0.989, fl2va 0.986 / 0.975, ref2va (its own GPTQ export) 0.946 / 0.972, both 0.940 / 0.978
print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)

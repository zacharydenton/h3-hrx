"""The vision tower in Loom against ComfyUI's (build/ref_truth): the resized fox frame -> merged embeds [n][5120] and the
three DeepStack embeds; the patch flatten is checked on the host too.    python3 tests/test_vision.py"""
import sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from h3pipe_loom import H3Pipe
T = ROOT / "build/ref_truth"; L = lambda n: np.load(T / f"{n}.npy")
img = L("image_resized").astype(np.float32)                      # [H][W][3] in [0,1]
H, W = img.shape[:2]; gh, gw = H // 16, W // 16
# the patch flatten as process_qwen2vl_images: CLIP mean/std, merge order, content (c, t, py, px)
mean = np.array([0.48145466, 0.4578275, 0.40821073], np.float32); std = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
x = ((img - mean) / std).transpose(2, 0, 1)[None].repeat(2, 0)    # [2][3][H][W], CLIP normalisation (process_qwen2vl_images)
p = x.reshape(2, 3, gh // 2, 2, 16, gw // 2, 2, 16).transpose(2, 5, 3, 6, 1, 0, 4, 7).reshape(gh * gw, 1536)
pref = L("vision_patches"); print("patches vs comfy: max|diff|", float(np.abs(p - pref).max()))
pipe = H3Pipe()
t0 = time.time(); merged, ds = pipe.vision_embed(img); dt = time.time() - t0
mref, dref = L("vision_merged"), L("vision_deepstack")
def cmp(name, a, b):
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel(); cos = float(a @ b / np.sqrt((a @ a) * (b @ b))); rel = float(np.linalg.norm(a - b) / np.linalg.norm(b))
    print(f"  {name}: cosine {cos:.6f}  rel rms {rel:.4f}"); return cos
c = [cmp("merged", merged, mref)] + [cmp(f"deepstack {j}", ds[j], dref[j]) for j in range(3)]
print(f"vision tower {merged.shape[0]} tokens in {dt:.2f} s (first call compiles)")
ok = min(c) > 0.995; print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)

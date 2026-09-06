"""The video VAE encoder in Loom against ComfyUI's (build/ref_truth): the resized fox frame -> [24][1][30][54] and the
17-frame clip -> [24][2][30][54].    python3 tests/test_vae_encoder.py [--clip]"""
import sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from h3pipe_loom import H3Pipe
from PIL import Image
T = ROOT / "build/ref_truth"; L = lambda n: np.load(T / f"{n}.npy")
pipe = H3Pipe()
def cmp(name, a, b):
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel(); cos = float(a @ b / np.sqrt((a @ a) * (b @ b))); rel = float(np.linalg.norm(a - b) / np.linalg.norm(b))
    print(f"  {name}: cosine {cos:.6f}  rel rms {rel:.4f}"); return cos
img = L("image_resized").astype(np.float32)
t0 = time.time(); z = pipe.encode_video(img); print(f"image {img.shape[:2]} -> {z.shape} in {time.time() - t0:.2f} s (first call compiles)")
c1 = cmp("image latents", z, L("image_z"))
ok = c1 > 0.995
if "--clip" in sys.argv:
    frames = np.stack([np.asarray(Image.open(ROOT / f"build/refs/fox_{i:02d}.png").convert("RGB"), dtype=np.float32) / 255.0 for i in range(1, 18)])
    t0 = time.time(); zc = pipe.encode_video(frames); print(f"clip {frames.shape} -> {zc.shape} in {time.time() - t0:.2f} s")
    c2 = cmp("clip latents", zc, L("clip_z")); ok = ok and c2 > 0.995
print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)

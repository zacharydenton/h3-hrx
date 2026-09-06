"""The audio VAE encoder in Loom against ComfyUI's (build/ref_truth from tools/ref_truth_comfy.py): the fox clip's
stereo wav -> [2][32][T] latents.    python3 tests/test_audio_encoder.py"""
import sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from h3pipe_loom import H3Pipe
truth = ROOT / "build/ref_truth"
wav = np.load(truth / "audio_wav.npy").astype(np.float32)          # [2][L]
z_ref = np.load(truth / "audio_z.npy").astype(np.float32)          # [32][2][T] -> [2][32][T]
z_ref = np.transpose(z_ref, (1, 0, 2))
pipe = H3Pipe()
t0 = time.time(); z = pipe.encode_audio(wav); dt = time.time() - t0
print(f"encoded {wav.shape[1]} samples -> {z.shape} in {dt:.2f} s (first call compiles)")
t0 = time.time(); z = pipe.encode_audio(wav); print(f"second call {time.time() - t0:.3f} s")
assert z.shape == z_ref.shape, (z.shape, z_ref.shape)
x, y = z.astype(np.float64).ravel(), z_ref.astype(np.float64).ravel()
cos = float(x @ y / np.sqrt((x @ x) * (y @ y))); err = float(np.abs(x - y).max()); rel = float(np.linalg.norm(x - y) / np.linalg.norm(y))
print(f"cosine {cos:.7f}  max|diff| {err:.4e}  rel rms {rel:.3e}")
ok = cos > 0.9999 and rel < 1e-2
print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)

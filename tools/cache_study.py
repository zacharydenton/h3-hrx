"""The first-block step cache's trade: for each threshold, denoise the same prompt/seed through libh3pipe,
report the evaluations skipped, the time, and the latent agreement with the uncached run (plus frame PSNR
after the Loom decode). Small clip by default so a run is a minute.
    python3 tools/cache_study.py [--thresholds 0,0.05,0.1,0.15,0.2] [--frames 22] [--steps 30] [--height 480 --width 864]"""
import argparse, math, os, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from h3pipe_loom import H3Pipe
from encode_prompt import TOK
PROMPT = "A red fox trotting through a snowy forest at dawn, cinematic"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--thresholds", default="0,0.05,0.1,0.15,0.2"); ap.add_argument("--frames", type=int, default=22); ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--height", type=int, default=480); ap.add_argument("--width", type=int, default=864); ap.add_argument("--prompt", default=PROMPT); ap.add_argument("--decode", action="store_true")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    ids = AutoTokenizer.from_pretrained(str(TOK))(a.prompt, add_special_tokens=False)["input_ids"]
    os.environ["H3_CACHE_TRACE"] = "1"
    pipe = H3Pipe(); base = None
    for th in [float(v) for v in a.thresholds.split(",")]:
        p = H3Pipe.params(height=a.height, width=a.width, frames=a.frames, steps=a.steps, seed=0, cache_threshold=th); sh = pipe.shape(p)
        t0 = time.time(); video, audio = pipe.denoise(ids, p); dt = time.time() - t0
        if base is None:
            base = (video.copy(), audio.copy()); frames0 = pipe.decode_video(p, video) if a.decode else None
            print(f"  threshold {th:.2f}: {dt:6.1f} s (reference, {sh.latent_t}x{sh.lat_h}x{sh.lat_w} latents)"); continue
        cv = float(np.dot(video.ravel(), base[0].ravel()) / (np.linalg.norm(video) * np.linalg.norm(base[0]))); ca = float(np.dot(audio.ravel(), base[1].ravel()) / (np.linalg.norm(audio) * np.linalg.norm(base[1])))
        line = f"  threshold {th:.2f}: {dt:6.1f} s, video latent cosine {cv:.4f}, audio {ca:.4f}"
        if a.decode:
            fr = pipe.decode_video(p, video); mse = float(((fr.astype(np.float32) - frames0.astype(np.float32)) ** 2).mean()); line += f", frame PSNR vs uncached {10 * math.log10(255 ** 2 / max(mse, 1e-9)):.1f} dB"
        print(line, flush=True)
    pipe.close()


if __name__ == "__main__":
    main()

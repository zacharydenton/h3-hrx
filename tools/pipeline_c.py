"""Text -> video + audio through libh3pipe.so alone: the tokenizer here, everything else in the C
library (every kernel in Loom). Writes <out>.mp4 (+ .wav) through ffmpeg.
    python3 tools/pipeline_c.py "a red fox ..." [--frames 124 --steps 50 --height 480 --width 864 --seed 0 --out build/clip_c.mp4]"""
import argparse, subprocess, sys, time, wave
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from h3pipe_loom import H3Pipe
from encode_prompt import TOK
FPS, RATE = 24, 32000


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("prompt"); ap.add_argument("--height", type=int, default=480); ap.add_argument("--width", type=int, default=864)
    ap.add_argument("--frames", type=int, default=124); ap.add_argument("--steps", type=int, default=50); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-threshold", type=float, default=0.0, help="first-block step cache threshold (0 = off)"); ap.add_argument("--vae-bits", type=int, default=8); ap.add_argument("--out", default=str(ROOT / "build/clip_c.mp4")); ap.add_argument("--latents-out", default=None)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    ids = AutoTokenizer.from_pretrained(str(TOK))(a.prompt, add_special_tokens=False)["input_ids"]
    t0 = time.time(); pipe = H3Pipe(vae_bits=a.vae_bits); print(f"session in {time.time() - t0:.1f} s; {len(ids)} prompt tokens", flush=True)
    p = H3Pipe.params(height=a.height, width=a.width, frames=a.frames, steps=a.steps, seed=a.seed, cache_threshold=a.cache_threshold); sh = pipe.shape(p)
    print(f"{sh.frames} frames at {a.width}x{a.height}: {sh.latent_t}x{sh.lat_h}x{sh.lat_w} latents, {sh.audio_t} audio latents", flush=True)
    t0 = time.time()
    video, audio = pipe.denoise(ids, p, progress=lambda step, n, sec: print(f"  step {step}/{n}  {sec:.1f} s", flush=True) or 0)
    print(f"denoised in {time.time() - t0:.1f} s", flush=True)
    if a.latents_out: np.savez(a.latents_out, video=video, audio=audio)
    t0 = time.time(); frames = pipe.decode_video(p, video); print(f"video decoded in {time.time() - t0:.1f} s", flush=True)
    t0 = time.time(); samples = pipe.decode_audio(audio); print(f"audio decoded in {time.time() - t0:.1f} s", flush=True)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(samples.T, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(out.with_suffix(".wav")), "wb") as wf: wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(RATE); wf.writeframes(pcm.tobytes())
    raw = out.with_suffix(".rgb"); raw.write_bytes(frames.tobytes())
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{a.width}x{a.height}", "-r", str(FPS), "-i", str(raw), "-i", str(out.with_suffix(".wav")),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-c:a", "aac", "-b:a", "192k", "-shortest", str(out)], check=True)
    raw.unlink(); print(f"wrote {out}")


if __name__ == "__main__":
    main()

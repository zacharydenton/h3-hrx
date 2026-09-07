"""Text -> video + audio through libh3pipe.so alone: the tokenizer here, everything else in the C
library (every kernel in Loom). Writes <out>.mp4 (+ .wav) through ffmpeg.
    python3 tools/pipeline_c.py "a red fox ..." [--frames 124 --steps 50 --height 480 --width 864 --seed 0 --out build/clip_c.mp4]"""
import argparse, os, math, subprocess, sys, time, wave
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from h3pipe_loom import H3Pipe
FPS, RATE = 24, 32000


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("prompt"); ap.add_argument("--height", type=int, default=480); ap.add_argument("--width", type=int, default=864)
    ap.add_argument("--frames", type=int, default=124); ap.add_argument("--steps", type=int, default=31, help="sigma grid points = evaluations + 1; 31 = 30 evaluations (ComfyUI's stock workflows use 21)"); ap.add_argument("--sampler", choices=["euler", "res_multistep"], default="res_multistep", help="res_multistep is the ComfyUI workflows' sampler"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-threshold", type=float, default=0.0, help="first-block step cache threshold (0 = off)"); ap.add_argument("--vae-bits", type=int, default=8); ap.add_argument("--out", default=str(ROOT / "build/clip_c.mp4")); ap.add_argument("--latents-out", default=None); ap.add_argument("--no-decode", action="store_true", help="stop after denoising (timing runs)")
    ap.add_argument("--ref-image", action="append", default=[], help="reference image (png/jpg) for ref2va: presented as <Picture i> and encoded by the VAE encoder; repeatable")
    ap.add_argument("--ref-audio", action="append", default=[], help="reference wav (32 kHz stereo/mono) for ref2va: <Audio j>; repeatable")
    ap.add_argument("--first-frame", default=None, help="keyframe image for fl2va (resized to the canvas)")
    ap.add_argument("--blocks", default=None, help="override the block weights directory"); ap.add_argument("--attn", choices=["f16", "i8", "i4"], default="i8", help="the DiT attention's QK^T operands (i8: the parity path; i4 ghosts keyframe/reference clips)"); ap.add_argument("--glue", default=None)
    ap.add_argument("--base-weights", action="store_true", help="run reference files on the base checkpoint when the ref2va exports are absent (otherwise an error)")
    a = ap.parse_args()
    from h3tok_ids import encode_presentation
    # reference runs take the ref2va checkpoint's exports (README, Weights: export_weights.py --ckpt <ref2va> --out build/weights_i8_ref2va, export_glue.py --ckpt <ref2va> --out build/weights_glue_ref2va)
    wdir = "weights_i8"   # the checkpoint's int8 rows
    want_refs = bool(a.ref_image or a.ref_audio)
    have_ref2va = (ROOT / f"build/{wdir}_ref2va/manifest.txt").exists() and (ROOT / "build/weights_glue_ref2va/manifest.txt").exists()
    if want_refs and not have_ref2va and not a.base_weights and not (a.blocks and a.glue):
        raise SystemExit(f"references need the ref2va exports, build/{wdir}_ref2va and build/weights_glue_ref2va (README, Weights); --base-weights runs the base checkpoint anyway, or give --blocks and --glue")
    suffix = "_ref2va" if want_refs and have_ref2va and not a.base_weights else ""
    blocks = a.blocks or str(ROOT / ("build/" + wdir + suffix)); attn = a.attn
    glue = a.glue or (str(ROOT / "build/weights_glue_ref2va") if suffix else None)
    t0 = time.time(); pipe = H3Pipe(vae_bits=a.vae_bits, blocks=blocks, glue=glue, attn=attn); print(f"session in {time.time() - t0:.1f} s ({os.path.basename(blocks)}, {attn} attention{', ref2va glue' if glue else ''})", flush=True)
    p = H3Pipe.params(height=a.height, width=a.width, frames=a.frames, steps=a.steps, seed=a.seed, cache_threshold=a.cache_threshold, sampler=a.sampler); sh = pipe.shape(p)
    print(f"{sh.frames} frames at {a.width}x{a.height}: {sh.latent_t}x{sh.lat_h}x{sh.lat_w} latents, {sh.audio_t} audio latents", flush=True)
    # references (ref2va) and the keyframe (fl2va): images resized as ComfyUI's nodes do, encoded by the Loom encoders
    refs, kfs, image_tokens = [], [], []
    def load_image(path, tw, th):
        from PIL import Image
        im = Image.open(path).convert("RGB").resize((tw, th), Image.BILINEAR); return np.asarray(im, dtype=np.float32) / 255.0
    if a.first_frame:
        img = load_image(a.first_frame, a.width, a.height); z = pipe.encode_video(img)
        kfs.append({"frame_index": 0, "video": z, "pixels": img}); image_tokens.append((a.height // 32) * (a.width // 32))
    for path in a.ref_image:
        from PIL import Image
        w, h = Image.open(path).size; scale = min(1.0, math.sqrt((a.width * a.height) / (w * h)))
        tw, th = max(32, round(w * scale / 32) * 32), max(32, round(h * scale / 32) * 32)
        img = load_image(path, tw, th); z = pipe.encode_video(img)
        refs.append({"kind": "image", "video": z, "pixels": img}); image_tokens.append((th // 32) * (tw // 32))
    for path in a.ref_audio:
        with wave.open(path, "rb") as wf:
            sr, ch, sw, n = wf.getframerate(), wf.getnchannels(), wf.getsampwidth(), wf.getnframes(); raw = wf.readframes(n)
        if sr != 32000: raise SystemExit(f"{path}: {sr} Hz; reference audio must be 32 kHz")
        pcm = np.frombuffer(raw, dtype={1: np.int8, 2: np.int16, 4: np.int32}[sw]).astype(np.float32) / float(2 ** (8 * sw - 1)); pcm = pcm.reshape(n, ch).T
        if ch == 1: pcm = np.concatenate([pcm, pcm])
        refs.append({"kind": "audio", "audio": pipe.encode_audio(pcm[:2])})
    ids = encode_presentation(a.prompt, images=image_tokens, audios=len(a.ref_audio))
    print(f"{len(ids)} prompt tokens ({len(kfs)} keyframe, {len(a.ref_image)} reference images, {len(a.ref_audio)} reference audio)", flush=True)
    t0 = time.time()
    video, audio = pipe.denoise(ids, p, progress=lambda step, n, sec: print(f"  step {step}/{n}  {sec:.1f} s", flush=True) or 0, refs=refs, keyframes=kfs)
    print(f"denoised in {time.time() - t0:.1f} s", flush=True)
    if a.latents_out: np.savez(a.latents_out, video=video, audio=audio)
    if a.no_decode: return
    t0 = time.time(); frames = pipe.decode_video(p, video); print(f"video decoded in {time.time() - t0:.1f} s", flush=True)
    t0 = time.time(); samples = pipe.decode_audio(audio); print(f"audio decoded in {time.time() - t0:.1f} s", flush=True)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(samples.T, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(out.with_suffix(".wav")), "wb") as wf: wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(RATE); wf.writeframes(pcm.tobytes())
    raw = out.with_suffix(".rgb"); raw.write_bytes(frames.tobytes())
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{a.width}x{a.height}", "-r", str(FPS), "-i", str(raw), "-i", str(out.with_suffix(".wav")),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-c:a", "aac", "-b:a", "192k", "-shortest", str(out.with_suffix(".mp4"))], check=True)   # --out with or without .mp4
    raw.unlink(); print(f"wrote {out.with_suffix('.mp4')}")


if __name__ == "__main__":
    main()

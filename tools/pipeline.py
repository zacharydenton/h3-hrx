"""Text -> video + audio with MiniMax H3, the 50 blocks in Loom.

Everything around the blocks is torch: the cached prompt conditioning (tools/encode_prompt.py),
the packed layout and per-row modulation classes (reference/h3_ref.py), diffusers'
MiniMaxH3Scheduler for the two schedules (video shift 12, audio shift 3), the reference's
embeddings and final layer, and diffusers' two VAEs. The velocity handed to the scheduler is
the head's raw data-ward output (diffusers' convention; ComfyUI negates it).

    python3 tools/pipeline.py "prompt" [--height 480 --width 864 --frames 124 --steps 50 --seed 0 --out build/clip.mp4]
Frames snap up to the next 17n + 5 the video VAE decodes; the model is trained for 5-15 s.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "reference")); sys.path.insert(0, str(ROOT / "tools"))
import h3_ref as R
from h3_loom import H3Blocks, mods_table
from encode_prompt import prompt_path

MODELS = Path.home() / "h3-models"
FPS, AUDIO_RATE, AUDIO_LATENTS_PER_S = 24, 32000, 40


def align_frames(n: int) -> int:
    while n % 17 != 5:
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--height", type=int, default=480); ap.add_argument("--width", type=int, default=864)
    ap.add_argument("--frames", type=int, default=124)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weights", default=str(ROOT / "build/weights_gptq"))
    ap.add_argument("--out", default=str(ROOT / "build/clip.mp4"))
    ap.add_argument("--latents-out", default=None)
    ap.add_argument("--profile", action="store_true")
    a = ap.parse_args()
    dev = "cuda"
    from diffusers import MiniMaxH3Scheduler, AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio

    # --- conditioning and layout ---
    pp = prompt_path(a.prompt, ROOT / "build/prompts")
    if not pp.exists():
        raise SystemExit(f"encode the prompt first: python3 tools/encode_prompt.py {a.prompt!r}")
    prompt = torch.load(pp)
    frames = align_frames(a.frames)
    latent_t = (frames - 5) // 17 * 5 + 2
    lat_h, lat_w = a.height // 16, a.width // 16
    audio_t = int(round(frames / FPS * AUDIO_LATENTS_PER_S))
    text_len = prompt["embeds"].shape[0]
    layout = R.Layout(text_len, latent_t, lat_h, lat_w, audio_t)
    print(f"{frames} frames ({frames / FPS:.2f} s) at {a.width}x{a.height}: {latent_t}x{lat_h}x{lat_w} latents, {audio_t} audio latents, "
          f"{text_len} text rows -> {layout.seq_len} packed rows")

    ckpt = R.Checkpoint(device=dev, dtype=torch.bfloat16)
    ref = R.H3Ref(ckpt, quant="none")
    with torch.no_grad():
        text_x = ref.text_in(prompt["embeds"].to(dev))                       # [L, 5376], once
    cos, sin = R.rope_tables(layout.position_ids, ref.inv_freq, dev)
    rows = layout.adaln_rows.to(dev); tclass = layout.tclass.to(dev)

    # --- schedules ---
    video_sched = MiniMaxH3Scheduler(shift=12.0); audio_sched = MiniMaxH3Scheduler(shift=3.0)
    video_sched.set_timesteps(a.steps, device=dev); audio_sched.set_timesteps(a.steps, device=dev)
    steps = list(zip(video_sched.timesteps.tolist(), audio_sched.timesteps.tolist()))

    # --- noise (the same draw order as diffusers: video latent tensor, then audio rows) ---
    gen = torch.Generator(dev).manual_seed(a.seed)
    latents = torch.randn((1, 24, latent_t, lat_h, lat_w), generator=gen, device=dev, dtype=torch.float32)
    video_rows = latents.reshape(1, 24, latent_t, lat_h // 2, 2, lat_w // 2, 2).permute(0, 2, 3, 5, 1, 4, 6).reshape(-1, 96)
    audio_rows = torch.randn((audio_t * 2, 32), generator=gen, device=dev, dtype=torch.float32)

    blocks = H3Blocks(layout.seq_len, layers=50, weights=a.weights)
    if a.profile:
        blocks.profile(True)
    t_start = time.time()
    for i, (t_v, t_a) in enumerate(steps):
        t0 = time.time()
        temb = ref.t_emb(torch.tensor([t_v, t_a]))
        with torch.no_grad():
            x = torch.cat([text_x, ref.audio_in(audio_rows), ref.video_in(video_rows)], dim=0)
            mods = mods_table(ref, temb, 50)
            y = blocks.forward(x, rows, mods, cos, sin).to(dev)
            v_rows, a_rows = ref.final(y, temb, tclass, layout)                   # data-ward velocity rows
        video_rows = video_sched.step(v_rows.float(), video_sched.timesteps[i], video_rows, return_dict=False)[0]
        audio_rows = audio_sched.step(a_rows.float(), audio_sched.timesteps[i], audio_rows, return_dict=False)[0]
        print(f"  step {i + 1}/{len(steps)}  t_v {t_v:.4f} t_a {t_a:.4f}  {time.time() - t0:.1f} s", flush=True)
    blocks.close()
    print(f"denoised in {time.time() - t_start:.0f} s")
    latents = video_rows.reshape(1, latent_t, lat_h // 2, lat_w // 2, 24, 1, 2, 2).permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(1, 24, latent_t, lat_h, lat_w)
    audio = audio_rows.reshape(2, audio_t, 32).permute(0, 2, 1)                     # [2 (stereo), 32, audio_t]: the audio VAE takes each channel as a batch entry
    if a.latents_out:
        torch.save(dict(video=latents.cpu(), audio=audio.cpu(), frames=frames, size=(a.height, a.width)), a.latents_out)

    # --- decode ---
    del blocks, ref, ckpt; torch.cuda.empty_cache()
    vae = AutoencoderKLMiniMaxH3.from_pretrained(str(MODELS / "vae"), torch_dtype=torch.float16).to(dev).eval()
    mean = torch.tensor(vae.config.latents_mean, device=dev).view(1, -1, 1, 1, 1); std = torch.tensor(vae.config.latents_std, device=dev).view(1, -1, 1, 1, 1)
    with torch.no_grad():
        video = vae.decode((latents * std + mean).to(torch.float16), return_dict=False)[0]     # [1, 3, F, H, W]
    del vae; torch.cuda.empty_cache()
    avae = AutoencoderKLMiniMaxH3Audio.from_pretrained(str(MODELS / "audio_vae"), torch_dtype=torch.float32).to(dev).eval()
    amean = torch.tensor(avae.config.latents_mean, device=dev).view(1, -1, 1); astd = torch.tensor(avae.config.latents_std, device=dev).view(1, -1, 1)
    with torch.no_grad():
        wave = avae.decode((audio * astd + amean).float(), return_dict=False)[0]              # [2, 1, samples]
    print("video", tuple(video.shape), "audio", tuple(wave.shape))
    write_clip(video, wave, Path(a.out))


def write_clip(video: torch.Tensor, wave: torch.Tensor, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    frames = ((video[0].float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()   # [F, H, W, 3]
    f, h, w, _ = frames.shape
    wav = wave.float().cpu().reshape(-1, wave.shape[-1]).T.contiguous().numpy()        # [samples, 2]
    pcm = (np.clip(wav, -1, 1) * 32767).astype(np.int16)
    wav_path = out.with_suffix(".wav"); raw_path = out.with_suffix(".rgb")
    import wave as wavmod
    with wavmod.open(str(wav_path), "wb") as wf:
        wf.setnchannels(pcm.shape[1]); wf.setsampwidth(2); wf.setframerate(AUDIO_RATE); wf.writeframes(pcm.tobytes())
    raw_path.write_bytes(frames.tobytes())
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", str(raw_path),
                    "-i", str(wav_path), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-c:a", "aac", "-b:a", "192k", "-shortest", str(out)], check=True)
    raw_path.unlink()
    print(f"wrote {out} ({f} frames, {w}x{h}) and {wav_path}")


if __name__ == "__main__":
    main()

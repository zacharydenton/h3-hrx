"""Decode saved latents (tools/pipeline.py --latents-out) to an mp4: the torch baseline for the
Loom decoder, timed per stage.
    python3 tools/decode_latents.py build/fox_480p_5s_latents.pt --out build/redecode.mp4"""
import argparse, sys, time
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from pipeline import write_clip, MODELS

ap = argparse.ArgumentParser(); ap.add_argument("latents"); ap.add_argument("--out", default=str(ROOT / "build/redecode.mp4"))
a = ap.parse_args(); dev = "cuda"
from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio
fx = torch.load(a.latents)
latents, audio = fx["video"].to(dev), fx["audio"].to(dev)
t0 = time.time()
vae = AutoencoderKLMiniMaxH3.from_pretrained(str(MODELS / "vae"), torch_dtype=torch.float16).to(dev).eval()
print(f"video vae loaded {time.time() - t0:.1f} s; latents {tuple(latents.shape)}", flush=True)
mean = torch.tensor(vae.config.latents_mean, device=dev).view(1, -1, 1, 1, 1); std = torch.tensor(vae.config.latents_std, device=dev).view(1, -1, 1, 1, 1)
torch.cuda.synchronize(); t0 = time.time()
with torch.no_grad():
    video = vae.decode((latents * std + mean).to(torch.float16), return_dict=False)[0]
torch.cuda.synchronize(); print(f"video decode {time.time() - t0:.1f} s -> {tuple(video.shape)}", flush=True)
del vae; torch.cuda.empty_cache()
t0 = time.time()
avae = AutoencoderKLMiniMaxH3Audio.from_pretrained(str(MODELS / "audio_vae"), torch_dtype=torch.float32).to(dev).eval()
amean = torch.tensor(avae.config.latents_mean, device=dev).view(1, -1, 1); astd = torch.tensor(avae.config.latents_std, device=dev).view(1, -1, 1)
if audio.dim() == 4: audio = audio[0].permute(1, 0, 2)
with torch.no_grad():
    wave = avae.decode((audio * astd + amean).float(), return_dict=False)[0]
torch.cuda.synchronize(); print(f"audio decode (incl. load) {time.time() - t0:.1f} s -> {tuple(wave.shape)}", flush=True)
write_clip(video, wave, Path(a.out))

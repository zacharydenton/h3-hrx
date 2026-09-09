"""A whole ComfyUI MiniMax H3 clip with the stock workflows' settings (res_multistep, 'simple', 20 evaluations,
cfg 1, the model's shifts): t2va, or fl2va from --first-frame. Ground truth for what a conditioned clip should
look like, and the fixtures `scripts/parity.py gate` compares against.

This is the half of the parity check that cannot live in `scripts/parity.py`: it runs inside the
Strix Halo ComfyUI image, against ComfyUI's own modules, where nothing else here runs.

    podman run --rm -v "$HOME:$HOME" -w "$PWD" -e PYTHONPATH=/opt/ComfyUI \
      --entrypoint /opt/venv/bin/python docker.io/kyuz0/amd-strix-halo-comfyui:latest \
      scripts/comfy_dump.py --dump-steps --dump-blocks 0,1,2,5,10,20,30,40,49 \
      --steps 2 --out build/comfy_t2va_blocks

Writes frames.npy [F,H,W,3] uint8, audio.wav, video_latent.npy, audio_latent.npy; mp4 muxing is left
to ffmpeg outside the container."""
import argparse, logging, os, sys, time, wave
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--comfy", type=Path, default=Path("/opt/ComfyUI")); ap.add_argument("--models", type=Path, default=Path("/mnt/usb/models/comfy"))
ap.add_argument("--model", default="minimax_h3_fl2va_pruned_int8_convrot.safetensors")
ap.add_argument("--width", type=int, default=864); ap.add_argument("--height", type=int, default=480); ap.add_argument("--length", type=int, default=22)
ap.add_argument("--steps", type=int, default=20); ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--sampler", default="res_multistep"); ap.add_argument("--scheduler", default="simple")
ap.add_argument("--prompt", default="A red fox trotting through a snowy forest at dawn, cinematic")
ap.add_argument("--first-frame", default=None); ap.add_argument("--dump-blocks", default="", help="comma-separated block indices whose output rows to save at the first evaluation (plus the refined text and the embedded rows)"); ap.add_argument("--dump-steps", action="store_true", help="save the noise and, per evaluation, the sampler state x_k and denoised d_k (video part)"); ap.add_argument("--out", type=Path, default=Path("build/comfy_clip"))
a = ap.parse_args()
sys.path.insert(0, str(a.comfy)); sys.argv = [sys.argv[0]]
import comfy.options; comfy.options.enable_args_parsing()
logging.basicConfig(level=logging.WARNING)
from comfy.cli_args import args as comfy_args, enables_dynamic_vram
import comfy_aimdo.control
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
if enables_dynamic_vram(): comfy_aimdo.control.init(simple_vram_headroom=None, nvml_pressure=not comfy_args.disable_nvml_pressure)
import torch, comfy.sd, comfy.sample, comfy.samplers, comfy.model_management, comfy.utils, comfy.memory_management, comfy.model_patcher
if enables_dynamic_vram() and comfy.model_management.rocm_version >= (7, 14):
    if comfy_aimdo.control.init_devices((d.index, int(comfy_args.vram_headroom * 1024**3)) for d in comfy.model_management.get_all_torch_devices()):
        comfy.memory_management.aimdo_enabled = True; comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
from PIL import Image
import comfy_extras.nodes_minimax_h3 as H3
from comfy_extras.nodes_audio import vae_decode_audio
a.out.mkdir(parents=True, exist_ok=True)


def stamp(): torch.cuda.synchronize(); return time.perf_counter()


with torch.inference_mode():
    t0 = stamp()
    clip = comfy.sd.load_clip([str(a.models / "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors")], clip_type=comfy.sd.CLIPType.MINIMAX)
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(a.models / "vae/minimax_h3_video_vae_fp16.safetensors")))
    audio_vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(a.models / "vae/minimax_h3_audio_vae_fp32.safetensors")))
    if a.first_frame:
        img = torch.from_numpy(np.asarray(Image.open(a.first_frame).convert("RGB"), dtype=np.float32) / 255.0)[None]
        out = H3.MiniMaxH3ImageToVideo.execute(clip, vae, a.prompt, a.width, a.height, a.length, first_frame=img)
        positive, latent = out.args[0], out.args[1]
        for kf in positive[0][1].get("minimax_keyframes", []): print("keyframe at frame", kf["resolved_frame_index"], flush=True)
    else:
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(a.prompt)); latent, _ = H3._empty_av_latent(a.width, a.height, a.length)
    print(f"conditioning in {stamp() - t0:.1f} s", flush=True)
    clip = None; comfy.model_management.unload_all_models(); comfy.model_management.soft_empty_cache()
    model = comfy.sd.load_diffusion_model(str(a.models / "diffusion_models" / a.model))
    noise = comfy.sample.prepare_noise(latent["samples"], a.seed)
    marks = [stamp()]
    if a.dump_steps:
        nv, na = noise.unbind(); np.save(a.out / "noise_video.npy", nv[0].float().cpu().numpy()); np.save(a.out / "noise_audio.npy", na[0].float().cpu().numpy())
    def cb(step, x0, x, total):
        marks.append(stamp()); print(f"  step {step + 1}/{total}  {marks[-1] - marks[-2]:.1f} s", flush=True)
        if a.dump_steps:   # k_diffusion's callback: x is the state entering evaluation `step`, x0 its denoised prediction
            np.save(a.out / f"x_{step:02d}.npy", x.unbind()[0][0].float().cpu().numpy()); np.save(a.out / f"d_{step:02d}.npy", x0.unbind()[0][0].float().cpu().numpy())
    if a.dump_blocks:   # layerwise truth at the first evaluation: hook the token refiner, the embedding (block 0's input) and the chosen blocks' outputs
        bd = a.out / "blocks"; bd.mkdir(exist_ok=True); dm = model.model.diffusion_model; seen = set()
        def hook(mod, name, pick=lambda args, out: out):
            orig = mod.forward
            def f(*args, **kw):
                out = orig(*args, **kw)
                if name not in seen:
                    seen.add(name); t = pick(args, out); np.save(bd / f"{name}.npy", t.detach().float().cpu().numpy()); print("saved", name, tuple(t.shape), flush=True)
                return out
            mod.forward = f
        hook(dm.token_refiner, "refined_text")
        for i in [int(v) for v in a.dump_blocks.split(",")]:
            hook(dm.blocks[i], f"blk_{i:02d}")
            if i == 0: hook(dm.blocks[0], "h_in", pick=lambda args, out: args[0])
    samples = comfy.sample.sample(model, noise, a.steps, 1.0, a.sampler, a.scheduler, positive, positive, latent["samples"], seed=a.seed, callback=cb, disable_pbar=True)
    print(f"sampled {a.steps} evaluations ({a.sampler}, {a.scheduler}) in {stamp() - marks[0]:.1f} s", flush=True)
    model = None; comfy.model_management.unload_all_models(); comfy.model_management.soft_empty_cache()
    v, au = samples.unbind() if getattr(samples, "is_nested", False) else (samples, None)
    np.save(a.out / "video_latent.npy", v[0].float().cpu().numpy())
    images = vae.decode(v)
    if images.ndim == 5: images = images.reshape(-1, *images.shape[-3:])
    frames = (images.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy(); np.save(a.out / "frames.npy", frames); print("frames", frames.shape, flush=True)
    if au is not None:
        np.save(a.out / "audio_latent.npy", au[0].float().cpu().numpy())
        audio = vae_decode_audio(audio_vae, {"samples": au}); wav = audio["waveform"][0].float().cpu().numpy(); sr = int(audio["sample_rate"])
        with wave.open(str(a.out / "audio.wav"), "wb") as wf:
            wf.setnchannels(wav.shape[0]); wf.setsampwidth(2); wf.setframerate(sr); wf.writeframes((np.clip(wav.T, -1, 1) * 32767).astype(np.int16).tobytes())
        print("audio", wav.shape, sr, flush=True)
    print("done", flush=True)

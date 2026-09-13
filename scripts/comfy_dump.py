"""A whole ComfyUI MiniMax H3 clip with the stock workflows' settings (res_multistep, 'simple', 20 evaluations,
cfg 1, the model's shifts): t2va, or fl2va from --first-frame. Ground truth for what a conditioned clip should
look like, and the fixtures `scripts/parity.py gate` compares against.

This is the half of the parity check that cannot live in `scripts/parity.py`: it runs inside the
Strix Halo ComfyUI image, with numpy and huggingface_hub installed, against
ComfyUI's own modules.

    podman run --rm -v "$HOME:$HOME" -w "$PWD" -e PYTHONPATH=/opt/ComfyUI \
      -e HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}/hub}" \
      --entrypoint /opt/venv/bin/python docker.io/kyuz0/amd-strix-halo-comfyui:latest \
      scripts/comfy_dump.py --dump-steps --dump-blocks 0,1,2,5,10,20,30,40,49 \
      --steps 2 --out build/comfy_t2va_blocks

Checkpoints must already be downloaded into the standard Hugging Face Hub cache.
If the cache is outside $HOME, bind-mount its directory into the container too.

Writes frames.npy [F,H,W,3] uint8, audio.wav, video_latent.npy, audio_latent.npy by default.
With --video-out, writes MP4/WAV instead; requires FFmpeg with libx264 and AAC encoders."""
import argparse, json, logging, os, subprocess, sys, time, wave
from pathlib import Path
import numpy as np
from parity import model_path
ap = argparse.ArgumentParser()
ap.add_argument("--comfy", type=Path, default=Path("/opt/ComfyUI"))
ap.add_argument("--model", default="minimax_h3_fl2va_pruned_int8_convrot.safetensors")
ap.add_argument("--width", type=int, default=864); ap.add_argument("--height", type=int, default=480); ap.add_argument("--length", type=int, default=22)
ap.add_argument("--steps", type=int, default=20); ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--sampler", default="res_multistep"); ap.add_argument("--scheduler", default="simple")
ap.add_argument("--video-out", type=Path, help="write MP4/WAV instead of final NPY exports (intermediate dump flags still apply)")
ap.add_argument("--prompt-file", type=Path, help="read the structured prompt from a UTF-8 file")
ap.add_argument("--prompt", default="A red fox trotting through a snowy forest at dawn, cinematic")
ap.add_argument("--first-frame", default=None); ap.add_argument("--dump-blocks", default="", help="comma-separated block indices whose output rows to save at the first evaluation (plus the refined text and the embedded rows)"); ap.add_argument("--dump-steps", action="store_true", help="save the noise and, per evaluation, the sampler state x_k and denoised d_k (video part)"); ap.add_argument("--out", type=Path, default=Path("build/comfy_clip"))
a = ap.parse_args()
if a.prompt_file is not None: a.prompt = a.prompt_file.read_text(encoding="utf-8").strip()
if a.video_out is not None:
    # Fail before loading models if this environment cannot produce the requested output.
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=s=32x32:r=24",
        "-f", "lavfi", "-i", "anullsrc=r=32000:cl=stereo", "-t", "0.1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-f", "mp4",
        "-movflags", "frag_keyframe+empty_moov", "-",
    ], stdout=subprocess.DEVNULL, check=True)
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
torch.cuda.reset_peak_memory_stats()
metrics = {
    "torch": torch.__version__, "hip": torch.version.hip,
    "comfy_commit": subprocess.check_output(["git", "-C", str(a.comfy), "rev-parse", "HEAD"], text=True).strip(),
    "gpu": torch.cuda.get_device_name(0), "width": a.width, "height": a.height,
    "frames": a.length, "evaluations": a.steps, "seed": a.seed,
    "sampler": a.sampler, "schedule": a.scheduler, "cfg": 1.0,
    "checkpoint": str(model_path(f"diffusion_models/{a.model}")),
    "prompt": a.prompt, "step_seconds": [],
}


def save_metrics():
    metrics["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    metrics["torch_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    (a.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


def stamp(): torch.cuda.synchronize(); return time.perf_counter()


with torch.inference_mode():
    t0 = stamp()
    clip = comfy.sd.load_clip([str(model_path("text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"))], clip_type=comfy.sd.CLIPType.MINIMAX)
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(model_path("vae/minimax_h3_video_vae_fp16.safetensors"))))
    audio_vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(model_path("vae/minimax_h3_audio_vae_fp32.safetensors"))))
    if a.first_frame:
        img = torch.from_numpy(np.asarray(Image.open(a.first_frame).convert("RGB"), dtype=np.float32) / 255.0)[None]
        out = H3.MiniMaxH3ImageToVideo.execute(clip, vae, a.prompt, a.width, a.height, a.length, first_frame=img)
        positive, latent = out.args[0], out.args[1]
        for kf in positive[0][1].get("minimax_keyframes", []): print("keyframe at frame", kf["resolved_frame_index"], flush=True)
    else:
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(a.prompt)); latent, _ = H3._empty_av_latent(a.width, a.height, a.length)
    metrics["conditioning_seconds"] = stamp() - t0
    print(f"conditioning in {metrics['conditioning_seconds']:.1f} s", flush=True)
    clip = None; comfy.model_management.unload_all_models(); comfy.model_management.soft_empty_cache()
    model = comfy.sd.load_diffusion_model(str(model_path(f"diffusion_models/{a.model}")))
    metrics["dit_inference_dtype"] = str(model.model.get_dtype_inference())
    noise = comfy.sample.prepare_noise(latent["samples"], a.seed)
    marks = [stamp()]
    if a.dump_steps:
        nv, na = noise.unbind(); np.save(a.out / "noise_video.npy", nv[0].float().cpu().numpy()); np.save(a.out / "noise_audio.npy", na[0].float().cpu().numpy())
    def cb(step, x0, x, total):
        marks.append(stamp())
        metrics["step_seconds"].append(marks[-1] - marks[-2]); save_metrics()
        print(f"  step {step + 1}/{total}  {marks[-1] - marks[-2]:.1f} s", flush=True)
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
    metrics["sampling_seconds"] = stamp() - marks[0]; save_metrics()
    print(f"sampled {a.steps} evaluations ({a.sampler}, {a.scheduler}) in {metrics['sampling_seconds']:.1f} s", flush=True)
    model = None; comfy.model_management.unload_all_models(); comfy.model_management.soft_empty_cache()
    v, au = samples.unbind() if getattr(samples, "is_nested", False) else (samples, None)
    if a.video_out is None: np.save(a.out / "video_latent.npy", v[0].float().cpu().numpy())
    decode_start = stamp()
    images = vae.decode(v)
    if images.ndim == 5: images = images.reshape(-1, *images.shape[-3:])
    frames = (images.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
    if a.video_out is None: np.save(a.out / "frames.npy", frames)
    print("frames", frames.shape, flush=True)
    if au is not None:
        if a.video_out is None: np.save(a.out / "audio_latent.npy", au[0].float().cpu().numpy())
        audio = vae_decode_audio(audio_vae, {"samples": au}); wav = audio["waveform"][0].float().cpu().numpy(); sr = int(audio["sample_rate"])
        with wave.open(str(a.out / "audio.wav"), "wb") as wf:
            wf.setnchannels(wav.shape[0]); wf.setsampwidth(2); wf.setframerate(sr); wf.writeframes((np.clip(wav.T, -1, 1) * 32767).astype(np.int16).tobytes())
        print("audio", wav.shape, sr, flush=True)
    metrics["decode_and_save_seconds"] = stamp() - decode_start; save_metrics()
    if a.video_out is not None:
        if au is None: raise RuntimeError("MP4/WAV comparison requires the audio output")
        a.video_out.parent.mkdir(parents=True, exist_ok=True)
        raw_frames = np.ascontiguousarray(frames)
        mux_start = time.perf_counter()
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{a.width}x{a.height}", "-r", "24", "-i", "-", "-i", str(a.out / "audio.wav"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k", "-shortest", str(a.video_out),
        ], input=memoryview(raw_frames).cast("B"), check=True)
        metrics["mux_seconds"] = time.perf_counter() - mux_start
        metrics["video_out"] = str(a.video_out)
        save_metrics()
    print("done", flush=True)

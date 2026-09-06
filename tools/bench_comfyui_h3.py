"""ComfyUI's MiniMax H3 path, timed per denoising step: its standard loaders (int8 ConvRot checkpoint, bf16
compute), the minimax CLIP (Qwen3-VL-32B), the AV latent node, and the stock Euler sampler on the 'simple'
schedule with the model's own shifts. No server, graph cache, previews or decode. Run inside the Strix Halo
ComfyUI image (see docs/notes.md for the podman command); the local checkout is not needed.
    /opt/venv/bin/python tools/bench_comfyui_h3.py --width 1344 --height 768 --length 124 --steps 3"""
import argparse, json, logging, os, sys, time
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("--comfy", type=Path, default=Path("/opt/ComfyUI")); ap.add_argument("--models", type=Path, default=Path.home() / "comfy-models")
ap.add_argument("--model", default="minimax_h3_fl2va_pruned_int8_convrot.safetensors"); ap.add_argument("--clip", default="qwen3vl_32b_minimax_h3_int8_convrot.safetensors")
ap.add_argument("--width", type=int, default=1344); ap.add_argument("--height", type=int, default=768); ap.add_argument("--length", type=int, default=124)
ap.add_argument("--steps", type=int, default=3); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--prompt", default="A red fox trotting through a snowy forest at dawn, cinematic")
a = ap.parse_args()
sys.path.insert(0, str(a.comfy)); sys.argv = [sys.argv[0]]   # mmap the checkpoints: with --disable-mmap the 48 GB of int8 weights sit in RAM next to their device copies and the run is OOM-killed
import comfy.options; comfy.options.enable_args_parsing()
logging.basicConfig(level=logging.INFO)
from comfy.cli_args import args as comfy_args, enables_dynamic_vram
import comfy_aimdo.control
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
if enables_dynamic_vram(): comfy_aimdo.control.init(simple_vram_headroom=None, nvml_pressure=not comfy_args.disable_nvml_pressure)
import torch, comfy.sd, comfy.sample, comfy.model_management, comfy.utils, comfy.memory_management, comfy.model_patcher
if enables_dynamic_vram() and comfy.model_management.rocm_version >= (7, 14):
    if comfy_aimdo.control.init_devices((d.index, int(comfy_args.vram_headroom * 1024**3)) for d in comfy.model_management.get_all_torch_devices()):
        comfy_aimdo.control.set_log_info(); comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic; comfy.memory_management.aimdo_enabled = True
from comfy_extras.nodes_minimax_h3 import _empty_av_latent


def stamp(): torch.cuda.synchronize(); return time.perf_counter()


with torch.inference_mode():
    t0 = stamp()
    model = comfy.sd.load_diffusion_model(str(a.models / "diffusion_models" / a.model))
    clip = comfy.sd.load_clip([str(a.models / "text_encoders" / a.clip)], clip_type=comfy.sd.CLIPType.MINIMAX)
    print(json.dumps(dict(load_seconds=stamp() - t0, torch=torch.__version__, model_dtype=str(model.model.get_dtype()), dynamic_vram=comfy.memory_management.aimdo_enabled)), flush=True)
    latent, frames = _empty_av_latent(a.width, a.height, a.length)
    print(f"{frames} frames at {a.width}x{a.height}", flush=True)
    t1 = stamp(); cond = clip.encode_from_tokens_scheduled(clip.tokenize(a.prompt)); t2 = stamp()
    print(f"text encode {t2 - t1:.1f} s", flush=True)
    clip = None; comfy.model_management.unload_all_models(); comfy.model_management.soft_empty_cache()   # the 27 GB text encoder is not needed for sampling
    noise = comfy.sample.prepare_noise(latent["samples"], a.seed)
    marks = [stamp()]
    def cb(step, x0, x, total):
        marks.append(stamp()); print(f"  step {step + 1}/{total}  {marks[-1] - marks[-2]:.1f} s  (cumulative {marks[-1] - marks[0]:.1f} s)", flush=True)
    t3 = stamp()
    comfy.sample.sample(model, noise, a.steps, 1.0, "euler", "simple", cond, cond, latent["samples"], seed=a.seed, callback=cb, disable_pbar=True)
    t4 = stamp()
    print(json.dumps(dict(steps=a.steps, denoise_seconds=t4 - t3, per_step=[round(marks[i + 1] - marks[i], 1) for i in range(len(marks) - 1)], peak_gpu_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)), flush=True)

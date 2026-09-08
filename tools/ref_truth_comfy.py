"""Ground truth for the reference-conditioning port, from ComfyUI's own H3 path (run inside the Strix Halo
image, see CONTRIBUTING.md). Writes build/ref_truth/*.npy:
  audio_z [32,2,T] for --wav; image_z [24,1,h,w] for --image (ref2va 'match' sizing at --width x --height);
  clip_z [24,T,h,w] for the 17 frames --frames-dir/fox_%02d.png; vision_merged [n,5120], vision_deepstack
  [3,n,5120], presentation ids/tags and text states for the prompt with the image; one-step denoised video
  and audio (and the noise) for ref2va with image+audio refs and with the audio ref alone.
    /opt/venv/bin/python tools/ref_truth_comfy.py --models /mnt/usb/models/comfy"""
import argparse, json, logging, os, sys
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--comfy", type=Path, default=Path("/opt/ComfyUI")); ap.add_argument("--models", type=Path, default=Path("/mnt/usb/models/comfy"))
ap.add_argument("--out", type=Path, default=Path("build/ref_truth")); ap.add_argument("--wav", default="build/clip_hip.wav")
ap.add_argument("--image", default="build/refs/fox_frame.png"); ap.add_argument("--frames-dir", default="build/refs")
ap.add_argument("--width", type=int, default=864); ap.add_argument("--height", type=int, default=480); ap.add_argument("--length", type=int, default=22)
ap.add_argument("--prompt", default="<Picture 1> is the fox. A red fox trotting through a snowy forest at dawn, cinematic, with the sound of <Audio 1>")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--skip-dit", action="store_true"); ap.add_argument("--fl2va", action="store_true", help="also a first-frame keyframe case (needs --model-fl2va for its checkpoint)"); ap.add_argument("--only", default="", help="comma list of cases to run (both,audio,fl2va)"); ap.add_argument("--skip-encoders", action="store_true", help="only the DiT one-step cases (the encoder outputs already saved)")
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
a.out.mkdir(parents=True, exist_ok=True)
def save(name, t):
    arr = t.detach().float().cpu().numpy() if torch.is_tensor(t) else np.asarray(t); np.save(a.out / f"{name}.npy", arr); print(name, arr.shape, flush=True)
def load_png(p): return torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0)[None]   # [1,H,W,3]

with torch.inference_mode():
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(a.models / "vae/minimax_h3_video_vae_fp16.safetensors")))
    audio_vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(a.models / "vae/minimax_h3_audio_vae_fp32.safetensors")))
    import wave
    with wave.open(a.wav, "rb") as wf:   # torchaudio.load needs torchcodec, absent from the image
        sr, ch, sw, n = wf.getframerate(), wf.getnchannels(), wf.getsampwidth(), wf.getnframes(); raw = wf.readframes(n)
    pcm = np.frombuffer(raw, dtype={1: np.int8, 2: np.int16, 4: np.int32}[sw]).astype(np.float32) / float(2 ** (8 * sw - 1))
    wav = torch.from_numpy(pcm.reshape(n, ch).T.copy())
    audio = {"waveform": wav[None], "sample_rate": sr}
    if not a.skip_encoders:
        # audio encoder
      z, T = H3._encode_ref_audio(audio_vae, audio); save("audio_z", z[0]); save("audio_wav", wav); print("audio sr", sr, "T", T)
      # image encoder with ref2va 'match' sizing
      img = load_png(a.image); h, w = img.shape[1], img.shape[2]
      import math
      scale = min(1.0, math.sqrt((a.width * a.height) / (w * h)))
      tw = max(H3.CANVAS_MULTIPLE, round(w * scale / H3.CANVAS_MULTIPLE) * H3.CANVAS_MULTIPLE); th = max(H3.CANVAS_MULTIPLE, round(h * scale / H3.CANVAS_MULTIPLE) * H3.CANVAS_MULTIPLE)
      resized = H3._resize(img, tw, th, "disabled"); save("image_resized", resized[0]); z = vae.encode(resized); save("image_z", z[0])
      # 17-frame clip encoder
      frames = torch.cat([load_png(Path(a.frames_dir) / f"fox_{i:02d}.png") for i in range(1, 18)]); save("clip_frames_shape", frames.shape)
      zc = vae.encode(frames); save("clip_z", zc[0])
      # text encoder with the image: presentation, tags, vision outputs, text states
      clip = comfy.sd.load_clip([str(a.models / "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors")], clip_type=comfy.sd.CLIPType.MINIMAX)
      ref_items = [{"type": "image", "data": resized}, {"type": "audio"}]
      tokens = clip.tokenize(a.prompt, minimax_ref_items=ref_items)
      entries = tokens["qwen3vl_32b"][0]
      ids = [(e[0] if isinstance(e[0], int) else -1) for e in entries]; save("pres_ids", np.array(ids, dtype=np.int64))
      cond = clip.encode_from_tokens_scheduled(tokens)
      states = cond[0][0]; save("text_states", states[0]); extra = cond[0][1]
      inner = getattr(clip.cond_stage_model, "qwen3vl_32b", clip.cond_stage_model)   # SD1ClipModel keeps the clip under its name
      tags = getattr(inner.transformer, "last_token_tags", None)
      if tags is not None: save("text_tags", tags)
      for k, v in extra.items():
          if torch.is_tensor(v): save("cond_extra_" + k, v)
      # the vision tower alone on the same image
      import comfy.text_encoders.qwen_vl as QV
      flatten, grid = QV.process_qwen2vl_images(resized, patch_size=16)
      visual = inner.transformer.visual
      dev = comfy.model_management.get_torch_device(); visual.to(dev)
      merged, deepstack = visual(flatten.to(dev, dtype=torch.float32), grid.to(dev))
      save("vision_grid", grid); save("vision_patches", flatten); save("vision_merged", merged); save("vision_deepstack", torch.stack(deepstack))
    if a.skip_dit: sys.exit(0)
    if a.skip_encoders:
        clip = comfy.sd.load_clip([str(a.models / "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors")], clip_type=comfy.sd.CLIPType.MINIMAX)
        img = load_png(a.image); h, w = img.shape[1], img.shape[2]
        import math
        scale = min(1.0, math.sqrt((a.width * a.height) / (w * h)))
        tw = max(H3.CANVAS_MULTIPLE, round(w * scale / H3.CANVAS_MULTIPLE) * H3.CANVAS_MULTIPLE); th = max(H3.CANVAS_MULTIPLE, round(h * scale / H3.CANVAS_MULTIPLE) * H3.CANVAS_MULTIPLE)
        resized = H3._resize(img, tw, th, "disabled")
    # conditionings for both cases first (the text encoder and VAEs are needed only here), then the DiT alone
    cases = {}
    for tag, refs in (("both", dict(ref_images={"ref_image_1": img}, ref_audios={"ref_audio_1": audio})), ("audio", dict(ref_audios={"ref_audio_1": audio}))):
        prompt = a.prompt if tag == "both" else a.prompt.replace("<Picture 1> is the fox. ", "")
        out = H3.MiniMaxH3ReferenceToVideo.execute(clip, prompt, a.width, a.height, a.length, "match", vae, audio_vae, **refs)
        positive, latent = out.args[0], out.args[1]
        pres = clip.tokenize(prompt, minimax_ref_items=([{"type": "image", "data": resized}] if tag == "both" else []) + [{"type": "audio"}])
        save(f"{tag}_pres_ids", np.array([(e[0] if isinstance(e[0], int) else -1) for e in pres["qwen3vl_32b"][0]], dtype=np.int64))
        save(f"{tag}_text_states", positive[0][0][0])
        for r in positive[0][1].get("minimax_refs", []):
            if "latent" in r: save(f"{tag}_ref_latent", r["latent"][0])
            if r.get("audio_latent") is not None: save(f"{tag}_ref_audio_latent", r["audio_latent"][0])
        cases[tag] = (positive, latent)
    if a.fl2va:   # first-frame keyframe with the fl2va checkpoint (our default weights): the frame resized to the canvas
        out = H3.MiniMaxH3ImageToVideo.execute(clip, vae, a.prompt.replace("<Picture 1> is the fox. ", "").replace(", with the sound of <Audio 1>", ""), a.width, a.height, a.length, first_frame=img)
        positive, latent = out.args[0], out.args[1]
        canvas = H3._resize(img[:1], a.width, a.height, "disabled"); save("fl2va_image_resized", canvas[0])
        pres = clip.tokenize(a.prompt.replace("<Picture 1> is the fox. ", "").replace(", with the sound of <Audio 1>", ""), images=[canvas])
        save("fl2va_pres_ids", np.array([(e[0] if isinstance(e[0], int) else -1) for e in pres["qwen3vl_32b"][0]], dtype=np.int64))
        for kf in positive[0][1].get("minimax_keyframes", []): save("fl2va_keyframe_latent", kf["latent"][0]); print("keyframe index", kf["resolved_frame_index"])
        cases["fl2va"] = (positive, latent)
    if "t2va" in a.only.split(","):   # no references: the baseline agreement of the pipeline with ComfyUI (fl2va checkpoint)
        prompt = a.prompt.replace("<Picture 1> is the fox. ", "").replace(", with the sound of <Audio 1>", "")
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt)); latent, _ = H3._empty_av_latent(a.width, a.height, a.length)
        save("t2va_pres_ids", np.array([(e[0] if isinstance(e[0], int) else -1) for e in clip.tokenize(prompt)["qwen3vl_32b"][0]], dtype=np.int64)); save("t2va_text_states", positive[0][0][0])
        cases["t2va"] = (positive, latent)
    clip = None; vae = None; audio_vae = None
    comfy.model_management.unload_all_models(); comfy.model_management.soft_empty_cache()
    only = [x for x in a.only.split(",") if x]
    by_model = {"ref2va": [t for t in ("both", "audio") if t in cases and (not only or t in only)], "fl2va": [t for t in ("fl2va", "t2va") if t in cases and (not only or t in only)]}
    for model_tag, tags in by_model.items():
      if not tags: continue
      model = comfy.sd.load_diffusion_model(str(a.models / f"diffusion_models/minimax_h3_{model_tag}_pruned_int8_convrot.safetensors"))
      sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), "simple", 1); save("sigmas", sigmas)
      for tag in tags:
        positive, latent = cases[tag]
        noise = comfy.sample.prepare_noise(latent["samples"], a.seed)
        v, au = noise.unbind(); save(f"{tag}_noise_video", v[0]); save(f"{tag}_noise_audio", au[0])
        got = {}
        def cb(step, x0, x, total): got["x0"] = x0; got["x"] = x
        dm = model.model.diffusion_model; orig_forward = dm.forward   # the network's own inputs and raw outputs at the first call
        def hooked(x, timestep, context, *args, **kw):
            out = orig_forward(x, timestep, context, *args, **kw)
            if "raw" not in got: got["raw"] = (x[0].detach().float().cpu().clone(), x[1].detach().float().cpu().clone(), out[0].detach().float().cpu().clone(), out[1].detach().float().cpu().clone(), timestep.detach().float().cpu().clone())
            return out
        dm.forward = hooked
        comfy.sample.sample(model, noise, 1, 1.0, "euler", "simple", positive, positive, latent["samples"], seed=a.seed, callback=cb, disable_pbar=True)
        dv, da = got["x0"].unbind(); save(f"{tag}_denoised_video", dv[0]); save(f"{tag}_denoised_audio", da[0])
        xv, xa = got["x"].unbind(); save(f"{tag}_x_video", xv[0]); save(f"{tag}_x_audio", xa[0])
        dm.forward = orig_forward
        iv, ia, ov, oa, ts = got["raw"]; save(f"{tag}_in_video", iv[0]); save(f"{tag}_in_audio", ia[0]); save(f"{tag}_out_video", ov[0]); save(f"{tag}_out_audio", oa[0]); save(f"{tag}_timestep", ts)
      model = None; comfy.model_management.unload_all_models(); comfy.model_management.soft_empty_cache()
    print("done", flush=True)

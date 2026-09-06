"""The C pipeline (libh3pipe.so) against the Python reference, stage by stage:
  text_in: the refined text rows for a cached prompt vs reference/h3_ref.py's text_in on the same embeddings.
    python3 tests/test_pipe.py [--prompt-file build/prompts/<hash>.pt]"""
import argparse, glob, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "reference")); sys.path.insert(0, str(ROOT / "tools"))
import h3_ref as R
from h3pipe_loom import H3Pipe


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--prompt-file", default=None); ap.add_argument("--step", action="store_true", help="one denoising step vs the Python pipeline"); ap.add_argument("--attrib", action="store_true", help="also the Python step with the C path's int8 numerics per stage"); ap.add_argument("--decode", action="store_true", help="the C video decoder vs tools/decode_loom.py on the fox latents (first 22 frames)"); ap.add_argument("--audio", action="store_true", help="the C audio decoder (BigVGAN in Loom) vs diffusers on the fox audio latents")
    ap.add_argument("--height", type=int, default=480); ap.add_argument("--width", type=int, default=864); ap.add_argument("--frames", type=int, default=22); a = ap.parse_args()
    pf = Path(a.prompt_file) if a.prompt_file else min((Path(p) for p in glob.glob(str(ROOT / "build/prompts/*.pt"))), key=lambda p: p.stat().st_size)
    prompt = torch.load(pf); ids = prompt["ids"].numpy().astype(np.int32); print(f"prompt {prompt['prompt'][:60]!r}: {ids.size} tokens")
    ok = True
    ckpt = R.Checkpoint(device="cuda", dtype=torch.bfloat16); ref = R.H3Ref(ckpt, quant="none")
    with torch.no_grad(): want = ref.text_in(prompt["embeds"].cuda()).float().cpu().numpy()
    del ref, ckpt; torch.cuda.empty_cache()
    t0 = time.time(); pipe = H3Pipe(); print(f"session in {time.time() - t0:.1f} s")
    t0 = time.time(); got = pipe.text_in(ids); print(f"text_in in {time.time() - t0:.2f} s")
    c = float(np.dot(got.ravel(), want.ravel()) / (np.linalg.norm(got) * np.linalg.norm(want) + 1e-30)); err = float(np.linalg.norm(got - want) / (np.linalg.norm(want) + 1e-30))
    print(f"  {'PASS' if c > 0.999 else 'FAIL'} text_in: cosine {c:.5f}, rel err {err:.4f}  (the C path re-encodes the prompt in Loom; the reference uses the cached Loom embeddings)")
    ok &= c > 0.999
    if a.step or a.attrib:
        ok &= denoise_step(pipe, prompt, ids, a)
    if a.decode:
        ok &= decode_video(pipe, a)
    if a.audio:
        ok &= decode_audio(pipe)
    pipe.close()
    return 0 if ok else 1


def glue_linear(name, k_true):
    """The C path's int8 weights (rotated along K, per-row scales) from build/weights_glue, dequantised: [N][K] f32 in rotated space."""
    spans = {l.split()[0]: l.split() for l in (ROOT / "build/weights_glue/manifest.txt").read_text().splitlines()}
    raw = np.memmap(ROOT / "build/weights_glue/weights.bin", dtype=np.uint8, mode="r")
    q = np.frombuffer(raw[int(spans[name + ".q"][1]):int(spans[name + ".q"][1]) + int(spans[name + ".q"][2])], dtype=np.int8).reshape([int(v) for v in spans[name + ".q"][4].split("x")])
    s = np.frombuffer(raw[int(spans[name + ".s"][1]):int(spans[name + ".s"][1]) + int(spans[name + ".s"][2])], dtype=np.float32)
    b = np.frombuffer(raw[int(spans[name + ".b"][1]):int(spans[name + ".b"][1]) + int(spans[name + ".b"][2])], dtype=np.float32)
    return torch.from_numpy(q.astype(np.float32) * s[:, None]).cuda(), torch.from_numpy(b.copy()).cuda()


def w8a8(x, w_rot, b, kpad=None):
    """x [S][K] float -> pad K, rotate (group 256), per-token int8, times the dequantised rotated weights."""
    h = R.hadamard(256).to(x.device)
    xf = x.float()
    if kpad and xf.shape[1] < kpad: xf = torch.cat([xf, xf.new_zeros(xf.shape[0], kpad - xf.shape[1])], 1)
    xr = R.rotate_groups(xf, h)
    s = xr.abs().amax(dim=1, keepdim=True).clamp_min(1e-30) / 127.0
    xq = (xr / s).round().clamp(-127, 127) * s
    return xq @ w_rot.T + b


def decode_video(pipe, a):
    """The C decoder (chunking, heads, blending, ImageNet mapping) vs the Python Loom decoder on the same latents."""
    import math
    from diffusers import AutoencoderKLMiniMaxH3
    from pipeline import MODELS
    from decode_loom import LoomClipDecoder
    dev = "cuda"; fx = torch.load(ROOT / "build/fox_480p_5s_latents.pt"); frames = 22
    p = H3Pipe.params(height=480, width=864, frames=frames, steps=2); sh = pipe.shape(p)
    z = fx["video"][0, :, :sh.latent_t].float().numpy()
    t0 = time.time(); got = pipe.decode_video(p, z); print(f"  C decode {frames} frames in {time.time() - t0:.1f} s")
    vae = AutoencoderKLMiniMaxH3.from_pretrained(str(MODELS / "vae"), torch_dtype=torch.float32).to(dev).eval()
    mean = torch.tensor(vae.config.latents_mean, device=dev).view(1, -1, 1, 1, 1); std = torch.tensor(vae.config.latents_std, device=dev).view(1, -1, 1, 1, 1)
    vae.decoder.forward = LoomClipDecoder(vae, weights=str(ROOT / "build/weights_vae_i8"), bits=8).forward
    with torch.no_grad():
        t0 = time.time(); video = vae._decode(torch.from_numpy(z)[None].to(dev) * std + mean); torch.cuda.synchronize(); print(f"  Python Loom decode in {time.time() - t0:.1f} s")
    imstd = torch.tensor((0.229, 0.224, 0.225), device=dev).view(1, 3, 1, 1, 1); immean = torch.tensor((0.485, 0.456, 0.406), device=dev).view(1, 3, 1, 1, 1)
    want = ((video.float() * imstd + immean).clamp(0, 1)[0] * 255).round().to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
    assert got.shape == want.shape, (got.shape, want.shape)
    mse = float(((got.astype(np.float32) - want.astype(np.float32)) ** 2).mean()); psnr = 10 * math.log10(255.0 ** 2 / max(mse, 1e-9))
    print(f"  {'PASS' if psnr > 35 else 'FAIL'} C video decoder vs Python Loom decoder: PSNR {psnr:.2f} dB over {got.shape[0]} frames (max abs {np.abs(got.astype(int) - want.astype(int)).max()})")
    try:
        import imageio.v2 as iio
        strip = np.concatenate([got[2], got[11], got[21]], axis=1); iio.imwrite(str(ROOT / "build/pipe_decode_strip.jpg"), strip)
    except Exception: pass
    del vae; torch.cuda.empty_cache()
    return psnr > 35


def decode_audio(pipe):
    import math
    from diffusers import AutoencoderKLMiniMaxH3Audio
    from pipeline import MODELS
    dev = "cuda"; fx = torch.load(ROOT / "build/fox_480p_5s_latents.pt"); audio = np.nan_to_num(fx["audio"].float().numpy())    # [2][32][T] model space (one saved frame is NaN)
    t0 = time.time(); got = pipe.decode_audio(audio); print(f"  C audio decode ({audio.shape[-1]} latents -> {got.shape[-1]} samples) in {time.time() - t0:.1f} s")
    avae = AutoencoderKLMiniMaxH3Audio.from_pretrained(str(MODELS / "audio_vae"), torch_dtype=torch.float32).to(dev).eval()
    amean = torch.tensor(avae.config.latents_mean, device=dev).view(1, -1, 1); astd = torch.tensor(avae.config.latents_std, device=dev).view(1, -1, 1)
    with torch.no_grad(): want = avae.decode((torch.from_numpy(audio).to(dev) * astd + amean).float(), return_dict=False)[0][:, 0].cpu().numpy()
    err = float(np.linalg.norm(got - want)); snr = 20 * math.log10(np.linalg.norm(want) / max(err, 1e-12))
    print(f"  {'PASS' if snr > 30 else 'FAIL'} C audio decoder vs diffusers: SNR {snr:.1f} dB, max abs {np.abs(got - want).max():.4f}, rms {np.sqrt((want ** 2).mean()):.4f}")
    del avae; torch.cuda.empty_cache()
    return snr > 30


def denoise_step(pipe, prompt, ids, a):
    """One Euler step from shared noise: the C pipeline vs tools/pipeline.py's loop (the Loom blocks
    through the Python session, the reference's embedders / final layer, diffusers' scheduler)."""
    from diffusers import MiniMaxH3Scheduler
    from h3_loom import H3Blocks, mods_table
    from pipeline import align_frames
    dev = "cuda"; p = H3Pipe.params(height=a.height, width=a.width, frames=a.frames, steps=2, seed=1)
    s = pipe.shape(p); frames = align_frames(a.frames); assert frames == s.frames
    rng = np.random.default_rng(1)
    nv = rng.standard_normal((24, s.latent_t, s.lat_h, s.lat_w)).astype(np.float32); na = rng.standard_normal((2, 32, s.audio_t)).astype(np.float32)
    # --- Python ---
    layout = R.Layout(ids.size, s.latent_t, s.lat_h, s.lat_w, s.audio_t)
    ckpt = R.Checkpoint(device=dev, dtype=torch.bfloat16); ref = R.H3Ref(ckpt, quant="none")
    with torch.no_grad(): text_x = ref.text_in(prompt["embeds"].to(dev))
    cos, sin = R.rope_tables(layout.position_ids, ref.inv_freq, dev); rows = layout.adaln_rows.to(dev); tclass = layout.tclass.to(dev)
    vs, as_ = MiniMaxH3Scheduler(shift=12.0), MiniMaxH3Scheduler(shift=3.0); vs.set_timesteps(2, device=dev); as_.set_timesteps(2, device=dev)
    latents = torch.from_numpy(nv).to(dev)[None]
    video_rows = latents.reshape(1, 24, s.latent_t, s.lat_h // 2, 2, s.lat_w // 2, 2).permute(0, 2, 3, 5, 1, 4, 6).reshape(-1, 96)
    audio_rows = torch.from_numpy(na).to(dev).permute(0, 2, 1).reshape(-1, 32).contiguous()
    blocks = H3Blocks(layout.seq_len, layers=50, weights=str(ROOT / "build/weights_gptq"))
    variants = [("float", False, False, False)]
    if a.attrib: variants += [("int8 embedders", True, False, False), ("C text rows + int8 embedders", True, False, True), ("perturbed 0.1%", False, False, False)]
    wants = {}
    with torch.no_grad():
        temb = ref.t_emb(torch.tensor([vs.timesteps[0].item(), as_.timesteps[0].item()]))
        mods = mods_table(ref, temb, 50)
        text_c = torch.from_numpy(pipe.text_in(ids)).to(dev) if a.attrib else None
        for name, q_embed, q_final, q_text in variants:
            if q_embed:
                wv, bv = glue_linear("h3.video_in", 96); wa, ba = glue_linear("h3.audio_in", 32)
                x = torch.cat([text_c.to(text_x.dtype) if q_text else text_x, w8a8(audio_rows, wa, ba, 256).to(text_x.dtype), w8a8(video_rows, wv, bv, 256).to(text_x.dtype)], dim=0)
            else:
                x = torch.cat([text_c.to(text_x.dtype) if q_text else text_x, ref.audio_in(audio_rows), ref.video_in(video_rows)], dim=0)
            if name.startswith("perturbed"):
                g = torch.Generator(dev).manual_seed(7); x = x.float(); x = x + torch.randn(x.shape, generator=g, device=dev) * x.abs() * (0.004 if "0.4" in name else 0.001)
            y = blocks.forward(x, rows, mods, cos, sin).to(dev)
            bad = ~torch.isfinite(y).all(1)
            if bad.any(): print(f"    [{name}] Python session: {int(bad.sum())} non-finite rows, first {bad.nonzero().flatten()[:6].tolist()} (text {layout.text_len}, audio {layout.audio_rows}, video {layout.video_rows}); x max {x.float().abs().max():.3g}")
            if q_final:
                fm = ref.final_mods(temb)[tclass]
                h = R.rms_norm(y, ref.t("final_layer.norm.weight"), ref.eps) * (1.0 + fm[:, 1].to(y.dtype)) + fm[:, 0].to(y.dtype)
                wf, bf = glue_linear("h3.final.out", 5376); o = w8a8(h, wf, bf)
                a0, a1 = layout.text_len, layout.text_len + layout.audio_rows
                v_rows, a_rows = o[a1:, :96], o[a0:a1, 96:]
            else:
                v_rows, a_rows = ref.final(y, temb, tclass, layout)
            vr = vs.step(v_rows.float(), vs.timesteps[0], video_rows, return_dict=False)[0]; ar = as_.step(a_rows.float(), as_.timesteps[0], audio_rows, return_dict=False)[0]
            vs.set_timesteps(2, device=dev); as_.set_timesteps(2, device=dev)
            wants[name] = (vr.reshape(1, s.latent_t, s.lat_h // 2, s.lat_w // 2, 24, 2, 2).permute(0, 4, 1, 2, 5, 3, 6).reshape(24, s.latent_t, s.lat_h, s.lat_w).cpu().numpy(),
                           ar.reshape(2, s.audio_t, 32).permute(0, 2, 1).cpu().numpy())
    want_v, want_a = wants["float"]
    blocks.close(); del blocks, ref, ckpt; torch.cuda.empty_cache()
    # --- C ---
    t0 = time.time(); got_v, got_a = pipe.denoise(ids, p, noise_video=nv, noise_audio=na, progress=lambda st, n, sec: print(f"    step {st}/{n} {sec:.1f} s") or 0); print(f"  C denoise (1 step, {layout.seq_len} rows) in {time.time() - t0:.1f} s")
    ok = True
    for vname, (want_v, want_a) in wants.items():
        for name, got, want, noise in (("video", got_v, want_v, nv), ("audio", got_a, want_a, na)):
            c = float(np.dot(got.ravel(), want.ravel()) / (np.linalg.norm(got) * np.linalg.norm(want) + 1e-30)); err = float(np.linalg.norm(got - want) / (np.linalg.norm(want) + 1e-30))
            upd_c = float(np.dot((got - noise).ravel(), (want - noise).ravel()) / (np.linalg.norm(got - noise) * np.linalg.norm(want - noise) + 1e-30))
            print(f"  {'PASS' if upd_c > 0.98 else 'FAIL'} one step vs Python [{vname}], {name}: cosine {c:.5f}, rel err {err:.4f}, update cosine {upd_c:.5f}")
            if vname == "float": ok &= upd_c > 0.98         # the int4 blocks' own sensitivity to a 0.1% input change is ~0.994 (--attrib)
    return ok


if __name__ == "__main__":
    sys.exit(main())

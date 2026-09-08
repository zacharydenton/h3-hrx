"""The C pipeline (libh3.so) against the Python reference, stage by stage:
  text_in: the refined text rows for a cached prompt vs reference/h3_ref.py's text_in on the same embeddings;
  --decode / --audio: the video and audio decoders against the reference decoders on the fox latents.
    python3 tests/test_pipe.py [--decode] [--audio] [--prompt-file build/prompts/<hash>.pt]"""
import argparse, glob, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "reference")); sys.path.insert(0, str(ROOT / "tools"))
import h3_ref as R
from h3_loom import H3


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--prompt-file", default=None); ap.add_argument("--step", action="store_true", help="one denoising step vs the Python pipeline"); ap.add_argument("--attrib", action="store_true", help="also the Python step with the C path's int8 numerics per stage"); ap.add_argument("--decode", action="store_true", help="the C video decoder vs tools/decode_loom.py on the fox latents (first 22 frames)"); ap.add_argument("--audio", action="store_true", help="the C audio decoder (BigVGAN in Loom) vs diffusers on the fox audio latents")
    ap.add_argument("--height", type=int, default=480); ap.add_argument("--width", type=int, default=864); ap.add_argument("--frames", type=int, default=22); a = ap.parse_args()
    pf = Path(a.prompt_file) if a.prompt_file else min((Path(p) for p in glob.glob(str(ROOT / "build/prompts/*.pt"))), key=lambda p: p.stat().st_size)
    prompt = torch.load(pf); ids = prompt["ids"].numpy().astype(np.int32); print(f"prompt {prompt['prompt'][:60]!r}: {ids.size} tokens")
    ok = True
    ckpt = R.Checkpoint(device="cuda", dtype=torch.bfloat16); ref = R.H3Ref(ckpt, quant="none")
    with torch.no_grad(): want = ref.text_in(prompt["embeds"].cuda()).float().cpu().numpy()
    del ref, ckpt; torch.cuda.empty_cache()
    t0 = time.time(); pipe = H3(); print(f"session in {time.time() - t0:.1f} s")
    t0 = time.time(); got = pipe.text_in(ids); print(f"text_in in {time.time() - t0:.2f} s")
    c = float(np.dot(got.ravel(), want.ravel()) / (np.linalg.norm(got) * np.linalg.norm(want) + 1e-30)); err = float(np.linalg.norm(got - want) / (np.linalg.norm(want) + 1e-30))
    print(f"  {'PASS' if c > 0.999 else 'FAIL'} text_in: cosine {c:.5f}, rel err {err:.4f}  (the C path re-encodes the prompt in Loom; the reference uses the cached Loom embeddings)")
    ok &= c > 0.999
    if a.decode:
        ok &= decode_video(pipe, a)
    if a.audio:
        ok &= decode_audio(pipe)
    pipe.close()
    return 0 if ok else 1


def decode_video(pipe, a):
    """The C decoder (the f16 blocks, chunking, heads, blending, ImageNet mapping) vs diffusers' f32 decoder on the same latents."""
    import math
    fx = torch.load(ROOT / "build/fox_480p_5s_latents.pt"); frames = 22
    p = H3.params(height=480, width=864, frames=frames, steps=2); sh = pipe.shape(p)
    z = fx["video"][0, :, :sh.latent_t].float().numpy()
    t0 = time.time(); got = pipe.decode_video(p, z); print(f"  C decode {frames} frames in {time.time() - t0:.1f} s")
    from test_decoder_tiles import diffusers_decode
    t0 = time.time(); want = diffusers_decode(z, frames)
    print(f"  diffusers f32 decode in {time.time() - t0:.1f} s")
    assert got.shape == want.shape, (got.shape, want.shape)
    mse = float(((got.astype(np.float32) - want.astype(np.float32)) ** 2).mean()); psnr = 10 * math.log10(255.0 ** 2 / max(mse, 1e-9))
    print(f"  {'PASS' if psnr > 35 else 'FAIL'} C video decoder vs diffusers: PSNR {psnr:.2f} dB over {got.shape[0]} frames (max abs {np.abs(got.astype(int) - want.astype(int)).max()})")
    try:
        import imageio.v2 as iio
        strip = np.concatenate([got[2], got[11], got[21]], axis=1); iio.imwrite(str(ROOT / "build/pipe_decode_strip.jpg"), strip)
    except Exception: pass
    torch.cuda.empty_cache()
    return psnr > 35


def decode_audio(pipe):
    import math
    from diffusers import AutoencoderKLMiniMaxH3Audio
    MODELS = Path.home() / "h3-models"   # the reference tier's diffusers oracles (CONTRIBUTING.md)
    dev = "cuda"; fx = torch.load(ROOT / "build/fox_480p_5s_latents.pt"); audio = np.nan_to_num(fx["audio"].float().numpy())    # [2][32][T] model space (one saved frame is NaN)
    t0 = time.time(); got = pipe.decode_audio(audio); print(f"  C audio decode ({audio.shape[-1]} latents -> {got.shape[-1]} samples) in {time.time() - t0:.1f} s")
    avae = AutoencoderKLMiniMaxH3Audio.from_pretrained(str(MODELS / "audio_vae"), torch_dtype=torch.float32).to(dev).eval()
    amean = torch.tensor(avae.config.latents_mean, device=dev).view(1, -1, 1); astd = torch.tensor(avae.config.latents_std, device=dev).view(1, -1, 1)
    with torch.no_grad(): want = avae.decode((torch.from_numpy(audio).to(dev) * astd + amean).float(), return_dict=False)[0][:, 0].cpu().numpy()
    err = float(np.linalg.norm(got - want)); snr = 20 * math.log10(np.linalg.norm(want) / max(err, 1e-12))
    print(f"  {'PASS' if snr > 30 else 'FAIL'} C audio decoder vs diffusers: SNR {snr:.1f} dB, max abs {np.abs(got - want).max():.4f}, rms {np.sqrt((want ** 2).mean()):.4f}")
    del avae; torch.cuda.empty_cache()
    return snr > 30


if __name__ == "__main__":
    sys.exit(main())

"""Decode saved latents with the ViT decoder blocks in Loom (int4), everything else from
diffusers' AutoencoderKLMiniMaxH3: post_quant_conv, proj_in, the register and cls tokens,
the rotary grid, norm_out + proj_out, unpatchify, spatial tiles and temporal chunk blending.
    python3 tools/decode_loom.py build/fox_480p_5s_latents.pt [--out build/loom_decode.mp4] [--clips N] [--compare]"""
import argparse, math, sys, time
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from pipeline import write_clip, MODELS
from h3vae_loom import H3VaeBlocks


def decoder_tokens(vae, z, *, post_quantized=False):
    """diffusers' decoder forward up to the blocks: tokens [1, N, 2048] and the rotary tables."""
    dec = vae.decoder
    hs = z if post_quantized else vae.post_quant_conv(z)
    b, c, f, h, w = hs.shape
    hs = hs.permute(0, 2, 3, 4, 1).reshape(b, f * h * w, c)
    hs = dec.proj_in(hs)
    num_patches = hs.shape[1]
    hs = torch.cat([hs, dec.register_tokens.expand(b, -1, -1), torch.zeros_like(hs[:, :1])], dim=1)
    grids = [2.0 * (torch.arange(0.5, size, dtype=torch.float32, device=hs.device) / size) - 1.0 for size in (f, h, w)]
    pos = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).flatten(0, 2)[None]
    pos = torch.cat([pos, pos.new_zeros((b, dec.num_register_tokens + 1, 3))], dim=1)
    cos, sin = dec.rope(pos)                      # [1, N, 1, 48]: duplicated halves
    return hs, cos[0, :, 0, :24].contiguous(), sin[0, :, 0, :24].contiguous(), num_patches, (f, h, w)


def decoder_head(vae, hs, num_patches, fhw):
    dec = vae.decoder
    f, h, w = fhw
    hs = dec.proj_out(dec.norm_out(hs))[:, :num_patches]
    hs = hs.view(1, f, h, w, dec.out_channels, dec.patch_size_t, dec.patch_size, dec.patch_size).permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
    return hs.reshape(1, dec.out_channels, f * dec.patch_size_t, h * dec.patch_size, w * dec.patch_size)


class LoomClipDecoder:
    def __init__(self, vae, layers=36, profile=False, weights=None, bits=4):
        self.vae, self.layers, self.profile, self.sessions, self.weights, self.bits = vae, layers, profile, {}, weights, bits
    def close(self):
        for session in self.sessions.values():
            session.close()
        self.sessions.clear()
        if getattr(self.vae.decoder.forward, "__self__", None) is self:
            del self.vae.decoder.forward  # break the decoder -> wrapper -> VAE ownership cycle
    def __call__(self, z):
        return self._forward(z, post_quantized=False)
    def forward(self, z):
        """Replace decoder.forward so the VAE retains spatial tiles and temporal blending."""
        return self._forward(z, post_quantized=True)
    def _forward(self, z, *, post_quantized):
        hs, cos, sin, num_patches, fhw = decoder_tokens(self.vae, z, post_quantized=post_quantized)
        n = hs.shape[1]
        if n not in self.sessions:
            self.sessions[n] = H3VaeBlocks(n, layers=self.layers, weights=self.weights, bits=self.bits)
            if self.profile: self.sessions[n].profile(True)
        y = self.sessions[n].forward(hs[0], cos, sin).to(hs.device, hs.dtype)[None]
        return decoder_head(self.vae, y, num_patches, fhw)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("latents"); ap.add_argument("--out", default=str(ROOT / "build/loom_decode.mp4"))
    ap.add_argument("--clips", type=int, default=None, help="decode only the first N latent chunks (debug)")
    ap.add_argument("--compare", action="store_true", help="also decode the first clip in torch fp32 and report PSNR")
    ap.add_argument("--profile", action="store_true"); ap.add_argument("--weights", default=None, help="build/weights_vae (RTN) or build/weights_vae_gptq"); ap.add_argument("--bits", type=int, default=4)
    a = ap.parse_args(); dev = "cuda"
    from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio
    fx = torch.load(a.latents); latents, audio = fx["video"].to(dev), fx["audio"].to(dev)
    from reference.vae_decoder import DecoderWeights
    reference_weights = DecoderWeights()
    vae = reference_weights.heads(dev)
    mean = torch.tensor(vae.config.latents_mean, device=dev).view(1, -1, 1, 1, 1); std = torch.tensor(vae.config.latents_std, device=dev).view(1, -1, 1, 1, 1)
    z = (latents * std + mean).float()
    loom = LoomClipDecoder(vae, profile=a.profile, weights=a.weights, bits=a.bits)
    if a.compare:
        zc = z[:, :, :vae.tokens_chunk_size + vae.token_overlap]
        def reference_forward(post_quantized):
            hs, cos, sin, patches, grid = decoder_tokens(vae, post_quantized, post_quantized=True)
            _, result = next(reference_weights.blocks(hs, cos, sin, [36]))
            return decoder_head(vae, result.to(dev), patches, grid)
        vae.decoder.forward = reference_forward
        with torch.no_grad():
            t0 = time.time(); ref = vae._decode_clip(zc); torch.cuda.synchronize(); t_ref = time.time() - t0
            vae.decoder.forward = loom.forward
            t0 = time.time(); got = vae._decode_clip(zc); torch.cuda.synchronize(); t_loom = time.time() - t0
        imstd = torch.tensor((0.229, 0.224, 0.225), device=dev).view(1, 3, 1, 1, 1); immean = torch.tensor((0.485, 0.456, 0.406), device=dev).view(1, 3, 1, 1, 1)
        p, q = (ref * imstd + immean).clamp(0, 1), (got * imstd + immean).clamp(0, 1)
        mse = ((p - q) ** 2).mean().item(); print(f"first clip: torch fp32 {t_ref:.1f} s, loom {t_loom:.1f} s, PSNR {10 * math.log10(1 / max(mse, 1e-12)):.2f} dB")
    if a.clips is not None:
        z = z[:, :, : a.clips * vae.tokens_chunk_size]
    vae.decoder.forward = loom.forward              # preserve spatial tiling and temporal blending
    with torch.no_grad():
        try:
            t0 = time.time(); video = vae._decode(z); torch.cuda.synchronize(); print(f"loom decode {time.time() - t0:.1f} s -> {tuple(video.shape)}", flush=True)
        finally:
            loom.close()
        # write_clip converts the ImageNet-normalised decoder output to pixels.
        avae = AutoencoderKLMiniMaxH3Audio.from_pretrained(str(MODELS / "audio_vae"), torch_dtype=torch.float32).to(dev).eval()
        amean = torch.tensor(avae.config.latents_mean, device=dev).view(1, -1, 1); astd = torch.tensor(avae.config.latents_std, device=dev).view(1, -1, 1)
        if audio.dim() == 4: audio = audio[0].permute(1, 0, 2)
        wave = avae.decode((audio * astd + amean).float(), return_dict=False)[0]
    write_clip(video[:, :, : wave.shape[-1] * 24 // 32000 + 1] if False else video, wave, Path(a.out))


if __name__ == "__main__":
    main()

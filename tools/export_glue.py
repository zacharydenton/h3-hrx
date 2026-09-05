"""Everything outside the three block stacks, as one weights dir for the C pipeline
(host/h3pipe.cpp): build/weights_glue/{weights.bin,manifest.txt}.

H3 side (from the ComfyUI int8 checkpoint): the AdaLN curve table and the per-layer /
final AdaLN projections in f32 (the host evaluates them on the CPU per step), rope.inv_freq,
and the float linears rotated along K by the group-256 Hadamard and quantised to int8 per
row so they run on the runtime's W8A8 GEMMs: condition_proj, the two token-refiner blocks
(qkv, out, gate|up interleaved, down) with their norms, the patch embedders with K padded to
256 (video 96 -> 256, audio 32 -> 256; the pad columns are zero before rotation) and the
final layer's video_out | audio_out stacked into one N = 128 GEMM (96 video columns, 32 audio).
Text side: the embedding table as stored (bf16). Video VAE side: post_quant_conv (f32),
proj_in (K 24 -> 256, int8), register tokens, norm_out (LayerNorm weight, bias), proj_out
(int8, N 3072), latents mean/std. Audio side: the BigVGAN decoder with weight norm folded
and Snake parameters exponentiated, f32 throughout, plus dec_in_proj and latents mean/std."""
import json, struct, sys, time
from pathlib import Path
import torch
from safetensors import safe_open
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reference")); sys.path.insert(0, str(ROOT / "tools"))
import h3_ref as R
from export_weights import interleave_gate_up
H3 = R.CKPT
TE = Path.home() / "comfy-models/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
VAE = Path.home() / "h3-models/vae"
AVAE = Path.home() / "h3-models/audio_vae"
KPAD = 256


def main():
    out = ROOT / "build/weights_glue"; out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    blobs = []
    def add(name, t): blobs.append((name, t.detach().contiguous().cpu()))
    h = R.hadamard(R.HADAMARD_GROUP).double()
    def rot_i8(name, w, bias=None, kpad=None):
        """w [N][K] float -> rotate along K (K padded to kpad with zero columns first), int8 per row."""
        w = w.double()
        if kpad is not None and w.shape[1] < kpad:
            w = torch.cat([w, torch.zeros(w.shape[0], kpad - w.shape[1], dtype=w.dtype)], 1)
        assert w.shape[1] % R.HADAMARD_GROUP == 0, (name, w.shape)
        wr = R.rotate_groups(w, h)
        q, s = R.quant_int8_rows(wr.float())
        add(name + ".q", q.round().to(torch.int8)); add(name + ".s", s.view(-1))
        if bias is not None: add(name + ".b", bias.float())
    ck = safe_open(str(H3), "pt", device="cpu")
    g = lambda n: ck.get_tensor(n)
    # --- H3 conditioning tables ---
    add("h3.adaln_t_table", g("adaln_t_table").float())
    add("h3.rope_inv_freq", g("rope.inv_freq").float())
    for i in range(50):
        add(f"h3.blocks.{i}.adaln.w", g(f"blocks.{i}.adaln_proj.linear.weight").float()); add(f"h3.blocks.{i}.adaln.b", g(f"blocks.{i}.adaln_proj.linear.bias").float())
    add("h3.final.adaln.w", g("final_layer.adaln_proj.linear.weight").float()); add("h3.final.adaln.b", g("final_layer.adaln_proj.linear.bias").float())
    add("h3.final.norm", g("final_layer.norm.weight").float())
    rot_i8("h3.final.out", torch.cat([g("final_layer.video_out.weight").float(), g("final_layer.audio_out.weight").float()], 0),
           torch.cat([g("final_layer.video_out.bias").float(), g("final_layer.audio_out.bias").float()], 0))
    # --- embedders and the refiner ---
    rot_i8("h3.cond", g("condition_proj.weight").float(), g("condition_proj.bias").float())
    rot_i8("h3.video_in", g("video_patch_proj.weight").float(), g("video_patch_proj.bias").float(), kpad=KPAD)
    rot_i8("h3.audio_in", g("audio_patch_proj.weight").float(), g("audio_patch_proj.bias").float(), kpad=KPAD)
    for j in range(2):
        p = f"token_refiner.blocks.{j}"
        rot_i8(f"h3.refiner.{j}.qkv", g(f"{p}.attn.qkv_proj.weight").float())
        rot_i8(f"h3.refiner.{j}.out", g(f"{p}.attn.out_proj.weight").float())
        rot_i8(f"h3.refiner.{j}.gu", interleave_gate_up(g(f"{p}.mlp.fc1.weight").float()))
        rot_i8(f"h3.refiner.{j}.down", g(f"{p}.mlp.fc2.weight").float())
        for n in ("norm1", "norm2"): add(f"h3.refiner.{j}.{n}", g(f"{p}.{n}.weight").float())
        add(f"h3.refiner.{j}.qnorm", g(f"{p}.attn.q_norm.weight").float()); add(f"h3.refiner.{j}.knorm", g(f"{p}.attn.k_norm.weight").float())
    add("h3.refiner.final_norm", g("token_refiner.final_norm.weight").float())
    print(f"  h3 glue: {time.time() - t0:.0f} s", flush=True)
    # --- the text encoder's embedding table, as stored ---
    add("te.embed", safe_open(str(TE), "pt", device="cpu").get_tensor("model.embed_tokens.weight"))
    print(f"  embed table: {time.time() - t0:.0f} s", flush=True)
    # --- video VAE glue ---
    cfg = json.loads((VAE / "config.json").read_text())
    vs = {}
    for shard in sorted(VAE.glob("*.safetensors")):
        f = safe_open(str(shard), "pt", device="cpu")
        for k in f.keys():
            if not k.startswith("encoder.") and not k.startswith("decoder.transformer_blocks."): vs[k] = f.get_tensor(k)
    add("vae.post_quant_conv.w", vs["post_quant_conv.weight"].float().reshape(24, 24)); add("vae.post_quant_conv.b", vs["post_quant_conv.bias"].float())
    rot_i8("vae.proj_in", vs["decoder.proj_in.weight"].float(), vs["decoder.proj_in.bias"].float(), kpad=KPAD)
    add("vae.register_tokens", vs["decoder.register_tokens"].float().reshape(-1, 2048))
    add("vae.norm_out.w", vs["decoder.norm_out.weight"].float()); add("vae.norm_out.b", vs["decoder.norm_out.bias"].float())
    rot_i8("vae.proj_out", vs["decoder.proj_out.weight"].float(), vs["decoder.proj_out.bias"].float())
    add("vae.latents_mean", torch.tensor(cfg["latents_mean"], dtype=torch.float32)); add("vae.latents_std", torch.tensor(cfg["latents_std"], dtype=torch.float32))
    print(f"  video vae glue: {time.time() - t0:.0f} s  (keys: {sorted(k for k in vs if 'decoder' in k)[:6]}...)", flush=True)
    # --- audio decoder (BigVGAN), weight norm folded ---
    acfg = json.loads((AVAE / "config.json").read_text())
    af = safe_open(str(next(AVAE.glob("*.safetensors"))), "pt", device="cpu")
    ak = set(af.keys())
    def wn(prefix):
        v = af.get_tensor(prefix + ".weight_v").double(); gg = af.get_tensor(prefix + ".weight_g").double()
        norm = v.flatten(1).norm(dim=1).view(-1, *([1] * (v.ndim - 1)))
        return (gg * v / norm).float()
    def conv(name, prefix, bias=True):
        add(name + ".w", wn(prefix))
        if bias: add(name + ".b", af.get_tensor(prefix + ".bias").float())
    add("audio.latents_mean", torch.tensor(acfg["latents_mean"], dtype=torch.float32)); add("audio.latents_std", torch.tensor(acfg["latents_std"], dtype=torch.float32))
    add("audio.dec_in_proj.w", af.get_tensor("dec_in_proj.weight").float().reshape(2048, 32)); add("audio.dec_in_proj.b", af.get_tensor("dec_in_proj.bias").float())
    conv("audio.conv_pre", "decoder.conv_pre")
    rates, kernels = acfg["decoder_rates"], acfg["decoder_kernel_sizes"]
    rk, rd = acfg["resblock_kernel_sizes"], acfg["resblock_dilation_sizes"]
    for i, (rate, k) in enumerate(zip(rates, kernels)):
        conv(f"audio.ups.{i}", f"decoder.ups.{i}.0")
        for j, (kk, dil) in enumerate(zip(rk, rd)):
            r = i * len(rk) + j; p = f"decoder.resblocks.{r}"
            for d in range(len(dil)):
                conv(f"audio.res.{r}.c1.{d}", f"{p}.convs1.{d}"); conv(f"audio.res.{r}.c2.{d}", f"{p}.convs2.{d}")
            for a in range(2 * len(dil)):
                add(f"audio.res.{r}.act.{a}.alpha", af.get_tensor(f"{p}.activations.{a}.act.alpha").float().exp())
                add(f"audio.res.{r}.act.{a}.beta", af.get_tensor(f"{p}.activations.{a}.act.beta").float().exp())
    add("audio.post.alpha", af.get_tensor("decoder.activation_post.act.alpha").float().exp()); add("audio.post.beta", af.get_tensor("decoder.activation_post.act.beta").float().exp())
    add("audio.fir", af.get_tensor("decoder.activation_post.upsample.filter").float().reshape(-1))          # the same 12-tap Kaiser-sinc in every activation
    for kk in ak:
        if kk.endswith(".filter"):
            assert torch.equal(af.get_tensor(kk).reshape(-1), af.get_tensor("decoder.activation_post.upsample.filter").reshape(-1)), kk
    conv("audio.conv_post", "decoder.conv_post", bias=False)
    meta = dict(audio_rates=rates, audio_kernels=kernels, res_kernels=rk, res_dilations=rd, audio_channels=acfg["decoder_dim"], hop=acfg["hop_length"] if "hop_length" in acfg else 800,
                kpad=KPAD, vae_tokens_chunk=5, vae_token_overlap=2, vae_clip_length=cfg["clip_length"], vae_token_drop=cfg["token_drop"])
    manifest, offset = [], 0
    with open(out / "weights.bin", "wb") as fh:
        for name, t in blobs:
            b = t.numpy().tobytes() if t.dtype != torch.bfloat16 else t.view(torch.int16).numpy().tobytes()
            manifest.append(f"{name} {offset} {len(b)} {t.dtype} {'x'.join(map(str, t.shape))}"); fh.write(b); offset += len(b)
    (out / "manifest.txt").write_text("\n".join(manifest) + "\n")
    (out / "config.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {out}/weights.bin: {offset / 1e9:.2f} GB, {len(manifest)} tensors, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

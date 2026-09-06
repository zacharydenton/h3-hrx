"""The audio VAE's encoder (DAC conv stack + posterior head) from ComfyUI's minimax_h3_audio_vae_fp32.safetensors
into build/weights_aenc/{weights.bin,manifest.txt} (the Blob format libh3pipe reads; f32, no torch needed).
    python3 tools/export_audio_encoder.py [--src /mnt/usb/models/comfy/vae/minimax_h3_audio_vae_fp32.safetensors]"""
import argparse, json, struct, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser(); ap.add_argument("--src", default="/mnt/usb/models/comfy/vae/minimax_h3_audio_vae_fp32.safetensors"); ap.add_argument("--out", default=str(ROOT / "build/weights_aenc"))
a = ap.parse_args()
DT = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16}
with open(a.src, "rb") as fh:
    n = struct.unpack("<Q", fh.read(8))[0]; hdr = json.loads(fh.read(n)); base = 8 + n
    hdr.pop("__metadata__", None)
    def get(k):
        v = hdr[k]; s, e = v["data_offsets"]; fh.seek(base + s)
        arr = np.frombuffer(fh.read(e - s), dtype=DT[v["dtype"]]).reshape(v["shape"])
        assert v["dtype"] == "F32", (k, v["dtype"]); return arr.astype(np.float32)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    manifest, offset = [], 0; t0 = time.time()
    with open(out / "weights.bin", "wb") as wb:
        def add(name, arr):
            global offset
            arr = np.ascontiguousarray(arr, dtype=np.float32); b = arr.tobytes()
            manifest.append(f"{name} {offset} {len(b)} torch.float32 {'x'.join(map(str, arr.shape))}"); wb.write(b); offset += len(b)
        def conv(name, prefix):
            w = get(prefix + ".weight"); add(name + ".w", w.reshape(w.shape[0], -1)); add(name + ".b", get(prefix + ".bias"))
            return w.shape
        def snake(name, prefix): add(name, get(prefix + ".alpha").reshape(-1))
        meta = {"rates": [2, 4, 4, 5, 5], "dims": [], "blocks": []}
        conv("aenc.conv_in", "encoder.block.0")
        for i in range(1, 6):
            p = f"encoder.block.{i}"; dims = []
            for r in range(3):
                q = f"{p}.block.{r}.block"
                snake(f"aenc.b{i}.r{r}.act0", f"{q}.0"); s1 = conv(f"aenc.b{i}.r{r}.c1", f"{q}.1"); snake(f"aenc.b{i}.r{r}.act1", f"{q}.2"); conv(f"aenc.b{i}.r{r}.c2", f"{q}.3")
                dims.append([int(s1[0]), int(s1[2])])
            snake(f"aenc.b{i}.act", f"{p}.block.3"); sd = conv(f"aenc.b{i}.down", f"{p}.block.4")
            meta["blocks"].append({"res": dims, "down": [int(sd[0]), int(sd[1]), int(sd[2])]})
        snake("aenc.act_out", "encoder.block.6"); conv("aenc.conv_out", "encoder.block.7")
        for nm in ("norm1", "norm2", "norm3"): add(f"aenc.pre.{nm}.w", get(f"pre_block.{nm}.weight")); add(f"aenc.pre.{nm}.b", get(f"pre_block.{nm}.bias"))
        add("aenc.pre.qkv.w", get("pre_block.attn.qkv.weight"))
        add("aenc.pre.qkv.b", np.concatenate([get("pre_block.attn.q_bias"), np.zeros(2048, np.float32), get("pre_block.attn.v_bias")]))
        add("aenc.pre.attn_proj.w", get("pre_block.attn.proj.weight")); add("aenc.pre.attn_proj.b", get("pre_block.attn.proj.bias"))
        add("aenc.pre.proj.w", get("pre_block.proj.weight")); add("aenc.pre.proj.b", get("pre_block.proj.bias"))
        add("aenc.pre.mlp.norm.w", get("pre_block.mlp.norm.weight")); add("aenc.pre.mlp.norm.b", get("pre_block.mlp.norm.bias"))
        for nm in ("w0", "w1", "w2"): add(f"aenc.pre.mlp.{nm}.w", get(f"pre_block.mlp.{nm}.weight")); add(f"aenc.pre.mlp.{nm}.b", get(f"pre_block.mlp.{nm}.bias"))
        w = get("mean_proj.weight"); add("aenc.mean_proj.w", w.reshape(32, 32)); add("aenc.mean_proj.b", get("mean_proj.bias"))
        add("aenc.latents_mean", get("latents_mean")); add("aenc.latents_std", get("latents_std"))
    (out / "manifest.txt").write_text("\n".join(manifest) + "\n"); (out / "config.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {out}/weights.bin: {offset / 1e6:.1f} MB, {len(manifest)} tensors, {time.time() - t0:.1f} s; blocks {meta['blocks']}")

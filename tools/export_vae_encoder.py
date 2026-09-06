"""The video VAE's causal 3-D conv encoder (encoder.*, quant_conv in minimax_h3_video_vae_fp16.safetensors) into
build/weights_venc/{weights.bin,manifest.txt} for libh3pipe: every conv as f16 [Cout_pad][taps*Cin_pad] in the implicit-GEMM
order k = ((dt*3 + dy)*3 + dx)*Cin_pad + c (".w3", 27 taps) and as the last temporal tap only (".w2", 9 taps, single
frames); Cin_pad a multiple of 8, Cout_pad of 64, K_pad of 32; biases padded f32; GroupNorm gamma/beta f32; the 1x1
shortcuts and quant_conv as [Nout_pad][Kin_pad] matmul weights. numpy only.
    python3 tools/export_vae_encoder.py"""
import argparse, json, struct, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser(); ap.add_argument("--src", default="/mnt/usb/models/comfy/vae/minimax_h3_video_vae_fp16.safetensors"); ap.add_argument("--out", default=str(ROOT / "build/weights_venc"))
a = ap.parse_args()
def up(n, m): return (n + m - 1) // m * m
with open(a.src, "rb") as fh:
    n = struct.unpack("<Q", fh.read(8))[0]; hdr = json.loads(fh.read(n)); base = 8 + n; hdr.pop("__metadata__", None)
    def get(k):
        v = hdr[k]; s, e = v["data_offsets"]; fh.seek(base + s)
        dt = {"F16": np.float16, "F32": np.float32, "BF16": np.uint16}[v["dtype"]]; raw = np.frombuffer(fh.read(e - s), dtype=dt).reshape(v["shape"])
        if v["dtype"] == "BF16": raw = (raw.astype(np.uint32) << 16).view(np.float32)
        return raw.astype(np.float32)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); manifest, offset = [], 0; t0 = time.time(); meta = {"convs": {}}
    with open(out / "weights.bin", "wb") as wb:
        def add(name, arr, dtype):
            global offset
            arr = np.ascontiguousarray(arr, dtype=dtype); b = arr.tobytes()
            manifest.append(f"{name} {offset} {len(b)} {'torch.float16' if dtype == np.float16 else 'torch.float32'} {'x'.join(map(str, arr.shape))}"); wb.write(b); offset += len(b)
        def conv(name, prefix):
            w = get(prefix + ".weight"); b = get(prefix + ".bias")            # [Cout][Cin][kt][kh][kw]
            cout, cin, kt, kh, kw = w.shape; cin_pad, cout_pad = up(cin, 8), up(cout, 64)
            if kt == 1 and kh == 1:                                          # 1x1x1: a matmul over channels
                wm = np.zeros((cout_pad, up(cin_pad, 32)), np.float32); wm[:cout, :cin] = w.reshape(cout, cin)
                add(name + ".wm", wm, np.float16); add(name + ".b", np.pad(b, (0, cout_pad - cout)), np.float32)
                meta["convs"][name] = dict(kind="1x1", cin=cin, cout=cout, cin_pad=cin_pad, cout_pad=cout_pad, k=up(cin_pad, 32)); return
            assert (kt, kh, kw) == (3, 3, 3), (prefix, w.shape)
            w3 = np.zeros((cout_pad, 27, cin_pad), np.float32); w3[:cout, :, :cin] = w.transpose(0, 2, 3, 4, 1).reshape(cout, 27, cin)
            k3 = up(27 * cin_pad, 32); m3 = np.zeros((cout_pad, k3), np.float32); m3[:, :27 * cin_pad] = w3.reshape(cout_pad, -1); add(name + ".w3", m3, np.float16)
            w2 = np.zeros((cout_pad, 9, cin_pad), np.float32); w2[:cout, :, :cin] = w[:, :, 2].transpose(0, 2, 3, 1).reshape(cout, 9, cin)
            k2 = up(9 * cin_pad, 32); m2 = np.zeros((cout_pad, k2), np.float32); m2[:, :9 * cin_pad] = w2.reshape(cout_pad, -1); add(name + ".w2", m2, np.float16)
            add(name + ".b", np.pad(b, (0, cout_pad - cout)), np.float32)
            meta["convs"][name] = dict(kind="3x3x3", cin=cin, cout=cout, cin_pad=cin_pad, cout_pad=cout_pad, k3=k3, k2=k2)
        def norm(name, prefix): add(name + ".g", get(prefix + ".weight"), np.float32); add(name + ".b", get(prefix + ".bias"), np.float32)
        conv("venc.conv_in", "encoder.conv_in")
        levels = []
        for i in range(6):
            blocks = []
            for r in range(2):
                p = f"encoder.down.{i}.block.{r}"; norm(f"venc.l{i}.r{r}.norm1", p + ".norm1"); conv(f"venc.l{i}.r{r}.conv1", p + ".conv1"); norm(f"venc.l{i}.r{r}.norm2", p + ".norm2"); conv(f"venc.l{i}.r{r}.conv2", p + ".conv2")
                short = (p + ".nin_shortcut.weight") in hdr
                if short: conv(f"venc.l{i}.r{r}.nin", p + ".nin_shortcut")
                blocks.append(dict(shortcut=short))
            down = (f"encoder.down.{i}.downsample.conv.weight") in hdr
            if down: conv(f"venc.l{i}.down", f"encoder.down.{i}.downsample.conv")
            levels.append(dict(blocks=blocks, down=down))
        norm("venc.norm_out", "encoder.norm_out"); conv("venc.conv_out", "encoder.conv_out"); conv("venc.quant", "quant_conv")
        add("venc.latents_mean", get("latents_mean"), np.float32); add("venc.latents_std", get("latents_std"), np.float32)
        meta.update(levels=levels, space_down=[2, 2, 2, 2, 1, 1], time_down=[1, 2, 2, 1, 1, 1], groups=32, eps=1e-6)
    (out / "manifest.txt").write_text("\n".join(manifest) + "\n"); (out / "config.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {out}/weights.bin: {offset / 1e9:.2f} GB, {len(manifest)} tensors, {time.time() - t0:.1f} s; levels {[ (l['down'], [b['shortcut'] for b in l['blocks']]) for l in levels]}")

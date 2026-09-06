"""Qwen3-VL-32B's vision tower (visual.* in the H3 text-encoder checkpoint, bf16) into build/weights_vision/{weights.bin,
manifest.txt} for libh3pipe: f16 weights [n][k] with the head dimension padded 72 -> 128 (qkv rows, proj columns) and the MLP
hidden 4304 -> 4352, f32 biases and norms, the 48x48 position table f32. numpy only.
    python3 tools/export_vision.py"""
import argparse, json, struct, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser(); ap.add_argument("--src", default="/mnt/usb/models/comfy/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"); ap.add_argument("--out", default=str(ROOT / "build/weights_vision"))
a = ap.parse_args()
HID, HEADS, HD, HDP, MLP, MLPP, OUT = 1152, 16, 72, 128, 4304, 4352, 5120
with open(a.src, "rb") as fh:
    n = struct.unpack("<Q", fh.read(8))[0]; hdr = json.loads(fh.read(n)); base = 8 + n; hdr.pop("__metadata__", None)
    def get(k):
        v = hdr[k]; s, e = v["data_offsets"]; fh.seek(base + s); raw = np.frombuffer(fh.read(e - s), dtype=np.uint16).reshape(v["shape"])
        assert v["dtype"] == "BF16", (k, v["dtype"]); return (raw.astype(np.uint32) << 16).view(np.float32)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); manifest, offset = [], 0; t0 = time.time()
    with open(out / "weights.bin", "wb") as wb:
        def add(name, arr, dtype):
            global offset
            arr = np.ascontiguousarray(arr, dtype=dtype); b = arr.tobytes()
            manifest.append(f"{name} {offset} {len(b)} {'torch.float16' if dtype == np.float16 else 'torch.float32'} {'x'.join(map(str, arr.shape))}"); wb.write(b); offset += len(b)
        def lin(name, prefix, npad=None, kpad=None):
            w = get(prefix + ".weight"); b = get(prefix + ".bias")
            if npad and npad > w.shape[0]: w = np.concatenate([w, np.zeros((npad - w.shape[0], w.shape[1]), np.float32)]); b = np.concatenate([b, np.zeros(npad - b.shape[0], np.float32)])
            if kpad and kpad > w.shape[1]: w = np.concatenate([w, np.zeros((w.shape[0], kpad - w.shape[1]), np.float32)], 1)
            add(name + ".w", w, np.float16); add(name + ".b", b, np.float32)
        def norm(name, prefix): add(name + ".w", get(prefix + ".weight"), np.float32); add(name + ".b", get(prefix + ".bias"), np.float32)
        w = get("visual.patch_embed.proj.weight").reshape(HID, -1); add("vis.patch.w", w, np.float16); add("vis.patch.b", get("visual.patch_embed.proj.bias"), np.float32)
        add("vis.pos", get("visual.pos_embed.weight"), np.float32)
        for i in range(27):
            p = f"visual.blocks.{i}"; norm(f"vis.b{i}.norm1", p + ".norm1"); norm(f"vis.b{i}.norm2", p + ".norm2")
            w = get(p + ".attn.qkv.weight").reshape(3, HEADS, HD, HID); b = get(p + ".attn.qkv.bias").reshape(3, HEADS, HD)
            wp = np.zeros((3, HEADS, HDP, HID), np.float32); wp[:, :, :HD] = w; bp = np.zeros((3, HEADS, HDP), np.float32); bp[:, :, :HD] = b
            add(f"vis.b{i}.qkv.w", wp.reshape(3 * HEADS * HDP, HID), np.float16); add(f"vis.b{i}.qkv.b", bp.reshape(-1), np.float32)
            w = get(p + ".attn.proj.weight").reshape(HID, HEADS, HD); wp = np.zeros((HID, HEADS, HDP), np.float32); wp[:, :, :HD] = w
            add(f"vis.b{i}.proj.w", wp.reshape(HID, HEADS * HDP), np.float16); add(f"vis.b{i}.proj.b", get(p + ".attn.proj.bias"), np.float32)
            lin(f"vis.b{i}.fc1", p + ".mlp.linear_fc1", npad=MLPP); lin(f"vis.b{i}.fc2", p + ".mlp.linear_fc2", kpad=MLPP)
        norm("vis.merger.norm", "visual.merger"[:0] + "visual.merger.norm"); lin("vis.merger.fc1", "visual.merger.linear_fc1"); lin("vis.merger.fc2", "visual.merger.linear_fc2")
        for j in range(3):
            p = f"visual.deepstack_merger_list.{j}"; norm(f"vis.ds{j}.norm", p + ".norm"); lin(f"vis.ds{j}.fc1", p + ".linear_fc1"); lin(f"vis.ds{j}.fc2", p + ".linear_fc2")
    (out / "manifest.txt").write_text("\n".join(manifest) + "\n")
    (out / "config.json").write_text(json.dumps(dict(hidden=HID, heads=HEADS, head_dim=HD, head_pad=HDP, mlp=MLP, mlp_pad=MLPP, out=OUT, blocks=27, deepstack=[8, 16, 24], patch=16, merge=2), indent=1))
    print(f"wrote {out}/weights.bin: {offset / 1e9:.2f} GB, {len(manifest)} tensors, {time.time() - t0:.1f} s")

"""Export the video VAE's ViT decoder blocks as the runtime's int4 operands, from the
diffusers-format fp32 weights in ~/h3-models/vae: per block
  qkv  : to_q | to_k | to_v          [6144][2048] + bias
  out  : to_out.0                    [2048][2048] + bias, layer scale scale1 -> gate table [1][2048]
  gu   : ff.net.0.proj               [16384][2048] + bias, rows interleaved in 16-row groups (linear | gate)
  down : ff.net.2                    [2048][8192] + bias, layer scale scale2 -> gate table
plus norm1 / norm2 weights (f32). Weights are rotated by the group-256 Hadamard along K and
quantised int4 per output row (round to nearest); activations are rotated by the prepare
kernels. Output: build/weights_vae/weights.bin + manifest.txt.
"""
import json
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reference")); sys.path.insert(0, str(ROOT / "tools"))
import h3_ref as R
from export_weights import pack_i4, interleave_gate_up

VAE = Path.home() / "h3-models/vae"
LAYERS = 36


def main() -> None:
    out = ROOT / "build/weights_vae"; out.mkdir(parents=True, exist_ok=True)
    index = json.loads((VAE / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    files = {}
    def get(name):
        f = index[name]
        if f not in files:
            files[f] = safe_open(str(VAE / f), "pt", device="cpu")
        return files[f].get_tensor(name)
    h = R.hadamard(R.HADAMARD_GROUP).cuda()
    def quant(w):
        wr = R.rotate_groups(w.cuda().float(), h)
        q, s = R.quant_int4_rows(wr)
        return pack_i4(q.to(torch.int16).cpu()), s.view(-1).cpu()
    blobs = []
    def add(name, t): blobs.append((name, t.contiguous().cpu()))
    t0 = time.time()
    for i in range(LAYERS):
        p = f"decoder.transformer_blocks.{i}"
        qkv = torch.cat([get(f"{p}.attn.to_q.weight"), get(f"{p}.attn.to_k.weight"), get(f"{p}.attn.to_v.weight")], dim=0)
        qkv_b = torch.cat([get(f"{p}.attn.to_q.bias"), get(f"{p}.attn.to_k.bias"), get(f"{p}.attn.to_v.bias")], dim=0)
        q, s = quant(qkv); add(f"blocks.{i}.qkv.q", q); add(f"blocks.{i}.qkv.s", s); add(f"blocks.{i}.qkv.b", qkv_b.float())
        q, s = quant(get(f"{p}.attn.to_out.0.weight")); add(f"blocks.{i}.out.q", q); add(f"blocks.{i}.out.s", s); add(f"blocks.{i}.out.b", get(f"{p}.attn.to_out.0.bias").float())
        gu = get(f"{p}.ff.net.0.proj.weight"); gu_b = get(f"{p}.ff.net.0.proj.bias")
        q, s = quant(gu); add(f"blocks.{i}.gu.q", interleave_gate_up(q)); add(f"blocks.{i}.gu.s", interleave_gate_up(s.view(-1, 1)).view(-1)); add(f"blocks.{i}.gu.b", interleave_gate_up(gu_b.float().view(-1, 1)).view(-1))
        q, s = quant(get(f"{p}.ff.net.2.weight")); add(f"blocks.{i}.down.q", q); add(f"blocks.{i}.down.s", s); add(f"blocks.{i}.down.b", get(f"{p}.ff.net.2.bias").float())
        add(f"blocks.{i}.norm1", get(f"{p}.norm1.weight").float()); add(f"blocks.{i}.norm2", get(f"{p}.norm2.weight").float())
        add(f"blocks.{i}.scale1", get(f"{p}.scale1").float()); add(f"blocks.{i}.scale2", get(f"{p}.scale2").float())
        print(f"  block {i}: {time.time() - t0:.0f} s", flush=True)
    manifest, offset = [], 0
    with open(out / "weights.bin", "wb") as f:
        for name, t in blobs:
            b = t.numpy().tobytes()
            manifest.append(f"{name} {offset} {len(b)} {t.dtype} {'x'.join(map(str, t.shape))}")
            f.write(b); offset += len(b)
    (out / "manifest.txt").write_text("\n".join(manifest) + "\n")
    (out / "config.json").write_text(json.dumps(dict(layers=LAYERS, hidden=2048, heads=32, head_dim=64, ffn=8192, rope_dim=48, group=256), indent=1))
    print(f"wrote {out}/weights.bin: {offset / 1e9:.2f} GB, {len(manifest)} tensors, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

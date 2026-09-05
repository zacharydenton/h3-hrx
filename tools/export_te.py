"""Export the Qwen3-VL-32B text encoder's 50 layers as the runtime's int8 operands, straight from
ComfyUI's int8 ConvRot file (rows already rotated and quantised per row: nothing is requantised).
Per layer: qkv = q_proj | k_proj | v_proj [10240][5120], o_proj [5120][8192], gate|up interleaved
in 16-row groups [51200][5120], down [5120][25600], plus input/post-attention norms and the
q/k norms. Output: build/weights_te/weights.bin + manifest.txt."""
import json, sys, time
from pathlib import Path
import torch
from safetensors import safe_open
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from export_weights import interleave_gate_up
TE = Path.home() / "comfy-models/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
LAYERS = 50


def main():
    out = ROOT / "build/weights_te"; out.mkdir(parents=True, exist_ok=True)
    f = safe_open(str(TE), "pt", device="cpu")
    blobs = []
    def add(name, t): blobs.append((name, t.contiguous().cpu()))
    def lin(name): return f.get_tensor(f"model.layers.{i}.{name}.weight"), f.get_tensor(f"model.layers.{i}.{name}.weight_scale").float().view(-1)
    t0 = time.time()
    for i in range(LAYERS):
        q, qs = lin("self_attn.q_proj"); k, ks = lin("self_attn.k_proj"); v, vs = lin("self_attn.v_proj")
        add(f"blocks.{i}.qkv.q", torch.cat([q, k, v], 0)); add(f"blocks.{i}.qkv.s", torch.cat([qs, ks, vs], 0))
        o, os_ = lin("self_attn.o_proj"); add(f"blocks.{i}.out.q", o); add(f"blocks.{i}.out.s", os_)
        g, gs = lin("mlp.gate_proj"); u, us = lin("mlp.up_proj")
        add(f"blocks.{i}.gu.q", interleave_gate_up(torch.cat([g, u], 0))); add(f"blocks.{i}.gu.s", interleave_gate_up(torch.cat([gs, us], 0).view(-1, 1)).view(-1))
        d, ds = lin("mlp.down_proj"); add(f"blocks.{i}.down.q", d); add(f"blocks.{i}.down.s", ds)
        add(f"blocks.{i}.norm1", f.get_tensor(f"model.layers.{i}.input_layernorm.weight").float())
        add(f"blocks.{i}.norm2", f.get_tensor(f"model.layers.{i}.post_attention_layernorm.weight").float())
        add(f"blocks.{i}.qnorm", f.get_tensor(f"model.layers.{i}.self_attn.q_norm.weight").float())
        add(f"blocks.{i}.knorm", f.get_tensor(f"model.layers.{i}.self_attn.k_norm.weight").float())
        if i % 10 == 9: print(f"  layer {i}: {time.time() - t0:.0f} s", flush=True)
    manifest, offset = [], 0
    with open(out / "weights.bin", "wb") as fh:
        for name, t in blobs:
            b = t.numpy().tobytes(); manifest.append(f"{name} {offset} {len(b)} {t.dtype} {'x'.join(map(str, t.shape))}"); fh.write(b); offset += len(b)
    (out / "manifest.txt").write_text("\n".join(manifest) + "\n")
    (out / "config.json").write_text(json.dumps(dict(layers=LAYERS, hidden=5120, heads=64, kv_heads=8, head_dim=128, ffn=25600, rope_dim=128, group=256, bits=8), indent=1))
    print(f"wrote {out}/weights.bin: {offset / 1e9:.2f} GB, {len(manifest)} tensors, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

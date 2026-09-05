"""The MiniMax H3 text conditioning: the prompt through the Qwen3-VL-32B encoder cut to 50
layers (ComfyUI's int8 ConvRot file, dequantised to bf16), taking the unnormalised hidden
state after layer 50. The t2va presentation is the raw prompt tokens: no chat template, no
special tokens; every row carries modality tag 1 (text). Saves build/prompts/<hash>.pt.

    python3 tools/encode_prompt.py "a red fox ..." [--out build/prompts]
"""
import argparse
import hashlib
import json
import struct
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
TE = Path.home() / "comfy-models/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
CFG = Path.home() / "h3-models/text_encoder"
TOK = Path.home() / "h3-models/tokenizer"
LAYERS = 50


def prompt_path(prompt: str, out: Path) -> Path:
    return out / (hashlib.sha1(prompt.encode()).hexdigest()[:12] + ".pt")


def load_encoder(device="cuda"):
    from safetensors import safe_open
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLConfig
    cfg = Qwen3VLConfig.from_pretrained(str(CFG))
    cfg.text_config.num_hidden_layers = LAYERS
    with torch.device("meta"):
        model = Qwen3VLForConditionalGeneration(cfg)
    model.model.language_model.norm = torch.nn.Identity()       # the conditioning is layer 50's raw output
    model.model.visual = torch.nn.Identity()                     # text-only presentation: no vision tower
    model.lm_head = torch.nn.Identity()
    rot = model.model.language_model.rotary_emb
    with torch.device(device):
        model.model.language_model.rotary_emb = type(rot)(config=cfg.text_config)   # its inv_freq buffer, off the meta device
    with open(TE, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; header = json.loads(fh.read(n))
    f = safe_open(str(TE), "pt", device="cpu")
    sd = {}
    t0 = time.time()
    for name in header:
        if name == "__metadata__" or name.endswith(".comfy_quant") or name.endswith(".weight_scale") or name.startswith("visual."):
            continue
        t = f.get_tensor(name)
        if t.dtype == torch.int8:
            s = f.get_tensor(name.replace(".weight", ".weight_scale")).float().view(-1, 1)
            t = (t.to(device).float() * s.to(device)).to(torch.bfloat16)      # per-row int8 -> bf16 (rotated along K:
        else:                                                                  # ConvRot rotates the *input*; see below)
            t = t.to(device, torch.bfloat16)
        hf = name
        if hf.startswith("model.layers."): hf = "model.language_model.layers." + hf[len("model.layers."):]
        elif hf == "model.embed_tokens.weight": hf = "model.language_model.embed_tokens.weight"
        elif hf.startswith("visual."): hf = "model." + hf
        sd[hf] = t
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    print(f"encoder loaded in {time.time() - t0:.0f} s; missing {len(missing)}, unexpected {len(unexpected)}", file=sys.stderr)
    if unexpected: print("  unexpected:", unexpected[:5], file=sys.stderr)
    meta = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.device.type == "meta"]
    if meta:
        raise SystemExit(f"tensors left on the meta device: {meta[:8]}")
    return model.eval()


class RotatedLinear(torch.nn.Module):
    """ConvRot: the file's rows are W H (group-256 regular Hadamard along K); the input must be
    rotated by the same H before the matmul."""
    def __init__(self, weight, h):
        super().__init__(); self.weight = torch.nn.Parameter(weight, requires_grad=False); self.h = h
    def forward(self, x):
        g = self.h.shape[0]
        xr = (x.float().reshape(*x.shape[:-1], x.shape[-1] // g, g) @ self.h).reshape(x.shape).to(x.dtype)
        return torch.nn.functional.linear(xr, self.weight)


def rotate_inputs(model, device):
    sys.path.insert(0, str(ROOT / "reference"))
    import h3_ref as R
    h = R.hadamard(R.HADAMARD_GROUP).to(device)
    n = 0
    for layer in model.model.language_model.layers:
        for parent, names in ((layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")), (layer.mlp, ("gate_proj", "up_proj", "down_proj"))):
            for nm in names:
                lin = getattr(parent, nm)
                setattr(parent, nm, RotatedLinear(lin.weight.data, h)); n += 1
    print(f"  {n} linears take rotated inputs", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--out", default=str(ROOT / "build/prompts"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    path = prompt_path(a.prompt, out)
    if path.exists():
        print(f"cached: {path}"); return
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(TOK))
    ids = tok(a.prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
    device = "cuda"
    model = load_encoder(device)
    rotate_inputs(model, device)
    with torch.no_grad():
        outputs = model.model.language_model(input_ids=ids.to(device), output_hidden_states=True)
        embeds = outputs.hidden_states[LAYERS][0].float().cpu()            # [L, 5120], after layer 50, no norm
    torch.save(dict(prompt=a.prompt, ids=ids[0], embeds=embeds, tags=torch.ones(embeds.shape[0], dtype=torch.long)), path)
    print(f"{path}: {embeds.shape[0]} tokens, rms {embeds.pow(2).mean().sqrt():.3f}")


if __name__ == "__main__":
    main()

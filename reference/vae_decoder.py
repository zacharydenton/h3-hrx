"""Small VAE heads and a block-at-a-time floating-point decoder reference."""
import json
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors import safe_open
from diffusers import AutoencoderKLMiniMaxH3
import diffusers.models.autoencoders.autoencoder_kl_minimax_h3 as model_impl


class DecoderWeights:
    def __init__(self, directory=None):
        self.directory = Path(directory or Path.home() / "h3-models/vae")
        self.config = json.loads((self.directory / "config.json").read_text())
        self.index = json.loads((self.directory / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]

    def tensor(self, name, device):
        with safe_open(str(self.directory / self.index[name]), framework="pt", device="cpu") as f:
            # Clone CPU tensors so no whole-shard mappings remain resident between blocks.
            value = f.get_tensor(name)
            return value.clone() if str(device) == "cpu" else value.to(device)

    def full(self, device="cuda"):
        """The whole decoder in f32 (the independent oracle: diffusers' own blocks, not this repository's)."""
        vae = AutoencoderKLMiniMaxH3.from_config(self.config)
        state = {name: self.tensor(name, device) for name in self.index if name.startswith("post_quant_conv.") or name.startswith("decoder.")}
        del vae.encoder, vae.quant_conv
        vae.load_state_dict(state, strict=True, assign=True)
        return vae.to(device).eval()

    def heads(self, device="cuda"):
        with torch.device("meta"):
            vae = AutoencoderKLMiniMaxH3.from_config(self.config)
        del vae.encoder, vae.quant_conv
        vae.decoder.transformer_blocks = torch.nn.ModuleList()
        state = {name: self.tensor(name, device) for name in self.index
                 if name.startswith("post_quant_conv.") or
                 (name.startswith("decoder.") and not name.startswith("decoder.transformer_blocks."))}
        vae.load_state_dict(state, strict=True, assign=True)
        dim = int(self.config["decoder_attention_head_dim"] * self.config["decoder_rope_dim_ratio"])
        vae.decoder.rope = model_impl.MiniMaxH3VideoRotaryPosEmbed(dim, theta=self.config["decoder_rope_theta"]).to(device)
        return vae.eval()

    @staticmethod
    def attention(query, key, value, **kwargs):
        """Float32 SDPA, bounded to 128 query rows against all keys at a time."""
        q, k, v = [a.transpose(1, 2).contiguous() for a in (query, key, value)]
        result = torch.empty_like(q)
        for start in range(0, q.shape[2], 128):
            result[:, :, start:start + 128] = torch.nn.functional.scaled_dot_product_attention(q[:, :, start:start + 128], k, v)
        return result.transpose(1, 2)

    def blocks(self, x, cos, sin, depths):
        """Yield CPU snapshots at requested depths; only one float block is resident."""
        c = self.config
        with torch.device("meta"):
            block = model_impl.MiniMaxH3VideoTransformerBlock(
                c["decoder_num_attention_heads"] * c["decoder_attention_head_dim"],
                c["decoder_num_attention_heads"], c["decoder_attention_head_dim"],
                ffn_mult=c["decoder_ffn_mult"], eps=c["decoder_norm_eps"])
        rope = tuple(torch.cat([a, a], dim=-1)[None, :, None] for a in (cos, sin))
        with torch.no_grad():
            for i in range(max(depths)):
                prefix = f"decoder.transformer_blocks.{i}."
                state = {name[len(prefix):]: self.tensor(name, x.device)
                         for name in self.index if name.startswith(prefix)}
                block.load_state_dict(state, strict=True, assign=True)
                del state
                with patch.object(model_impl, "dispatch_attention_fn", self.attention):
                    x = block(x, rope)
                if i + 1 in depths:
                    yield i + 1, x.cpu()

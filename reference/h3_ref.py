"""MiniMax H3's transformer on the ComfyUI checkpoint names, in plain torch.

Own implementation (no ComfyUI import) of what `comfy/ldm/minimax/model.py` computes for the
`pruned_int8_convrot` checkpoints: the packed sequence layout for a text -> video+audio request,
the AdaLN curve table, the token refiner, the 50 blocks and the final layer. Validated bit for bit
against ComfyUI's `MiniMaxH3Model` at toy size by `tests/test_ref_vs_comfy.py`.

Quantisation modes (the block linears only; everything else stays in float):
  none : the checkpoint's int8 ConvRot weights dequantised per row, activations rotated in float.
         This is what ComfyUI's int8 path computes (its int8_tensorwise layout does not quantise
         the input), so it is the production baseline on this box.
  w4a4 : the same rotated weights requantised to int4 per output row (absmax / 7) and the rotated
         activations quantised to int4 per token: the arithmetic the Loom kernels implement.

Rotation: the checkpoint's weights are already rotated along K by the normalised Kronecker power
of H4 (group 256, 1/16); the activations are rotated by the same matrix here, and by the prepare
kernels in Loom.
"""
from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import torch
import torch.nn.functional as F

HIDDEN, HEADS, HEAD_DIM, FFN = 5376, 56, 128, 14336
INNER = HEADS * HEAD_DIM                       # 7168
TEXT_DIM, VIDEO_PATCH, AUDIO_CH = 5120, 96, 32
ROPE_FREQS, ROPE_DIM = 16, 96                  # 3 axes x 16 frequencies, duplicated halves -> 96 of 128 channels
HADAMARD_GROUP = 256
MODALITIES = 3                                 # AdaLN rows per timestep class: 0 video, 1 text, 2 audio
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0
SPATIAL_SCALE = 32.0
CKPT = Path.home() / "comfy-models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"


# --- rotation and quantisation (as krea2-loom) ---------------------------------------------------

def hadamard(n: int) -> torch.Tensor:
    """comfy-kitchen's regular Hadamard: the Kronecker power of the regular H4 (one negative entry
    per row), normalised by 1/sqrt(n). Orthogonal and symmetric. The kernels' four radix-4
    stages compute the same matrix with the 1/16 folded into the scale."""
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float64)
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.kron(h, h4)
    assert h.shape[0] == n
    return (h / math.sqrt(n)).float()


def rotate_groups(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    g = h.shape[0]
    return (x.reshape(*x.shape[:-1], x.shape[-1] // g, g) @ h.to(x.dtype)).reshape(x.shape)


def quant_int4_rows(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric int4 per row: codes in [-7, 7] (float tensor), scale = absmax / 7."""
    s = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30) / 7.0
    return torch.round(w / s).clamp(-7, 7), s


class QuantLinear:
    """y = x_rot @ W_rot^T with the checkpoint's rotated weights, in one of the modes above."""

    def __init__(self, w_rot: torch.Tensor, h: torch.Tensor, quant: str):
        self.h, self.quant = h, quant
        if quant == "w4a4":
            q, s = quant_int4_rows(w_rot.float())
            self.w = (q * s).to(w_rot.dtype)          # exact int4 codes times per-row scale
        else:
            self.w = w_rot

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        xr = rotate_groups(x.float(), self.h)
        if self.quant == "w4a4":
            q, s = quant_int4_rows(xr)
            xr = q * s
        return (xr.to(self.w.dtype) @ self.w.t()).to(x.dtype)


# --- checkpoint ---------------------------------------------------------------------------------

class Checkpoint:
    """Lazy reader of the ComfyUI safetensors file; int8 ConvRot linears come back dequantised."""

    def __init__(self, path: Path = CKPT, device="cpu", dtype=torch.bfloat16):
        from safetensors import safe_open
        self.f = safe_open(str(path), "pt", device="cpu")
        self.device, self.dtype = device, dtype
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            self.header = json.loads(fh.read(n))
        self.keys = set(self.header) - {"__metadata__"}

    def raw(self, name: str) -> torch.Tensor:
        return self.f.get_tensor(name)

    def tensor(self, name: str, dtype=None) -> torch.Tensor:
        return self.raw(name).to(self.device, dtype or self.dtype)

    def linear(self, name: str) -> torch.Tensor:
        """`name` without `.weight`: dequantised [N][K] in self.dtype (rotated along K)."""
        w = self.raw(name + ".weight")
        if w.dtype == torch.int8:
            s = self.raw(name + ".weight_scale").float().view(-1, 1)
            w = w.float() * s
        return w.to(self.device, self.dtype)


# --- the packed layout for text -> video + audio (no keyframes, no references) --------------------

def _axis(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    ratio, n = dim / sqrt_area, dim // patch
    return (torch.arange(n, dtype=torch.float64) * (ratio / n) + (1.0 - ratio) / 2.0) * SPATIAL_SCALE


def frame_grid(lat_h: int, lat_w: int) -> tuple[torch.Tensor, torch.Tensor]:
    area = math.sqrt(lat_h * lat_w)
    hh, ww = torch.meshgrid(_axis(lat_h, 2, area), _axis(lat_w, 2, area), indexing="ij")
    return torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1), _axis(lat_w, 2, area)


def video_t_grid(n: int, origin: float) -> torch.Tensor:
    spans = torch.tensor([FRAME_RESCALE * FRAME_PER_TOKEN[k % 5] for k in range(n)], dtype=torch.float64)
    return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


class Layout:
    """[text | audio target (2 * audio_t rows, channel-major stereo) | video target] with the
    float64 (t, h, w) rotary grid and, per row, the modality tag and the timestep class."""

    def __init__(self, text_len: int, latent_t: int, lat_h: int, lat_w: int, audio_t: int, text_tags=None):
        frame, w_grid = frame_grid(lat_h, lat_w)
        pos, tags, tclass = [], [], []
        g = torch.zeros(text_len, 3, dtype=torch.float64); g[:, 0] = torch.arange(text_len, dtype=torch.float64)
        pos.append(g); tags.append(torch.ones(text_len, dtype=torch.long) if text_tags is None else text_tags.long()); tclass.append(torch.zeros(text_len, dtype=torch.long))
        cursor = float(text_len)
        a = torch.zeros(audio_t * 2, 3, dtype=torch.float64)
        a[:, 0] = (cursor + torch.arange(audio_t, dtype=torch.float64)).repeat(2)
        a[:audio_t, 2] = float(w_grid[0]); a[audio_t:, 2] = float(w_grid[-1])
        pos.append(a); tags.append(torch.full((audio_t * 2,), 2, dtype=torch.long)); tclass.append(torch.ones(audio_t * 2, dtype=torch.long))
        v = torch.empty(latent_t, frame.shape[0], 3, dtype=torch.float64)
        v[:, :, 0] = video_t_grid(latent_t, cursor)[:, None]; v[:, :, 1:] = frame[None]
        pos.append(v.reshape(-1, 3)); tags.append(torch.zeros(latent_t * frame.shape[0], dtype=torch.long)); tclass.append(torch.zeros(latent_t * frame.shape[0], dtype=torch.long))
        self.text_len, self.audio_rows, self.video_rows = text_len, audio_t * 2, latent_t * frame.shape[0]
        self.position_ids = torch.cat(pos)
        self.tags = torch.cat(tags)
        self.tclass = torch.cat(tclass)          # 0: the video timestep, 1: the audio timestep
        self.seq_len = self.position_ids.shape[0]

    @property
    def adaln_rows(self) -> torch.Tensor:
        """Row of the (timestep class, modality) table for every packed row."""
        return self.tclass * MODALITIES + self.tags


def rope_tables(position_ids: torch.Tensor, inv_freq: torch.Tensor, device) -> tuple[torch.Tensor, torch.Tensor]:
    """[S, 3] float64 -> cos, sin [S, 48] f32: the angle of each rotated pair (channel i, i + 48)."""
    pos = position_ids.to(torch.float32).to(device)
    per_axis = pos.unsqueeze(-1) * inv_freq.to(device).view(1, 1, -1)       # [S, 3, 16]
    half = per_axis.reshape(pos.shape[0], -1)                                  # [S, 48] (t | h | w)
    return torch.cos(half), torch.sin(half)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [S, heads, 128]: rotate-half on the first 96 channels, the rest pass through."""
    half = ROPE_DIM // 2
    x1, x2, rest = x[..., :half], x[..., half:ROPE_DIM], x[..., ROPE_DIM:]
    c, s = cos[:, None, :].to(x.dtype), sin[:, None, :].to(x.dtype)
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c, rest], dim=-1)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (y * weight.float()).to(x.dtype)


def time_shift_sigma(sigma: float, from_shift: float, to_shift: float) -> float:
    """Move a sigma from one exponential shift to another (as ComfyUI's time_shift_sigma)."""
    base = sigma / (from_shift - (from_shift - 1.0) * sigma)
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


# --- the model ----------------------------------------------------------------------------------

class H3Ref:
    def __init__(self, ckpt: Checkpoint, quant: str = "none", layers: int = 50, eps: float = 1e-5, cache_linears: bool = True):
        assert quant in ("none", "w4a4")
        self.ckpt, self.quant, self.layers, self.eps = ckpt, quant, layers, eps
        self.cache_linears = cache_linears          # False: a block's 770 MB of bf16 weights are dropped after use
        self.device, self.dtype = ckpt.device, ckpt.dtype
        self.h = hadamard(HADAMARD_GROUP).to(ckpt.device)
        self.table = ckpt.tensor("adaln_t_table", torch.float32)              # [1025, 8]
        self.inv_freq = ckpt.tensor("rope.inv_freq", torch.float32)
        self._lin = {}

    # --- weights ---
    def t(self, name: str, dtype=None) -> torch.Tensor:
        return self.ckpt.tensor(name, dtype)

    def lin(self, name: str) -> QuantLinear:
        if name not in self._lin:
            self._lin[name] = QuantLinear(self.ckpt.linear(name), self.h, self.quant)
        return self._lin[name]

    def plain_linear(self, name: str, x: torch.Tensor, bias: bool = True, dtype=None) -> torch.Tensor:
        w = self.t(name + ".weight", dtype)
        b = self.t(name + ".bias", dtype) if bias else None
        return F.linear(x.to(w.dtype), w, b)

    # --- conditioning ---
    def t_emb(self, t_values: torch.Tensor) -> torch.Tensor:
        """Distinct timesteps in [0, 1] (1 = clean) -> [M, 8] rows of the AdaLN curve, by lerp."""
        n = self.table.shape[0]
        pos = t_values.to(torch.float32).to(self.device).clamp(0.0, 1.0) * (n - 1)
        i0 = pos.floor().long().clamp(max=n - 2)
        return torch.lerp(self.table[i0], self.table[i0 + 1], (pos - i0).unsqueeze(1))

    def block_mods(self, i: int, temb: torch.Tensor) -> torch.Tensor:
        """[M*3, 6, hidden] f32: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp per (timestep, modality)."""
        w = self.t(f"blocks.{i}.adaln_proj.linear.weight", torch.float32); b = self.t(f"blocks.{i}.adaln_proj.linear.bias", torch.float32)
        x = F.linear(temb, w, b)                                              # [M, 6*3*hidden]
        return x.view(-1, 6 * HIDDEN).view(-1, 6, HIDDEN)                      # rows: m*3 + modality

    def final_mods(self, temb: torch.Tensor) -> torch.Tensor:
        w = self.t("final_layer.adaln_proj.linear.weight", torch.float32); b = self.t("final_layer.adaln_proj.linear.bias", torch.float32)
        return F.linear(temb, w, b).view(-1, 2, HIDDEN)                       # [M, 2 (shift, scale), hidden]

    # --- embeddings ---
    def text_in(self, text_states: torch.Tensor) -> torch.Tensor:
        """[L, 5120] -> [L, hidden] through condition_proj and the token refiner (all bf16)."""
        x = self.plain_linear("condition_proj", text_states.to(self.dtype))
        for j in range(2):
            p = f"token_refiner.blocks.{j}"
            x = x + self.refiner_attention(p, rms_norm(x, self.t(f"{p}.norm1.weight"), self.eps))
            h = rms_norm(x, self.t(f"{p}.norm2.weight"), self.eps)
            x = x + self.plain_linear(f"{p}.mlp.fc2", self.swiglu(self.plain_linear(f"{p}.mlp.fc1", h, bias=False)), bias=False)
        return rms_norm(x, self.t("token_refiner.final_norm.weight"), self.eps)

    def video_in(self, rows: torch.Tensor) -> torch.Tensor:
        return self.plain_linear("video_patch_proj", rows.float(), dtype=torch.float32).to(self.dtype)

    def audio_in(self, rows: torch.Tensor) -> torch.Tensor:
        return self.plain_linear("audio_patch_proj", rows.float(), dtype=torch.float32).to(self.dtype)

    # --- pieces of a block ---
    @staticmethod
    def swiglu(gu: torch.Tensor) -> torch.Tensor:
        gate, up = gu.chunk(2, dim=-1)
        return F.silu(gate) * up

    def qkv_heads(self, p: str, qkv: torch.Tensor, cos, sin):
        s = qkv.shape[0]
        q, k, v = qkv.split(INNER, dim=-1)
        q = rms_norm(q.reshape(s, HEADS, HEAD_DIM), self.t(f"{p}.q_norm.weight"), self.eps)
        k = rms_norm(k.reshape(s, HEADS, HEAD_DIM), self.t(f"{p}.k_norm.weight"), self.eps)
        v = v.reshape(s, HEADS, HEAD_DIM)
        if cos is not None:
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        return q, k, v

    @staticmethod
    def sdpa(q, k, v) -> torch.Tensor:
        s = q.shape[0]
        o = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None])
        return o[0].transpose(0, 1).reshape(s, INNER)

    def refiner_attention(self, p: str, h: torch.Tensor) -> torch.Tensor:
        qkv = self.plain_linear(f"{p}.attn.qkv_proj", h, bias=False)
        q, k, v = self.qkv_heads(f"{p}.attn", qkv, None, None)
        return self.plain_linear(f"{p}.attn.out_proj", self.sdpa(q, k, v), bias=False)

    def block(self, i: int, x: torch.Tensor, mods: torch.Tensor, rows: torch.Tensor, cos, sin) -> torch.Tensor:
        """x [S, hidden]; mods [R, 6, hidden] f32 (from block_mods); rows [S] the table row of each token."""
        p = f"blocks.{i}"
        m = mods[rows]                                                        # [S, 6, hidden]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (m[:, j].to(x.dtype) for j in range(6))
        h = rms_norm(x, self.t(f"{p}.norm1.weight"), self.eps) * (1.0 + scale_msa) + shift_msa
        q, k, v = self.qkv_heads(f"{p}.attn", self.lin(f"{p}.attn.qkv_proj")(h), cos, sin)
        x = x + gate_msa * self.lin(f"{p}.attn.out_proj")(self.sdpa(q, k, v))
        h = rms_norm(x, self.t(f"{p}.norm2.weight"), self.eps) * (1.0 + scale_mlp) + shift_mlp
        return x + gate_mlp * self.lin(f"{p}.mlp.fc2")(self.swiglu(self.lin(f"{p}.mlp.fc1")(h)))

    def blocks_forward(self, x: torch.Tensor, temb: torch.Tensor, rows: torch.Tensor, cos, sin, layers: int | None = None) -> torch.Tensor:
        for i in range(self.layers if layers is None else layers):
            x = self.block(i, x, self.block_mods(i, temb), rows, cos, sin)
            if not self.cache_linears:
                for key in [k for k in self._lin if k.startswith(f"blocks.{i}.")]:
                    del self._lin[key]
        return x

    def final(self, x: torch.Tensor, temb: torch.Tensor, tclass: torch.Tensor, layout: Layout):
        """-> (video rows [Nv, 96], audio rows [Na, 32]) in f32; the caller negates for velocity."""
        fm = self.final_mods(temb)[tclass]                                    # [S, 2, hidden]
        h = rms_norm(x, self.t("final_layer.norm.weight"), self.eps) * (1.0 + fm[:, 1].to(x.dtype)) + fm[:, 0].to(x.dtype)
        h = h.float()
        a0, a1 = layout.text_len, layout.text_len + layout.audio_rows
        video = self.plain_linear("final_layer.video_out", h[a1:], dtype=torch.float32)
        audio = self.plain_linear("final_layer.audio_out", h[a0:a1], dtype=torch.float32)
        return video, audio

    def forward(self, text_states: torch.Tensor, video_rows: torch.Tensor, audio_rows: torch.Tensor,
                layout: Layout, t_video: float, t_audio: float):
        """One packed forward. Returns (-video velocity rows, -audio velocity rows) as ComfyUI does."""
        temb = self.t_emb(torch.tensor([t_video, t_audio]))
        x = torch.cat([self.text_in(text_states), self.audio_in(audio_rows), self.video_in(video_rows)], dim=0)
        cos, sin = rope_tables(layout.position_ids, self.inv_freq, self.device)
        x = self.blocks_forward(x, temb, layout.adaln_rows.to(self.device), cos, sin)
        v, a = self.final(x, temb, layout.tclass.to(self.device), layout)
        return -v, -a

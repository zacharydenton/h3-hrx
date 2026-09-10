#!/usr/bin/env python3
"""Parity against the reference implementation: does this host compute what MiniMax's does?

Self-contained — the C ABI binding, the prompt presentation, the PyTorch reference and the checks are
all in this file. It is deliberately not part of `scripts/test.sh`: it needs the checkpoints, a GPU,
and for the gate a directory of dumps produced inside the ComfyUI container by `scripts/comfy_dump.py`.
Run it before a release, not on every change.

    python3 scripts/parity.py gate [--require]   the release gate: this host vs ComfyUI's own run
    python3 scripts/parity.py compare --truth DIR --mode blocks|trajectory     one comparison, verbose
    python3 scripts/parity.py stack --stack dit|te      locate a regression to a block
    python3 scripts/parity.py vae                      the decoder vs diffusers on MiniMax's weights
    python3 scripts/parity.py dit                      the blocks vs diffusers on MiniMax's weights
    python3 scripts/parity.py convert                  ComfyUI's conversion vs the released weights
    python3 scripts/parity.py te                       the text encoder vs the released Qwen3-VL

`gate` needs no torch. `stack` needs torch, and `--stack dit` also needs diffusers.

Everything else in this repository checks that the model agrees with its own earlier output. This is
the only check against something outside itself, which is why it is worth keeping even though it
cannot run unattended.

Assembled mechanically from the six modules this replaced, at commit 2ce3d0b: h3_loom.py,
tools/h3tok_ids.py, reference/h3_ref.py, tools/compare_comfy.py, tests/test_comfy_parity.py and
tests/test_stack_parity.py, in that order. That is why the sections below keep their original
terse style, and why `R` is bound to this module: the stack checks call the reference section
`R.Checkpoint`, `R.Layout` and so on, as they did when it was a separate import.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# torch is needed only by `stack`, and a ROCm torch takes tens of seconds to import, so it is loaded
# on demand rather than here. Every annotation below is a string (see the __future__ import), so
# nothing evaluates `torch` when this module is read.
torch = F = None


def require_torch() -> bool:
    """Import torch into this module's namespace, for the checks whose oracle it is."""
    global torch, F
    if torch is None:
        try:
            import torch as _torch
            import torch.nn.functional as _F
        except ImportError:
            return False
        torch, F = _torch, _F
    return True


ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------------------------------
# The host, through h3-dev parity-dump
# --------------------------------------------------------------------------------------------------

MODELS = Path(os.environ.get("H3_MODELS") or Path.home() / "comfy-models")
DIT = MODELS / "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
DIT_REF = MODELS / "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"
TE = MODELS / "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
VIDEO_VAE = MODELS / "vae/minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = MODELS / "vae/minimax_h3_audio_vae_fp32.safetensors"

DUMP = ROOT / "target/release/h3-dev"


class H3Error(RuntimeError):
    pass


def host_command(command: str) -> list[str]:
    if not DUMP.is_file():
        raise H3Error(f"{DUMP} is missing; cargo build --release --features internals --bin h3-dev")
    return [str(DUMP), "parity-dump", command]


def host(command: str, out: Path, dumps: Path | None = None, env: dict | None = None, **flags) -> Path:
    """Run one `h3-dev parity-dump` command. Python never links the library: it writes the inputs as files,
    the host writes its artefacts as files, and everything below reads them back. There is no
    foreign-function boundary here to drift out of step with the crate."""
    argv = host_command(command) + ["--out", str(out)]
    for key, value in flags.items():
        if value is None:
            continue
        argv += [f"--{key.replace('_', '-')}", str(value)]
    environment = dict(os.environ)
    environment.update(env or {})
    if dumps is not None:
        dumps.mkdir(parents=True, exist_ok=True)
        environment["H3_DUMP_BLOCKS"] = str(dumps)
        environment["H3_DUMP_CALL"] = "0"
    out.mkdir(parents=True, exist_ok=True)
    done = subprocess.run(argv, capture_output=True, text=True, env=environment)
    if done.returncode:
        raise H3Error((done.stderr or done.stdout).strip() or f"{command} exited {done.returncode}")
    return out


def host_shape(height: int, width: int, frames: int) -> dict:
    """The model's own sizing, so nothing here reimplements the 17n+5 snapping or the /16 grids."""
    argv = host_command("shape") + ["--height", str(height), "--width", str(width), "--frames", str(frames)]
    done = subprocess.run(argv, capture_output=True, text=True)
    if done.returncode:
        raise H3Error((done.stderr or done.stdout).strip())
    return json.loads(done.stdout)


def read_f32(path: Path, width: int | None = None) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    return values.reshape(-1, width) if width else values


# --------------------------------------------------------------------------------------------------
# The PyTorch reference implementation
# --------------------------------------------------------------------------------------------------

HIDDEN, HEADS, HEAD_DIM, FFN = 5376, 56, 128, 14336
INNER = HEADS * HEAD_DIM                       # 7168
TEXT_DIM, VIDEO_PATCH, AUDIO_CH = 5120, 96, 32
ROPE_FREQS, ROPE_DIM = 16, 96                  # 3 axes x 16 frequencies, duplicated halves -> 96 of 128 channels
HADAMARD_GROUP = 256
MODALITIES = 3                                 # AdaLN rows per timestep class: 0 video, 1 text, 2 audio
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0
SPATIAL_SCALE = 32.0
CKPT = Path(os.environ["H3_CKPT"]) if os.environ.get("H3_CKPT") else Path.home() / "comfy-models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"   # H3_CKPT: the ref2va file for its export


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


def quant_int8_rows(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    s = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30) / 127.0
    return torch.round(w / s).clamp(-127, 127), s


def quant_int4_groups(w: torch.Tensor, g: int) -> torch.Tensor:
    """int4 with a scale per (row, K-group of g): returns the dequantised tensor."""
    wg = w.reshape(*w.shape[:-1], w.shape[-1] // g, g)
    q, s = quant_int4_rows(wg)
    return (q * s).reshape(w.shape)


def factor_group_scales(w: torch.Tensor, g: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """int4 weights whose per-(row, K-group) scale is factored s[n][g] ~= r[n] * t[g]: t (per group,
    shared by all rows) folds into the activations, r stays per row, so the GEMM is plain int4.
    Returns (dequantised weight, r [N, 1], t [K])."""
    n, k = w.shape
    wg = w.reshape(n, k // g, g)
    amax = wg.abs().amax(dim=-1).clamp_min(1e-30)                     # [N, K/g]
    t = torch.exp(torch.log(amax).mean(dim=0))                        # geometric mean over rows -> [K/g]
    t = t / t.mean()
    r = (amax / t[None]).amax(dim=1, keepdim=True) / 7.0              # per row: largest group after t
    scale = r * t[None]                                               # [N, K/g]
    q = torch.round(wg / scale[..., None]).clamp(-7, 7)
    return (q * scale[..., None]).reshape(n, k), r, t.repeat_interleave(g)


class QuantLinear:
    """y = x_rot @ W_rot^T with the checkpoint's rotated weights. The mode string names the
    weight and activation treatment, e.g. "w4a4" (the kernels today), "w4g128a8", "w8a8":
      w4     int4 per output row            w4g<G>  int4 per (row, K-group of G)
      w4r<G> int4 per row with a per-group factor t[g] (rank-1 of the (row, group) scales) folded into x
      w8     the checkpoint's int8 rows     none    no activation quantisation, int8 rows
      a4     int4 per token                 a4g     int4 per (token, 256-group)
      a8     int8 per token
    """

    def __init__(self, w_rot: torch.Tensor, h: torch.Tensor, quant: str):
        self.h, self.quant = h, quant
        wq, self.aq = ("w8", "none") if quant == "none" else quant.split("a")
        if wq == "w4":
            q, s = quant_int4_rows(w_rot.float())
            self.w = (q * s).to(w_rot.dtype)          # exact int4 codes times per-row scale
        elif wq.startswith("w4g"):
            self.w = quant_int4_groups(w_rot.float(), int(wq[3:])).to(w_rot.dtype)
        elif wq.startswith("w4r"):
            w, r, t = factor_group_scales(w_rot.float(), int(wq[3:]))
            self.w = (w / t[None]).to(w_rot.dtype)     # the GEMM sees q * r; t is applied to the activations
            self.t = t
        else:
            self.w = w_rot                            # w8 / none: the checkpoint's int8 rows as they are

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        xr = rotate_groups(x.float(), self.h)
        if hasattr(self, "t"):
            xr = xr * self.t
        if self.aq == "4":
            q, s = quant_int4_rows(xr); xr = q * s
        elif self.aq == "4g":
            g = self.h.shape[0]
            xg = xr.reshape(*xr.shape[:-1], xr.shape[-1] // g, g)
            q, s = quant_int4_rows(xg); xr = (q * s).reshape(xr.shape)
        elif self.aq == "8":
            q, s = quant_int8_rows(xr); xr = q * s
        return (xr.to(self.w.dtype) @ self.w.t()).to(x.dtype)


# --- checkpoint ---------------------------------------------------------------------------------

class Checkpoint:
    """Lazy reader of the ComfyUI safetensors file; int8 ConvRot linears come back dequantised."""

    def __init__(self, path: Path = CKPT, device="cpu", dtype=None):
        # torch is imported on demand, so the default cannot be a default argument.
        dtype = torch.bfloat16 if dtype is None else dtype
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
        elif name.startswith("blocks.") and w.shape[-1] % HADAMARD_GROUP == 0:
            # a bf16 (pruned_bf16) checkpoint: unrotated rows. Rotate along K as the ConvRot export did, so the same
            # rotated-activation path (QuantLinear) applies unchanged; the rotation is orthogonal, the product is identical
            w = rotate_groups(w.float(), hadamard(HADAMARD_GROUP))
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


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_dim: int = ROPE_DIM) -> torch.Tensor:
    """x [S, heads, D]: rotate-half on the first rope_dim channels, the rest pass through."""
    half = rope_dim // 2
    x1, x2, rest = x[..., :half], x[..., half:rope_dim], x[..., rope_dim:]
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
        assert quant == "none" or ("a" in quant and quant.startswith("w")), quant
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
            mode = self.quant
            for suffix, override in getattr(self, "quant_overrides", {}).items():   # e.g. {"mlp.fc2": "w8a8"}
                if name.endswith(suffix):
                    mode = override
            self._lin[name] = QuantLinear(self.ckpt.linear(name), self.h, mode)
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

    attn_quant = None          # attention operand study: "a8" (q, k int8 per token), "a4r" (rotated, K smoothed, int4 per token),
    attn_exact_from = 99       # "a4r32" (32-channel groups); layers >= attn_exact_from stay exact

    def sdpa(self, q, k, v, layer: int = 0) -> torch.Tensor:
        s = q.shape[0]
        if self.attn_quant and layer < self.attn_exact_from:
            mode = self.attn_quant
            if not hasattr(self, "_h128"):
                h4 = hadamard(4).double(); h2 = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float64) / math.sqrt(2.0)
                self._h128 = torch.kron(torch.kron(torch.kron(h4, h4), h4), h2).float().to(q.device)
            H = self._h128
            def qr(x, bits, group=None):
                qmax = 7 if bits == 4 else 127
                if group:
                    xg = x.reshape(*x.shape[:-1], x.shape[-1] // group, group); sc = xg.abs().amax(-1, keepdim=True).clamp_min(1e-12) / qmax
                    return ((xg / sc).round().clamp(-qmax, qmax) * sc).reshape(x.shape)
                sc = x.abs().amax(-1, keepdim=True).clamp_min(1e-12) / qmax
                return (x / sc).round().clamp(-qmax, qmax) * sc
            qf, kf = q.float(), k.float()
            if mode == "a8":
                q, k = qr(qf, 8).to(q.dtype), qr(kf, 8).to(k.dtype)
            elif mode in ("a4r", "a4r32"):
                kf = kf - kf.mean(0, keepdim=True); g = 32 if mode == "a4r32" else None
                q, k = (qr(qf @ H, 4, g) @ H.T).to(q.dtype), (qr(kf @ H, 4, g) @ H.T).to(k.dtype)
            elif mode.startswith("a4rs"):                        # int4 rotated + the kernel's tile skip (running-max rule, 16x16 tiles, tau = the number after a4rs)
                tau = float(mode[4:]); q4 = (qr(qf @ H, 4) @ H.T); k4 = (qr(kf @ H, 4) @ H.T)
                S = q4.shape[0]; nq = (S + 15) // 16; nk = nq; out = torch.empty_like(v.float())
                for hd in range(HEADS):
                    sc = (q4[:, hd] @ k4[:, hd].T) / math.sqrt(HEAD_DIM)
                    sp = F.pad(sc, (0, nk * 16 - S, 0, nq * 16 - S), value=-1e9).reshape(nq, 16, nk, 16)
                    row_tile_max = sp.amax(dim=3); running = torch.cummax(row_tile_max, dim=2).values
                    prev = torch.cat([torch.full_like(running[:, :, :1], -1e9), running[:, :, :-1]], dim=2)
                    skip = (row_tile_max < (prev - tau)).all(dim=1)
                    mask = skip[:, None, :, None].expand(nq, 16, nk, 16).reshape(nq * 16, nk * 16)[:S, :S]
                    out[:, hd] = torch.softmax(sc.masked_fill(mask, -1e9), -1) @ v[:, hd].float()
                return out.reshape(S, INNER).to(v.dtype)
            elif mode in ("a4c16", "a4c128", "a4cg", "a4cr16", "a4crg"):   # SageAttention2: Q and K centred, int4 per token, exact correction restored (a4cr*: also rotated)
                qb = {"a4c16": 16, "a4c128": 128, "a4cg": 0, "a4cr16": 16, "a4crg": 0}[mode]
                qm = torch.cat([blk.mean(0, keepdim=True).expand_as(blk) for blk in qf.split(qb, dim=0)], 0) if qb else qf.mean(0, keepdim=True).expand_as(qf)
                if mode.startswith("a4cr"):
                    kc = qr((kf - kf.mean(0, keepdim=True)) @ H, 4) @ H.T; qc = qr((qf - qm) @ H, 4) @ H.T
                else:
                    kc = qr(kf - kf.mean(0, keepdim=True), 4); qc = qr(qf - qm, 4)
                scores = (torch.einsum("thd,shd->hts", qc, kc) + torch.einsum("thd,shd->hts", qm, kc)) / math.sqrt(HEAD_DIM)
                o = (torch.softmax(scores, -1) @ v.float().transpose(0, 1)).transpose(0, 1)
                return o.reshape(s, INNER).to(v.dtype)
            else:
                raise ValueError(mode)
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
        x = x + gate_msa * self.lin(f"{p}.attn.out_proj")(self.sdpa(q, k, v, i))
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


R = sys.modules[__name__]   # the reference section above, under the name the checks call it


# --------------------------------------------------------------------------------------------------
# Comparison against ComfyUI's own run
# --------------------------------------------------------------------------------------------------

def comparison_options() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["t2va", "fl2va"], default="t2va"); ap.add_argument("--mode", choices=["trajectory", "blocks"], default="trajectory")
    ap.add_argument("--truth", required=True, help="scripts/comfy_dump.py --out directory"); ap.add_argument("--dit", default=None, help="the DiT checkpoint (default: the fl2va file under $H3_MODELS)"); ap.add_argument("--attn", choices=["i4", "i8", "f16"], default="i8")
    ap.add_argument("--prompt", default="A red fox trotting through a snowy forest at dawn, cinematic"); ap.add_argument("--first-frame", default=str(ROOT / "build/refs/fox_clean.png"))
    ap.add_argument("--width", type=int, default=864); ap.add_argument("--height", type=int, default=480); ap.add_argument("--frames", type=int, default=22); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dump", default=None, help="directory for the per-step / per-block dumps (default: <truth>/mine_<blocks>_<attn>)")
    ap.add_argument("--block-ids", default="0,1,2,5,10,20,30,40,49")
    return ap


def compare(a) -> list:
    """One comparison against a dump directory; returns the lines it printed."""
    from PIL import Image
    lines = []

    def emit(*parts, **kw):
        text = " ".join(str(p) for p in parts)
        lines.append(text)
        print(text, **{k: v for k, v in kw.items() if k != "end"})

    D = Path(a.truth); dit = a.dit or str(DIT)
    dump = Path(a.dump or D / f"mine_{Path(dit).stem}_{a.attn}"); dump.mkdir(parents=True, exist_ok=True)
    work = dump / "host"; work.mkdir(parents=True, exist_ok=True)
    sh = host_shape(a.height, a.width, a.frames)
    steps = 2 if a.mode == "blocks" else 21
    nv = np.load(D / "noise_video.npy").astype(np.float32); na = np.load(D / "noise_audio.npy").astype(np.float32)   # ComfyUI's pack: video [24,T,H,W], audio [32, 2, A]
    nv.tofile(work / "nv.f32")
    np.ascontiguousarray(na.transpose(1, 0, 2)).tofile(work / "na.f32")   # the host takes [2][32][A]
    keyframe = {}
    if a.case == "fl2va":
        img = np.asarray(Image.open(a.first_frame).convert("RGB").resize((a.width, a.height), Image.BILINEAR), dtype=np.float32) / 255.0
        img.tofile(work / "kf.f32")
        host("encode", work, pixels=work / "kf.f32", height=a.height, width=a.width, dit=dit, attn=a.attn)
        keyframe = {"keyframe": work / "kf.f32", "keyframe_latents": work / "latents.f32"}
    env = {"H3_DUMP_DIR": str(dump)}   # the per-evaluation trajectory; the residual stream is H3_DUMP_BLOCKS
    t0 = time.time()
    host("denoise", work, dumps=dump if a.mode == "blocks" else None, env=env,
         prompt=a.prompt, height=a.height, width=a.width, frames=a.frames, steps=steps,
         seed=a.seed, sampler="res_multistep", dit=dit, attn=a.attn,
         noise_video=work / "nv.f32", noise_audio=work / "na.f32", **keyframe)
    emit(f"host denoised in {time.time() - t0:.1f} s ({Path(dit).stem}, {a.attn} attention)", flush=True)
    ids = np.fromfile(work / "ids.i32", dtype=np.int32)
    cos = lambda x, y: float((x * y).sum() / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-30))
    T, Hh, Ww = sh["latent_t"], sh["lat_h"], sh["lat_w"]
    if a.mode == "trajectory":
        for k in range(0, 20):
            c = np.fromfile(dump / f"x_{k:02d}.f32", dtype=np.float32).reshape(24, T, Hh, Ww).astype(np.float64); y = np.load(D / f"x_{k:02d}.npy").astype(np.float64)
            emit(f"x_{k:02d}: cosine {cos(c, y):.4f}  rel err {np.linalg.norm(c - y) / np.linalg.norm(y):.4f}", flush=True)
        c = np.fromfile(dump / "x_20.f32", dtype=np.float32).reshape(24, T, Hh, Ww).astype(np.float64); y = np.load(D / "video_latent.npy").astype(np.float64)
        emit(f"final: cosine {cos(c, y):.4f}  rel err {np.linalg.norm(c - y) / np.linalg.norm(y):.4f}", flush=True)
    else:
        L = len(ids); Na = sh["audio_t"] * 2; Nv = sh["latent_t"] * (sh["lat_h"] // 2) * (sh["lat_w"] // 2)
        host("text", work, prompt=a.prompt, dit=dit, attn=a.attn)
        txt = read_f32(work / "text_in.f32", HIDDEN).astype(np.float64)
        ref = np.load(D / "blocks/refined_text.npy").astype(np.float64).reshape(-1, txt.shape[-1])
        emit(f"refined text: cosine {cos(txt, ref):.5f} rel err {np.linalg.norm(txt - ref) / np.linalg.norm(ref):.4f}", flush=True)
        segs = {"text": (0, L), "audio": (L, L + Na), "video": (L + Na, L + Na + Nv)}
        for i in [int(x) for x in a.block_ids.split(",")]:
            name = f"blk_{i:02d}"; m = np.fromfile(dump / f"dit_{name}.f32", np.float32).reshape(-1, 5376).astype(np.float64); y = np.load(D / f"blocks/{name}.npy").astype(np.float64).reshape(-1, 5376)
            parts = "  ".join(f"{k} {cos(m[s0:s1], y[s0:s1]):.4f}" for k, (s0, s1) in segs.items())
            emit(f"{name}: cosine {cos(m, y):.4f} rel err {np.linalg.norm(m - y) / np.linalg.norm(y):.4f}  [{parts}]", flush=True)
    return lines


# --------------------------------------------------------------------------------------------------
# The release gate
# --------------------------------------------------------------------------------------------------

# The blocks scripts/comfy_dump.py --dump-blocks writes; a comparison that skips one is a failure.
REQUIRED_BLOCKS = (0, 1, 2, 5, 10, 20, 30, 40, 49)


def run(args) -> list:
    """One `compare` run's lines, or None if it raised — which is a failure, never a pass."""
    try:
        return compare(comparison_options().parse_args(args))
    except Exception as e:
        print(f"  FAIL compare {' '.join(args)}: {e}", file=sys.stderr)
        return None


def gate(require: bool) -> int:
    missing = [str(f) for f in (ROOT / "build/comfy_t2va_blocks/blocks/blk_49.npy", ROOT / "build/comfy_fl2va/x_19.npy", DIT) if not f.exists()]
    if missing:
        print(f"{'FAIL' if require else 'SKIP'}: missing {', '.join(missing)} (scripts/comfy_dump.py --dump-steps --dump-blocks 0,1,2,5,10,20,30,40,49 --steps 2 --out build/comfy_t2va_blocks; README, Weights)")
        return 1 if require else 0
    ok = True
    cases = [("build/comfy_t2va_blocks", "f16"), ("build/comfy_t2va_blocks", "i8")]   # int8 QK^T operands are the parity path too
    for truth, attn in cases:
        lines = run(["--case", "t2va", "--mode", "blocks", "--truth", truth, "--attn", attn])
        if lines is None: ok = False; continue
        seen = set()
        for line in lines:
            if line.startswith("blk_"):
                blk = int(line.split(":")[0][4:]); video = float(line.split("video ")[1].rstrip("]")); seen.add(blk)
                good = video >= (0.999 if blk <= 20 else 0.99)   # measured 0.9990 / 0.9988 at block 30 (f16 / int8), 0.9935 at 40, 0.9992 at 49
                ok &= good; print(f"  {'PASS' if good else 'FAIL'} the checkpoint's int8 rows + {attn} attention {line.split(':')[0]} video rows {video:.4f}")
        absent = [b for b in REQUIRED_BLOCKS if b not in seen]
        if absent: ok = False; print(f"  FAIL {attn} attention: no result for blocks {absent}")
    lines = run(["--case", "fl2va", "--mode", "trajectory", "--truth", "build/comfy_fl2va", "--attn", "f16"])
    if lines is None: ok = False
    else:
        x05 = [line for line in lines if line.startswith("x_05")]
        if not x05: ok = False; print("  FAIL trajectory: no x_05 result")
        for line in x05:
            err = float(line.split("rel err ")[1]); good = err <= 0.02; ok &= good; print(f"  {'PASS' if good else 'FAIL'} trajectory after five evaluations: rel err {err:.4f}")
    print("parity gate passed" if ok else "PARITY GATE FAILED")
    return 0 if ok else 1


# --------------------------------------------------------------------------------------------------
# Block-by-block stack parity
# --------------------------------------------------------------------------------------------------

def cosine(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def dumped(dump, tag, name, width):
    return np.fromfile(dump / f"{tag}_{name}.f32", dtype=np.float32).reshape(-1, width)


def stack_dit(depths, tokens_hw):
    """The DiT stack: the host's packed rows through the PyTorch reference above (the same int8 checkpoint, dequantised)."""
    height, width, frames = tokens_hw
    prompt = "A red fox trotting through a snowy forest at dawn, cinematic"
    with tempfile.TemporaryDirectory(prefix="h3-dit-parity-") as tmp:
        tmp = Path(tmp)
        s = host_shape(height, width, frames)
        rng = np.random.default_rng(1)
        nv = rng.standard_normal((24, s["latent_t"], s["lat_h"], s["lat_w"])).astype(np.float32)
        na = rng.standard_normal((2, 32, s["audio_t"])).astype(np.float32)
        nv.tofile(tmp / "nv.f32"); na.tofile(tmp / "na.f32")
        t0 = time.time()
        host("denoise", tmp, dumps=tmp, prompt=prompt, height=height, width=width, frames=frames,
             steps=2, seed=1, noise_video=tmp / "nv.f32", noise_audio=tmp / "na.f32")
        print(f"host evaluation in {time.time() - t0:.1f} s")
        ids = np.fromfile(tmp / "ids.i32", dtype=np.int32)
        x0 = dumped(tmp, "dit", "h_in", R.HIDDEN)
        want_last = {d: dumped(tmp, "dit", f"blk_{d - 1:02d}", R.HIDDEN) for d in depths}
    layout = R.Layout(ids.size, s["latent_t"], s["lat_h"], s["lat_w"], s["audio_t"])
    ckpt = R.Checkpoint(device="cuda", dtype=torch.bfloat16); ref = R.H3Ref(ckpt, quant="none")
    cos, sin = R.rope_tables(layout.position_ids, ref.inv_freq, "cuda"); rows = layout.adaln_rows.to("cuda")
    from diffusers import MiniMaxH3Scheduler
    sv = MiniMaxH3Scheduler(shift=12.0); sa = MiniMaxH3Scheduler(shift=3.0); sv.set_timesteps(2, device="cuda"); sa.set_timesteps(2, device="cuda")
    ok = True
    with torch.no_grad():
        temb = ref.t_emb(torch.tensor([sv.timesteps[0].item(), sa.timesteps[0].item()]))
        x = torch.from_numpy(x0).to("cuda").to(torch.bfloat16)
        for d in sorted(depths):
            y = ref.blocks_forward(x.clone(), temb, rows, cos, sin, layers=d).float().cpu().numpy()
            got = want_last[d]
            L, Na = layout.text_len, layout.audio_rows
            segs = {"text": (0, L), "audio": (L, L + Na), "video": (L + Na, got.shape[0])}
            parts = "  ".join(f"{k} {cosine(got[a:b], y[a:b]):.4f}" for k, (a, b) in segs.items())
            c = cosine(got, y); good = c > 0.99; ok &= good
            print(f"  {'PASS' if good else 'FAIL'} after {d:2d} blocks: cosine {c:.5f}  [{parts}]")
    del ref, ckpt; torch.cuda.empty_cache()
    return ok


def stack_te(depths):
    """The text encoder: the host's embedding rows through transformers' bf16 layers on the same checkpoint's weights."""
    cache = ROOT / "build/te_hidden.pt"
    with tempfile.TemporaryDirectory(prefix="h3-te-parity-") as tmp:
        tmp = Path(tmp)
        t0 = time.time()
        host("text", tmp, dumps=tmp, prompt="A red fox trotting through a snowy forest at dawn, cinematic")
        print(f"host text_in in {time.time() - t0:.1f} s")
        got = {d: dumped(tmp, "te", f"blk_{d - 1:02d}", TEXT_DIM) for d in depths}
        x0 = dumped(tmp, "te", "h_in", TEXT_DIM)
    if not cache.exists():
        print(f"SKIP: no {cache} (the transformers reference: see docs/archive/notes.md, the text encoder)"); return True
    want = torch.load(cache)   # [layers + 1][tokens][5120] bf16 hidden states from transformers on the same ids
    ok = True
    for d in sorted(depths):
        y = want[d].float().numpy(); c = cosine(got[d], y); good = c > 0.99; ok &= good
        print(f"  {'PASS' if good else 'FAIL'} after {d:2d} layers: cosine {c:.5f}")
    print(f"  (x_in cosine {cosine(x0, want[0].float().numpy()):.5f})")
    return ok


# --------------------------------------------------------------------------------------------------
# The video VAE against diffusers on MiniMax's own weights
# --------------------------------------------------------------------------------------------------

def official_vae(directory: Path, device: str):
    """diffusers' `AutoencoderKLMiniMaxH3` decoder in f32, on the weights MiniMax released.

    This is the one oracle here that is genuinely upstream: not ComfyUI's conversion of the model and
    not a reimplementation of it, but the reference implementation reading the reference weights. The
    host decodes ComfyUI's f16 conversion of the same VAE, so the gap this measures is that narrowing
    plus whatever the Loom decoder does differently.
    """
    import json

    from diffusers import AutoencoderKLMiniMaxH3
    from safetensors import safe_open

    config = json.loads((directory / "config.json").read_text())
    index = json.loads(
        (directory / "diffusion_pytorch_model.safetensors.index.json").read_text()
    )["weight_map"]

    def tensor(name):
        with safe_open(str(directory / index[name]), framework="pt", device="cpu") as f:
            return f.get_tensor(name).to(device)

    vae = AutoencoderKLMiniMaxH3.from_config(config)
    wanted = [name for name in index if name.startswith(("post_quant_conv.", "decoder."))]
    del vae.encoder, vae.quant_conv
    vae.load_state_dict(dict((name, tensor(name)) for name in wanted), strict=True, assign=True)
    return vae.to(device).eval(), config


def vae_parity(a) -> int:
    """The host's tiled f16 decoder against that reference, on latents both are given."""
    import torch

    z = (
        np.load(a.latents).astype(np.float32)
        if a.latents
        else np.random.default_rng(0).normal(size=(24, 7, 20, 24)).astype(np.float32)
    )
    _, t, h, w = z.shape
    if t < 7 or (t - 2) % 5 or h <= 16 or w <= 16:
        print("latents must cross a tile and a temporal boundary", file=sys.stderr)
        return 1
    frames = (t - 2) // 5 * 17 + 5

    with tempfile.TemporaryDirectory(prefix="h3-vae-parity-") as tmp:
        tmp = Path(tmp)
        z.tofile(tmp / "latents.f32")
        # a decoder-only run: the DiT and text checkpoints are never opened
        host("decode", tmp, latents=tmp / "latents.f32", height=h * 16, width=w * 16,
             frames=frames, dit="/unused/dit.safetensors", te="/unused/te.safetensors")
        got = np.fromfile(tmp / "frames.rgb", dtype=np.uint8).reshape(frames, h * 16, w * 16, 3)

    vae, config = official_vae(Path(a.official), a.device)
    with torch.no_grad():
        zt = torch.from_numpy(z)[None].to(a.device)
        mean = zt.new_tensor(config["latents_mean"])[None, :, None, None, None]
        std = zt.new_tensor(config["latents_std"])[None, :, None, None, None]
        video = vae._decode(zt * std + mean)
        # the decoder emits ImageNet-normalised pixels; the host writes RGB8
        vmean = video.new_tensor((0.485, 0.456, 0.406))[None, :, None, None, None]
        vstd = video.new_tensor((0.229, 0.224, 0.225))[None, :, None, None, None]
        want = (
            ((video * vstd + vmean).clamp(0, 1) * 255)
            .round()
            .to(torch.uint8)[0]
            .permute(1, 2, 3, 0)
            .cpu()
            .numpy()
        )
    del vae
    torch.cuda.empty_cache()

    if got.shape != want.shape:
        print("shape %s against %s" % (got.shape, want.shape), file=sys.stderr)
        return 1
    mse = float(np.mean((got.astype(np.float64) - want.astype(np.float64)) ** 2))
    psnr = 10 * math.log10(255**2 / max(mse, 1e-12))
    good = psnr > a.floor
    verdict = "PASS" if good else "FAIL"
    print(
        "  %s video decoder vs diffusers on MiniMax's weights: %.2f dB over %d frames at %dx%d (floor %s)"
        % (verdict, psnr, got.shape[0], w * 16, h * 16, a.floor)
    )
    return 0 if good else 1


# --------------------------------------------------------------------------------------------------
# The DiT blocks against diffusers on MiniMax's unquantized weights
# --------------------------------------------------------------------------------------------------

OFFICIAL = Path.home() / ".cache/huggingface/hub/models--MiniMaxAI--MiniMax-H3"


def official_transformer(directory: Path, device: str):
    """diffusers' `MiniMaxH3Transformer3DModel` pieces, with one block resident at a time.

    Returns the model shell, its weight map and a loader. The shell is built on the meta device, so
    `rope` and `time_proj` — whose tables are computed in `__init__` rather than stored — are
    reconstructed rather than moved: `to_empty` would hand them uninitialised memory, which shows up
    as plausible text and audio rows and nonsense video ones.
    """
    import json

    import diffusers.models.transformers.transformer_minimax_h3 as impl
    from diffusers import MiniMaxH3Transformer3DModel
    from safetensors import safe_open

    config = json.loads((directory / "config.json").read_text())
    index = json.loads(
        (directory / "diffusion_pytorch_model.safetensors.index.json").read_text()
    )["weight_map"]

    def tensor(name):
        with safe_open(str(directory / index[name]), framework="pt", device="cpu") as f:
            return f.get_tensor(name).to(device)

    with torch.device("meta"):
        model = MiniMaxH3Transformer3DModel.from_config(config)
    embed = dict(
        (name[len("time_embedder."):], tensor(name))
        for name in index
        if name.startswith("time_embedder.")
    )
    model.time_embedder.load_state_dict(embed, strict=True, assign=True)
    model.rope = impl.MiniMaxH3RotaryPosEmbed(config["rope_freq_dim"], config["rope_theta"]).to(device)
    model.time_proj = type(model.time_proj)(
        config["freq_dim"], flip_sin_to_cos=True, downscale_freq_shift=0
    ).to(device)
    return model, index, tensor


def dit_parity(a) -> int:
    """The host's packed rows through the released bf16 blocks, block by block.

    The host runs the int8 ConvRot conversion of these weights, so the agreement below is the
    quantisation and this implementation together, measured against the reference rather than against
    ComfyUI's independent reimplementation of it.
    """
    directory = Path(a.official)
    if not (directory / "config.json").is_file():
        snapshots = sorted((OFFICIAL / "snapshots").glob("*/transformer"))
        if not snapshots:
            print(f"no released transformer under {directory}", file=sys.stderr)
            return 1
        directory = snapshots[-1]

    depths = [int(x) for x in a.depths.split(",") if x]
    prompt = "A red fox trotting through a snowy forest at dawn, cinematic"
    with tempfile.TemporaryDirectory(prefix="h3-dit-official-") as tmp:
        tmp = Path(tmp)
        shape = host_shape(a.height, a.width, a.frames)
        rng = np.random.default_rng(1)
        nv = rng.standard_normal(
            (24, shape["latent_t"], shape["lat_h"], shape["lat_w"])
        ).astype(np.float32)
        na = rng.standard_normal((2, 32, shape["audio_t"])).astype(np.float32)
        nv.tofile(tmp / "nv.f32")
        na.tofile(tmp / "na.f32")
        started = time.time()
        host("denoise", tmp, dumps=tmp, prompt=prompt, height=a.height, width=a.width,
             frames=a.frames, steps=2, seed=1,
             noise_video=tmp / "nv.f32", noise_audio=tmp / "na.f32")
        print(f"host evaluation in {time.time() - started:.1f} s")
        ids = np.fromfile(tmp / "ids.i32", dtype=np.int32)
        x0 = dumped(tmp, "dit", "h_in", HIDDEN)
        want = dict((d, dumped(tmp, "dit", f"blk_{d - 1:02d}", HIDDEN)) for d in depths)

    layout = Layout(ids.size, shape["latent_t"], shape["lat_h"], shape["lat_w"], shape["audio_t"])
    model, index, tensor = official_transformer(directory, a.device)
    from diffusers import MiniMaxH3Scheduler

    sv = MiniMaxH3Scheduler(shift=12.0)
    sa = MiniMaxH3Scheduler(shift=3.0)
    sv.set_timesteps(2, device=a.device)
    sa.set_timesteps(2, device=a.device)
    ok = True
    with torch.no_grad():
        steps = torch.tensor(
            [sv.timesteps[0].item(), sa.timesteps[0].item()], device=a.device
        )
        embed_dtype = next(model.time_embedder.parameters()).dtype
        temb = model.time_embedder(model.time_proj(steps).to(embed_dtype))
        rope = model.rope(layout.position_ids.to(a.device))
        rows = layout.adaln_rows.to(a.device)
        x = torch.from_numpy(x0).to(a.device).to(torch.bfloat16)[None]
        block = model.transformer_blocks[0]
        for i in range(max(depths)):
            prefix = f"transformer_blocks.{i}."
            state = dict(
                (name[len(prefix):], tensor(name))
                for name in index
                if name.startswith(prefix)
            )
            block.load_state_dict(state, strict=True, assign=True)
            del state
            x = block(x, temb, rows, rope)
            if i + 1 in depths:
                y = x[0].float().cpu().numpy()
                got = want[i + 1]
                text, audio = layout.text_len, layout.audio_rows
                segments = {
                    "text": (0, text),
                    "audio": (text, text + audio),
                    "video": (text + audio, got.shape[0]),
                }
                parts = "  ".join(
                    f"{k} {cosine(got[lo:hi], y[lo:hi]):.4f}"
                    for k, (lo, hi) in segments.items()
                )
                c = cosine(got, y)
                good = c > a.floor
                ok &= good
                verdict = "PASS" if good else "FAIL"
                print(f"  {verdict} after {i + 1:2d} blocks: cosine {c:.5f}  [{parts}]")
            if a.device == "cuda":
                torch.cuda.empty_cache()
    return 0 if ok else 1


# --------------------------------------------------------------------------------------------------
# ComfyUI's conversion against the weights it was converted from
# --------------------------------------------------------------------------------------------------

# What ComfyUI's `comfy_quant` descriptor says each quantised tensor is, and where the same weights
# live in the released checkpoint. `qkv_proj` fuses the three projections the reference keeps apart.
CONVERTED = {
    "attn.qkv_proj": ["attn.to_q", "attn.to_k", "attn.to_v"],
    "attn.out_proj": ["attn.to_out.0"],
    "mlp.fc1": ["ff.net.0.proj"],
    "mlp.fc2": ["ff.net.2"],
}


def conversion_parity(a) -> int:
    """Is ComfyUI's int8 ConvRot conversion faithful to the weights MiniMax released?

    The conversion rotates each row within groups of `convrot_groupsize` channels, quantises to int8
    and reorders rows to suit the kernels, so the stored numbers are deliberately not the released
    ones. Two invariants survive all of that. A rotation is orthogonal, so the norm of every row is
    unchanged — compared as a multiset, that holds whatever order the rows end up in and whatever
    Hadamard convention was used. Where the rows also stay in place, every 256-channel group norm
    holds too, which is the stricter statement and the one the attention projections satisfy.

    Between them these catch a tensor mapped to the wrong place, a mis-scaled row, a dropped shard
    and a truncated conversion. Neither can catch a rotation applied with the wrong sign convention;
    that is what `parity.py dit` measures end to end.
    """
    import json

    from safetensors import safe_open

    directory = Path(a.official)
    if not (directory / "config.json").is_file():
        found = sorted((OFFICIAL / "snapshots").glob("*/transformer"))
        if not found:
            print(f"no released transformer under {directory}", file=sys.stderr)
            return 1
        directory = found[-1]
    index = json.loads(
        (directory / "diffusion_pytorch_model.safetensors.index.json").read_text()
    )["weight_map"]

    def official(name):
        with safe_open(str(directory / index[name]), framework="pt", device="cpu") as f:
            return f.get_tensor(name).float()

    worst, aligned, checked, unmapped = 0.0, 0, 0, []
    with safe_open(str(a.converted), framework="pt", device="cpu") as f:
        for name in sorted(f.keys()):
            if not name.endswith(".weight_scale"):
                continue
            stem = name[: -len(".weight_scale")]
            _, _, rest = stem.partition(".")
            depth, _, kind = rest.partition(".")
            if kind not in CONVERTED:
                unmapped.append(stem)
                continue
            size = json.loads(
                bytes(f.get_tensor(stem + ".comfy_quant").numpy().tobytes())
            ).get("convrot_groupsize", 256)
            got = f.get_tensor(stem + ".weight").float() * f.get_tensor(name).float()
            want = torch.cat(
                [official(f"transformer_blocks.{depth}.{part}.weight") for part in CONVERTED[kind]],
                0,
            )
            if got.shape != want.shape:
                print(f"  FAIL {stem}: {tuple(got.shape)} against {tuple(want.shape)}")
                return 1
            rows = got.norm(dim=1).sort().values, want.norm(dim=1).sort().values
            error = float((rows[0] - rows[1]).abs().mean() / rows[1].mean())
            if error > a.floor:
                print(f"  FAIL {stem}: row norms differ by {error:.5f} even as a multiset")
                return 1
            worst = max(worst, error)
            checked += 1
            groups = (
                got.reshape(got.shape[0], -1, size).norm(dim=2),
                want.reshape(want.shape[0], -1, size).norm(dim=2),
            )
            if float((groups[0] - groups[1]).abs().mean() / groups[1].mean()) <= a.floor:
                aligned += 1
            del got, want, groups

    if unmapped:
        print(f"  FAIL {len(unmapped)} quantised tensors have no mapping, e.g. {unmapped[:3]}")
        return 1
    print(
        f"  PASS {checked} converted tensors carry the released weights: worst row-norm error "
        f"{worst:.5f} (floor {a.floor}); {aligned} of them also keep their rows in place, so their "
        f"256-channel group norms match too"
    )
    return 0


# --------------------------------------------------------------------------------------------------
# The text encoder against the released Qwen3-VL
# --------------------------------------------------------------------------------------------------

def te_parity(a) -> int:
    """The host's embedding rows through the released bf16 Qwen3-VL layers, one layer resident.

    MiniMax-H3 reads the unnormalised hidden state after the 50th of 64 decoder layers, so this walks
    the same 50 and compares where the host dumped. The host runs ComfyUI's int8 conversion of these
    weights; the agreement is that quantisation and this implementation together against the release.
    """
    import json

    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_vl import modeling_qwen3_vl as impl

    directory = Path(a.official)
    if not (directory / "config.json").is_file():
        found = sorted((OFFICIAL / "snapshots").glob("*/text_encoder"))
        if not found:
            print(f"no released text encoder under {directory}", file=sys.stderr)
            return 1
        directory = found[-1]

    depths = [int(x) for x in a.depths.split(",") if x]
    with tempfile.TemporaryDirectory(prefix="h3-te-official-") as tmp:
        tmp = Path(tmp)
        started = time.time()
        host("text", tmp, dumps=tmp, prompt=a.prompt)
        print(f"host text_in in {time.time() - started:.1f} s")
        x0 = dumped(tmp, "te", "h_in", TEXT_DIM)
        want = dict((d, dumped(tmp, "te", f"blk_{d - 1:02d}", TEXT_DIM)) for d in depths)

    config = AutoConfig.from_pretrained(str(directory))
    text = getattr(config, "text_config", config)
    index = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = "model.language_model.layers."
    if not any(name.startswith(prefix) for name in index):
        prefix = "model.layers."

    def tensor(name):
        with safe_open(str(directory / index[name]), framework="pt", device="cpu") as f:
            return f.get_tensor(name).to(a.device)

    with torch.device("meta"):
        layer = impl.Qwen3VLTextDecoderLayer(text, 0)
    rotary = impl.Qwen3VLTextRotaryEmbedding(text).to(a.device)

    rows = x0.shape[0]
    ok = True
    with torch.no_grad():
        x = torch.from_numpy(x0).to(a.device).to(torch.bfloat16)[None]
        # mrope carries three axes; a text-only presentation puts them all on one line
        position = torch.arange(rows, device=a.device)[None].expand(3, 1, rows)
        rope = rotary(x, position)
        mask = torch.full(
            (rows, rows), float("-inf"), device=a.device, dtype=x.dtype
        ).triu(1)[None, None]
        for i in range(max(depths)):
            at = f"{prefix}{i}."
            state = dict(
                (name[len(at):], tensor(name)) for name in index if name.startswith(at)
            )
            layer.load_state_dict(state, strict=True, assign=True)
            del state
            x = layer(x, rope, attention_mask=mask, position_ids=position)
            if isinstance(x, tuple):
                x = x[0]
            if i + 1 in depths:
                c = cosine(want[i + 1], x[0].float().cpu().numpy())
                good = c > a.floor
                ok &= good
                print(f"  {'PASS' if good else 'FAIL'} after {i + 1:2d} layers: cosine {c:.5f}")
            if a.device == "cuda":
                torch.cuda.empty_cache()
    return 0 if ok else 1


# --------------------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    g = sub.add_parser("gate", help="the release gate: this host vs ComfyUI's own run")
    g.add_argument("--require", action="store_true", help="a missing dump or checkpoint is a failure, not a skip")
    sub.add_parser("compare", parents=[comparison_options()], add_help=False, help="one comparison, verbose")
    v = sub.add_parser("vae", help="the video decoder against diffusers on MiniMax's own weights")
    v.add_argument("--official", default=str(Path.home() / "h3-models/vae"), help="the released VAE, diffusers layout")
    v.add_argument("--latents", type=Path, help="a .npy of [24][t][h][w]; a fixed random draw by default")
    v.add_argument("--device", default="cuda")
    v.add_argument("--floor", type=float, default=40.0, help="PSNR the decoder must clear")
    d = sub.add_parser("dit", help="the blocks against diffusers on MiniMax's unquantized weights")
    d.add_argument("--official", default=str(OFFICIAL), help="the released transformer, diffusers layout")
    d.add_argument("--depths", default="1,10,25,50", help="block counts to compare")
    d.add_argument("--height", type=int, default=64)
    d.add_argument("--width", type=int, default=96)
    d.add_argument("--frames", type=int, default=22)
    d.add_argument("--device", default="cuda")
    d.add_argument("--floor", type=float, default=0.99, help="cosine every depth must clear")
    c = sub.add_parser("convert", help="ComfyUI's int8 conversion against the released weights")
    c.add_argument("--official", default=str(OFFICIAL), help="the released transformer, diffusers layout")
    c.add_argument("--converted", default=str(DIT), help="the ComfyUI checkpoint the host runs")
    c.add_argument("--floor", type=float, default=0.005, help="relative group-norm error allowed")
    e = sub.add_parser("te", help="the text encoder against the released Qwen3-VL")
    e.add_argument("--official", default=str(OFFICIAL), help="the released text encoder")
    e.add_argument("--depths", default="1,10,25,50", help="layer counts to compare")
    e.add_argument("--prompt", default="A red fox trotting through a snowy forest at dawn, cinematic")
    e.add_argument("--device", default="cuda")
    e.add_argument("--floor", type=float, default=0.99, help="cosine every depth must clear")
    s = sub.add_parser("stack", help="locate a regression to a block")
    s.add_argument("--stack", choices=["dit", "te"], required=True)
    s.add_argument("--depths", default="1,10,25,50", help="block counts to compare")
    s.add_argument("--height", type=int, default=64)
    s.add_argument("--width", type=int, default=96)
    s.add_argument("--frames", type=int, default=22)
    a = ap.parse_args()
    if a.command == "gate":
        return gate(a.require or os.environ.get("H3_REQUIRE_PARITY") == "1")
    if a.command == "compare":
        compare(a)
        return 0
    if a.command == "vae":
        if not require_torch():
            print("the VAE check needs torch and diffusers", file=sys.stderr)
            return 1
        return vae_parity(a)
    if a.command == "dit":
        if not require_torch():
            print("the DiT check needs torch and diffusers", file=sys.stderr)
            return 1
        return dit_parity(a)
    if a.command == "convert":
        if not require_torch():
            print("the conversion check needs torch", file=sys.stderr)
            return 1
        return conversion_parity(a)
    if a.command == "te":
        if not require_torch():
            print("the text encoder check needs torch and transformers", file=sys.stderr)
            return 1
        return te_parity(a)
    if not require_torch():
        print("stack parity needs torch (and diffusers for --stack dit)", file=sys.stderr)
        return 1
    depths = [int(x) for x in a.depths.split(",") if x]
    ok = stack_dit(depths, (a.height, a.width, a.frames)) if a.stack == "dit" else stack_te(depths)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""The text encoder's Qwen3 layers in Loom: a resident session per prompt token count."""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
_ABI, _ERR = 1, 4096
_F32P = ctypes.POINTER(ctypes.c_float)
HIDDEN, HEAD_DIM, ROPE_HALF, LAYERS, ROPE_THETA = 5120, 128, 64, 50, 5_000_000.0


class H3TeError(RuntimeError):
    pass


def rope_tables(tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Qwen3-VL's text-only rotary tables (the three mrope axes coincide for text): [tokens][64] f32."""
    inv = ROPE_THETA ** (-torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    ang = torch.arange(tokens, dtype=torch.float32)[:, None] * inv[None]
    return torch.cos(ang).contiguous(), torch.sin(ang).contiguous()


class H3TeBlocks:
    def __init__(self, tokens: int, layers: int = LAYERS, weights: str | Path | None = None, library: str | Path | None = None):
        self.tokens, self.layers = tokens, layers
        weights = Path(weights or ROOT / "build/weights_te"); library = Path(library or ROOT / "build/libh3te.so")
        kernels = ROOT / "build/kernels_te" / f"T{tokens}"
        if not (kernels / "capacity.txt").exists():
            subprocess.run([sys.executable, str(ROOT / "scripts/build_kernels_te.py"), str(tokens)], check=True, capture_output=True)
        native = ctypes.CDLL(str(library))
        native.h3te_abi_version.restype = ctypes.c_uint32
        if native.h3te_abi_version() != _ABI:
            raise H3TeError("ABI mismatch; rebuild with scripts/build_host.sh")
        native.h3te_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_size_t]
        native.h3te_run.argtypes = [ctypes.c_void_p, _F32P, ctypes.c_size_t, _F32P, _F32P, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t]
        native.h3te_profile.argtypes = [ctypes.c_void_p, ctypes.c_int]
        native.h3te_destroy.argtypes = [ctypes.c_void_p]
        self._native = native
        handle = ctypes.c_void_p(); err = ctypes.create_string_buffer(_ERR)
        if native.h3te_create(os.fsencode(weights), os.fsencode(kernels), tokens, layers, ctypes.byref(handle), err, _ERR):
            raise H3TeError(err.value.decode())
        self._handle = handle

    def close(self):
        if getattr(self, "_handle", None):
            self._native.h3te_destroy(self._handle); self._handle = None

    def __del__(self):
        try: self.close()
        except Exception: pass

    def forward(self, x: torch.Tensor, cos: torch.Tensor | None = None, sin: torch.Tensor | None = None) -> torch.Tensor:
        """x [tokens][5120] embeddings -> f32 hidden state after `layers` layers (no final norm)."""
        if cos is None: cos, sin = rope_tables(self.tokens)
        xa = np.ascontiguousarray(x.detach().to(torch.float32).cpu().numpy())
        ca = np.ascontiguousarray(cos.detach().float().cpu().numpy()); sa = np.ascontiguousarray(sin.detach().float().cpu().numpy())
        assert xa.shape == (self.tokens, HIDDEN) and ca.shape == sa.shape == (self.tokens, ROPE_HALF)
        err = ctypes.create_string_buffer(_ERR)
        if self._native.h3te_run(self._handle, xa.ctypes.data_as(_F32P), xa.size, ca.ctypes.data_as(_F32P), sa.ctypes.data_as(_F32P), ca.size, err, _ERR):
            raise H3TeError(err.value.decode())
        return torch.from_numpy(xa.copy())

    def profile(self, enable: bool = True):
        self._native.h3te_profile(self._handle, int(enable))

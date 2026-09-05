"""The video VAE's ViT decoder blocks in Loom: a resident session per clip token count."""
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
HIDDEN, ROPE_HALF = 2048, 24


class H3VaeError(RuntimeError):
    pass


class H3VaeBlocks:
    def __init__(self, tokens: int, layers: int = 36, weights: str | Path | None = None, library: str | Path | None = None):
        self.tokens, self.layers = tokens, layers
        weights = Path(weights or ROOT / "build/weights_vae"); library = Path(library or ROOT / "build/libh3vae.so")
        kernels = ROOT / "build/kernels_vae" / f"T{tokens}"
        if not (kernels / "attention_waves.txt").exists():
            subprocess.run([sys.executable, str(ROOT / "scripts/build_kernels_vae.py"), str(tokens)], check=True, capture_output=True)
        native = ctypes.CDLL(str(library))
        native.h3vae_abi_version.restype = ctypes.c_uint32
        if native.h3vae_abi_version() != _ABI:
            raise H3VaeError("ABI mismatch; rebuild with scripts/build_host.sh")
        native.h3vae_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_size_t]
        native.h3vae_run.argtypes = [ctypes.c_void_p, _F32P, ctypes.c_size_t, _F32P, _F32P, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t]
        native.h3vae_profile.argtypes = [ctypes.c_void_p, ctypes.c_int]
        native.h3vae_destroy.argtypes = [ctypes.c_void_p]
        self._native = native
        handle = ctypes.c_void_p(); err = ctypes.create_string_buffer(_ERR)
        if native.h3vae_create(os.fsencode(weights), os.fsencode(kernels), tokens, layers, ctypes.byref(handle), err, _ERR):
            raise H3VaeError(err.value.decode())
        self._handle = handle

    def close(self):
        if getattr(self, "_handle", None):
            self._native.h3vae_destroy(self._handle); self._handle = None

    def __del__(self):
        try: self.close()
        except Exception: pass

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """x [tokens][2048] -> f32 stream after `layers` blocks; cos/sin [tokens][24] f32."""
        xa = np.ascontiguousarray(x.detach().to(torch.float32).cpu().numpy())
        ca = np.ascontiguousarray(cos.detach().float().cpu().numpy()); sa = np.ascontiguousarray(sin.detach().float().cpu().numpy())
        assert xa.shape == (self.tokens, HIDDEN) and ca.shape == sa.shape == (self.tokens, ROPE_HALF)
        err = ctypes.create_string_buffer(_ERR)
        if self._native.h3vae_run(self._handle, xa.ctypes.data_as(_F32P), xa.size, ca.ctypes.data_as(_F32P), sa.ctypes.data_as(_F32P), ca.size, err, _ERR):
            raise H3VaeError(err.value.decode())
        return torch.from_numpy(xa.copy())

    def profile(self, enable: bool = True):
        self._native.h3vae_profile(self._handle, int(enable))

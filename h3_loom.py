"""The MiniMax H3 transformer blocks in Loom, behind the same shape of API as the sibling
repos: a resident session per packed-sequence length, one ctypes call per forward."""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "reference"))
import h3_ref as R

_ABI, _CLASSES = 1, 12
_ERR = 4096
_U16P, _F32P, _I32P = ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int32)


class H3Error(RuntimeError):
    pass


def mods_table(ref: R.H3Ref, temb: torch.Tensor, layers: int) -> torch.Tensor:
    """The ABI's per-layer modulation tables from the reference's block_mods:
    [layers][6 * 12][5376] f32 = per layer (scale_msa, shift_msa) x 12 classes, gate_msa x 12,
    (scale_mlp, shift_mlp) x 12, gate_mlp x 12. Classes beyond the distinct timesteps are zero."""
    out = torch.zeros(layers, 6 * _CLASSES, R.HIDDEN, dtype=torch.float32)
    for i in range(layers):
        m = ref.block_mods(i, temb).float().cpu()                  # [R, 6, hidden]: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        r = m.shape[0]
        t = out[i]
        t[0:2 * r].view(r, 2, R.HIDDEN)[:, 0] = m[:, 1]; t[0:2 * r].view(r, 2, R.HIDDEN)[:, 1] = m[:, 0]
        t[2 * _CLASSES:2 * _CLASSES + r] = m[:, 2]
        base = 3 * _CLASSES
        t[base:base + 2 * r].view(r, 2, R.HIDDEN)[:, 0] = m[:, 4]; t[base:base + 2 * r].view(r, 2, R.HIDDEN)[:, 1] = m[:, 3]
        t[5 * _CLASSES:5 * _CLASSES + r] = m[:, 5]
    return out


class H3Blocks:
    def __init__(self, tokens: int, layers: int = 50, weights: str | Path | None = None, library: str | Path | None = None):
        self.tokens, self.layers = tokens, layers
        weights = Path(weights or ROOT / "build/weights")
        library = Path(library or ROOT / "build/libh3.so")
        kernels = ROOT / "build/kernels" / f"T{tokens}"
        if not (kernels / "attention_waves.txt").exists():
            subprocess.run([sys.executable, str(ROOT / "scripts/build_kernels.py"), str(tokens)], check=True, capture_output=True)
        native = ctypes.CDLL(str(library))
        native.h3_abi_version.restype = ctypes.c_uint32
        if native.h3_abi_version() != _ABI:
            raise H3Error("ABI mismatch; rebuild with scripts/build_host.sh")
        native.h3_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_size_t]
        native.h3_run.argtypes = [ctypes.c_void_p, _F32P, ctypes.c_size_t, _I32P, ctypes.c_size_t, _F32P, ctypes.c_size_t, _F32P, _F32P, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t]
        native.h3_profile.argtypes = [ctypes.c_void_p, ctypes.c_int]
        native.h3_destroy.argtypes = [ctypes.c_void_p]
        self._native = native
        handle = ctypes.c_void_p(); err = ctypes.create_string_buffer(_ERR)
        if native.h3_create(os.fsencode(weights), os.fsencode(kernels), tokens, layers, ctypes.byref(handle), err, _ERR):
            raise H3Error(err.value.decode())
        self._handle = handle

    def close(self):
        if getattr(self, "_handle", None):
            self._native.h3_destroy(self._handle); self._handle = None

    def __del__(self):
        try: self.close()
        except Exception: pass

    def forward(self, x: torch.Tensor, cls: torch.Tensor, mods: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """x [tokens][5376] (any float dtype) -> f32 residual stream after `layers` blocks.
        cls [tokens] int (AdaLN row class), mods [layers][72][5376] f32 (mods_table), cos/sin [tokens][48] f32."""
        timing = os.environ.get("H3_TIMING") == "1"
        t0 = time.time()
        xa = np.ascontiguousarray(x.detach().to(torch.float32).cpu().numpy())
        ca_ = np.ascontiguousarray(cls.detach().to(torch.int32).cpu().numpy())
        ma = np.ascontiguousarray(mods.detach().float().cpu().numpy()[: self.layers])
        ca = np.ascontiguousarray(cos.detach().float().cpu().numpy()); sa = np.ascontiguousarray(sin.detach().float().cpu().numpy())
        t1 = time.time()
        assert xa.shape == (self.tokens, R.HIDDEN) and ca_.shape == (self.tokens,) and ma.shape == (self.layers, 6 * _CLASSES, R.HIDDEN) and ca.shape == sa.shape == (self.tokens, 48)
        err = ctypes.create_string_buffer(_ERR)
        rc = self._native.h3_run(self._handle, xa.ctypes.data_as(_F32P), xa.size, ca_.ctypes.data_as(_I32P), ca_.size, ma.ctypes.data_as(_F32P), ma.size,
                                 ca.ctypes.data_as(_F32P), sa.ctypes.data_as(_F32P), ca.size, err, _ERR)
        if rc:
            raise H3Error(err.value.decode())
        t2 = time.time()
        out = torch.from_numpy(xa.copy())
        if timing:
            print(f"  loom wrapper: to-host {t1 - t0:.3f} s, h3_run {t2 - t1:.3f} s, from-host {time.time() - t2:.3f} s", file=sys.stderr)
        return out

    def profile(self, enable: bool = True):
        self._native.h3_profile(self._handle, int(enable))

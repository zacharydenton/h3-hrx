"""ctypes wrapper of libh3pipe.so: the whole pipeline (prompt ids -> latents -> frames / samples)."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
_ABI, _ERR = 2, 4096
_F32P, _U8P, _I32P = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_int32)
PROGRESS = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_double)


class Config(ctypes.Structure):
    _fields_ = [("glue_dir", ctypes.c_char_p), ("blocks_dir", ctypes.c_char_p), ("te_dir", ctypes.c_char_p), ("vae_dir", ctypes.c_char_p),
                ("kernel_sources", ctypes.c_char_p), ("cache_dir", ctypes.c_char_p), ("loom_compile", ctypes.c_char_p), ("vae_bits", ctypes.c_int)]


class Params(ctypes.Structure):
    _fields_ = [("height", ctypes.c_int), ("width", ctypes.c_int), ("frames", ctypes.c_int), ("steps", ctypes.c_int), ("seed", ctypes.c_uint64),
                ("video_shift", ctypes.c_float), ("audio_shift", ctypes.c_float), ("cache_threshold", ctypes.c_float)]


class Shape(ctypes.Structure):
    _fields_ = [("frames", ctypes.c_int), ("latent_t", ctypes.c_int), ("lat_h", ctypes.c_int), ("lat_w", ctypes.c_int), ("audio_t", ctypes.c_int), ("text_rows_max", ctypes.c_int)]


class H3PipeError(RuntimeError):
    pass


def default_loom_compile() -> str:
    if os.environ.get("LOOM_COMPILE"): return os.environ["LOOM_COMPILE"]
    return str(Path.home() / "code/hrx-system/build-cuda/loom/src/loom/tools/loom-compile/loom-compile")


class H3Pipe:
    def __init__(self, glue=None, blocks=None, te=None, vae=None, vae_bits=8, cache=None, library=None):
        native = ctypes.CDLL(str(library or os.environ.get("H3PIPE_LIB") or ROOT / "build/libh3pipe.so"))   # H3PIPE_LIB=build/libh3pipe_hrx.so: the libhrx build
        native.h3pipe_abi_version.restype = ctypes.c_uint32
        if native.h3pipe_abi_version() != _ABI: raise H3PipeError("ABI mismatch; rebuild with scripts/build_host.sh")
        native.h3pipe_create.argtypes = [ctypes.POINTER(Config), ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_size_t]
        native.h3pipe_destroy.argtypes = [ctypes.c_void_p]
        native.h3pipe_shape_for.argtypes = [ctypes.POINTER(Params), ctypes.POINTER(Shape)]
        native.h3pipe_text_in.argtypes = [ctypes.c_void_p, _I32P, ctypes.c_int, _F32P, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t]
        native.h3pipe_denoise.argtypes = [ctypes.c_void_p, _I32P, ctypes.c_int, ctypes.POINTER(Params), _F32P, _F32P, _F32P, ctypes.c_size_t, _F32P, ctypes.c_size_t, PROGRESS, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
        native.h3pipe_decode_video.argtypes = [ctypes.c_void_p, ctypes.POINTER(Params), _F32P, ctypes.c_size_t, _U8P, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t]
        native.h3pipe_decode_audio.argtypes = [ctypes.c_void_p, _F32P, ctypes.c_size_t, ctypes.c_int, _F32P, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t]
        self._native = native
        cfg = Config(os.fsencode(glue or ROOT / "build/weights_glue"), os.fsencode(blocks or ROOT / "build/weights_gptq"), os.fsencode(te or ROOT / "build/weights_te"),
                     os.fsencode(vae or ROOT / ("build/weights_vae_i8" if vae_bits == 8 else "build/weights_vae_gptq")), os.fsencode(ROOT / "kernels"),
                     os.fsencode(cache or ROOT / "build/kernel_cache"), os.fsencode(default_loom_compile()), vae_bits)
        handle = ctypes.c_void_p(); err = ctypes.create_string_buffer(_ERR)
        if native.h3pipe_create(ctypes.byref(cfg), ctypes.byref(handle), err, _ERR): raise H3PipeError(err.value.decode())
        self._handle = handle

    def close(self):
        if getattr(self, "_handle", None): self._native.h3pipe_destroy(self._handle); self._handle = None

    def __del__(self):
        try: self.close()
        except Exception: pass

    @staticmethod
    def params(height=480, width=864, frames=124, steps=50, seed=0, video_shift=0.0, audio_shift=0.0, cache_threshold=0.0) -> Params:
        return Params(height, width, frames, steps, seed, video_shift, audio_shift, cache_threshold)

    def shape(self, p: Params) -> Shape:
        s = Shape(); self._native.h3pipe_shape_for(ctypes.byref(p), ctypes.byref(s)); return s

    def text_in(self, ids) -> np.ndarray:
        ids = np.ascontiguousarray(np.asarray(ids, dtype=np.int32)); out = np.zeros((ids.size, 5376), np.float32); err = ctypes.create_string_buffer(_ERR)
        if self._native.h3pipe_text_in(self._handle, ids.ctypes.data_as(_I32P), ids.size, out.ctypes.data_as(_F32P), out.size, err, _ERR): raise H3PipeError(err.value.decode())
        return out

    def denoise(self, ids, p: Params, noise_video=None, noise_audio=None, progress=None):
        """-> (video latents [24][T][H][W], audio latents [2][32][audio_t]) in model space."""
        ids = np.ascontiguousarray(np.asarray(ids, dtype=np.int32)); s = self.shape(p)
        video = np.zeros((24, s.latent_t, s.lat_h, s.lat_w), np.float32); audio = np.zeros((2, 32, s.audio_t), np.float32)
        nv = None if noise_video is None else np.ascontiguousarray(np.asarray(noise_video, dtype=np.float32).reshape(video.shape))
        na = None if noise_audio is None else np.ascontiguousarray(np.asarray(noise_audio, dtype=np.float32).reshape(audio.shape))
        cb = PROGRESS(lambda user, step, steps, sec: int(bool(progress(step, steps, sec))) if progress else 0)
        err = ctypes.create_string_buffer(_ERR)
        rc = self._native.h3pipe_denoise(self._handle, ids.ctypes.data_as(_I32P), ids.size, ctypes.byref(p),
                                          None if nv is None else nv.ctypes.data_as(_F32P), None if na is None else na.ctypes.data_as(_F32P),
                                          video.ctypes.data_as(_F32P), video.size, audio.ctypes.data_as(_F32P), audio.size, cb, None, err, _ERR)
        if rc: raise H3PipeError(err.value.decode())
        return video, audio

    def decode_video(self, p: Params, video) -> np.ndarray:
        s = self.shape(p); v = np.ascontiguousarray(np.asarray(video, dtype=np.float32)); frames = np.zeros((s.frames, p.height, p.width, 3), np.uint8); err = ctypes.create_string_buffer(_ERR)
        if self._native.h3pipe_decode_video(self._handle, ctypes.byref(p), v.ctypes.data_as(_F32P), v.size, frames.ctypes.data_as(_U8P), frames.size, err, _ERR): raise H3PipeError(err.value.decode())
        return frames

    def decode_audio(self, audio) -> np.ndarray:
        a = np.ascontiguousarray(np.asarray(audio, dtype=np.float32)); audio_t = a.shape[-1]; samples = np.zeros((2, audio_t * 800), np.float32); err = ctypes.create_string_buffer(_ERR)
        if self._native.h3pipe_decode_audio(self._handle, a.ctypes.data_as(_F32P), a.size, audio_t, samples.ctypes.data_as(_F32P), samples.size, err, _ERR): raise H3PipeError(err.value.decode())
        return samples

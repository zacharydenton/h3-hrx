"""ctypes wrapper of libh3pipe.so: the whole pipeline (prompt ids -> latents -> frames / samples)."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
_ABI, _ERR = 6, 4096
_F32P, _U8P, _I32P = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_int32)
PROGRESS = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_double)


class Config(ctypes.Structure):
    _fields_ = [("glue_dir", ctypes.c_char_p), ("blocks_dir", ctypes.c_char_p), ("te_dir", ctypes.c_char_p), ("vae_dir", ctypes.c_char_p),
                ("kernel_sources", ctypes.c_char_p), ("cache_dir", ctypes.c_char_p), ("loom_compile", ctypes.c_char_p), ("vae_bits", ctypes.c_int), ("aenc_dir", ctypes.c_char_p), ("vision_dir", ctypes.c_char_p), ("venc_dir", ctypes.c_char_p), ("attn_qk_bits", ctypes.c_int)]


class Ref(ctypes.Structure):
    _fields_ = [("kind", ctypes.c_int), ("video_latent", _F32P), ("latent_t", ctypes.c_int), ("lat_h", ctypes.c_int), ("lat_w", ctypes.c_int), ("audio_latent", _F32P), ("audio_t", ctypes.c_int), ("pixels", _F32P), ("height", ctypes.c_int), ("width", ctypes.c_int)]


class Keyframe(ctypes.Structure):
    _fields_ = [("frame_index", ctypes.c_int), ("video_latent", _F32P), ("pixels", _F32P), ("height", ctypes.c_int), ("width", ctypes.c_int), ("audio_latent", _F32P), ("audio_t", ctypes.c_int)]


class Params(ctypes.Structure):
    _fields_ = [("height", ctypes.c_int), ("width", ctypes.c_int), ("frames", ctypes.c_int), ("steps", ctypes.c_int), ("seed", ctypes.c_uint64),
                ("video_shift", ctypes.c_float), ("audio_shift", ctypes.c_float), ("sampler", ctypes.c_int), ("cache_threshold", ctypes.c_float)]


class Shape(ctypes.Structure):
    _fields_ = [("frames", ctypes.c_int), ("latent_t", ctypes.c_int), ("lat_h", ctypes.c_int), ("lat_w", ctypes.c_int), ("audio_t", ctypes.c_int), ("text_rows_max", ctypes.c_int)]


class H3PipeError(RuntimeError):
    pass


def default_loom_compile() -> str:
    if os.environ.get("LOOM_COMPILE"): return os.environ["LOOM_COMPILE"]
    return str(Path.home() / "code/hrx-system/build-cuda/loom/src/loom/tools/loom-compile/loom-compile")


class H3Pipe:
    def __init__(self, glue=None, blocks=None, te=None, vae=None, vae_bits=8, cache=None, library=None, aenc=None, vision=None, venc=None, attn="i4"):
        """blocks: build/weights_i8 (int8 rows, ComfyUI parity) or build/weights_gptq (int4, the fast path); attn: "f16", "i8" or "i4" QK^T. Both int4 choices ghost conditioned clips (docs/notes.md)."""
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
        native.h3pipe_denoise_refs.argtypes = [ctypes.c_void_p, _I32P, ctypes.c_int, ctypes.POINTER(Params), ctypes.POINTER(Keyframe), ctypes.c_int, ctypes.POINTER(Ref), ctypes.c_int, _F32P, _F32P, _F32P, ctypes.c_size_t, _F32P, ctypes.c_size_t, PROGRESS, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
        native.h3pipe_encode_video.argtypes = [ctypes.c_void_p, _F32P, ctypes.c_int, ctypes.c_int, ctypes.c_int, _F32P, ctypes.c_size_t, ctypes.POINTER(ctypes.c_int), ctypes.c_char_p, ctypes.c_size_t]
        native.h3pipe_vision_embed.argtypes = [ctypes.c_void_p, _F32P, ctypes.c_int, ctypes.c_int, _F32P, ctypes.c_size_t, _F32P, ctypes.c_size_t, ctypes.POINTER(ctypes.c_int), ctypes.c_char_p, ctypes.c_size_t]
        native.h3pipe_encode_audio.argtypes = [ctypes.c_void_p, _F32P, ctypes.c_int, _F32P, ctypes.c_size_t, ctypes.POINTER(ctypes.c_int), ctypes.c_char_p, ctypes.c_size_t]
        self._native = native
        cfg = Config(os.fsencode(glue or ROOT / "build/weights_glue"), os.fsencode(blocks or ROOT / "build/weights_gptq"), os.fsencode(te or ROOT / "build/weights_te"),
                     os.fsencode(vae or ROOT / ("build/weights_vae_i8" if vae_bits == 8 else "build/weights_vae_gptq")), os.fsencode(ROOT / "kernels"),
                     os.fsencode(cache or ROOT / "build/kernel_cache"), os.fsencode(default_loom_compile()), vae_bits, os.fsencode(aenc or ROOT / "build/weights_aenc"), os.fsencode(vision or ROOT / "build/weights_vision"), os.fsencode(venc or ROOT / "build/weights_venc"), {"i4": 4, "i8": 8, "f16": 16}[attn])
        handle = ctypes.c_void_p(); err = ctypes.create_string_buffer(_ERR)
        if native.h3pipe_create(ctypes.byref(cfg), ctypes.byref(handle), err, _ERR): raise H3PipeError(err.value.decode())
        self._handle = handle

    def close(self):
        if getattr(self, "_handle", None): self._native.h3pipe_destroy(self._handle); self._handle = None

    def __del__(self):
        try: self.close()
        except Exception: pass

    @staticmethod
    def params(height=480, width=864, frames=124, steps=31, seed=0, video_shift=0.0, audio_shift=0.0, cache_threshold=0.0, sampler="res_multistep") -> Params:
        """steps = sigma grid points (steps - 1 evaluations); the defaults are res_multistep on the simple schedule with 30 evaluations (ComfyUI's stock workflows use 20)."""
        return Params(height, width, frames, steps, seed, video_shift, audio_shift, {"euler": 0, "res_multistep": 1}[sampler], cache_threshold)

    def shape(self, p: Params) -> Shape:
        s = Shape(); self._native.h3pipe_shape_for(ctypes.byref(p), ctypes.byref(s)); return s

    def text_in(self, ids) -> np.ndarray:
        ids = np.ascontiguousarray(np.asarray(ids, dtype=np.int32)); out = np.zeros((ids.size, 5376), np.float32); err = ctypes.create_string_buffer(_ERR)
        if self._native.h3pipe_text_in(self._handle, ids.ctypes.data_as(_I32P), ids.size, out.ctypes.data_as(_F32P), out.size, err, _ERR): raise H3PipeError(err.value.decode())
        return out

    def denoise(self, ids, p: Params, noise_video=None, noise_audio=None, progress=None, refs=None, keyframes=None):
        """-> (video latents [24][T][H][W], audio latents [2][32][audio_t]) in model space. refs: list of dicts in presentation
        order, {"kind": "image"|"audio"|"video", "video": [24][T][h][w] latents, "audio": [2][32][audio_t] latents} (ref2va)."""
        ids = np.ascontiguousarray(np.asarray(ids, dtype=np.int32)); s = self.shape(p)
        video = np.zeros((24, s.latent_t, s.lat_h, s.lat_w), np.float32); audio = np.zeros((2, 32, s.audio_t), np.float32)
        nv = None if noise_video is None else np.ascontiguousarray(np.asarray(noise_video, dtype=np.float32).reshape(video.shape))
        na = None if noise_audio is None else np.ascontiguousarray(np.asarray(noise_audio, dtype=np.float32).reshape(audio.shape))
        cb = PROGRESS(lambda user, step, steps, sec: int(bool(progress(step, steps, sec))) if progress else 0)
        err = ctypes.create_string_buffer(_ERR)
        keep = []; rarr = (Ref * max(1, len(refs or [])))()
        for i, r in enumerate(refs or []):
            kind = {"image": 0, "audio": 1, "video": 2}[r["kind"]]; rarr[i].kind = kind
            if kind != 1:
                v = np.ascontiguousarray(np.asarray(r["video"], dtype=np.float32)); assert v.ndim == 4 and v.shape[0] == 24, v.shape; keep.append(v)
                rarr[i].video_latent = v.ctypes.data_as(_F32P); rarr[i].latent_t, rarr[i].lat_h, rarr[i].lat_w = int(v.shape[1]), int(v.shape[2]), int(v.shape[3])
            if kind == 0 and r.get("pixels") is not None:
                px = np.ascontiguousarray(np.asarray(r["pixels"], dtype=np.float32)); assert px.ndim == 3 and px.shape[2] == 3, px.shape; keep.append(px)
                rarr[i].pixels = px.ctypes.data_as(_F32P); rarr[i].height, rarr[i].width = int(px.shape[0]), int(px.shape[1])
            if kind != 0 and r.get("audio") is not None:
                a = np.ascontiguousarray(np.asarray(r["audio"], dtype=np.float32)); assert a.ndim == 3 and a.shape[:2] == (2, 32), a.shape; keep.append(a)
                rarr[i].audio_latent = a.ctypes.data_as(_F32P); rarr[i].audio_t = int(a.shape[2])
        karr = (Keyframe * max(1, len(keyframes or [])))()
        for i, k in enumerate(keyframes or []):
            karr[i].frame_index = int(k["frame_index"])
            v = np.ascontiguousarray(np.asarray(k["video"], dtype=np.float32))
            expected = (24, 1, s.lat_h, s.lat_w)
            if v.shape != expected: raise ValueError(f"keyframe video must have shape {expected}, got {v.shape}")
            keep.append(v); karr[i].video_latent = v.ctypes.data_as(_F32P)
            if k.get("pixels") is not None:
                px = np.ascontiguousarray(np.asarray(k["pixels"], dtype=np.float32))
                if px.ndim != 3 or px.shape[2] != 3: raise ValueError("keyframe pixels must have shape [H][W][3]")
                keep.append(px); karr[i].pixels = px.ctypes.data_as(_F32P); karr[i].height, karr[i].width = int(px.shape[0]), int(px.shape[1])
            if k.get("audio") is not None:
                a = np.ascontiguousarray(np.asarray(k["audio"], dtype=np.float32))
                if a.ndim != 3 or a.shape[:2] != (2, 32) or a.shape[2] < 1: raise ValueError("keyframe audio must have shape [2][32][audio_t >= 1]")
                keep.append(a); karr[i].audio_latent = a.ctypes.data_as(_F32P); karr[i].audio_t = int(a.shape[2])
        if refs or keyframes:
            rc = self._native.h3pipe_denoise_refs(self._handle, ids.ctypes.data_as(_I32P), ids.size, ctypes.byref(p), karr, len(keyframes or []), rarr, len(refs or []),
                                                   None if nv is None else nv.ctypes.data_as(_F32P), None if na is None else na.ctypes.data_as(_F32P),
                                                   video.ctypes.data_as(_F32P), video.size, audio.ctypes.data_as(_F32P), audio.size, cb, None, err, _ERR)
        else:
            rc = self._native.h3pipe_denoise(self._handle, ids.ctypes.data_as(_I32P), ids.size, ctypes.byref(p),
                                              None if nv is None else nv.ctypes.data_as(_F32P), None if na is None else na.ctypes.data_as(_F32P),
                                              video.ctypes.data_as(_F32P), video.size, audio.ctypes.data_as(_F32P), audio.size, cb, None, err, _ERR)
        if rc: raise H3PipeError(err.value.decode())
        return video, audio

    def decode_video(self, p: Params, video) -> np.ndarray:
        s = self.shape(p); v = np.ascontiguousarray(np.asarray(video, dtype=np.float32)); frames = np.zeros((s.frames, p.height, p.width, 3), np.uint8); err = ctypes.create_string_buffer(_ERR)
        if self._native.h3pipe_decode_video(self._handle, ctypes.byref(p), v.ctypes.data_as(_F32P), v.size, frames.ctypes.data_as(_U8P), frames.size, err, _ERR): raise H3PipeError(err.value.decode())
        return frames

    def encode_video(self, pixels) -> np.ndarray:
        """pixels [frames][H][W][3] (or [H][W][3]) in [0, 1] -> model-space video latents [24][latent_t][H/16][W/16] from the VAE encoder in Loom."""
        x = np.ascontiguousarray(np.asarray(pixels, dtype=np.float32)); x = x[None] if x.ndim == 3 else x; F, H, W = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
        TL = 1 if F == 1 else 5 * ((F + 16) // 17) - 3; z = np.zeros((24, TL, H // 16, W // 16), np.float32); t = ctypes.c_int(0); err = ctypes.create_string_buffer(_ERR)
        if self._native.h3pipe_encode_video(self._handle, x.ctypes.data_as(_F32P), F, H, W, z.ctypes.data_as(_F32P), z.size, ctypes.byref(t), err, _ERR): raise H3PipeError(err.value.decode())
        return z[:, :t.value]

    def vision_embed(self, pixels):
        """pixels [H][W][3] in [0, 1] (H, W multiples of 32) -> (merged [tokens][5120], deepstack [3][tokens][5120]) from the vision tower."""
        x = np.ascontiguousarray(np.asarray(pixels, dtype=np.float32)); H, W = int(x.shape[0]), int(x.shape[1]); m = (H // 32) * (W // 32)
        merged = np.zeros((m, 5120), np.float32); ds = np.zeros((3, m, 5120), np.float32); t = ctypes.c_int(0); err = ctypes.create_string_buffer(_ERR)
        if self._native.h3pipe_vision_embed(self._handle, x.ctypes.data_as(_F32P), H, W, merged.ctypes.data_as(_F32P), merged.size, ds.ctypes.data_as(_F32P), ds.size, ctypes.byref(t), err, _ERR): raise H3PipeError(err.value.decode())
        return merged, ds

    def encode_audio(self, samples) -> np.ndarray:
        """Stereo float samples [2][n] at 32 kHz -> model-space audio latents [2][32][ceil(n / 800)] (the audio VAE's encoder in Loom)."""
        x = np.ascontiguousarray(np.asarray(samples, dtype=np.float32)); assert x.ndim == 2 and x.shape[0] == 2, x.shape
        n = x.shape[1]; T = (n + 799) // 800; z = np.zeros((2, 32, T), np.float32); t_out = ctypes.c_int(0); err = ctypes.create_string_buffer(_ERR)
        if self._native.h3pipe_encode_audio(self._handle, x.ctypes.data_as(_F32P), n, z.ctypes.data_as(_F32P), z.size, ctypes.byref(t_out), err, _ERR): raise H3PipeError(err.value.decode())
        return z[:, :, :t_out.value]

    def decode_audio(self, audio) -> np.ndarray:
        a = np.ascontiguousarray(np.asarray(audio, dtype=np.float32)); audio_t = a.shape[-1]; samples = np.zeros((2, audio_t * 800), np.float32); err = ctypes.create_string_buffer(_ERR)
        if self._native.h3pipe_decode_audio(self._handle, a.ctypes.data_as(_F32P), a.size, audio_t, samples.ctypes.data_as(_F32P), samples.size, err, _ERR): raise H3PipeError(err.value.decode())
        return samples

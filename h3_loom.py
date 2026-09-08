"""ctypes wrapper of libh3.so: the whole pipeline (prompt ids -> latents -> frames / samples)."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
_ABI = 8
_F32P, _U8P, _I32P = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_int32)
PROGRESS = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_double)


class Config(ctypes.Structure):
    _fields_ = [("dit_file", ctypes.c_char_p), ("te_file", ctypes.c_char_p), ("video_vae_file", ctypes.c_char_p), ("audio_vae_file", ctypes.c_char_p),
                ("kernel_sources", ctypes.c_char_p), ("cache_dir", ctypes.c_char_p), ("loom_compile", ctypes.c_char_p), ("attn_qk_bits", ctypes.c_int)]


MODELS = Path(os.environ.get("H3_MODELS") or Path.home() / "comfy-models")   # ComfyUI's models directory: the four checkpoints, read as they are
DIT = MODELS / "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
REF2VA = MODELS / "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"
TE = MODELS / "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
VIDEO_VAE = MODELS / "vae/minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = MODELS / "vae/minimax_h3_audio_vae_fp32.safetensors"


class Ref(ctypes.Structure):
    _fields_ = [("kind", ctypes.c_int), ("video_latent", _F32P), ("latent_t", ctypes.c_int), ("lat_h", ctypes.c_int), ("lat_w", ctypes.c_int), ("audio_latent", _F32P), ("audio_t", ctypes.c_int), ("pixels", _F32P), ("height", ctypes.c_int), ("width", ctypes.c_int)]


class Keyframe(ctypes.Structure):
    _fields_ = [("frame_index", ctypes.c_int), ("video_latent", _F32P), ("pixels", _F32P), ("height", ctypes.c_int), ("width", ctypes.c_int), ("audio_latent", _F32P), ("audio_t", ctypes.c_int)]


class Params(ctypes.Structure):
    _fields_ = [("height", ctypes.c_int), ("width", ctypes.c_int), ("frames", ctypes.c_int), ("steps", ctypes.c_int), ("seed", ctypes.c_uint64),
                ("video_shift", ctypes.c_float), ("audio_shift", ctypes.c_float), ("sampler", ctypes.c_int), ("cache_threshold", ctypes.c_float)]


class Shape(ctypes.Structure):
    _fields_ = [("frames", ctypes.c_int), ("latent_t", ctypes.c_int), ("lat_h", ctypes.c_int), ("lat_w", ctypes.c_int), ("audio_t", ctypes.c_int), ("text_rows_max", ctypes.c_int)]


class H3Error(RuntimeError):
    pass


def _f32(value, name: str, ndim, last=None, first=None) -> np.ndarray:
    """A contiguous float32 array whose rank (one of `ndim`) and leading/trailing extents match, or a ValueError naming it.
    Every array crosses the C boundary as a raw pointer, so the shape is checked here explicitly (never by assert: python -O drops those)."""
    x = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
    dims = (ndim,) if isinstance(ndim, int) else tuple(ndim)
    if x.ndim not in dims: raise ValueError(f"{name} must have {' or '.join(map(str, dims))} dimensions, got shape {x.shape}")
    if first is not None and tuple(x.shape[:len(first)]) != tuple(first): raise ValueError(f"{name} must start with shape {tuple(first)}, got {x.shape}")
    if last is not None and x.shape[-1] != last: raise ValueError(f"{name} must end in {last} channels, got shape {x.shape}")
    if x.size == 0: raise ValueError(f"{name} is empty: shape {x.shape}")
    return x


def _canvas(name: str, height: int, width: int, limit: int = 2048):
    if height % 32 or width % 32 or height < 32 or width < 32 or height > limit or width > limit:
        raise ValueError(f"{name} height and width must be multiples of 32 up to {limit}, got {height}x{width}")


def default_loom_compile() -> str:
    if os.environ.get("LOOM_COMPILE"): return os.environ["LOOM_COMPILE"]
    return str(Path.home() / "code/hrx-system/build-cuda/loom/src/loom/tools/loom-compile/loom-compile")


def _last_error(lib) -> str:
    """The library's message for the last failing call on this thread."""
    lib.h3_last_error.restype = ctypes.c_char_p
    return (lib.h3_last_error() or b"").decode(errors="replace") or "unknown error"


class H3:
    def __init__(self, dit=None, te=None, video_vae=None, audio_vae=None, cache=None, library=None, attn="i8"):
        """The four ComfyUI checkpoints as they are (h3_loom.DIT / REF2VA / TE / VIDEO_VAE / AUDIO_VAE are the defaults under
        $H3_MODELS or ~/comfy-models); attn: the DiT attention's QK^T operands, "f16", "i8" (the parity path) or "i4" (int4 ghosts conditioned clips, docs/archive/notes.md)."""
        native = ctypes.CDLL(str(library or os.environ.get("H3_LIB") or ROOT / "build/libh3.so"))   # H3_LIB=<path> selects another build
        native.h3_abi_version.restype = ctypes.c_uint32
        native.h3_last_error.restype = ctypes.c_char_p
        if native.h3_abi_version() != _ABI: raise H3Error("ABI mismatch; rebuild with scripts/build_host.sh")
        native.h3_create.argtypes = [ctypes.POINTER(Config), ctypes.POINTER(ctypes.c_void_p)]
        native.h3_destroy.argtypes = [ctypes.c_void_p]
        native.h3_shape_for.argtypes = [ctypes.POINTER(Params), ctypes.POINTER(Shape)]
        native.h3_text_in.argtypes = [ctypes.c_void_p, _I32P, ctypes.c_int, _F32P, ctypes.c_size_t]
        native.h3_decode_video.argtypes = [ctypes.c_void_p, ctypes.POINTER(Params), _F32P, ctypes.c_size_t, _U8P, ctypes.c_size_t]
        native.h3_decode_audio.argtypes = [ctypes.c_void_p, _F32P, ctypes.c_size_t, ctypes.c_int, _F32P, ctypes.c_size_t]
        native.h3_denoise.argtypes = [ctypes.c_void_p, _I32P, ctypes.c_int, ctypes.POINTER(Params), ctypes.POINTER(Keyframe), ctypes.c_int, ctypes.POINTER(Ref), ctypes.c_int, _F32P, _F32P, _F32P, ctypes.c_size_t, _F32P, ctypes.c_size_t, PROGRESS, ctypes.c_void_p]
        native.h3_encode_video.argtypes = [ctypes.c_void_p, _F32P, ctypes.c_int, ctypes.c_int, ctypes.c_int, _F32P, ctypes.c_size_t, ctypes.POINTER(ctypes.c_int)]
        native.h3_vision_embed.argtypes = [ctypes.c_void_p, _F32P, ctypes.c_int, ctypes.c_int, _F32P, ctypes.c_size_t, _F32P, ctypes.c_size_t, ctypes.POINTER(ctypes.c_int)]
        native.h3_encode_audio.argtypes = [ctypes.c_void_p, _F32P, ctypes.c_int, _F32P, ctypes.c_size_t, ctypes.POINTER(ctypes.c_int)]
        self._native = native
        cfg = Config(os.fsencode(dit or DIT), os.fsencode(te or TE), os.fsencode(video_vae or VIDEO_VAE), os.fsencode(audio_vae or AUDIO_VAE),
                     os.fsencode(ROOT / "h3/kernels"), os.fsencode(cache or ROOT / "build/kernel_cache"), os.fsencode(default_loom_compile()), {"i4": 4, "i8": 8, "f16": 16}[attn])
        handle = ctypes.c_void_p()
        if native.h3_create(ctypes.byref(cfg), ctypes.byref(handle)): raise H3Error(_last_error(native))
        self._handle = handle

    def close(self):
        if getattr(self, "_handle", None): self._native.h3_destroy(self._handle); self._handle = None

    def __del__(self):
        try: self.close()
        except Exception: pass

    @staticmethod
    def params(height=480, width=864, frames=124, steps=31, seed=0, video_shift=0.0, audio_shift=0.0, cache_threshold=0.0, sampler="res_multistep") -> Params:
        """steps = sigma grid points (steps - 1 evaluations); the defaults are res_multistep on the simple schedule with 30 evaluations (ComfyUI's stock workflows use 20)."""
        return Params(height, width, frames, steps, seed, video_shift, audio_shift, {"euler": 0, "res_multistep": 1}[sampler], cache_threshold)

    def shape(self, p: Params) -> Shape:
        s = Shape()
        if self._native.h3_shape_for(ctypes.byref(p), ctypes.byref(s)): raise ValueError(f"invalid parameters: height and width must be multiples of 32, frames >= 1 (got {p.height}x{p.width}, {p.frames} frames)")
        return s

    def text_in(self, ids) -> np.ndarray:
        ids = np.ascontiguousarray(np.asarray(ids, dtype=np.int32)); out = np.zeros((ids.size, 5376), np.float32)
        if self._native.h3_text_in(self._handle, ids.ctypes.data_as(_I32P), ids.size, out.ctypes.data_as(_F32P), out.size): raise H3Error(_last_error(self._native))
        return out

    def denoise(self, ids, p: Params, noise_video=None, noise_audio=None, progress=None, refs=None, keyframes=None):
        """-> (video latents [24][T][H][W], audio latents [2][32][audio_t]) in model space. refs: list of dicts in presentation
        order, {"kind": "image"|"audio"|"video", "video": [24][T][h][w] latents, "audio": [2][32][audio_t] latents} (ref2va)."""
        ids = np.ascontiguousarray(np.asarray(ids, dtype=np.int32)).ravel(); s = self.shape(p)
        if ids.size < 1: raise ValueError("ids must hold at least one token")
        video = np.zeros((24, s.latent_t, s.lat_h, s.lat_w), np.float32); audio = np.zeros((2, 32, s.audio_t), np.float32)
        nv = None if noise_video is None else _f32(noise_video, "noise_video", 4, first=video.shape)
        na = None if noise_audio is None else _f32(noise_audio, "noise_audio", 3, first=audio.shape)
        cb = PROGRESS(lambda user, step, steps, sec: int(bool(progress(step, steps, sec))) if progress else 0)
        keep = []; rarr = (Ref * max(1, len(refs or [])))()
        for i, r in enumerate(refs or []):
            if r.get("kind") not in ("image", "audio", "video"): raise ValueError(f"ref {i} kind must be image, audio or video, got {r.get('kind')!r}")
            kind = {"image": 0, "audio": 1, "video": 2}[r["kind"]]; rarr[i].kind = kind
            if kind != 1:
                v = _f32(r.get("video"), f"ref {i} video latents", 4, first=(24,)); keep.append(v)
                if kind == 0 and v.shape[1] != 1: raise ValueError(f"ref {i} image latents must have shape [24][1][lat_h][lat_w], got {v.shape}")
                if v.shape[2] < 2 or v.shape[3] < 2 or v.shape[2] % 2 or v.shape[3] % 2: raise ValueError(f"ref {i} video latents need even lat_h and lat_w of at least 2, got {v.shape}")
                rarr[i].video_latent = v.ctypes.data_as(_F32P); rarr[i].latent_t, rarr[i].lat_h, rarr[i].lat_w = int(v.shape[1]), int(v.shape[2]), int(v.shape[3])
            if kind == 0 and r.get("pixels") is not None:
                px = _f32(r["pixels"], f"ref {i} pixels", 3, last=3); _canvas(f"ref {i} pixels", px.shape[0], px.shape[1]); keep.append(px)
                if (px.shape[0] // 16, px.shape[1] // 16) != (int(v.shape[2]), int(v.shape[3])): raise ValueError(f"ref {i} pixels {px.shape[:2]} do not match its latents {v.shape}")
                rarr[i].pixels = px.ctypes.data_as(_F32P); rarr[i].height, rarr[i].width = int(px.shape[0]), int(px.shape[1])
            if kind == 1 and r.get("audio") is None: raise ValueError(f"ref {i} audio needs audio latents")
            if kind != 0 and r.get("audio") is not None:
                a = _f32(r["audio"], f"ref {i} audio latents", 3, first=(2, 32)); keep.append(a)
                rarr[i].audio_latent = a.ctypes.data_as(_F32P); rarr[i].audio_t = int(a.shape[2])
        karr = (Keyframe * max(1, len(keyframes or [])))()
        for i, k in enumerate(keyframes or []):
            karr[i].frame_index = int(k["frame_index"])
            v = np.ascontiguousarray(np.asarray(k.get("video"), dtype=np.float32))
            expected = (24, 1, s.lat_h, s.lat_w)
            if v.shape != expected: raise ValueError(f"keyframe video must have shape {expected}, got {v.shape}")
            keep.append(v); karr[i].video_latent = v.ctypes.data_as(_F32P)
            if k.get("pixels") is not None:
                px = np.ascontiguousarray(np.asarray(k["pixels"], dtype=np.float32))
                if px.ndim != 3 or px.shape[2] != 3 or px.shape[0] != p.height or px.shape[1] != p.width: raise ValueError(f"keyframe pixels must have shape [{p.height}][{p.width}][3] (the canvas), got {px.shape}")
                keep.append(px); karr[i].pixels = px.ctypes.data_as(_F32P); karr[i].height, karr[i].width = int(px.shape[0]), int(px.shape[1])
            if k.get("audio") is not None:
                a = np.ascontiguousarray(np.asarray(k["audio"], dtype=np.float32))
                if a.ndim != 3 or a.shape[:2] != (2, 32) or a.shape[2] < 1: raise ValueError("keyframe audio must have shape [2][32][audio_t >= 1]")
                keep.append(a); karr[i].audio_latent = a.ctypes.data_as(_F32P); karr[i].audio_t = int(a.shape[2])
        rc = self._native.h3_denoise(self._handle, ids.ctypes.data_as(_I32P), ids.size, ctypes.byref(p),
                                     karr if keyframes else None, len(keyframes or []),
                                     rarr if refs else None, len(refs or []),
                                     None if nv is None else nv.ctypes.data_as(_F32P), None if na is None else na.ctypes.data_as(_F32P),
                                     video.ctypes.data_as(_F32P), video.size, audio.ctypes.data_as(_F32P), audio.size, cb, None)
        if rc: raise H3Error(_last_error(self._native))
        return video, audio

    def decode_video(self, p: Params, video) -> np.ndarray:
        s = self.shape(p); v = _f32(video, "video latents", 4, first=(24, s.latent_t, s.lat_h, s.lat_w)); frames = np.zeros((s.frames, p.height, p.width, 3), np.uint8)
        if self._native.h3_decode_video(self._handle, ctypes.byref(p), v.ctypes.data_as(_F32P), v.size, frames.ctypes.data_as(_U8P), frames.size): raise H3Error(_last_error(self._native))
        return frames

    def encode_video(self, pixels) -> np.ndarray:
        """pixels [frames][H][W][3] (or [H][W][3]) in [0, 1] -> model-space video latents [24][latent_t][H/16][W/16] from the VAE encoder in Loom."""
        x = _f32(pixels, "pixels", (3, 4), last=3); x = x[None] if x.ndim == 3 else x; F, H, W = int(x.shape[0]), int(x.shape[1]), int(x.shape[2]); _canvas("pixels", H, W)
        TL = 1 if F == 1 else 5 * ((F + 16) // 17) - 3; z = np.zeros((24, TL, H // 16, W // 16), np.float32); t = ctypes.c_int(0)
        if self._native.h3_encode_video(self._handle, x.ctypes.data_as(_F32P), F, H, W, z.ctypes.data_as(_F32P), z.size, ctypes.byref(t)): raise H3Error(_last_error(self._native))
        return z[:, :t.value]

    def vision_embed(self, pixels):
        """pixels [H][W][3] in [0, 1] (H, W multiples of 32) -> (merged [tokens][5120], deepstack [3][tokens][5120]) from the vision tower."""
        x = _f32(pixels, "pixels", 3, last=3); H, W = int(x.shape[0]), int(x.shape[1]); _canvas("pixels", H, W); m = (H // 32) * (W // 32)
        merged = np.zeros((m, 5120), np.float32); ds = np.zeros((3, m, 5120), np.float32); t = ctypes.c_int(0)
        if self._native.h3_vision_embed(self._handle, x.ctypes.data_as(_F32P), H, W, merged.ctypes.data_as(_F32P), merged.size, ds.ctypes.data_as(_F32P), ds.size, ctypes.byref(t)): raise H3Error(_last_error(self._native))
        return merged, ds

    def encode_audio(self, samples) -> np.ndarray:
        """Stereo float samples [2][n] at 32 kHz -> model-space audio latents [2][32][ceil(n / 800)] (the audio VAE's encoder in Loom)."""
        x = _f32(samples, "samples", 2, first=(2,))
        n = x.shape[1]; T = (n + 799) // 800; z = np.zeros((2, 32, T), np.float32); t_out = ctypes.c_int(0)
        if self._native.h3_encode_audio(self._handle, x.ctypes.data_as(_F32P), n, z.ctypes.data_as(_F32P), z.size, ctypes.byref(t_out)): raise H3Error(_last_error(self._native))
        return z[:, :, :t_out.value]

    def decode_audio(self, audio) -> np.ndarray:
        a = _f32(audio, "audio latents", 3, first=(2, 32)); audio_t = a.shape[-1]; samples = np.zeros((2, audio_t * 800), np.float32)
        if self._native.h3_decode_audio(self._handle, a.ctypes.data_as(_F32P), a.size, audio_t, samples.ctypes.data_as(_F32P), samples.size): raise H3Error(_last_error(self._native))
        return samples

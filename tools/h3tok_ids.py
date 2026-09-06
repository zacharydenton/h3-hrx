"""Prompt ids from the C tokenizer in libh3pipe (h3tok), with H3's reference presentation: for each reference in
order images, videos, audio: "<Picture i>: " + <|vision_start|> + one placeholder id per merged vision token +
<|vision_end|>; "<Audio j>: "; "<Video k>: " + per 2-frame block "<t seconds>" + a vision span; then the prompt.
Placeholders are -1 (the host replaces those rows with the vision embeds).
    python3 tools/h3tok_ids.py "<prompt>"  [--images 405] [--audios 1]"""
import ctypes, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
VISION_START, VISION_END = 151652, 151653
_lib = None
def _native():
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(str(os.environ.get("H3PIPE_LIB") or ROOT / "build/libh3pipe.so"))
        _lib.h3tok_create.restype = ctypes.c_void_p; _lib.h3tok_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
        _lib.h3tok_encode.restype = ctypes.c_int; _lib.h3tok_encode.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t]
        err = ctypes.create_string_buffer(1024)
        _lib._tok = _lib.h3tok_create(os.fsencode(Path.home() / "h3-models/tokenizer/tokenizer.json"), err, 1024)
        if not _lib._tok: raise RuntimeError(err.value.decode())
    return _lib
def encode_text(text: str) -> list:
    lib = _native(); buf = (ctypes.c_int32 * 8192)(); n = lib.h3tok_encode(lib._tok, text.encode("utf-8"), buf, 8192)
    if n < 0: raise RuntimeError("h3tok_encode failed")
    return [int(buf[i]) for i in range(min(n, 8192))]
def encode_presentation(prompt: str, images=(), audios: int = 0, videos=()) -> list:
    """images: merged vision token counts per reference image; videos: lists of (token_count, timestamp) per block."""
    ids = []
    for i, n in enumerate(images):
        ids += encode_text("<Picture %d>: " % (i + 1)) + [VISION_START] + [-1] * int(n) + [VISION_END]
    for k, blocks in enumerate(videos):
        ids += encode_text("<Video %d>: " % (k + 1))
        for n, ts in blocks: ids += encode_text("<%.1f seconds>" % ts) + [VISION_START] + [-1] * int(n) + [VISION_END]
    for j in range(audios): ids += encode_text("<Audio %d>: " % (j + 1))
    return ids + encode_text(prompt)
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("prompt"); ap.add_argument("--images", default=""); ap.add_argument("--audios", type=int, default=0)
    a = ap.parse_args(); imgs = [int(x) for x in a.images.split(",") if x]
    print(" ".join(map(str, encode_presentation(a.prompt, imgs, a.audios))))

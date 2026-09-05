"""The C tokenizer (host/h3tok.cpp, in libh3pipe.so) against transformers' Qwen3-VL tokenizer on a set of prompts.
    python3 tests/test_tokenizer.py"""
import ctypes, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from encode_prompt import TOK

PROMPTS = [
    "A red fox trotting through a snowy forest at dawn, cinematic",
    "A small sailboat crossing a calm lake at sunset, warm light, gentle ripples, birds in the distance",
    "It's 3:45pm; we're 2,000 km away -- don't panic!  Two   spaces and a\nnewline.\n\nTabs\there.",
    "Café naïve résumé, Zürich, São Paulo: 日本語のテキスト and emoji 🦊🔥 mixed in.",
    "numbers 12345 67.89 and 1e-5, symbols #$%^&*() [brackets] {braces} <angles> 'quotes' \"double\"",
    "   leading spaces and trailing spaces   ",
    "ALL CAPS SHOUTING and MiXeD CaSe words, plus a hyphenated-word and under_score",
    "",
]


def main():
    from transformers import AutoTokenizer
    hf = AutoTokenizer.from_pretrained(str(TOK))
    lib = ctypes.CDLL(str(ROOT / "build/libh3pipe.so"))
    lib.h3tok_create.restype = ctypes.c_void_p; lib.h3tok_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
    lib.h3tok_encode.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t]; lib.h3tok_encode.restype = ctypes.c_int
    lib.h3tok_vocab_size.argtypes = [ctypes.c_void_p]; lib.h3tok_vocab_size.restype = ctypes.c_int
    err = ctypes.create_string_buffer(512); tok = lib.h3tok_create(str(TOK / "tokenizer.json").encode(), err, 512)
    assert tok, err.value.decode()
    print(f"vocab {lib.h3tok_vocab_size(tok)} (hf {len(hf)})")
    ok = True
    for text in PROMPTS:
        want = hf(text, add_special_tokens=False)["input_ids"]
        buf = (ctypes.c_int32 * 4096)(); n = lib.h3tok_encode(tok, text.encode(), buf, 4096); got = list(buf[:n]) if n >= 0 else None
        same = got == want
        ok &= same
        print(f"  {'PASS' if same else 'FAIL'} {text[:50]!r}: {len(want)} tokens" + ("" if same else f"\n       want {want}\n       got  {got}"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""Prompt -> Qwen3-VL token ids (no special tokens), one per line or space-separated.
    python3 tools/prompt_ids.py "a red fox ..." """
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from encode_prompt import TOK
from transformers import AutoTokenizer
print(" ".join(str(i) for i in AutoTokenizer.from_pretrained(str(TOK))(sys.argv[1], add_special_tokens=False)["input_ids"]))

#!/usr/bin/env bash
# Fetch original MiniMaxAI/MiniMax-H3 files for the reference tools into ~/h3-models:
# VAEs, tokenizer, processor and scheduler configs. Normal inference uses the four
# Comfy-Org/MiniMax-H3 checkpoints instead (docs/setup.md); this download is optional.
#   scripts/download.sh [DIR] [--all]
# --all adds the original FL2VA bf16 transformer shards (66 GB) and the bf16 Qwen3-VL-32B text encoder (64 GB), which only the
# reference tools read. `hf download` restarts from zero on Xet repos when interrupted, so the shards are fetched with curl -C -
# into a .part file, checked against their safetensors header, and only then published under their final name.
# Authentication: HF_TOKEN, else the token `hf auth login` stored (HF_HOME or ~/.cache/huggingface).
set -euo pipefail
dst=$HOME/h3-models; all=0
for a in "$@"; do case "$a" in --all) all=1;; -*) echo "usage: scripts/download.sh [DIR] [--all]" >&2; exit 64;; *) dst=$a;; esac; done
mkdir -p "$dst"
repo=MiniMaxAI/MiniMax-H3
hf download "$repo" --local-dir "$dst" --include "model_index.json" "FL2VA/model_index.json" "scheduler/*" "tokenizer/*" "processor/*" "vae/*" "audio_vae/*" "audio_scheduler/*" "FL2VA/transformer/config.json" "FL2VA/transformer/model.safetensors.index.json" "text_encoder/config.json" "LICENSE" "README.md"
if [ "$all" = 1 ]; then
  tok=${HF_TOKEN:-}
  if [ -z "$tok" ] && [ -f "${HF_HOME:-$HOME/.cache/huggingface}/token" ]; then tok=$(cat "${HF_HOME:-$HOME/.cache/huggingface}/token"); fi
  for i in $(seq -f "%05g" 1 13); do
    f="FL2VA/transformer/model-$i-of-00013.safetensors"; mkdir -p "$dst/FL2VA/transformer"
    if [ -f "$dst/$f" ]; then echo "have $f"; continue; fi
    part="$dst/$f.part"
    # -f: an HTTP error (401, 404, 5xx) is a failure and its body is never written; --retry covers dropped connections
    curl -fL --retry 5 --retry-all-errors -C - ${tok:+-H "Authorization: Bearer $tok"} -o "$part" "https://huggingface.co/$repo/resolve/main/$f"
    python3 - "$part" <<'PY'
import json, os, struct, sys
path = sys.argv[1]
with open(path, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]; header = json.loads(f.read(n))
end = 8 + n + max(v["data_offsets"][1] for k, v in header.items() if k != "__metadata__")
size = os.path.getsize(path)
if size != end: raise SystemExit(f"{path}: {size} bytes, the safetensors header says {end}: incomplete or not a model file")
PY
    mv "$part" "$dst/$f"
  done
  hf download "$repo" --local-dir "$dst" --include "text_encoder/*"
fi
echo done

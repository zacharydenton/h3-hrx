#!/usr/bin/env bash
# Fetch the FL2VA partition of MiniMaxAI/MiniMax-H3 into ~/h3-models (transformer 66 GB bf16,
# Qwen3-VL-32B encoder 64 GB, the two VAEs, tokenizer/processor/scheduler configs).
# `hf download` restarts from zero on Xet repos when interrupted; the transformer shards are
# fetched with curl -C - so a dropped connection resumes.
set -euo pipefail
dst=${1:-$HOME/h3-models}; mkdir -p "$dst"
repo=MiniMaxAI/MiniMax-H3
hf download "$repo" --local-dir "$dst" --include "model_index.json" "FL2VA/model_index.json" "scheduler/*" "tokenizer/*" "processor/*" "vae/*" "audio_vae/*" "audio_scheduler/*" "FL2VA/transformer/config.json" "FL2VA/transformer/model.safetensors.index.json" "text_encoder/config.json" "LICENSE" "README.md"
tok=$(cat ~/.cache/huggingface/token)
for i in $(seq -f "%05g" 1 13); do
  f="FL2VA/transformer/model-$i-of-00013.safetensors"; mkdir -p "$dst/FL2VA/transformer"
  curl -L -C - -H "Authorization: Bearer $tok" -o "$dst/$f" "https://huggingface.co/$repo/resolve/main/$f"
done
hf download "$repo" --local-dir "$dst" --include "text_encoder/*"
echo done

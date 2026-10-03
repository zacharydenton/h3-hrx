# Upstream presentation fixtures

Generated from ComfyUI `e9027f2b30f37bb3052714eb08fcf479542f4fc0`,
`comfy/text_encoders/minimax.py`. The source SHA-256 is embedded in both artifacts.

`video_pair.safetensors` runs upstream `process_video_block` unchanged on two
64×96 RGB frames: `arange(2*64*96*3) % 251 / 250`, float32. It contains inputs
and normalized temporal patches. No resize is needed for this case.
`video_resize.safetensors` repeats the same construction at 32×64; upstream
upsizes it to 64×96 to meet its minimum pixel count. The Rust test compares
normalized resized patches with a 2e-6 maximum absolute error tolerance.

`upstream.json` runs upstream `MiniMaxH3Tokenizer.tokenize_with_weights` and
`token_tags_from_embeds_info` unchanged. Only the text-tokenizer delegate is
replaced with the Rust project's embedded `assets/tokenizer.json`, without added
special tokens. Inputs: a 64×96 picture, audio label, three 64×96 video frames at
0, 0.5, 1 seconds, then `a ball`. Vision entries expand to six placeholders each.
This exercises paired frames, odd-tail padding, averaged timestamps, independent
media numbering, and modality tags including vision sentinels.

Reference reconstruction and copy numbering follow ComfyUI-MiniMaxH3Mod
`f9462081e28794389b5a6c5067eb327412ad8ee7`, `prompt.py` and `nodes.py`.

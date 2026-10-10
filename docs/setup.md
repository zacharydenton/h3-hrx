# Install and run

h3-hrx runs MiniMax H3 on Linux with AMD Strix Halo (`gfx1151`). Development and
[measured renders](showcase.md#wyvern-chase--anime-and-cinematic) use a 128 GB
unified-memory system. Other GPUs are unvalidated.

## Dependencies

- A working AMD driver and GPU access.
- A current stable Rust toolchain and ffmpeg.
- System libraries compatible with HRX's pinned native bundle: Ubuntu 26.04,
  glibc 2.43+, or a compatible environment. See
  [HRX native setup](https://github.com/zacharydenton/hrx-rs/blob/main/docs/GPU-NPU.md#native-setup).
- About 54 GB of disk space for base checkpoints, plus runtime caches and outputs.

HRX supplies the Loom compiler and GPU runtime. Building h3 requires no ROCm
headers, `hipcc` or local LLVM build.

## Toolchain and build

From the repository root:

```sh
cargo install --locked --path . --bin h3
h3 --width 1344 --height 768 --frames 124 --steps 21 --seed 0 \
  --weight-io native-direct --memory-budget-mib 49152 \
  --out clip.mp4 < docs/prompts/wyvern_cinematic.txt
```

This writes `clip.mp4` and `clip.wav`. First use fetches missing checkpoints and
HRX's verified native bundle, then compiles the Loom kernels. `--steps 21` gives
20 model evaluations. For development, use `cargo build --locked --release --bin h3`.

The default CLI canvas is 864×480 with 124 frames and 30 evaluations. The command
above selects the featured 768p workload. Dimensions must be multiples of 32;
frame counts round up to `17n + 5`. See [prompts](prompting.md) and
[runtime options](runtime-options.md) for inputs, memory limits and loading modes.

## Offline runtime

Provision the HRX version used by this checkout before going offline:

```sh
cargo install --locked hrx-rs --version 0.10.1
hrx prepare
hrx doctor
```

Alternatively, supply that version's matching native archive to
`HRX_OFFLINE=1 hrx prepare native.tar.gz`. Set `HRX_OFFLINE=1` to prevent native
bundle downloads. Checkpoint downloads have a separate offline switch below.
[Runtime overrides and cache locations](shared-hrx.md#provisioning-and-overrides).

## Checkpoints

Missing checkpoints download from
[Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3).
No conversion step is needed.

| Checkpoint | Contents | Disk size |
| --- | --- | ---: |
| `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | Base DiT | ~21 GB |
| `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | Text encoder and vision tower | ~27 GB |
| `minimax_h3_video_vae_fp16.safetensors` | Video encoder and decoder | ~5.2 GB |
| `minimax_h3_audio_vae_fp32.safetensors` | Audio encoder and decoder | ~0.6 GB |
| `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | Reference-conditioned DiT | ~21 GB additional |

The default cache is `~/.cache/huggingface/hub`. Precedence is `HF_HUB_CACHE`,
then `$HF_HOME/hub`, then `$XDG_CACHE_HOME/huggingface/hub`.
`--offline` or `HF_HUB_OFFLINE=1` requires cached checkpoints.
Use `--dit`, `--te`, `--video-vae` or `--audio-vae` for explicit file paths.
Model weights have separate license terms.

Reference requests select Ref2VA; text and keyframe-only requests select FL2VA.
`--base-weights` forces FL2VA with references. To prefetch files with the optional
Hugging Face CLI:

```sh
hf download Comfy-Org/MiniMax-H3 \
  --include '*minimax_h3_fl2va_pruned_int8_convrot.safetensors' \
            '*qwen3vl_32b_minimax_h3_int8_convrot.safetensors' \
            '*minimax_h3_video_vae_fp16.safetensors' \
            '*minimax_h3_audio_vae_fp32.safetensors'
# Additional model for references:
hf download Comfy-Org/MiniMax-H3 \
  --include '*minimax_h3_ref2va_pruned_int8_convrot.safetensors'
```

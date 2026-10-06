# Setup

Run commands from the repository root.

## Dependencies

- Linux with a working AMD Strix Halo (`gfx1151`) driver and GPU access.
  Development uses a 128 GB unified-memory system.
- A current stable Rust toolchain and ffmpeg.
- System libraries compatible with the pinned HRX bundle. It targets
  Ubuntu 26.04 with glibc 2.43+; see
  [HRX native setup](https://github.com/zacharydenton/hrx-rs/blob/main/docs/GPU-NPU.md#native-setup).

HRX supplies Loom and the native runtime. Building h3 requires no ROCm headers,
`hipcc`, or local LLVM build. Python is used only by optional validation tools.
Allow memory for activations and staging as well as model weights.

## Toolchain and build

```sh
cargo build --locked --release --bin h3
# Or install onto Cargo's binary path:
cargo install --locked --path . --bin h3
```

The first GPU operation downloads and verifies HRX's pinned native bundle.
To provision it ahead of time using the version in `Cargo.lock`:

```sh
cargo install --locked --git https://github.com/zacharydenton/hrx-rs --rev 8f0e084a9efd69ec0045b5cd7d032852d048195d hrx-rs
hrx prepare
hrx doctor
```

For offline installation, supply that version's matching native archive to
`HRX_OFFLINE=1 hrx prepare native.tar.gz`. After provisioning, set `HRX_OFFLINE=1`
to prevent native downloads. This is separate from checkpoint offline mode.

An installed binary embeds the Loom sources and tokenizer. `h3 --root DIR`
selects `DIR/kernels/` for development. HRX stores runtime bundles and compiled
kernels under `$XDG_CACHE_HOME/hrx`, falling back to `~/.cache/hrx`.
See [runtime overrides and caching](shared-hrx.md).

## Checkpoints

Missing checkpoints download into the standard Hugging Face Hub cache.
No conversion step is needed.

| Checkpoint | Contents | Approximate disk size |
| --- | --- | ---: |
| `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | Base DiT | 21 GB |
| `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | Text encoder and vision tower | 27 GB |
| `minimax_h3_video_vae_fp16.safetensors` | Video encoder and decoder | 5.2 GB |
| `minimax_h3_audio_vae_fp32.safetensors` | Audio encoder and decoder | 0.6 GB |
| `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | Reference-conditioned DiT | 21 GB |

The default cache is `~/.cache/huggingface/hub`. Precedence is `HF_HUB_CACHE`,
then `$HF_HOME/hub`, then `$XDG_CACHE_HOME/huggingface/hub`.
`--offline` or `HF_HUB_OFFLINE=1` restricts checkpoint resolution to local files.
Use `--dit`, `--te`, `--video-vae`, or `--audio-vae` to select individual files.

To prefetch the base files with the optional Hugging Face CLI:

```sh
hf download Comfy-Org/MiniMax-H3 \
  --include '*minimax_h3_fl2va_pruned_int8_convrot.safetensors' \
            '*qwen3vl_32b_minimax_h3_int8_convrot.safetensors' \
            '*minimax_h3_video_vae_fp16.safetensors' \
            '*minimax_h3_audio_vae_fp32.safetensors'
```

Reference requests select Ref2VA; keyframe-only requests select FL2VA.
`--base-weights` forces FL2VA with references. To prefetch Ref2VA:

```sh
hf download Comfy-Org/MiniMax-H3 \
  --include '*minimax_h3_ref2va_pruned_int8_convrot.safetensors'
```

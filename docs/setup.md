# Setup

Run commands from the repository root unless stated otherwise.

## Dependencies

- Linux and a Radeon 8060S (`gfx1151`). Development and measurements used a
  128 GB Strix Halo system. Allow room for activations as well as weights;
  smaller memory configurations have not been validated.
- ffmpeg for input decoding and MP4/WAV output.
- `cargo` (Rust). Everything outside `h3/kernels/` is Rust: the library, the `h3`
  command and the `loomrun` kernel launcher. No ROCm headers and no `hipcc` —
  the runtime surface is `libhrx`.
- A checkout of the [`hrx.rs`](shared-hrx.md) crate beside the directory holding
  this one. It is a path dependency, so nothing here builds without it, and it
  supplies the Loom compiler, `libhrx` and a compatible HSA as a pinned,
  digest-verified native bundle. No ROCm headers, no `hipcc`, no LLVM build.
- For downloads, the Hugging Face CLI (`python3 -m pip install huggingface_hub`
  in a Python environment). For development, Python 3 with NumPy. Inference
  itself runs without Python.

## Toolchain and build

The native bundle is a tested local candidate; its public release has not been
uploaded yet, so prepare it from the sibling checkout rather than letting HRX
fetch it:

```sh
cargo install --locked --path ../hrx.rs --features runner
hrx prepare ../hrx.rs/artifacts/hrx-linux-x86_64-gfx1151.tar.gz
bash scripts/build_host.sh
```

`HRX_RUNTIME_DIR` points at a native directory of your own, `HRX_BUNDLE_MANIFEST`
at a pinned mirror, and `HRX_OFFLINE` refuses the network outright. `LOOM_COMPILE`
selects a developer compiler in place of the bundle's.

The build produces `build/libh3.so`, `build/h3`, `build/loomrun` and
`include/h3.h`, the last copied from what the Cargo build generated — an ordinary
`cargo build` leaves the checkout alone. An installed binary carries the Loom
sources and the tokenizer inside it and caches compiled kernels per user under
`$XDG_CACHE_HOME/hrx`; `h3 --root DIR` opts back into a working tree's
`h3/kernels/` and `build/kernel_cache` instead.

Optional CLI installation:

```sh
cargo install --locked --path cli   # or: ln -s "$PWD/build/h3" ~/.local/bin/h3
```

Building Loom yourself, rather than taking the bundle's compiler, needs
[the included compiler patches](../patches/loom/README.md) — among them
`0003-amdgpu-pm4-emulation-query-optional.patch`, without which `hrx_gpu_initialize`
returns `INVALID_ARGUMENT` from `hsa_agent_get_info` on a ROCr older than the
`HSA_AMD_AGENT_INFO_PM4_EMULATION` attribute. The Python kernel tooling under
`tools/` still drives such a build, through `scripts/env.sh` and `HRX_BUILD`.

## Checkpoints

Follow the [base checkpoint download](../README.md#weights); `h3` fetches them on
first use if they are not already present. They are resolved from a models
directory, then the shared Hugging Face cache, then the hub, so a copy that
already exists anywhere is reused. `tools/link_hf_cache.py --apply` registers an
existing ComfyUI models directory in the cache by symlink, which costs no disk
and works when the two are on different filesystems. Approximate file sizes:

| Checkpoint | Contents | Disk size |
| --- | --- | ---: |
| `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` | Base DiT | 21 GB |
| `text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | Text encoder and vision tower | 27 GB |
| `vae/minimax_h3_video_vae_fp16.safetensors` | Video encoder and decoder | 5.2 GB |
| `vae/minimax_h3_audio_vae_fp32.safetensors` | Audio encoder and decoder | 0.6 GB |

There is no separate export step. The loader preserves the checkpoint's core
weight types and scales while arranging tensors for the kernels. Qwen's
tokenizer is embedded in the library.

Image and audio references need an additional 21 GB checkpoint:

```sh
hf download Comfy-Org/MiniMax-H3 --local-dir ~/comfy-models \
  --include diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors
```

The CLI chooses ref2va for positional reference files. `--first-frame` uses
the base fl2va checkpoint. `--base-weights` explicitly forces the base model
for references; use ref2va for normal reference-conditioned generation.

The default model directory is `~/comfy-models`. Set `H3_MODELS` or pass
`--models DIR`; `--dit`, `--te`, `--video-vae`, and `--audio-vae` override
individual files. All flags are listed by `./build/h3 --help`.

`scripts/download.sh` downloads the original MiniMax reference-model files
for development tools. It is not needed to generate clips with this host.

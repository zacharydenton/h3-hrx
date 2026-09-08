# Setup

Run commands from the repository root unless stated otherwise.

## Dependencies

- Linux and a Radeon 8060S (`gfx1151`). Development and measurements used a
  128 GB Strix Halo system. Allow room for activations as well as weights;
  smaller memory configurations have not been validated.
- ffmpeg for input decoding and MP4/WAV output.
- `cargo` (Rust). Everything outside `kernels/` is Rust: the library, the `h3`
  command and the `loomrun` kernel launcher. No ROCm headers and no `hipcc` —
  the runtime surface is `libhrx`.
- `cbindgen` (`cargo install cbindgen`) to regenerate `include/h3.h`. The build
  skips it when absent and uses the committed header.
- [Loom with the included compiler patches](../patches/loom/README.md).
  Everything dispatches through `libhrx` from that build.
- For downloads, the Hugging Face CLI (`python3 -m pip install huggingface_hub`
  in a Python environment). For development, Python 3 with NumPy. Inference
  itself runs without Python.

## Toolchain and build

```sh
export HRX_BUILD=/path/to/hrx-system/build
source scripts/env.sh
bash scripts/build_host.sh
```

`env.sh` derives tool paths from `HRX_BUILD`; individual `LOOM_COMPILE`,
`LOOM_FORMAT`, `LOOM_CHECK`, `IREE_TEST_LOOM`, and `IREE_BENCHMARK_LOOM`
overrides are respected. It retains the development checkout default
`~/code/hrx-system/build-cuda` when `HRX_BUILD` is unset.

The build produces `build/libh3.so`, `build/h3`, `build/loomrun` and
`include/h3.h`. Keep the repository's `kernels/` directory available: kernels
compile on first use for each shape and are cached in `build/kernel_cache`.
Use `h3 --root DIR` if you relocate the executable.

Optional CLI installation:

```sh
mkdir -p ~/.local/bin
ln -s "$PWD/build/h3" ~/.local/bin/h3
```

The optional HRX backend is built when `libhrx.so` is available under
`HRX_SYSTEM` (default `~/code/hrx-system`). The HIP backend is the normal CLI
path.

An unpatched HRX runtime fails to initialize on a ROCr older than the
`HSA_AMD_AGENT_INFO_PM4_EMULATION` attribute, including the distribution's
HSA 1.18: `hrx_gpu_initialize` returns `INVALID_ARGUMENT` from
`hsa_agent_get_info`. `patches/loom/0003-amdgpu-pm4-emulation-query-optional.patch`
fixes that. With it applied, HRX runs on the stock system runtime and produces
frames and samples bit-identical to the HIP backend, so no replacement runtime
and no `LD_LIBRARY_PATH` entry is needed.

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

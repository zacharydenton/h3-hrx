# Setup

Run commands from the repository root unless stated otherwise.

## Dependencies

- Linux and AMD Strix Halo (Radeon 8060S, `gfx1151`). Development and
  measurements used a 128 GB system. Allow room for activations as well as weights;
  smaller memory configurations have not been validated.
- ffmpeg for input decoding and MP4/WAV output.
- A current stable Rust toolchain (`cargo`). The library and the
  `h3` command build with Cargo. No ROCm headers and no `hipcc` — the runtime surface is `libhrx`,
  loaded on demand from the HRX bundle.
- Nothing else. [`hrx-rs`](shared-hrx.md) comes from crates.io like any other
  dependency, and provisions its own native bundle — Loom, `libhrx` and a
  compatible HSA — on first use, without a local LLVM or ROCm build.

`h3` fetches the checkpoints itself. Inference needs no Python; optional parity tools use it.

## Toolchain and build

```sh
cargo build --locked --release --bin h3
```

The first run that needs the GPU downloads the native
bundle the crate pins, verifies it file by file against the manifest compiled
into the crate, and caches it under `$XDG_CACHE_HOME/hrx`.

To provision it ahead of time, or on a machine that will not have the network
later, install the crate's runner and unpack the release yourself:

```sh
cargo install --locked hrx-rs --features runner
gh release download native-20260909-reviewed --repo zacharydenton/hrx-rs \
  --pattern hrx-linux-x86_64-gfx1151.tar.gz --dir build --clobber
hrx prepare build/hrx-linux-x86_64-gfx1151.tar.gz
```

`HRX_RUNTIME_DIR` points at a native directory of your own, `HRX_BUNDLE_MANIFEST`
at a pinned mirror, and `HRX_OFFLINE` refuses the network outright. `HRX_LOOM_LIBRARY`
selects a developer `libloomc.so` in place of the bundle’s.

The build produces `target/release/h3`. An installed binary carries the Loom
sources and the tokenizer inside it; `h3 --root DIR` opts back into a working tree's `kernels/`
instead. Compiled kernels go to one cache per user under `$XDG_CACHE_HOME/hrx` whichever sources
they came from, because an artifact's name already covers the compiler, the source, the export, the
target and the configuration. `hrx gc [DAYS]` sweeps it by last use.

Optional CLI installation:

```sh
cargo install --locked --path . --bin h3
```

For a local compiler build, follow
[HRX’s upstream pin and compiler patches](https://github.com/zacharydenton/hrx-rs/tree/main/patches/loom).
HRX owns the required native fixes, and the bundle it pins carries them; H3 loads the
resulting `libloomc.so` through `HRX_LOOM_LIBRARY`.

## Checkpoints

See the [base checkpoint list](../README.md#weights); `h3` fetches missing files
on first use into the standard Hugging Face Hub cache. The default is
`~/.cache/huggingface/hub`, shared with the `hf` CLI and other Hugging Face tools.
Set `HF_HUB_CACHE` to override it, or `HF_HOME` to use `$HF_HOME/hub`.
`HF_HUB_OFFLINE=1` or `--offline` restricts model resolution to local files.
Without `HF_HOME`, `XDG_CACHE_HOME` selects `$XDG_CACHE_HOME/huggingface/hub`.

The Hub manages snapshots and cached files automatically. The checkpoints stay
in their published Comfy-Org format. Approximate file sizes:

| Checkpoint | Contents | Disk size |
| --- | --- | ---: |
| `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | Base DiT | 21 GB |
| `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | Text encoder and vision tower | 27 GB |
| `minimax_h3_video_vae_fp16.safetensors` | Video encoder and decoder | 5.2 GB |
| `minimax_h3_audio_vae_fp32.safetensors` | Audio encoder and decoder | 0.6 GB |

To download these ahead of time with the optional Hugging Face CLI:

```sh
hf download Comfy-Org/MiniMax-H3 \
  --include '*minimax_h3_fl2va_pruned_int8_convrot.safetensors' \
            '*qwen3vl_32b_minimax_h3_int8_convrot.safetensors' \
            '*minimax_h3_video_vae_fp16.safetensors' \
            '*minimax_h3_audio_vae_fp32.safetensors'
```

There is no separate export step. The loader preserves the checkpoint's core
weight types and scales while arranging tensors for the kernels. Qwen's
tokenizer is embedded in the library.

Image and audio references need an additional 21 GB checkpoint:

```sh
hf download Comfy-Org/MiniMax-H3 \
  --include '*minimax_h3_ref2va_pruned_int8_convrot.safetensors'
```

The CLI chooses ref2va for positional reference files. `--first-frame` uses
the base fl2va checkpoint. `--base-weights` explicitly forces the base model
for references; use ref2va for normal reference-conditioned generation.

The library and CLI use the same Hugging Face cache. To select an individual
file explicitly, use `--dit`, `--te`, `--video-vae`, or `--audio-vae` with its
path; the file can be stored anywhere. All flags are listed by `h3 --help`.

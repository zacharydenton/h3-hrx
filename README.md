# minimax-h3-loom

MiniMax H3 (Hailuo 3.0) video and audio generation on AMD Strix Halo
(Radeon 8060S, `gfx1151`). Every GPU kernel is written in
[Loom](https://github.com/ROCm/hrx-system); everything else is Rust, behind both a
Rust API and a generated C one. Inference dispatches through `libhrx` and needs no
Python, PyTorch, Triton, or vendor math libraries — and no ROCm headers or `hipcc`
to build. The host reads ComfyUI-format checkpoints directly.

Text-to-video, first-frame animation, and image/audio reference conditioning
are supported. This is an experimental implementation tested on a 128 GB
Radeon 8060S system; other GPUs and memory configurations are unvalidated.

https://github.com/user-attachments/assets/41a98dcf-48f0-4328-a0f4-7f17119243e6

[Demo prompt](docs/prompts/cliff_rider_768p.txt) · [Prompt format](docs/prompting.md)

## Performance

Recorded **per model evaluation** on an idle Radeon 8060S, using int8 weights.
ComfyUI was measured on the same GPU. These are not whole-clip timings.

| Clip | Loom | ComfyUI | Speedup |
| --- | ---: | ---: | ---: |
| 1344×768, 124 frames | 101 s | 771 s | 7.6× |
| 864×480, 124 frames | 26.7 s | 103 s | 3.9× |
| 864×480, 22 frames | 4.2 s | 3.8 s | 0.9× |

A five-second clip at the default 30 evaluations takes roughly **15 minutes at
480p** or **55 minutes at 768p**, including setup and decoding. Outputs are
numerically checked against ComfyUI, but are not bit-identical.
See [performance and validation](docs/performance.md) for conditions, quality
measurements, and decoder timings.

## Setup

You need Linux, a `gfx1151` device with its kernel driver, Rust and ffmpeg.
The shared `hrx.rs` crate provisions prebuilt Loom, HRX and compatible HSA.
Nothing here needs Python.

With access to the private `hrx.rs` repository and an authenticated GitHub CLI:

```sh
cargo install --locked --git https://github.com/zacharydenton/hrx.rs \
  --rev be89b44652af6adf17c5c950d0759f92c2e88582 --features runner
mkdir -p build
gh release download native-ecaaf7376f7d-loomc --repo zacharydenton/hrx.rs \
  --pattern hrx-linux-x86_64-gfx1151.tar.gz --dir build --clobber
hrx prepare build/hrx-linux-x86_64-gfx1151.tar.gz
cargo install --locked --path cli
# Build the C library and headers too:
bash scripts/build_host.sh
```

The pinned native release includes the Loom consolidation fixes. It is hosted
in the private HRX repository; the commands above prepare it locally. See
[shared HRX integration](docs/shared-hrx.md) for overrides, offline use, C ABI and
Rustler. Kernel sources and the tokenizer are embedded in installed binaries;
the default compiler cache is in the writable per-user HRX cache. `--root`
selects a developer source/cache tree explicitly.

### Weights

`h3` fetches what it needs on first use, into the shared Hugging Face cache:

```sh
h3 -p "..."          # downloads the four checkpoints if they are not already here
```

Checkpoints are looked for in a models directory first (`--models DIR` or
`H3_MODELS`, default `~/comfy-models`), then in the Hugging Face cache, and only
then on the hub, so an existing copy is reused wherever it lives. `--offline`
stops at what is already on disk. To fetch them ahead of time:

```sh
hf download Comfy-Org/MiniMax-H3 \
  --include diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
            text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors \
            vae/minimax_h3_video_vae_fp16.safetensors \
            vae/minimax_h3_audio_vae_fp32.safetensors
```

If you already have them in a ComfyUI models directory, point `H3_MODELS` at it
or symlink them into the Hugging Face cache rather than downloading them again:
the resolver checks the models directory, then that cache, then the hub.
Reference images and audio additionally need
the ref2va checkpoint. See [setup details](docs/setup.md) for that download and
tool paths.
Model weights have their own license; consult the
[model repository](https://huggingface.co/Comfy-Org/MiniMax-H3).

## Generate a clip

H3 expects [structured prompts](docs/prompting.md). Start with the included one:

```sh
./build/h3 --width 864 --height 480 --frames 124 --steps 31 --seed 7 \
  --out clip.mp4 < docs/prompts/cliff_rider_768p.txt
./build/h3 --help
```

This writes `clip.mp4` and `clip.wav`. `--steps 31` means 30 model evaluations.
The first run of each shape compiles kernels into the shared per-user HRX cache.
For the demo's resolution, use `--width 1344 --height 768`.

With prompts written for the corresponding conditioning mode:

```sh
./build/h3 --first-frame image.png --out animated.mp4 < keyframe-prompt.txt
./build/h3 ref.jpg voice.wav --out referenced.mp4 < reference-prompt.txt
```

Sizes must be multiples of 32; frame counts round up to `17n + 5`.
The sampler defaults to `res_multistep` with the `simple` schedule and no
classifier-free guidance. Video references are available through the C API,
but not the CLI.

## Documentation and development

- [Setup](docs/setup.md) — dependencies, checkpoints, and configuration.
- [Prompting](docs/prompting.md) and [other modes](docs/tricks.md) — references, audio, and stills.
- [C API](docs/abi.md) and [examples](examples/README.md) — C, Rust, Go, and Elixir clients.
- [Contributing](CONTRIBUTING.md) — repository layout, tests, and benchmarking.
- [Performance](docs/performance.md) and [research archive](docs/archive/README.md).

Run `bash scripts/test.sh --cpu` for checks that need no GPU or weights.

## License and credits

Code: [Apache-2.0](LICENSE). Model weights are licensed separately by MiniMax.

Built on AMD's [Loom](https://github.com/ROCm/hrx-system) and MiniMax H3, using
ComfyUI as the numerical reference. The int8 attention follows SageAttention's
approach. Kernel and runtime work started in
[krea2-loom](https://github.com/zacharydenton/krea2-loom).

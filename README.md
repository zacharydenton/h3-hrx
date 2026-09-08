# minimax-h3-loom

MiniMax H3 (Hailuo 3.0) video and audio generation on AMD Strix Halo
(Radeon 8060S, `gfx1151`). Every GPU kernel is written in
[Loom](https://github.com/ROCm/hrx-system), with a C++ host and a C API.
Inference uses the HIP runtime; it needs no Python, PyTorch, Triton, or vendor
math libraries. The host reads ComfyUI-format checkpoints directly.

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

You need Linux, ROCm with `gfx1151` support (tested with the 7.1 series), a C++
compiler, ffmpeg, and a built Loom compiler. The measured kernels require
[two Loom patches included here](patches/loom/README.md).
Python 3 and NumPy are needed for development and CPU tests.

From the repository root, after [building Loom](patches/loom/README.md):

```sh
export HRX_BUILD=/path/to/hrx-system/build
source scripts/env.sh
bash scripts/build_host.sh
```

### Weights

Download the four checkpoints with the Hugging Face `hf` CLI:

```sh
hf download Comfy-Org/MiniMax-H3 --local-dir ~/comfy-models \
  --include diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
            text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors \
            vae/minimax_h3_video_vae_fp16.safetensors \
            vae/minimax_h3_audio_vae_fp32.safetensors
```

Use `--models DIR` or `H3_MODELS` for a different location. Reference images and
audio additionally need the ref2va checkpoint. See [setup details](docs/setup.md)
for that download, tool paths, and the optional runtime workaround.
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
The first run of each shape compiles kernels into `build/kernel_cache`.
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
- [C API](docs/abi.md) and [examples](examples/README.md) — C, Rust, Go, and Python clients.
- [Contributing](CONTRIBUTING.md) — repository layout, tests, and benchmarking.
- [Performance](docs/performance.md) and [research archive](docs/archive/README.md).

Run `bash scripts/test.sh --cpu` for checks that need no GPU or weights.

## License and credits

Code: [Apache-2.0](LICENSE). Model weights are licensed separately by MiniMax.

Built on AMD's [Loom](https://github.com/ROCm/hrx-system) and MiniMax H3, using
ComfyUI as the numerical reference. The int8 attention follows SageAttention's
approach. Kernel and runtime work started in
[krea2-loom](https://github.com/zacharydenton/krea2-loom).

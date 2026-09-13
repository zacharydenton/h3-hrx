# h3-hrx

**MiniMax H3 video and audio generation on AMD Strix Halo, powered by Loom and HRX.**

Run MiniMax H3 (Hailuo 3.0) locally with every GPU kernel written in
[Loom](https://github.com/ROCm/hrx-system) and compiled, cached, and dispatched
through [HRX](https://github.com/zacharydenton/hrx-rs). Load Comfy-Org quantized
checkpoints from the standard Hugging Face cache and generate video with sound
from text, a first frame, or image and audio references.

**3.04× faster steady denoising than ComfyUI** in a measured 480p comparison on
Strix Halo, with higher GPU-resident memory use. [Timings and memory](#performance).

The inference pipeline needs no Python, PyTorch, Triton, or vendor math libraries.
Building it needs no ROCm headers or `hipcc`. HRX provisions the native compiler
and runtime on first GPU use; the CLI embeds the Loom kernel sources and tokenizer.

**Experimental:** tested on Linux with AMD Strix Halo (`gfx1151`)
and 128 GB of unified memory. Other GPUs and memory configurations are unvalidated.

[![A whale glides above an alpine valley at sunrise](docs/media/showcase/alpine_whale.jpg)](docs/media/showcase/alpine_whale.mp4)

[Watch the surreal video showcase](docs/showcase.md) · [Demo prompt](docs/prompts/alpine_whale.txt) · [Setup](docs/setup.md) ·
[Prompt guide](docs/prompting.md) · [Performance](docs/performance.md) ·
[Rust and Elixir clients](clients/README.md)

## What you can do

- **Text to video with sound:** generate an MP4 and a separate WAV from a structured prompt.
- **Image to video:** animate a first frame or condition generation on reference images.
- **Audio references:** guide generation with voices and sounds alongside image references.
- **Audio and stills:** use the CLI's audio-only and single-frame output modes.
- **Embed inference:** use the Rust `Session` API or the example Rustler adapter for Elixir.

The project is named `h3-hrx`; its command is `h3` and its Rust library is `h3_hrx`.

## Quick start

You need Linux, a working AMD Strix Halo GPU driver, a current stable Rust
toolchain, and `ffmpeg` on your `PATH`. The base checkpoints occupy approximately
54 GB on disk; keep additional memory available for activations. See
[setup details](docs/setup.md) for the supported hardware and configuration.

```sh
git clone https://github.com/zacharydenton/h3-hrx.git
cd h3-hrx
cargo install --locked --path . --bin h3
h3 --help
```

Ensure Cargo's binary directory (normally `~/.cargo/bin`) is on your `PATH`.
Start with the included [structured prompt](docs/prompting.md):

```sh
h3 --width 864 --height 480 --frames 124 --steps 21 --seed 2718 \
  --out clip.mp4 < docs/prompts/alpine_whale.txt
```

This writes `clip.mp4` and `clip.wav`. The first run downloads missing checkpoints
and the pinned HRX native bundle, and compiles kernels for the requested shape.
`--steps 21` means 20 model evaluations. The measured 480p workload completed
in **12 minutes 12 seconds** with checkpoints already cached, including loading,
text encoding, sampling, decoding, and output encoding. First-run downloads and
compilation can take longer; see the [benchmark conditions](docs/benchmarks/20260913/README.md).

### Weights

`h3` stores and reuses checkpoints in the standard Hugging Face Hub cache,
**`~/.cache/huggingface/hub`** by default. Missing files are downloaded from
[Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3).
`HF_HUB_CACHE` overrides the cache directory; `HF_HOME` sets its parent
(the cache is then `$HF_HOME/hub`). `XDG_CACHE_HOME` is also respected.

The Hub manages repository snapshots and cached files automatically. No model
folder setup or conversion is needed.

The base pipeline needs these four files:

| Checkpoint | Purpose |
| --- | --- |
| `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | Video and audio diffusion model |
| `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | Text encoder and vision tower |
| `minimax_h3_video_vae_fp16.safetensors` | Video encoder and decoder |
| `minimax_h3_audio_vae_fp32.safetensors` | Audio encoder and decoder |

Image and audio references additionally need the ref2va checkpoint (about 21 GB);
the CLI downloads it when needed. See [checkpoint setup](docs/setup.md#checkpoints)
for manual downloads and file overrides. Model weights have their own license;
consult the [model repository](https://huggingface.co/Comfy-Org/MiniMax-H3).

For fully offline inference after provisioning, set `HRX_OFFLINE=1` and
`HF_HUB_OFFLINE=1` (or pass `--offline` for model files). HRX caches its native bundle and compiled kernels in the per-user
HRX cache. See [shared HRX integration](docs/shared-hrx.md) for cache locations,
runtime overrides, and offline provisioning. `--root DIR` selects a developer
source tree explicitly.

## Generation modes

Use prompts written for the corresponding [conditioning mode](docs/prompting.md):

```sh
h3 --first-frame image.png --out animated.mp4 < keyframe-prompt.txt
h3 ref.jpg voice.wav --out referenced.mp4 < reference-prompt.txt
```

Sizes must be multiples of 32; frame counts round up to `17n + 5`.
The sampler defaults to `res_multistep` with the `simple` schedule and no
classifier-free guidance. Video references are available through the Rust API.
See [other modes](docs/tricks.md) for audio-only output and still images.

## Performance

Measured on AMD Strix Halo with 128 GB unified memory: 864×480, 124 frames,
20 evaluations, the same prompt and Comfy-Org int8 ConvRot checkpoints.

| Measurement | h3-hrx (Loom / HRX) | ComfyUI |
| --- | ---: | ---: |
| Steady denoising, median per evaluation | **28.1 s** | 85.5 s |
| Observed evaluation range | 26.8–28.2 s | 84.5–86.4 s |
| Sampled peak GPU-resident buffers | 51.74 GiB | 25.45 GiB |
| Sampled peak process PSS | 1.32 GiB | 23.12 GiB |

**3.04× faster denoising, with about twice the GPU-resident buffer memory.**
PSS and GPU residency overlap on unified memory: do not add them or interpret
PSS alone as total RAM usage. Memory was sampled every five seconds through
loading, inference, decoding, and output attempts.

This is one sampling trajectory per engine, excluding the first evaluation from
steady timing. h3 uses int8 products and QK attention; ComfyUI uses bf16 compute
on the same quantized weights. Outputs are not identical. ComfyUI completed
sampling and decoding, but its container lacked the `libx264` encoder for MP4
output, so **no end-to-end speedup is claimed**.

See the [reproducible benchmark, raw measurements, and memory plot](docs/benchmarks/20260913/README.md)
and [performance and numerical validation](docs/performance.md).

## Library and development

The Cargo package is `h3-hrx`, with a library named `h3_hrx`. To use it from a checkout:

```toml
[dependencies]
h3-hrx = { path = "../h3-hrx", default-features = false }
```

`Session` provides denoising, encoding, and decoding. Disabling default features
omits the CLI dependencies. See [clients](clients/README.md) for runnable Rust
and Elixir examples; `cargo doc --no-deps --open` builds the API reference.

```sh
bash scripts/test.sh --cpu
```

CPU checks require no GPU, native runtime, or model weights. Hardware and
checkpoint tests are documented in [test coverage](docs/testing.md).

- [Contributing](CONTRIBUTING.md) — development, tests, and bug reports.
- [Shared HRX integration](docs/shared-hrx.md) — compilation, caching, runtime, and graphs.
- [Research archive](docs/archive/README.md) — historical kernel experiments and measurements.
- [Release checklist](docs/releasing.md) — packaging and validation gates.

## License and credits

Code: [Apache-2.0](LICENSE). Model weights are licensed separately by MiniMax.
The embedded Qwen tokenizer is Apache-2.0; see [asset attribution](assets/README.md).

Built on AMD's [Loom](https://github.com/ROCm/hrx-system),
[HRX](https://github.com/zacharydenton/hrx-rs), and MiniMax H3, using ComfyUI as
the numerical reference. The int8 attention follows SageAttention's approach.
Kernel and runtime work started in
[krea2-loom](https://github.com/zacharydenton/krea2-loom).

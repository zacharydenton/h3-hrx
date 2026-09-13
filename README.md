# h3-hrx

**MiniMax H3 video and audio generation on AMD Strix Halo, powered by Loom and HRX.**

Run MiniMax H3 (Hailuo 3.0) locally with every GPU kernel written in
[Loom](https://github.com/ROCm/hrx-system) and compiled, cached, and dispatched
through [HRX](https://github.com/zacharydenton/hrx-rs). Load ComfyUI-format
checkpoints directly and generate video with sound from text, a first frame,
or image and audio references.

The inference pipeline needs no Python, PyTorch, Triton, or vendor math libraries.
Building it needs no ROCm headers or `hipcc`. HRX provisions the native compiler
and runtime on first GPU use; the CLI embeds the Loom kernel sources and tokenizer.

**Experimental:** tested on Linux with AMD Strix Halo, Radeon 8060S (`gfx1151`),
and 128 GB of unified memory. Other GPUs and memory configurations are unvalidated.

https://github.com/user-attachments/assets/41a98dcf-48f0-4328-a0f4-7f17119243e6

[Demo prompt](docs/prompts/cliff_rider_768p.txt) · [Setup](docs/setup.md) ·
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
h3 --width 864 --height 480 --frames 124 --steps 31 --seed 7 \
  --out clip.mp4 < docs/prompts/cliff_rider_768p.txt
```

This writes `clip.mp4` and `clip.wav`. The first run downloads missing checkpoints
and the pinned HRX native bundle, and compiles kernels for the requested shape.
`--steps 31` means 30 model evaluations. A five-second clip takes roughly
**15 minutes at 480p** or **55 minutes at 768p**, including setup and decoding
in recorded runs. For the demo's resolution, use `--width 1344 --height 768`.

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

Recorded **per model evaluation** on an idle AMD Strix Halo system, using int8 weights.
ComfyUI was measured on the same GPU. These are development measurements, not
whole-clip timings or a fresh release benchmark.

| Clip | h3-hrx (Loom / HRX) | ComfyUI | Speedup |
| --- | ---: | ---: | ---: |
| 1344×768, 124 frames | 101 s | 771 s | 7.6× |
| 864×480, 124 frames | 26.7 s | 103 s | 3.9× |
| 864×480, 22 frames | 4.2 s | 3.8 s | 0.9× |

Outputs are numerically checked against ComfyUI, but are not bit-identical.
See [performance and validation](docs/performance.md) for conditions, quality
measurements, and decoder timings.

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

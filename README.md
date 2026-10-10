# h3-hrx

MiniMax H3 video and audio generation in Rust for AMD Strix Halo.
GPU kernels are written in [Loom](https://github.com/ROCm/hrx-system) and run
through [HRX](https://github.com/zacharydenton/hrx-rs).

A five-second 1344×768 clip with audio takes **about 32 minutes** on a 128 GB
Strix Halo system: 124 frames at 24 fps, 20 model evaluations, full attention.
The two renders below took 31m57s and 31m49s, including weight loading and output
encoding. Checkpoint downloads are excluded.

https://github.com/user-attachments/assets/7a3e95c2-0174-4da4-b2f7-8b7219c650a3

Anime, then cinematic. Each clip is trimmed to five seconds.
[Prompts and settings](docs/showcase.md#wyvern-chase--anime-and-cinematic) ·
[More videos](docs/showcase.md)

## Features

- Text-to-video, first/last keyframes, and image, audio and video references.
- INT8 ConvRot checkpoints and INT8 attention.
- [RefMods](docs/refmods.md) for reusable subjects and voices.
- [LoRAs](docs/loras.md) and [learned latent upscaling](docs/upscaling.md).
- [H3-World](docs/world.md) action control, saved sessions and branching.
- [Rust API and Elixir adapter](clients/README.md).

## Build and run

Requires Linux, Rust, ffmpeg and a working Strix Halo GPU driver.
See [setup](docs/setup.md) for system-library requirements. Other GPUs are untested.

```sh
git clone https://github.com/zacharydenton/h3-hrx.git
cd h3-hrx
cargo install --locked --path . --bin h3
```

Generate the cinematic clip above:

```sh
h3 --width 1344 --height 768 --frames 124 --steps 21 --seed 0 \
  --weight-io native-direct --memory-budget-mib 49152 \
  --out clip.mp4 < docs/prompts/wyvern_cinematic.txt
```

Writes `clip.mp4` and `clip.wav`. Missing model weights download automatically
(~54 GB for the base models). `--steps 21` gives 20 model evaluations.

H3 uses [structured prompts](docs/prompting.md). Use `h3 --help` for CLI options;
see [runtime configuration](docs/runtime-options.md) for loading and memory limits.

## Development

```sh
cargo test
cargo bench
```

Both include GPU workloads. [Tests](docs/testing.md) ·
[Benchmarks and profiling](docs/performance.md) ·
[Loom/HRX integration](docs/shared-hrx.md) · [Contributing](CONTRIBUTING.md)

## License

[Apache-2.0](LICENSE). Model weights are licensed separately by MiniMax.
[Upscaler attribution](THIRD_PARTY_NOTICES) · [Asset attribution](assets/README.md).

ComfyUI is used as a numerical reference. INT8 attention follows SageAttention's
approach. Kernel work started in [krea2-loom](https://github.com/zacharydenton/krea2-loom).

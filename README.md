# h3-hrx

**768p video with generated audio in ~32 minutes on AMD Strix Halo.**
MiniMax H3 in Rust, with custom [Loom](https://github.com/ROCm/hrx-system) GPU
kernels and the [HRX](https://github.com/zacharydenton/hrx-rs) runtime.

https://github.com/user-attachments/assets/7a3e95c2-0174-4da4-b2f7-8b7219c650a3

*Wyvern chase: anime → cinematic. Two generated clips, trimmed to five seconds each.*

| Render | End-to-end time |
| --- | ---: |
| Cinematic | **31m49s** |
| Anime | **31m57s** |

Both renders: **1344×768 · 124 frames at 24 fps · 20 model evaluations**,
full attention, no step caching. Linux, `gfx1151`, 128 GB unified memory.
Times include local weight loading through final video/audio encoding;
checkpoint downloads are excluded.

[Settings and prompts](docs/showcase.md#wyvern-chase--anime-and-cinematic) ·
[More examples](docs/showcase.md)

## Run

Install Rust and ffmpeg ([system requirements](docs/setup.md)), then:

```sh
git clone https://github.com/zacharydenton/h3-hrx.git
cd h3-hrx
cargo install --locked --path . --bin h3
h3 --width 1344 --height 768 --frames 124 --steps 21 --seed 0 \
  --weight-io native-direct --memory-budget-mib 49152 \
  --out clip.mp4 < docs/prompts/wyvern_cinematic.txt
```

Writes `clip.mp4` and `clip.wav`. First use downloads ~54 GB of base checkpoints
and the HRX runtime, then compiles kernels. `--steps 21` means 20 model evaluations.
Use [structured H3 prompts](docs/prompting.md). Other GPUs are unvalidated.

## Features and guides

- [Loom kernels and HRX execution](docs/shared-hrx.md) · [Measure performance](docs/performance.md)
- [Text, keyframes and image/audio/video references](docs/prompting.md)
- [Reusable references](docs/refmods.md) · [LoRAs](docs/loras.md) · [Latent upscaling](docs/upscaling.md)
- [Action control and branching](docs/world.md) · [Audio-only and stills](docs/tricks.md)
- [Rust and Elixir clients](clients/README.md) · [Shared HRX contexts](docs/shared-hrx.md)
- [Runtime options](docs/runtime-options.md) · [Checkpoint and offline setup](docs/setup.md#checkpoints)

## Develop

```sh
cargo test
cargo bench
```

[Criterion benchmarks](docs/performance.md) cover kernels, model stages, full
renders, loading and memory. [Test coverage](docs/testing.md) ·
[Contributing](CONTRIBUTING.md) · [Release checks](docs/releasing.md).

## License and credits

[Apache-2.0](LICENSE); [MIT upscaler attribution](THIRD_PARTY_NOTICES).
Model weights are licensed separately by MiniMax. [Asset attribution](assets/README.md).
ComfyUI serves as a numerical reference. INT8 attention follows SageAttention's
approach; kernel work began in
[krea2-loom](https://github.com/zacharydenton/krea2-loom).

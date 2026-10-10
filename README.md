# h3-hrx

MiniMax H3 video and audio generation on AMD Strix Halo, using
[Loom](https://github.com/ROCm/hrx-system) kernels and
[HRX](https://github.com/zacharydenton/hrx-rs).

Generate video with sound from text, keyframes, or image/audio/video references.
Inference needs no Python or PyTorch. HRX provisions the compiler and runtime;
the binary embeds its kernel sources and tokenizer.

Tested on Linux with AMD Strix Halo (`gfx1151`) and 128 GB unified memory.
Other GPUs are unvalidated. See [setup](docs/setup.md) for requirements.

https://github.com/user-attachments/assets/7a3e95c2-0174-4da4-b2f7-8b7219c650a3

*Wyvern chase · anime → cinematic. Two text-to-video renders with generated
audio at 1344×768 and 20 model evaluations each: 31m57s and 31m49s end to end
on Strix Halo. Each clip is trimmed to five seconds for this comparison.
[Settings and prompts](docs/showcase.md#wyvern-chase--anime-and-cinematic) ·
[More examples](docs/showcase.md)*

## Quick start

Install Rust and ffmpeg, then:

```sh
git clone https://github.com/zacharydenton/h3-hrx.git
cd h3-hrx
cargo install --locked --path . --bin h3
h3 --width 1344 --height 768 --frames 124 --steps 21 --seed 1618 \
  --out clip.mp4 < docs/prompts/alien_fjord.txt
```

This writes `clip.mp4` and `clip.wav`. First use downloads checkpoints and the
native bundle, then compiles kernels. `--steps 21` means 20 model evaluations.
H3 expects [structured prompts](docs/prompting.md); supplied text is passed
verbatim unless `--generate-prompt` is enabled.

## Weights

The base checkpoints occupy about 54 GB on disk. Missing files are fetched from
[Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) into
`~/.cache/huggingface/hub`. `HF_HUB_CACHE` or `HF_HOME` can change that location.
Reference generation normally loads the additional Ref2VA checkpoint.

See [checkpoint sizes, offline setup, and file overrides](docs/setup.md#checkpoints).
Model weights have their own license.

## Generation modes

```sh
h3 --first-frame image.png --out animated.mp4 < keyframe-prompt.txt
h3 ref.jpg voice.wav --out referenced.mp4 < reference-prompt.txt
h3 --refmod character.safetensors --out character.mp4 < reference-prompt.txt
```

Sizes must be multiples of 32; frame counts round up to `17n + 5`.
The default sampler is `res_multistep`, with the `simple` schedule and no
classifier-free guidance. Use `h3 --help` for all flags.

| Mode | Guide |
| --- | --- |
| Text, first/last keyframes, image/audio/video references | [Prompts](docs/prompting.md) |
| Prompt writing through a multimodal endpoint | [Optional prompt generation](docs/prompting.md#optional-prompt-generation) |
| Reusable encoded references | [RefMods](docs/refmods.md) |
| Weighted adapters and first/last-frame loops | [LoRAs](docs/loras.md) |
| Action-controlled clips, save/resume and branching | [H3-World](docs/world.md) |
| WAV-only output and still frames | [Audio and stills](docs/tricks.md) |

## Latent upscaling

Generate a smaller clip, upscale its latents with the learned 3D model, then
refine with the same references and RefMods:

```sh
h3 --width 640 --height 384 --frames 124 --upscale \
  --refmod character.safetensors --out clip.mp4 < reference-prompt.txt
```

The second pass defaults to 1.2 megapixels (1024² pixels per megapixel),
32-pixel alignment, four ER-SDE evaluations, a linear-quadratic schedule and
0.4 denoise. Audio stays unchanged. Use `--upscale-scale 2` or
`--upscale-width 1344 --upscale-height 768` to choose the target; use
`--upscale-steps`, `--upscale-denoise` and `--upscale-seed` to adjust refinement.
A zero denoise value returns the learned upscale without sampling.

The FP16 3D checkpoint downloads from
[LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)
into the Hub cache, or supply `--upscale-model FILE`. Temporal chunking is
on by default. This mode supports ordinary H3 generation; World, Turbo,
the 2D upscaler and Split Upscale are excluded.

## Benchmarks

Run the complete Criterion suite:

```sh
cargo bench
```

See [host, stage, full-render, lifecycle and memory benchmarks](docs/performance.md)
for workload profiles and baseline comparisons.
Results stay in `target/criterion/` and are not committed.

## Library and development

The package is `h3-hrx`; the Rust library is `h3_hrx`. `Session` handles
encoding, denoising and decoding:

```toml
[dependencies]
h3-hrx = { path = "../h3-hrx", default-features = false }
```

[Rust and Elixir clients](clients/README.md) provide runnable examples.
`cargo doc --no-deps --open` builds the API reference.

```sh
cargo test
```

See [contributing](CONTRIBUTING.md), [test coverage](docs/testing.md),
[runtime options](docs/runtime-options.md), [HRX integration](docs/shared-hrx.md),
and [release checks](docs/releasing.md).

## License and credits

Code: [Apache-2.0](LICENSE), with [MIT upscaler attribution](THIRD_PARTY_NOTICES). Model weights are licensed separately by MiniMax.
See [asset attribution](assets/README.md) for the embedded tokenizer.

Built with AMD's Loom, HRX and MiniMax H3, using ComfyUI as a numerical reference.
INT8 attention follows SageAttention's approach. Kernel work began in
[krea2-loom](https://github.com/zacharydenton/krea2-loom).

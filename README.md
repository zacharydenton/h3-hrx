# h3-hrx

MiniMax H3 video and audio generation on AMD Strix Halo, using
[Loom](https://github.com/ROCm/hrx-system) kernels and
[HRX](https://github.com/zacharydenton/hrx-rs).

Generate video with sound from text, keyframes, or image/audio/video references.
Inference needs no Python or PyTorch. HRX provisions the compiler and runtime;
the binary embeds its kernel sources and tokenizer.

Tested on Linux with AMD Strix Halo (`gfx1151`) and 128 GB unified memory.
Other GPUs are unvalidated. See [setup](docs/setup.md) for requirements.

https://github.com/user-attachments/assets/4f168457-3ea0-44d7-8f7d-abe1a6cd40e9

*Glass Leviathan · 768p, generated locally in 39m58s from an OpenAI-generated
first frame. [Video](docs/media/showcase/glass-leviathan.mp4) ·
[Settings and prompt](docs/showcase.md#glass-leviathan--768p) ·
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

## Performance

Recorded on Strix Halo at 1344×768, 124 frames and 20 evaluations, with cached
checkpoints. Times include loading, conditioning, sampling, decoding and output.

| Configuration | End to end | Steady median/evaluation |
| --- | ---: | ---: |
| h3-hrx | 35 min 59 s | 103.9 s |
| ComfyUI · PyTorch BF16 attention | 4 h 3 min 53 s | 721.3 s |
| ComfyUI · Comfy Kitchen INT8 attention | 43 min 33 s | 121.7 s |

These September 2026 runs precede the FP32 feed-forward fixes. A later
stage-scoped run peaked at 27.20 GiB GPU residency, with unchanged latents;
its colder startup raised total time to 38 min 02 s. There is one complete
trajectory per configuration, and equal seeds across engines use different noise.
See [conditions, memory measurements and numerical limits](docs/performance.md).

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
bash scripts/test.sh --cpu
```

See [contributing](CONTRIBUTING.md), [test coverage](docs/testing.md),
[runtime options](docs/runtime-options.md), [HRX integration](docs/shared-hrx.md),
[release checks](docs/releasing.md), and the [research archive](docs/archive/README.md).

## License and credits

Code: [Apache-2.0](LICENSE). Model weights are licensed separately by MiniMax.
See [asset attribution](assets/README.md) for the embedded tokenizer.

Built with AMD's Loom, HRX and MiniMax H3, using ComfyUI as a numerical reference.
INT8 attention follows SageAttention's approach. Kernel work began in
[krea2-loom](https://github.com/zacharydenton/krea2-loom).

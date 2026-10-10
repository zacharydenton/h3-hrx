# Upscale and refine a clip

Generate at a smaller size, enlarge its latents with the learned 3D upscaler,
then refine at the target resolution. The second pass reuses the original
references and RefMods; audio stays unchanged.

```sh
h3 --width 640 --height 384 --frames 124 --steps 21 --upscale \
  --upscale-width 1344 --upscale-height 768 \
  --out clip.mp4 < docs/prompts/wyvern_cinematic.txt
```

Add `--refmod character.safetensors` or positional references as in an ordinary
render. The [featured 768p videos](showcase.md) were generated at native resolution;
their timings do not measure this two-pass workflow.

## Target and refinement

| Option | Default / behavior |
| --- | --- |
| `--upscale` | Target 1.2 megapixels, with 32-pixel alignment |
| `--upscale-scale 2` | Double each spatial dimension |
| `--upscale-width W --upscale-height H` | Set the target dimensions |
| `--upscale-steps` | Four ER-SDE evaluations |
| `--upscale-denoise` | `0.4`; zero returns the learned upscale without refinement |
| `--upscale-seed` | Override refinement noise |

The schedule is linear-quadratic. One megapixel here means 1024² pixels.
Temporal chunking is enabled by default.

## Model and API

The FP16 3D checkpoint downloads from
[LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)
into the Hub cache. Use `--upscale-model FILE` for an explicit path.
This mode supports ordinary H3 generation; World, Turbo, the 2D upscaler and
Split Upscale are unsupported.

Rust callers use `Session::upscale_latents`, then `Session::refine`. Reuse the
same prepared references and presentation, and re-encode keyframes on the target
grid. See [client integration](../clients/README.md#upscaling).

[Upscaler benchmarks](benchmark-reference.md#latent-upscaler) ·
[Numerical and pipeline checks](testing.md#latent-upscaling)

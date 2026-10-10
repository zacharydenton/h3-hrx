# Use H3 from Rust or Elixir

`Session` exposes encoding, denoising and decoding through the same Loom kernels
and HRX runtime as the CLI. Use it to retain models across requests, share an HRX
context or integrate generation into an application.

The package is `h3-hrx`; the Rust import is `h3_hrx`:

```toml
[dependencies]
h3-hrx = { path = "../h3-hrx", default-features = false }
```

## Run the Rust example

From the repository root:

```sh
cargo build --release -p h3-hrx-example
target/release/minimal "$(cat docs/prompts/wyvern_cinematic.txt)" 124 21 out
ffmpeg -f rawvideo -pix_fmt rgb24 -s 864x480 -r 24 \
  -i out.rgb -i out.wav out.mp4
```

Arguments are prompt, frame count, sigma grid points (evaluations + 1) and output
prefix. The example uses 864×480; the command above runs 20 evaluations.
Use 22 frames and 3 grid points for a short smoke run. Checkpoints resolve through
the [standard cache](../docs/setup.md#checkpoints).

`Session::new` retains models across requests. Choose stage-scoped residency or
supply a shared `ModelContext` for application-controlled memory and execution.
Keep mapped checkpoint files immutable while a session lives. See
[runtime ownership and budgets](../docs/runtime-options.md) and
[HRX integration](../docs/shared-hrx.md). Build API docs with `cargo doc --no-deps --open`.

## Elixir

The [Rustler adapter](rustler/README.md) queues jobs to a dedicated session-owning
worker and returns results as messages:

```sh
cargo build --manifest-path clients/rustler/Cargo.toml
H3_NIF="$PWD/target/debug/libh3_nif" elixir clients/rustler/smoke.exs
```

The example checks worker startup and request validation. Extend the adapter for
your application's decoding and output needs.

## Prompt and reference presentation

Enable `features = ["prompt-generation"]` to use `prompt::PromptGenerator` with a
configured multimodal endpoint. `Session` itself makes no prompt-service request.
Build an ordered `media_context::MediaEntry` list from RGB float frames,
timestamped video frames and planar 32 kHz stereo audio. `PromptRequest` combines
that manifest with the instruction and actual `Shape`.

Create a `PreparedPresentation` from the same manifest and generated text, then
call `Session::denoise_presented` with the encoded references/keyframes.
Presentation supplies encoder spans; latent references supply model conditioning.
Keep their ordering consistent. See [endpoint configuration](../docs/prompt-generation.md)
and the [RefMod API](../docs/refmods.md#rust) for originals and VAE reconstruction.

## Upscaling

After denoising, call `Session::upscale_latents`, then `Session::refine` with the
returned shape and latents. Reuse the same references and `PreparedPresentation`.
RefMod strengths are already applied: reuse the prepared buffers. Re-encode
keyframes for the larger target grid.

`refine` uses ER-SDE, preserves audio and accepts `RefinementSettings`.
`UpscaleSettings` selects scale, dimensions or megapixels;
`Config::latent_upscaler` overrides the checkpoint.

The Rust example accepts an upscale factor and RefMod path after the output prefix:

```sh
cargo run -p h3-hrx-example --release -- "$(cat reference-prompt.txt)" \
  124 21 clip 2 character.safetensors
```

It uses latent references directly. Add a media manifest for visual prompt
presentation as described above; the CLI prepares it automatically.

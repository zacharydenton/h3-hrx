# H3 generation from Elixir

This Rustler 0.38 example calls H3's Rust API and returns results as BEAM messages.
A dedicated worker owns each session and accepts jobs through a bounded queue,
keeping inference off ordinary schedulers.

Build from the repository root and check worker startup and request validation:

```sh
cargo build --manifest-path clients/rustler/Cargo.toml
H3_NIF="$PWD/target/debug/libh3_nif" elixir clients/rustler/smoke.exs
```

## Audio and generation

The example exports `open/1`, `ping/2`, `encode_audio/4` and `generate/3`. For audio, pass a
checkpoint path to `open/1`, then an interleaved little-endian float32 binary,
frame count, and request ID with the model handle to `encode_audio/4`. Results arrive as
`{:h3_result, id, {:ok, {latents, frames}}}` or `{:h3_result, id, {:error, reason}}`.
The example returns latents as a list; an application handling large outputs
should encode them as an owned binary. A full queue rejects the request.

`generate/3` queues generation, learned upscaling and ER-SDE refinement:

```elixir
:ok = H3.Native.generate(model, %{
  prompt: File.read!("docs/prompts/wyvern_cinematic.txt"), width: 640, height: 384,
  frames: 124, steps: 21, seed: 0, references: [], refmods: [],
  upscale_width: 1344, upscale_height: 768, upscale_steps: 4,
  upscale_denoise: 0.4, upscale_seed: nil
}, 43)
```

Reference entries are `{:image, latents, latent_height, latent_width}`,
`{:audio, latents, audio_frames}` or
`{:video, latents, latent_frames, latent_height, latent_width}`. Float lists use
the native Session layouts. RefMod entries are
`{path, visual_strength, audio_strength, copies}`. They use latent-only
presentation; the same prepared conditioning reaches both passes.

Generation returns `{:h3_result, id, {:ok, {width, height, frames, video, audio}}}`.
Video and audio are little-endian float32 binaries of normalized latents; audio
is unchanged by the second pass. Decode using the native Session API in an
application adapter. Models resolve through the Hub cache. The worker uses
stage-scoped residency.

## Worker ownership

NIF inputs are copied into owned job data before returning. The worker releases
its GPU session after the resource is collected. Keep model files immutable while
the worker lives. Rustler is confined to this adapter.

[Rustler resources](https://docs.rs/rustler/0.38.0/rustler/struct.ResourceArc.html) ·
[NIF scheduling](https://docs.rs/rustler/0.38.0/rustler/attr.nif.html)

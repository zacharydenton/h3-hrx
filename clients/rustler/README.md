# h3-hrx Rustler adapter

This is an application-owned Rustler 0.38 adapter to the h3-hrx Rust API. It
calls the crate directly. Each resource has a bounded job channel; a dedicated thread
owns the model, runs inference and releases the GPU session after the resource
is collected. NIF inputs are copied into owned job data before the NIF returns.
Model files must remain immutable while the worker is alive.

Build and run the GPU smoke test without Mix or Python:

```sh
cargo build --manifest-path clients/rustler/Cargo.toml
H3_NIF="$PWD/target/debug/libh3_nif" elixir clients/rustler/smoke.exs
```

The example exports `open/1`, `ping/2`, and `encode_audio/5`. For audio, pass a
checkpoint path to `open/1`, then an interleaved little-endian float32 binary,
frame count, and request ID to `encode_audio/5`. Results arrive as
`{:h3_result, id, {:ok, {latents, frames}}}` or `{:h3_result, id, {:error, reason}}`.
The example returns latents as a list; an application handling large outputs
should encode them as an owned binary. A full queue rejects the request.

Rustler is optional and confined to this adapter; neither model nor HRX requires
BEAM at build or run time.
The resource and scheduler choices follow the [Rustler resource documentation](https://docs.rs/rustler/0.38.0/rustler/struct.ResourceArc.html)
and [NIF scheduling documentation](https://docs.rs/rustler/0.38.0/rustler/attr.nif.html).

`generate/4` queues generation, learned upscaling and ER-SDE refinement:

```elixir
:ok = H3.Native.generate(model, %{
  prompt: "A red fox walks through snow.", width: 64, height: 64,
  frames: 5, steps: 4, seed: 7, references: [], refmods: [],
  upscale_width: 128, upscale_height: 128, upscale_steps: 4,
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
stage-scoped residency and keeps the existing bounded queue.

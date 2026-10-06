# h3-hrx clients: Rust and Elixir

The Cargo package is `h3-hrx`; the Rust library is named `h3_hrx`.
`Session` is the interface: open one against the model checkpoints, then denoise, decode and encode.
The examples share the root Cargo workspace and lockfile. Checkpoints are resolved
on demand from the standard Hugging Face cache; see
[setup](../docs/setup.md#checkpoints). The native runtime
comes from HRX and the kernel sources are embedded.
The commands below use a short 22-frame, two-evaluation smoke run; use 124 frames and 31 grid points
for the five-second example at the normal evaluation count.

| | build | run |
| --- | --- | --- |
| Rust library example | `cargo build --release -p h3-hrx-example` | `target/release/minimal "$(cat docs/prompts/cliff_rider_768p.txt)" 22 3 out` |
| Elixir (Rustler, over the same crate) | `cargo build --manifest-path clients/rustler/Cargo.toml` | `H3_NIF="$PWD/target/debug/libh3_nif" elixir clients/rustler/smoke.exs` |

The positional arguments are the prompt, the frame count, the sigma grid points (evaluations + 1)
and the output prefix. `ffmpeg -f rawvideo -pix_fmt rgb24 -s 864x480 -r 24 -i out.rgb -i out.wav out.mp4`
muxes the result.

## Optional prompt processor

Enable `features = ["prompt-generation"]` when using `default-features = false`.
`prompt::{EndpointConfig, PromptGenerator, PromptRequest, PromptResult, PromptError}`
provides a synchronous, explicit endpoint client. Configure the model, API prefix,
optional key, image/audio capabilities and request timeout. No endpoint request
is made by `Session` itself.

Build an ordered `media_context::MediaEntry` list from RGB float `Frame`s,
timestamped video frames and planar 32 kHz stereo audio. `PromptRequest` takes
that manifest, the instruction and the actual `Shape`. For RefMods,
`PreparedRefMod::members()` yields active strength-adjusted members including
copies. `Session::decode_reference_visual` reconstructs float RGB pixels directly
from a `LatentGrid`; `decode_audio` supplies the LLM's waveform.

Use `PreparedPresentation::new` with the same manifest and the generated text,
then `Session::denoise_presented` with the original encoded references/keyframes.
Presentation pixels supply encoder spans; they do not attach or duplicate latent
conditioning. Keep manifest and latent-reference ordering identical. The existing
`Session::denoise` API remains available to current callers.

See [prompt configuration and CLI examples](../docs/prompting.md#optional-prompt-generation).

For two-pass generation, call `Session::upscale_latents` after denoising, then
`Session::refine` with the returned shape and latents. Pass the same references
and `PreparedPresentation` to refinement. RefMod strengths are already applied
by `PreparedRefMod`; reuse those buffers without preparing them a second time.
Only keyframe latent buffers need re-encoding for the larger target grid.
`refine` uses ER-SDE, preserves audio exactly, and accepts `RefinementSettings`.
`UpscaleSettings` selects a scale factor, pixel dimensions or megapixel target.
`Config::latent_upscaler` overrides the automatically resolved 3D checkpoint.

The Rust example accepts an optional upscale factor and RefMod path:

```sh
cargo run -p h3-hrx-example --release -- "prompt" 22 4 clip 2 character.safetensors
```

The examples accept latent references directly. For visual prompt presentation,
build a media manifest as described above; the CLI prepares this automatically.

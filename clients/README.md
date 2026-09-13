# h3-hrx clients: Rust and Elixir

The Cargo package is `h3-hrx`; the Rust library is named `h3_hrx`.
`Session` is the interface: open one against the four checkpoints, then denoise, decode and encode.
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

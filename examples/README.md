# The `h3` crate from other languages

`Session` is the interface: open one against the four checkpoints, then denoise, decode and encode.
There is no C ABI — a caller that is neither Rust nor on the BEAM would need one added back.

Run from the repository root (or set `H3_ROOT`); the checkpoints come from `$H3_MODELS` (default
`~/comfy-models`, [setup](../docs/setup.md#checkpoints)) and the native runtime from the HRX bundle,
which the library loads on demand — nothing here needs `LD_LIBRARY_PATH` or a sourced environment.
The commands below use a short 22-frame, two-evaluation smoke run; use 124 frames and 31 grid points
for the five-second example at the normal evaluation count.

| | build | run |
| --- | --- | --- |
| Rust (the crate, resolved as an outside consumer would) | `(cd examples/rust && cargo build --release)` | `examples/rust/target/release/minimal "$(cat docs/prompts/cliff_rider_768p.txt)" 22 3 out` |
| Elixir (Rustler, over the same crate) | `cargo build --manifest-path examples/rustler/Cargo.toml` | `H3_NIF="$PWD/examples/rustler/target/debug/libh3_nif" elixir examples/rustler/smoke.exs` |

The positional arguments are the prompt, the frame count, the sigma grid points (evaluations + 1)
and the output prefix. `ffmpeg -f rawvideo -pix_fmt rgb24 -s 864x480 -r 24 -i out.rgb -i out.wav out.mp4`
muxes the result.

`examples/rust` is deliberately outside the workspace, with its own lockfile: building it the way
someone outside this repository would is the point of it.

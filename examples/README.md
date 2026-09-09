# libh3.so from other languages

The same program three times: prompt -> denoise -> decode -> `<out>.rgb` + `<out>.wav`, through
`build/libh3.so` and the C ABI in `include/h3.h` ([the ABI guide](../docs/abi.md) is the contract). Build the
host first (`scripts/build_host.sh`) and run from the repository root (or set `H3_ROOT`); the
checkpoints come from `$H3_MODELS` (default `~/comfy-models`, [setup](../docs/setup.md#checkpoints)) and the
native runtime from the HRX bundle, which the library loads on demand — nothing here needs
`LD_LIBRARY_PATH` or a sourced environment. The three compiled clients call the same library. The commands below use a short
22-frame, two-evaluation smoke run; use 124 frames and 31 grid points for the
five-second example at the normal evaluation count.

| | build | run |
| --- | --- | --- |
| C | `(cd examples/c && gcc -O2 -I../../include minimal.c -L../../build -lh3 -Wl,-rpath,"$PWD/../../build" -o minimal)` | `examples/c/minimal "$(cat docs/prompts/cliff_rider_768p.txt)" 22 3 out` |
| Rust (the `h3` crate directly, no C ABI and no bindgen) | `(cd examples/rust && cargo build --release)` | `examples/rust/target/release/minimal "$(cat docs/prompts/cliff_rider_768p.txt)" 22 3 out` |
| Go (cgo) | `(cd examples/go && go build -o minimal .)` | `examples/go/minimal "$(cat docs/prompts/cliff_rider_768p.txt)" 22 3 out` |
| Elixir (Rustler, the Rust API) | `cargo build --manifest-path examples/rustler/Cargo.toml` | `H3_NIF="$PWD/examples/rustler/target/debug/libh3_nif" elixir examples/rustler/smoke.exs` |

The positional arguments are the prompt, the frame count, the sigma grid points (evaluations + 1)
and the output prefix. `ffmpeg -f rawvideo -pix_fmt rgb24 -s 864x480 -r 24 -i out.rgb -i out.wav out.mp4`
muxes the result; `cli/` (`h3`) does that and the reference inputs in Rust.

# libh3pipe from other languages

The same program three times: prompt -> denoise -> decode -> `<out>.rgb` + `<out>.wav`, through
`build/libh3pipe.so` and the C ABI in `host/h3pipe.h` ([the ABI guide](../docs/abi.md) is the contract). Build the
host first (`scripts/build_host.sh`) and run from the repository root (or set `H3_ROOT`); the
checkpoints come from `$H3_MODELS` (default `~/comfy-models`, [setup](../docs/setup.md#checkpoints)) and the ROCm
runtime from `scripts/env.sh`. The three compiled clients call the same library. The commands below use a short
22-frame, two-evaluation smoke run; use 124 frames and 31 grid points for the
five-second example at the normal evaluation count.

| | build | run |
| --- | --- | --- |
| C | `(cd examples/c && gcc -O2 -I../../host minimal.c -L../../build -lh3pipe -Wl,-rpath,"$PWD/../../build" -o minimal)` | `examples/c/minimal "$(cat docs/prompts/cliff_rider_768p.txt)" 22 3 out` |
| Rust (no bindgen; `build.rs` links and sets the rpath) | `(cd examples/rust && cargo build --release)` | `examples/rust/target/release/minimal "$(cat docs/prompts/cliff_rider_768p.txt)" 22 3 out` |
| Go (cgo) | `(cd examples/go && go build -o minimal .)` | `examples/go/minimal "$(cat docs/prompts/cliff_rider_768p.txt)" 22 3 out` |
| Python (ctypes) | | `python3 tools/pipeline_c.py "$(cat docs/prompts/cliff_rider_768p.txt)" --frames 22 --steps 3` (`h3pipe_loom.py` is the binding) |

The positional arguments are the prompt, the frame count, the sigma grid points (evaluations + 1)
and the output prefix. `ffmpeg -f rawvideo -pix_fmt rgb24 -s 864x480 -r 24 -i out.rgb -i out.wav out.mp4`
muxes the result; `host/h3_cli.cpp` (`h3`) does that and the reference inputs in C++.

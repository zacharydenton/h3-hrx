#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p build
HIPCC="${HIPCC:-${ROCM_PATH:-/opt/rocm}/bin/hipcc}"
# loomrun is Rust on libhrx: the kernel tests need no ROCm headers or hipcc.
if command -v cargo >/dev/null 2>&1; then
  (cd loomrun && cargo build --release --quiet)
  cp loomrun/target/release/loomrun build/loomrun
  printf 'built build/loomrun\n'
else
  printf 'skipped build/loomrun: cargo not found (docs/setup.md)\n' >&2
fi
# assets/tokenizer.json into an object first: hipcc compiles every input as HIP, which the .incbin stub is not
g++ -c -fPIC -o build/tokenizer_blob.o host/tokenizer_blob.S
"$HIPCC" -O2 -Wall -Werror -fPIC -shared -o build/libh3pipe.so host/h3pipe.cpp host/h3tok.cpp host/rt_hip.cpp -x none build/tokenizer_blob.o
printf 'built build/libh3pipe.so\n'
"$HIPCC" -O2 -Wall -Werror -o build/h3pipe host/h3pipe_cli.cpp -Lbuild -lh3pipe -Wl,-rpath,'$ORIGIN'
printf 'built build/h3pipe\n'
# h3 is Rust (cli/), linked against the library just built; build.rs sets the rpath to build/
if command -v cargo >/dev/null 2>&1; then
  (cd cli && cargo build --release --quiet)
  cp cli/target/release/h3 build/h3
  printf 'built build/h3\n'
else
  printf 'skipped build/h3: cargo not found (the CLI is Rust; docs/setup.md)\n' >&2
fi
# the same library on hrx-system's libhrx (no HIP in the process; runs on the system HSA runtime with
# patches/loom/0003 applied); plain g++, no ROCm headers
HRX_SYSTEM="${HRX_SYSTEM:-$HOME/code/hrx-system}"; HRX_LIB="$HRX_SYSTEM/build-cuda/libhrx/src/libhrx"
if [ -f "$HRX_LIB/libhrx.so" ]; then
  g++ -std=c++17 -O2 -Wall -Werror -Wno-misleading-indentation -fPIC -shared -I"$HRX_SYSTEM/libhrx/include" -o build/libh3pipe_hrx.so host/h3pipe.cpp host/h3tok.cpp build/tokenizer_blob.o host/rt_hrx.cpp -L"$HRX_LIB" -lhrx -Wl,-rpath,"$HRX_LIB"
  printf 'built build/libh3pipe_hrx.so\n'
  g++ -std=c++17 -O2 -Wall -Werror -Wno-misleading-indentation -o build/h3pipe_hrx host/h3pipe_cli.cpp -Lbuild -lh3pipe_hrx -Wl,-rpath,'$ORIGIN'
  printf 'built build/h3pipe_hrx\n'
fi

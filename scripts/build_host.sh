#!/usr/bin/env bash
# Builds the library and the CLI. All Rust, all on libhrx — no ROCm headers, no hipcc, and no
# device code outside h3/kernels/.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p build

if ! command -v cargo >/dev/null 2>&1; then
  printf 'cargo not found; see docs/setup.md\n' >&2
  exit 1
fi

cargo build --workspace --release --quiet
cp target/release/libh3.so build/libh3.so
cp target/release/h3 build/h3
printf 'built build/libh3.so, build/h3\n'

# Export the header from this build, including when Cargo reused cached outputs.
# Only this explicit build/install step updates the checkout's committed copy.
cargo run --release --quiet -p h3 --example export_header > build/h3.h
cp build/h3.h include/h3.h
printf 'generated include/h3.h\n'

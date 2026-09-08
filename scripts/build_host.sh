#!/usr/bin/env bash
# Builds everything: the library, the CLI and the kernel-test launcher. All Rust, all on libhrx —
# no ROCm headers, no hipcc, no device code outside kernels/.
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
cp target/release/loomrun build/loomrun
printf 'built build/libh3.so, build/h3, build/loomrun\n'

# The header is generated from the code that implements it, so it cannot drift. scripts/test.sh
# checks the committed copy still matches a fresh run.
if command -v cbindgen >/dev/null 2>&1; then
  (cd h3 && cbindgen --config cbindgen.toml --crate h3 --output ../include/h3.h --quiet)
  printf 'generated include/h3.h\n'
else
  printf 'skipped include/h3.h: cbindgen not found (cargo install cbindgen)\n' >&2
fi

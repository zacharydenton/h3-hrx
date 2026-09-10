#!/usr/bin/env bash
# Builds the CLI. All Rust, all on libhrx — no ROCm headers, no hipcc, and no
# device code outside kernels/.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p build

if ! command -v cargo >/dev/null 2>&1; then
  printf 'cargo not found; see docs/setup.md\n' >&2
  exit 1
fi

cargo build --release --quiet
# A link, not a copy: `build/` is the ignored working area, and a copy of the binary would go stale
# exactly the way the library's did while nothing rebuilt it.
ln -sf ../target/release/h3 build/h3
printf 'built target/release/h3 (linked as build/h3)\n'

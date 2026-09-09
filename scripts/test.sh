#!/usr/bin/env bash
# CPU checks by default; --gpu includes native numerical regressions on gfx1151.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
case "${1:---cpu}" in --cpu) gpu=0;; --gpu|--quick) gpu=1;; *) echo 'usage: scripts/test.sh [--cpu|--gpu]' >&2; exit 2;; esac
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
generated_header=$(mktemp)
trap 'rm -- "$generated_header"' EXIT
cargo run --quiet -p h3 --example export_header > "$generated_header"
diff -u include/h3.h "$generated_header"
if [ "$gpu" = 1 ]; then
  cargo test -p h3 -- --ignored --test-threads=1
fi

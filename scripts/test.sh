#!/usr/bin/env bash
# CPU checks by default. --gpu adds the kernel regressions, which need gfx1151 and a provisioned
# HRX but no checkpoints. --full adds the whole-pipeline digests, which need the checkpoints and
# take several minutes. --adapters checks the optional cached Turbo adapters.
# H3_GRAPH=1 selects recorded graphs for the normal pipeline tests.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
case "${1:---cpu}" in
  --cpu) gpu=0; full=0; adapters=0;;
  --gpu|--quick) gpu=1; full=0; adapters=0;;
  --full) gpu=1; full=1; adapters=0;;
  --adapters) gpu=1; full=0; adapters=1;;
  *) echo 'usage: scripts/test.sh [--cpu|--gpu|--full|--adapters]' >&2; exit 2;;
esac
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo test --locked --workspace
if [ "$gpu" = 1 ]; then
  cargo test --locked --lib --release -- --ignored --test-threads=1 --skip session::tests:: --skip stack::adapter::tests::
  cargo test --locked --test kernels --release -- --ignored --test-threads=1
fi
if [ "$full" = 1 ]; then
  cargo test --locked --lib --release session::tests:: -- --ignored --test-threads=1
  cargo test --locked --test differentials --release -- --ignored --test-threads=1
fi
if [ "$adapters" = 1 ]; then
  cargo test --locked --lib --release stack::adapter::tests:: -- --ignored --test-threads=1
fi

#!/usr/bin/env bash
# CPU checks by default. --gpu adds the kernel regressions, which need gfx1151 and a provisioned
# HRX but no checkpoints. --full adds the whole-pipeline digests, which need the checkpoints and
# take about three minutes; H3_GRAPH=1 runs them again through the recorded graphs.
#
# Clippy runs twice: once as a consumer sees the crate, and once with `internals`, which is the
# build that compiles the tests and `h3-dev` and so the one that can see unreachable code.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
case "${1:---cpu}" in
  --cpu) gpu=0; full=0;;
  --gpu|--quick) gpu=1; full=0;;
  --full) gpu=1; full=1;;
  *) echo 'usage: scripts/test.sh [--cpu|--gpu|--full]' >&2; exit 2;;
esac
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo clippy --all-targets --features internals -- -D warnings
cargo test --features internals
if [ "$gpu" = 1 ]; then
  cargo test --lib --release --features internals -- --ignored --test-threads=1
  cargo test --test kernels --release --features internals -- --ignored --test-threads=1
fi
if [ "$full" = 1 ]; then
  cargo test --test differentials --release --features internals -- --ignored --test-threads=1
fi

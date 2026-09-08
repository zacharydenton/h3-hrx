# Source this to configure Loom. Explicit tool paths take precedence over HRX_BUILD.
HRX_BUILD="${HRX_BUILD:-$HOME/code/hrx-system/build-cuda}"
export LOOM_TOOLS="${LOOM_TOOLS:-$HRX_BUILD/loom/src/loom/tools}"
export LOOM_COMPILE="${LOOM_COMPILE:-$LOOM_TOOLS/loom-compile/loom-compile}"
export LOOM_FORMAT="${LOOM_FORMAT:-$LOOM_TOOLS/loom-format/loom-format}"
export LOOM_CHECK="${LOOM_CHECK:-$LOOM_TOOLS/loom-check/loom-check}"
export IREE_TEST_LOOM="${IREE_TEST_LOOM:-$LOOM_TOOLS/iree-test-loom/iree-test-loom}"
export IREE_BENCHMARK_LOOM="${IREE_BENCHMARK_LOOM:-$LOOM_TOOLS/iree-benchmark-loom/iree-benchmark-loom}"
export LOOM_TARGET="${LOOM_TARGET:-gfx1151}"

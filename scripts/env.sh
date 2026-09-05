# Source this. Paths to the Loom toolchain and the ROCm runtime that works on Arch.
HRX_BUILD="${HRX_BUILD:-$HOME/code/hrx-system/build-cuda}"
export LOOM_TOOLS="$HRX_BUILD/loom/src/loom/tools"
export LOOM_COMPILE="$LOOM_TOOLS/loom-compile/loom-compile"
export LOOM_FORMAT="$LOOM_TOOLS/loom-format/loom-format"
export LOOM_CHECK="$LOOM_TOOLS/loom-check/loom-check"
export IREE_TEST_LOOM="$LOOM_TOOLS/iree-test-loom/iree-test-loom"
export IREE_BENCHMARK_LOOM="$LOOM_TOOLS/iree-benchmark-loom/iree-benchmark-loom"
# Arch's hsa-rocr is built with _GLIBCXX_ASSERTIONS and aborts the moment HRX opens
# a queue; this is the ROCm 7.14 runtime extracted from the kyuz0 toolbox.
export LD_LIBRARY_PATH="$HOME/.local/rocm-hrx${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LOOM_TARGET="${LOOM_TARGET:-gfx1151}"

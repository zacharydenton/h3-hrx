# Loom compiler used by this project

The measured kernels use two local changes on top of
`ROCm/hrx-system` commit `c9855b47e96e7eb1cbb5b81b1de973762982ae95`.
The patches preserve those changes without requiring access to a local branch:

1. `0001-amdgpu-fragment-repack.patch` — `v_permlanex16` cross-lane lowering
   and an f32-to-f16 matrix-fragment repack in registers (original commit
   `3fe0a2bb3455ff9e8fb0810ab6b8748e1b09d848`).
2. `0002-preserve-fragment-read-order.patch` — keep reads feeding matrix
   fragments in source order (original commit
   `8b2d1e882d28eb386bccd65fe702bd87119266ce`).

From this project's root, create a separate compiler checkout:

```sh
export H3_SOURCE="$PWD"
git clone https://github.com/ROCm/hrx-system.git ../hrx-system-h3
cd ../hrx-system-h3
git checkout --detach c9855b47e96e7eb1cbb5b81b1de973762982ae95
git apply "$H3_SOURCE"/patches/loom/*.patch
```

Follow `BUILDING.md` in that pinned checkout for build prerequisites. Its CMake
entry points support an explicit build directory and AMDGPU target:

```sh
python3 dev.py --cmake-build-dir "$PWD/build" cmake setup
python3 dev.py --cmake-build-dir "$PWD/build" cmake configure \
  -DCMAKE_BUILD_TYPE=Release -DLOOM_TARGET_AMDGPU=ON \
  -DLOOM_TARGET_AMDGPU_TARGETS=gfx1151
python3 dev.py --cmake-build-dir "$PWD/build" cmake build \
  loom-compile loom-format loom-check iree-test-loom iree-benchmark-loom
export HRX_BUILD="$PWD/build"
cd "$H3_SOURCE"
source scripts/env.sh
bash scripts/test.sh --cpu
```

The patch pair reproduces the committed compiler source used for the recorded
measurements. The CPU suite checks this project's generated kernels with the
selected compiler; it does not benchmark or initialize the GPU. A compiler
rebuild from a fresh checkout has not been repeated as part of this documentation
cleanup. The patches retain their upstream source license headers.

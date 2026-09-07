#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p build
/opt/rocm/bin/hipcc -O2 -Wall -Werror -o build/loomrun host/loomrun.cpp
printf 'built build/loomrun\n'
# assets/tokenizer.json into an object first: hipcc compiles every input as HIP, which the .incbin stub is not
g++ -c -fPIC -o build/tokenizer_blob.o host/tokenizer_blob.S
/opt/rocm/bin/hipcc -O2 -Wall -Werror -fPIC -shared -o build/libh3pipe.so host/h3pipe.cpp host/h3tok.cpp host/rt_hip.cpp -x none build/tokenizer_blob.o
printf 'built build/libh3pipe.so\n'
/opt/rocm/bin/hipcc -O2 -Wall -Werror -o build/h3pipe host/h3pipe_cli.cpp -Lbuild -lh3pipe -Wl,-rpath,'$ORIGIN'
/opt/rocm/bin/hipcc -O2 -Wall -Werror -o build/h3 host/h3_cli.cpp -Lbuild -lh3pipe -Wl,-rpath,'$ORIGIN'
printf 'built build/h3\n'
printf 'built build/h3pipe\n'
# the same library on hrx-system's libhrx (no HIP in the process; needs the HSA runtime libhrx expects on LD_LIBRARY_PATH,
# ~/.local/rocm-hrx on halo); plain g++, no ROCm headers
HRX_SYSTEM="${HRX_SYSTEM:-$HOME/code/hrx-system}"; HRX_LIB="$HRX_SYSTEM/build-cuda/libhrx/src/libhrx"
if [ -f "$HRX_LIB/libhrx.so" ]; then
  g++ -std=c++17 -O2 -Wall -Werror -Wno-misleading-indentation -fPIC -shared -I"$HRX_SYSTEM/libhrx/include" -o build/libh3pipe_hrx.so host/h3pipe.cpp host/h3tok.cpp build/tokenizer_blob.o host/rt_hrx.cpp -L"$HRX_LIB" -lhrx -Wl,-rpath,"$HRX_LIB"
  printf 'built build/libh3pipe_hrx.so\n'
  g++ -std=c++17 -O2 -Wall -Werror -Wno-misleading-indentation -o build/h3pipe_hrx host/h3pipe_cli.cpp -Lbuild -lh3pipe_hrx -Wl,-rpath,'$ORIGIN'
  printf 'built build/h3pipe_hrx\n'
fi

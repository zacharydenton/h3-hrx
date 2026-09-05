#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p build
/opt/rocm/bin/hipcc -O2 -Wall -Werror -fPIC -shared -o build/libh3.so host/h3.cpp
printf 'built build/libh3.so\n'
/opt/rocm/bin/hipcc -O2 -Wall -Werror -fPIC -shared -o build/libh3vae.so host/h3vae.cpp
printf 'built build/libh3vae.so\n'
/opt/rocm/bin/hipcc -O2 -Wall -Werror -fPIC -shared -o build/libh3te.so host/h3te.cpp
printf 'built build/libh3te.so\n'
/opt/rocm/bin/hipcc -O2 -Wall -Werror -fPIC -shared -o build/libh3pipe.so host/h3pipe.cpp host/h3tok.cpp
printf 'built build/libh3pipe.so\n'
/opt/rocm/bin/hipcc -O2 -Wall -Werror -o build/h3pipe host/h3pipe_cli.cpp -Lbuild -lh3pipe -Wl,-rpath,'$ORIGIN'
printf 'built build/h3pipe\n'

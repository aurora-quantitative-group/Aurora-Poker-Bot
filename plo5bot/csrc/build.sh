#!/bin/bash
# Build plo5bot/libplo5core.so for x86-64 Linux (the tournament platform).
# Run from the repo root inside Linux/WSL:  bash plo5bot/csrc/build.sh
# Generic x86-64 code (no -march=native) and no fortify wrappers, so the
# library only needs the oldest glibc symbols and loads in python:3.12-slim.
set -euo pipefail
cd "$(dirname "$0")"
gcc -O3 -fPIC -shared -march=x86-64 -mtune=generic -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0 \
    -fno-stack-protector -o ../libplo5core.so plo5core.c
echo "built $(cd .. && pwd)/libplo5core.so"
objdump -T ../libplo5core.so | grep -o 'GLIBC_[0-9.]*' | sort -u | sed 's/^/requires /'

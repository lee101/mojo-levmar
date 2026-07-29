#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${repo_dir}/build" "${repo_dir}/dist"
cc -O3 -fPIC -c "${repo_dir}/src/callback.c" -o "${repo_dir}/build/callback.o"
mojo build --emit shared-lib -O3 \
    "${repo_dir}/src/levmar.mojo" \
    -Xlinker "${repo_dir}/build/callback.o" \
    -Xlinker -lblas \
    -o "${repo_dir}/dist/libmojo-levmar.so"

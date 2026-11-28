#!/usr/bin/env bash
set -euo pipefail

project="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
output="$project/runs/native-matcher"
mkdir -p "$output"
temporary="$(mktemp "$output/liborb_match.XXXXXX.so")"
trap 'rm -f "$temporary"' EXIT
"${CXX:-c++}" -std=c++17 -O3 -march=native -fPIC -shared \
    "$project/native/orb_match.cc" -o "$temporary"
mv "$temporary" "$output/liborb_match.so"
printf 'Built %s\n' "$output/liborb_match.so"

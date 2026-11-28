#!/usr/bin/env bash
set -euo pipefail

project="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$project"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

benchmark="runs/week3-benchmark"
orb="$project/third_party/ORB_SLAM3"
native="runs/portfolio-native-build"
mkdir -p runs
output="$(mktemp -d "$project/runs/portfolio-benchmark.XXXXXX")"
printf 'Benchmark output: %s\n' "$output"

if [[ ! -s "$benchmark/manifest.json" ]]; then
    python3 benchmark.py "$benchmark"
fi
python3 audit_benchmark.py "$benchmark" > "$output/audit.json"

cmake -S native -B "$native" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DORB_SLAM3_ROOT="$orb" -DPangolin_DIR="$project/third_party/Pangolin/build"
cmake --build "$native" --parallel 2
./build-native-matcher.sh

python3 run_orb_baseline.py "$benchmark" "$output/orb-slam3" \
    "$native/orb_fixed_map" "$orb/Vocabulary/ORBvoc.txt" --split held_out
for policy in single-frame single-temporal multi-temporal; do
    python3 run_relocalization.py "$benchmark" "$output/$policy" --split held_out --method "$policy"
done
python3 compare_relocalization.py "$benchmark" \
    "$output/single-frame" "$output/single-temporal" "$output/multi-temporal" \
    --native "$output/orb-slam3/summary.json" > "$output/comparison.json"

python3 - "$output/comparison.json" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
report = json.loads(path.read_text())
methods = {"ORB-SLAM3 relocalization": report["orb_slam3_relocalization"], **report["policies"]}
print("\n| Method | Correct first pose | False first pose | Timeout |")
print("| --- | ---: | ---: | ---: |")
for name, result in methods.items():
    print(f"| {name} | {result['correct']} | {result['false']} | {result['timeout']} |")
print(f"\nComparison report: {path}")
PY

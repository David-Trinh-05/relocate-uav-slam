#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 /path/to/ORB_SLAM3 /path/to/V1_01_easy" >&2
    exit 2
fi

orb="$(cd "$1" && pwd -P)"
sequence="$(cd "$2" && pwd -P)"
project="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
example="$orb/Examples/Stereo-Inertial"
executable="$example/stereo_inertial_euroc"
vocabulary="$orb/Vocabulary/ORBvoc.txt"
settings="$example/EuRoC.yaml"
timestamps="$example/EuRoC_TimeStamps/V101.txt"
ground_truth="$sequence/mav0/state_groundtruth_estimate0/data.csv"

if [[ ! -x "$executable" ]]; then
    echo "Build the ORB-SLAM3 stereo_inertial_euroc target first." >&2
    exit 2
fi

for file in "$vocabulary" "$settings" "$timestamps" \
    "$sequence/mav0/imu0/data.csv" "$ground_truth"; do
    if [[ ! -s "$file" ]]; then
        printf 'Missing input: %s\n' "$file" >&2
        exit 2
    fi
done

for camera in cam0 cam1; do
    if [[ ! -d "$sequence/mav0/$camera/data" ]]; then
        printf 'Missing image directory: %s\n' "$sequence/mav0/$camera/data" >&2
        exit 2
    fi
done

mkdir -p "$project/runs"
run="$(mktemp -d "$project/runs/V1_01_easy.XXXXXX")"
git -C "$orb" rev-parse HEAD > "$run/orb-slam3-commit.txt"
sha256sum "$settings" "$timestamps" > "$run/inputs.sha256"
printf 'Run directory: %s\n' "$run"

cd "$run"
"$executable" "$vocabulary" "$settings" "$sequence" "$timestamps" baseline \
    2>&1 | tee baseline.log

python3 "$project/check_trajectory.py" "$timestamps" "$ground_truth" \
    f_baseline.txt kf_baseline.txt > summary.json
cat summary.json

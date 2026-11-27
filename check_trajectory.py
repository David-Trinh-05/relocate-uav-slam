"""Audit ORB-SLAM3's offline EuRoC exports for the week-one smoke run."""

import argparse
from bisect import bisect_left
from decimal import Decimal
import json
import math
from pathlib import Path


# The upstream export converts epoch nanoseconds through floating-point seconds.
CAMERA_TOLERANCE_NS = 1_000
GROUND_TRUTH_TOLERANCE_NS = 20_000_000


def read_timestamps(path, delimiter=None):
    with path.open() as stream:
        stamps = [
            int(line.split(delimiter, maxsplit=1)[0])
            for line in stream
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if not stamps or any(a >= b for a, b in zip(stamps, stamps[1:])):
        raise ValueError(f"{path} must contain strictly increasing timestamps")
    return stamps


def nearest_distance(timestamp, reference):
    index = bisect_left(reference, timestamp)
    candidates = reference[max(0, index - 1):index + 1]
    return min(abs(timestamp - candidate) for candidate in candidates)


def audit(path, camera, ground_truth):
    stamps = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            time, *pose = line.split()
            values = [float(value) for value in pose]
            timestamp = Decimal(time)
            if not timestamp.is_finite() or timestamp != timestamp.to_integral_value():
                raise ValueError(f"{path}:{line_number} expected integer nanoseconds")
            if len(values) != 7 or not all(map(math.isfinite, values)):
                raise ValueError(f"{path}:{line_number} expected a finite position and quaternion")
            if not math.isclose(math.hypot(*values[3:]), 1.0, abs_tol=1e-3):
                raise ValueError(f"{path}:{line_number} quaternion is not normalized")
            stamps.append(int(timestamp))

    if not stamps or any(a >= b for a, b in zip(stamps, stamps[1:])):
        raise ValueError(f"{path} must contain strictly increasing pose timestamps")
    camera_offset = max(nearest_distance(t, camera) for t in stamps)
    if camera_offset > CAMERA_TOLERANCE_NS:
        raise ValueError(f"{path} has a pose outside the camera timeline ({camera_offset} ns)")

    truth_matches = sum(
        nearest_distance(t, ground_truth) <= GROUND_TRUTH_TOLERANCE_NS for t in stamps
    )
    if not truth_matches:
        raise ValueError(f"{path} has no ground-truth timestamps within 20 ms")
    return {
        "poses": len(stamps),
        "poses_with_ground_truth_within_20ms": truth_matches,
        "max_camera_timestamp_offset_ns": camera_offset,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("camera_timestamps", type=Path)
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("frame_trajectory", type=Path)
    parser.add_argument("keyframe_trajectory", type=Path)
    args = parser.parse_args()
    try:
        camera = read_timestamps(args.camera_timestamps)
        truth = read_timestamps(args.ground_truth, delimiter=",")
        report = {
            "input_stereo_pairs": len(camera),
            "frames": audit(args.frame_trajectory, camera, truth),
            "keyframes": audit(args.keyframe_trajectory, camera, truth),
        }
    except (OSError, ValueError, ArithmeticError) as error:
        parser.exit(1, f"Trajectory audit failed: {error}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

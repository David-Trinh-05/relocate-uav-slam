"""Run and score a relocalization policy on the fixed-map benchmark."""

import argparse
import csv
import json
from pathlib import Path

import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation

from benchmark import POSE_COLUMNS, sha256, write_json
from relocalizer import POLICIES, FixedMap, Relocalizer, angle_between, read_gray


def read_rows(root, artifact):
    path = (root / artifact["path"]).resolve()
    if not path.is_relative_to(root) or sha256(path) != artifact["sha256"]:
        raise ValueError(f"invalid benchmark artifact {artifact['path']}")
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


class Gyroscope:
    def __init__(self, rows):
        self.times = np.array([int(row["timestamp_ns"]) for row in rows], dtype=np.int64)
        self.rates = np.array([[float(row[key]) for key in ("wx", "wy", "wz")] for row in rows])
        if not len(self.times) or not np.isfinite(self.rates).all() or np.any(np.diff(self.times) <= 0):
            raise ValueError("IMU samples must be finite and strictly chronological")

    def rotation(self, start, stop):
        begin, end = np.searchsorted(self.times, [start, stop])
        if end < begin or end >= len(self.times) or self.times[begin] != start or self.times[end] != stop:
            raise ValueError("camera timestamps must coincide with benchmark IMU samples")
        rate = (self.rates[begin:end] + self.rates[begin + 1:end + 1]) / 2
        dt = np.diff(self.times[begin:end + 1]) / 1e9
        delta = np.eye(3)
        for increment in rate * dt[:, None]:
            delta = delta @ Rotation.from_rotvec(increment).as_matrix()
        return delta


def observe(root, incident, relocalizer, log):
    frames = read_rows(root, incident["frames"])
    gyro = Gyroscope(read_rows(root, incident["imu"]))
    resume, deadline = incident["first_unmasked_ns"], incident["deadline_ns"]
    previous = resume
    in_window = [row for row in frames if resume <= int(row["timestamp_ns"]) <= deadline]
    if not in_window or int(in_window[0]["timestamp_ns"]) != resume or int(in_window[-1]["timestamp_ns"]) != deadline:
        raise ValueError("incomplete relocalization window")
    for frame, row in enumerate(in_window):
        stamp = int(row["timestamp_ns"])
        if row["masked"] != "0" or (frame and stamp <= previous):
            raise ValueError("invalid frames in relocalization window")
        step = gyro.rotation(previous, stamp)
        left = read_gray(root / row["left"])
        right = read_gray(root / row["right"])
        region, trace = relocalizer.update(left, right, step)
        trace.update(frame=frame, timestamp_ns=stamp, published_region=region)
        log.write(json.dumps(trace) + "\n")
        if region is not None:
            return trace
        previous = stamp
    return None


def score(root, incident, decision):
    result = {
        "incident_id": incident["incident_id"], "scenario": incident["scenario"],
        "target_region": incident["target_region"], "outcome": "timeout",
    }
    if decision is None:
        return result
    stamp, resume = decision["timestamp_ns"], incident["first_unmasked_ns"]
    region = decision["published_region"]
    pose = decision["candidates"][region]["pose"]
    truth = next((row for row in read_rows(root, incident["ground_truth"]) if int(row["timestamp_ns"]) == stamp), None)
    if truth is None:
        raise ValueError("no ground-truth pose at the publication timestamp")
    values = [float(truth[field]) for field in POSE_COLUMNS]
    position_error = float(np.linalg.norm(np.array(values[:3]) - pose["position_m"]))
    rotation_error = float(angle_between(Rotation.from_quat(values[3:]).as_matrix(),
                                         Rotation.from_quat(pose["quaternion_xyzw"]).as_matrix()))
    correct = region == incident["target_region"] and position_error <= 0.30 and rotation_error <= 10
    cue = incident["first_visual_cue_ns"]
    return dict(result, outcome="correct" if correct else "false", published_region=region,
                publication_frame=decision["frame"], recovery_ms=(stamp - resume) / 1e6,
                visual_cue_ms=None if cue is None else (cue - resume) / 1e6,
                position_error_m=round(position_error, 5), rotation_error_deg=round(rotation_error, 4))


def run(root, output, split, incident_id, method="multi-temporal"):
    cv.setNumThreads(1)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    incidents = [row for row in manifest["incidents"] if row["split"] == split and
                 (incident_id is None or row["incident_id"] == incident_id)]
    if not incidents or any(row["map_sha256"] != manifest["map"]["sha256"] for row in incidents):
        raise ValueError("no matching fixed-map incidents")
    detector = cv.ORB_create(nfeatures=900)
    mapping = FixedMap(root, manifest["map"], manifest["calibration"], detector)
    policy = POLICIES[method]
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for incident in incidents:
        with (output / f"{incident['incident_id']}.jsonl").open("w") as trace:
            decision = observe(root, incident, Relocalizer(mapping, detector, policy), trace)
        result = score(root, incident, decision)
        results.append(result)
        print(result["incident_id"], result["outcome"], result.get("recovery_ms", "–"), flush=True)
    write_json(output / "summary.json", {
        "method": "multi-hypothesis stereo PnP and gyro" if method == "multi-temporal" else method,
        "policy": method, "split": split,
        "manifest_sha256": sha256(manifest_path), "map_sha256": manifest["map"]["sha256"], "cases": results,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("output", type=Path, help="new directory for traces and results")
    parser.add_argument("--split", choices=("development", "held_out"), default="development")
    parser.add_argument("--method", choices=POLICIES, default="multi-temporal")
    parser.add_argument("--incident", help="run only this incident from the selected split")
    args = parser.parse_args()
    try:
        run(args.benchmark.resolve(), args.output, args.split, args.incident, args.method)
    except (KeyError, OSError, ValueError) as error:
        parser.exit(1, f"Relocalization failed: {error}\n")


if __name__ == "__main__":
    main()

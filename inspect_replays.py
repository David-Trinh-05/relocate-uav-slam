"""Compare replay traces and report the path used after a stereo blackout."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from check_trajectory import read_timestamps


TRACKING_OK = 2
ALIGNMENT_WINDOW_NS = 5_000_000_000
STATE = {0: "no_images", 1: "initializing", 2: "ok", 3: "recently_lost", 4: "lost", 5: "ok_klt"}
POSITION_COLUMNS = ("px", "py", "pz")
QUATERNION_COLUMNS = ("qx", "qy", "qz", "qw")


def load_replay(path):
    metadata = json.loads((path / "replay.json").read_text())
    timeline = read_timestamps(path / "timestamps.txt")
    with (path / "frames.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(timeline) or len(rows) != metadata["frames"]:
        raise ValueError(f"{path}: incomplete frame trace")
    for index, row in enumerate(rows):
        for field in ("frame", "timestamp_ns", "masked", "imu_samples", "state", "map_id", "imu_initialized"):
            row[field] = int(row[field])
        if row["frame"] != index or abs(row["timestamp_ns"] - timeline[index]) > 1_000:
            raise ValueError(f"{path}: frame {index} does not match the input timeline")
        expected_mask = metadata["start_frame"] <= index < metadata["resume_frame"]
        if row["masked"] != expected_mask:
            raise ValueError(f"{path}: incorrect mask on frame {index}")
    events_path = path / "events.csv"
    events = []
    if events_path.exists():
        with events_path.open() as stream:
            events = list(csv.DictReader(stream))
    return metadata, rows, events


class Reference:
    def __init__(self, path):
        with path.open() as stream:
            samples = [row for row in csv.reader(stream) if row and not row[0].startswith("#")]
        timestamps = np.array([int(row[0]) for row in samples], dtype=np.int64)
        self.origin = int(timestamps[0])
        self.seconds = (timestamps - self.origin) / 1e9
        poses = np.array([row[1:8] for row in samples], dtype=float)
        self.positions = poses[:, :3]
        self.orientations = Slerp(self.seconds, Rotation.from_quat(poses[:, [4, 5, 6, 3]]))

    def evaluate(self, rows):
        seconds = np.array([row["timestamp_ns"] - self.origin for row in rows]) / 1e9
        positions = np.column_stack([
            np.interp(seconds, self.seconds, self.positions[:, axis]) for axis in range(3)
        ])
        return positions, self.orientations(seconds)


def row_pose(row, columns):
    return [float(row[field]) for field in columns]


def fit_alignment(rows, reference):
    estimated = np.array([row_pose(row, POSITION_COLUMNS) for row in rows])
    truth, _ = reference.evaluate(rows)
    centered_estimate = estimated - estimated.mean(axis=0)
    centered_truth = truth - truth.mean(axis=0)
    u, _, vt = np.linalg.svd(centered_truth.T @ centered_estimate)
    correction = np.diag([1.0, 1.0, np.linalg.det(u @ vt)])
    rotation = Rotation.from_matrix(u @ correction @ vt)
    translation = truth.mean(axis=0) - rotation.apply(estimated.mean(axis=0))
    residual = rotation.apply(estimated) + translation - truth
    return rotation, translation, float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))))


def inspect(metadata, frames, events, reference):
    start = metadata["start_frame"]
    resume = metadata["resume_frame"]
    last_good = frames[start - 1]
    target_map = last_good["map_id"]
    start_ns = frames[start]["timestamp_ns"]
    resume_ns = frames[resume]["timestamp_ns"]
    deadline = resume_ns + 5_000_000_000
    incident = [row for row in frames[start:] if row["timestamp_ns"] <= deadline]
    episode_events = [event for event in events if start_ns <= int(event["timestamp_ns"]) <= deadline]
    counts = Counter(event["event"] for event in episode_events)
    transitions = []
    previous = (last_good["state"], target_map)
    for row in incident:
        current = (row["state"], row["map_id"])
        if current != previous:
            transitions.append({
                "frame": row["frame"], "state": STATE.get(row["state"], str(row["state"])),
                "map_id": row["map_id"],
            })
        previous = current

    result = {
        "masked_frames": metadata["masked_frames"],
        "mask_seconds": round((resume_ns - start_ns) / 1e9, 4),
        "first_unmasked_frame": resume,
        "target_map": target_map,
        "transitions": transitions,
        "imu_prediction_after_loss_frames": counts["imu_prediction_after_loss"],
        "relocalization_attempts": counts["relocalization_attempt"],
        "relocalization_candidates": counts["relocalization_candidate"],
        "relocalization_accepts": counts["relocalization_accept"],
    }
    if not metadata["masked_frames"]:
        return dict(result, outcome="control")
    if last_good["state"] != TRACKING_OK or not last_good["imu_initialized"]:
        return dict(result, outcome="not_tracking_before_mask")
    if not any(row["state"] != TRACKING_OK or row["map_id"] != target_map
               for row in incident if row["frame"] <= resume):
        return dict(result, outcome="no_tracking_loss")

    for row in incident:
        if row["map_id"] != target_map or row["state"] == 1:
            return dict(result, outcome="left_target_map")
        if row["frame"] < resume or row["state"] != TRACKING_OK or not row["px"]:
            continue
        history = [r for r in frames[:start] if r["map_id"] == target_map and r["px"] and
                   r["imu_initialized"] and 0 < start_ns - r["timestamp_ns"] <= ALIGNMENT_WINDOW_NS]
        if len(history) < 10:
            raise ValueError("insufficient pre-outage poses to fit a rigid alignment")
        rotation, translation, fit_rmse = fit_alignment(history, reference)
        true_position, true_rotation = reference.evaluate([row])
        position = rotation.apply(row_pose(row, POSITION_COLUMNS)) + translation
        orientation = rotation * Rotation.from_quat(row_pose(row, QUATERNION_COLUMNS))
        position_error = float(np.linalg.norm(position - true_position[0]))
        angle_error = float(np.rad2deg((true_rotation[0].inv() * orientation).magnitude()))
        accepted = any(event["event"] == "relocalization_accept" and
                       int(event["map_id"]) == target_map and
                       int(event["timestamp_ns"]) == row["timestamp_ns"] for event in episode_events)
        return dict(
            result,
            outcome="relocalization_accepted" if accepted else "same_map_tracking_resumed",
            recovery_frame=row["frame"],
            latency_ms=round((row["timestamp_ns"] - resume_ns) / 1e6, 3),
            position_error_m=round(position_error, 4),
            rotation_error_deg=round(angle_error, 3),
            pre_outage_alignment_rmse_m=round(fit_rmse, 4),
            within_protocol_thresholds=position_error <= 0.30 and angle_error <= 10,
        )

    return dict(result, outcome="timeout")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("control", type=Path)
    parser.add_argument("outages", type=Path, nargs="+")
    args = parser.parse_args()
    try:
        reference = Reference(args.ground_truth)
        control, control_rows, control_events = load_replay(args.control)
        if control["masked_frames"]:
            raise ValueError("expected a control replay with zero masked frames")
        signature = [(row["timestamp_ns"], row["imu_samples"]) for row in control_rows]
        results = [inspect(control, control_rows, control_events, reference)]
        for path in args.outages:
            metadata, rows, events = load_replay(path)
            if metadata["imu_sha256"] != control["imu_sha256"] or [
                (row["timestamp_ns"], row["imu_samples"]) for row in rows
            ] != signature:
                raise ValueError(f"{path}: image timestamps or IMU stream differ from the control")
            if metadata["start_frame"] != control["start_frame"] or not metadata["masked_frames"]:
                raise ValueError(f"{path}: expected an outage starting at the control frame")
            results.append(inspect(metadata, rows, events, reference))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Replay audit failed: {error}\n")
    print(json.dumps({"sequence": control["sequence"], "mode": control["mode"],
                      "input_frames_per_run": control["frames"], "identical_camera_timeline_and_imu": True,
                      "alignment_window_seconds": ALIGNMENT_WINDOW_NS / 1e9,
                      "cases": results}, indent=2))


if __name__ == "__main__":
    main()

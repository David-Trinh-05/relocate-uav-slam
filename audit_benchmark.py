"""Validate benchmark coverage, fixed inputs, motion, and stereo ambiguity."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import cv2 as cv
import numpy as np

from ambiguity_scene import BASELINE_M, HEIGHT, K, ROOM_OFFSET_M, WIDTH, yaw_rotation
from benchmark import CAMERA_PERIOD_NS, GRAVITY, IMU_PERIOD_NS, MAP_VERSION, OBSERVATION_NS
from benchmark import POSE_COLUMNS, PRE_ROLL_FRAMES, Motion, flight_plan, setup_scene, sha256, stereo_views


def require(condition, message):
    if not condition:
        raise ValueError(message)


def artifact(root, record, expected_path=None):
    relative = record["path"]
    require(expected_path is None or relative == expected_path, f"unexpected path: {relative}")
    path = (root / relative).resolve()
    require(path.is_relative_to(root) and path.is_file(), f"missing or invalid artifact: {relative}")
    require(sha256(path) == record["sha256"], f"checksum mismatch: {relative}")
    return path


def csv_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def image(path):
    pixels = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    require(pixels is not None and pixels.shape == (HEIGHT, WIDTH), f"invalid image: {path}")
    return pixels


def audit_map(root, manifest):
    mapping = json.loads(artifact(root, manifest["map"], "map/index.json").read_text())
    require(mapping["version"] == MAP_VERSION, "unexpected map version")
    keys = mapping["keyframes"]
    require(Counter(row["region"] for row in keys) == {"A": 21, "B": 21}, "map has incomplete region coverage")
    require(len({row["keyframe_id"] for row in keys}) == len(keys), "map has duplicate keyframes")
    for key in keys:
        prefix = f"map/{key['keyframe_id']}"
        image(artifact(root, key["left"], f"{prefix}-left.png"))
        image(artifact(root, key["right"], f"{prefix}-right.png"))
        depth = np.load(artifact(root, key["left_optical_z_m"], f"{prefix}-depth.npy"), allow_pickle=False)
        require(depth.shape == (HEIGHT, WIDTH) and np.isfinite(depth).all() and np.min(depth) > 0,
                f"invalid depth in {prefix}")
    return len(keys)


def check_images(root, manifest, flight, incident, frames, frame_times):
    prefix = f"queries/{flight.flight_id}"
    hashes = hashlib.sha256()
    masked_count = 0
    for index, (frame, stamp) in enumerate(zip(frames, frame_times)):
        masked = incident["mask_start_ns"] <= stamp < incident["first_unmasked_ns"]
        require(frame["masked"] == str(int(masked)), f"incorrect frame mask in {prefix}")
        masked_count += int(masked)
        for eye in ("left", "right"):
            clean = f"{prefix}/{index:03d}-{eye}.png"
            actual = manifest["blank_image"]["path"] if masked else clean
            require((frame[f"clean_{eye}"], frame[eye]) == (clean, actual), f"incorrect image routing in {prefix}")
            hashes.update(bytes.fromhex(sha256(root / clean)))
    require(masked_count == flight.mask_frames, f"mask duration mismatch in {prefix}")
    require(hashes.hexdigest() == incident["clean_images_sha256"], f"image checksum mismatch in {prefix}")


def check_motion(truth, imu):
    poses = np.array([[float(row[column]) for column in POSE_COLUMNS] for row in truth])
    require(np.isfinite(poses).all() and np.allclose(np.linalg.norm(poses[:, 3:], axis=1), 1),
            "ground-truth poses must be finite with unit quaternions")
    aligned_imu = imu[::CAMERA_PERIOD_NS // IMU_PERIOD_NS]
    imu_vectors = np.array([[float(row[key]) for key in ("wx", "wy", "wz", "ax", "ay", "az")] for row in aligned_imu])
    yaw = np.unwrap(2 * np.arctan2(poses[:, 4], poses[:, 6]))
    dt = CAMERA_PERIOD_NS / 1e9
    numeric_rate = (yaw[2:] - yaw[:-2]) / (2 * dt)
    numeric_accel = (poses[2:, :3] - 2 * poses[1:-1, :3] + poses[:-2, :3]) / dt ** 2
    world_accel = np.array([
        yaw_rotation(np.rad2deg(angle)) @ row[3:] + GRAVITY for angle, row in zip(yaw, imu_vectors)
    ])
    gyro_error = np.max(np.abs(imu_vectors[1:-1, 1] - numeric_rate))
    accel_error = np.max(np.linalg.norm(world_accel[1:-1] - numeric_accel, axis=1))
    require(gyro_error < 0.015 and accel_error < 0.01, "IMU is inconsistent with finite differences of ground truth")
    return float(gyro_error), float(accel_error)


def stereo_difference(root, scene, motion, flight, frame, seconds, resume_seconds):
    center, _, yaw, _ = motion.pose(seconds, resume_seconds)
    other_left, other_right, _ = stereo_views(scene, 1 - flight.room, center, yaw_rotation(yaw))
    left = image(root / frame["left"])
    right = image(root / frame["right"])
    return int(np.count_nonzero(left != other_left) + np.count_nonzero(right != other_right))


def audit_visibility(root, scene, flight, incident, frames, frame_times):
    motion = Motion(flight)
    resume_index = PRE_ROLL_FRAMES + flight.mask_frames
    time_origin = frame_times[0]
    resume_seconds = (frame_times[resume_index] - time_origin) / 1e9
    cue = None
    for index in range(resume_index, len(frames)):
        seconds = (frame_times[index] - time_origin) / 1e9
        _, _, yaw, _ = motion.pose(seconds, resume_seconds)
        # Forward-facing rays all have positive z at these angles. They cannot hit the rear cue.
        if index != resume_index and abs(yaw) <= 45:
            continue
        if stereo_difference(root, scene, motion, flight, frames[index], seconds, resume_seconds):
            cue = frame_times[index]
            break
    require(cue == incident["first_visual_cue_ns"], f"visual cue mismatch in {flight.flight_id}")
    deadline_seconds = (frame_times[-1] - time_origin) / 1e9
    deadline_difference = stereo_difference(root, scene, motion, flight, frames[-1], deadline_seconds, resume_seconds)
    minimum_visible_pixels = WIDTH * HEIGHT * 2 * 0.01
    if flight.scenario == "unresolved":
        require(cue is None and deadline_difference == 0, f"unexpected distinction in {flight.flight_id}")
    else:
        require(cue is not None and deadline_difference >= minimum_visible_pixels,
                f"missing distinguishing panel in {flight.flight_id}")
        if flight.scenario == "immediate":
            require(cue == incident["first_unmasked_ns"] and stereo_difference(
                root, scene, motion, flight, frames[resume_index], resume_seconds, resume_seconds
            ) >= minimum_visible_pixels, f"expected a visible cue on resumption in {flight.flight_id}")
        else:
            require(incident["first_unmasked_ns"] < cue < incident["deadline_ns"],
                    f"expected a delayed cue in {flight.flight_id}")
    return None if cue is None else (cue - incident["first_unmasked_ns"]) / 1e9


def audit_flight(root, manifest, flight, incident, scene):
    require(incident["flight_id"] == flight.flight_id and incident["incident_id"] == flight.flight_id + "-outage",
            f"unexpected IDs in {flight.flight_id}")
    require((incident["split"], incident["scenario"], incident["motion_seed"], incident["target_region"]) ==
            (flight.split, flight.scenario, flight.seed, "AB"[flight.room]), f"incorrect split or plan in {flight.flight_id}")
    require(incident["map_version"] == MAP_VERSION and incident["map_sha256"] == manifest["map"]["sha256"],
            f"different map in {flight.flight_id}")
    prefix = f"queries/{flight.flight_id}"
    frames = csv_rows(artifact(root, incident["frames"], f"{prefix}/frames.csv"))
    imu = csv_rows(artifact(root, incident["imu"], f"{prefix}/imu.csv"))
    truth = csv_rows(artifact(root, incident["ground_truth"], f"ground_truth/{flight.flight_id}.csv"))
    origin = flight.seed * 20_000_000_000
    start = origin + PRE_ROLL_FRAMES * CAMERA_PERIOD_NS
    resume = start + flight.mask_frames * CAMERA_PERIOD_NS
    deadline = resume + OBSERVATION_NS
    require((incident["mask_start_ns"], incident["first_unmasked_ns"], incident["deadline_ns"]) ==
            (start, resume, deadline), f"incorrect incident timing in {flight.flight_id}")
    frame_times = list(range(origin, deadline + 1, CAMERA_PERIOD_NS))
    require([int(row["timestamp_ns"]) for row in frames] == frame_times and
            [int(row["timestamp_ns"]) for row in truth] == frame_times, f"stereo or truth gap in {flight.flight_id}")
    require([int(row["timestamp_ns"]) for row in imu] == list(range(origin, deadline + 1, IMU_PERIOD_NS)),
            f"IMU gap in {flight.flight_id}")
    region_x = ROOM_OFFSET_M * flight.room
    require(all(abs(float(row["px"]) - region_x) < 1.65 for row in truth), f"incorrect region in {flight.flight_id}")
    check_images(root, manifest, flight, incident, frames, frame_times)
    gyro_error, accel_error = check_motion(truth, imu)
    cue_delay = audit_visibility(root, scene, flight, incident, frames, frame_times)
    return {
        "incident_id": incident["incident_id"], "flight_id": flight.flight_id, "split": flight.split,
        "scenario": flight.scenario, "target_region": incident["target_region"],
        "mask_seconds": (resume - start) / 1e9, "cue_delay_seconds": cue_delay,
        "stereo_pairs": len(frames), "imu_samples": len(imu),
        "max_gyro_check_error_rad_s": round(gyro_error, 6),
        "max_accel_check_error_m_s2": round(accel_error, 6),
    }


def audit(root):
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    calibration = manifest["calibration"]
    require(manifest["schema_version"] == 1 and manifest["map_version"] == MAP_VERSION and
            manifest["observation_window_ns"] == OBSERVATION_NS, "unexpected manifest version or observation window")
    require(calibration["camera_intrinsics"] == K.tolist() and calibration["image_size"] == [WIDTH, HEIGHT] and
            calibration["right_from_left_translation_m"] == [-BASELINE_M, 0, 0] and
            calibration["gravity_world_m_s2"] == GRAVITY.tolist() and calibration["imu_body_frame"] == "left_camera" and
            (calibration["camera_period_ns"], calibration["imu_period_ns"]) == (CAMERA_PERIOD_NS, IMU_PERIOD_NS),
            "unexpected sensor calibration")
    require(not image(artifact(root, manifest["blank_image"], "blank.png")).any(), "masked image must be black")
    keyframes = audit_map(root, manifest)
    flights = list(flight_plan())
    incidents = manifest["incidents"]
    require(len(incidents) == len(flights), "incomplete flight plan")
    require(len({row["motion_seed"] for row in incidents}) == len(flights), "trajectory seed reuse between flights")
    require(len({row["imu"]["sha256"] for row in incidents}) == len(flights), "duplicate IMU streams between flights")
    scene = setup_scene()
    verified = [audit_flight(root, manifest, flight, incident, scene) for flight, incident in zip(flights, incidents)]
    return {
        "manifest_sha256": sha256(manifest_path), "map_sha256": manifest["map"]["sha256"],
        "map_keyframes": keyframes, "flights": len(verified), "incidents": len(verified),
        "stereo_pairs": sum(row["stereo_pairs"] for row in verified),
        "imu_samples": sum(row["imu_samples"] for row in verified),
        "split_counts": dict(Counter(row["split"] for row in verified)),
        "scenario_counts": dict(Counter(row["scenario"] for row in verified)),
        "cases": verified,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(audit(args.benchmark.resolve()), indent=2))
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"Benchmark audit failed: {error}\n")


if __name__ == "__main__":
    main()

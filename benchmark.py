"""Build the fixed-map stereo–inertial synthetic benchmark."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import NamedTuple

import cv2 as cv
import numpy as np

from ambiguity_scene import BASELINE_M, HEIGHT, K, ROOM_OFFSET_M, WIDTH
from ambiguity_scene import make_textures, pixel_rays, render, save_image, yaw_rotation


SCENARIOS = ("immediate", "delayed", "unresolved")
SPLITS = ("development", "held_out")
MAP_VERSION = "corridors-v1"
CAMERA_PERIOD_NS = 50_000_000
IMU_PERIOD_NS = 5_000_000
OBSERVATION_NS = 5_000_000_000
PRE_ROLL_FRAMES = 30
GRAVITY = np.array([0.0, 9.80665, 0.0])
POSE_COLUMNS = ("px", "py", "pz", "qx", "qy", "qz", "qw")


class Flight(NamedTuple):
    split: str
    scenario: str
    number: int
    room: int
    mask_frames: int
    seed: int

    @property
    def flight_id(self):
        return f"{self.split}-{self.scenario}-{self.number:02d}"


def flight_plan():
    for split_index, split in enumerate(SPLITS):
        for scenario_index, scenario in enumerate(SCENARIOS):
            for number, mask_frames in enumerate((10, 20, 40), 1):
                seed = 301 + split_index * 9 + scenario_index * 3 + number
                room = (split_index + scenario_index + number) % 2
                yield Flight(split, scenario, number, room, mask_frames, seed)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def write_csv(path, header, rows):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def indexed_file(root, path):
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256(path)}


def room_offset(room):
    return np.array([room * ROOM_OFFSET_M, 0.0, 0.0])


def quaternion(yaw_deg):
    angle = np.deg2rad(yaw_deg) * 0.5
    return [0.0, float(np.sin(angle)), 0.0, float(np.cos(angle))]


def setup_scene():
    textures, panels = make_textures()
    return textures, panels, pixel_rays()


def stereo_views(scene, room, center, rotation):
    textures, panels, rays = scene
    left, depth = render(textures, panels, room, center, rotation, rays)
    right, _ = render(textures, panels, room, center + BASELINE_M * rotation[:, 0], rotation, rays)
    return left, right, depth


def build_map(root):
    target = root / "map"
    target.mkdir()
    keyframes = []
    scene = setup_scene()
    for room, label in enumerate(("A", "B")):
        for distance_index, distance in enumerate((1.65, 2.32, 3.12)):
            center = np.array([0.0, 0.0, distance])
            for yaw in (0, 25, 55, 85, 115, 145, 175):
                keyframe_id = f"{label}-{distance_index}-{yaw:03d}"
                left, right, depth = stereo_views(scene, room, center, yaw_rotation(yaw))
                left_path = target / f"{keyframe_id}-left.png"
                right_path = target / f"{keyframe_id}-right.png"
                depth_path = target / f"{keyframe_id}-depth.npy"
                save_image(left_path, left)
                save_image(right_path, right)
                np.save(depth_path, depth.astype(np.float32), allow_pickle=False)
                keyframes.append({
                    "keyframe_id": keyframe_id,
                    "region": label,
                    "world_position_m": (center + room_offset(room)).tolist(),
                    "world_from_body_xyzw": quaternion(yaw),
                    "left": indexed_file(root, left_path),
                    "right": indexed_file(root, right_path),
                    "left_optical_z_m": indexed_file(root, depth_path),
                })
    index = target / "index.json"
    write_json(index, {"version": MAP_VERSION, "keyframes": keyframes})
    return indexed_file(root, index)


class Motion:
    def __init__(self, flight):
        rng = np.random.default_rng(flight.seed)
        self.center = rng.uniform([-0.16, -0.13, 1.95], [0.16, 0.13, 2.70])
        self.amplitude = rng.uniform([0.035, 0.025, 0.12], [0.11, 0.08, 0.25])
        self.rates = rng.uniform(0.65, 1.25, size=3)
        self.phase = rng.uniform(-np.pi, np.pi, size=3)
        self.yaw_bias = rng.uniform(-3.0, 3.0)
        self.turn_start = rng.uniform(0.25, 0.65)
        self.turn_duration = rng.uniform(1.65, 2.50)
        self.flight = flight

    def pose(self, seconds, resume_seconds):
        phase = self.rates * seconds + self.phase
        center = self.center + self.amplitude * np.sin(phase)
        acceleration = -self.rates ** 2 * self.amplitude * np.sin(phase)
        yaw = self.yaw_bias + 3.0 * np.sin(seconds * 0.8)
        yaw_rate = 2.4 * np.cos(seconds * 0.8)
        if self.flight.scenario == "immediate":
            yaw += 173.0
        if self.flight.scenario == "delayed":
            turn = np.clip((seconds - resume_seconds - self.turn_start) / self.turn_duration, 0.0, 1.0)
            yaw += 173.0 * turn ** 3 * (10 - 15 * turn + 6 * turn ** 2)
            yaw_rate += 173.0 * 30 * turn ** 2 * (1 - turn) ** 2 / self.turn_duration
        return center, acceleration, float(yaw), float(yaw_rate)


def build_flight(task):
    root, map_hash, flight = task
    cv.setNumThreads(1)
    target = root / "queries" / flight.flight_id
    target.mkdir(parents=True)
    scene = setup_scene()
    trajectory = Motion(flight)
    origin_ns = flight.seed * 20_000_000_000
    start_ns = origin_ns + PRE_ROLL_FRAMES * CAMERA_PERIOD_NS
    resume_ns = start_ns + flight.mask_frames * CAMERA_PERIOD_NS
    stop_ns = resume_ns + OBSERVATION_NS
    resume_seconds = (resume_ns - origin_ns) / 1e9
    frame_rows, truth_rows = [], []
    first_visual_cue_ns = None
    image_hashes = hashlib.sha256()

    for frame_index, stamp in enumerate(range(origin_ns, stop_ns + 1, CAMERA_PERIOD_NS)):
        seconds = (stamp - origin_ns) / 1e9
        center, _, yaw, _ = trajectory.pose(seconds, resume_seconds)
        rotation = yaw_rotation(yaw)
        left, right, _ = stereo_views(scene, flight.room, center, rotation)
        masked = start_ns <= stamp < resume_ns
        clean_paths = [target / f"{frame_index:03d}-{eye}.png" for eye in ("left", "right")]
        clean_images = [left, right]
        for path, image in zip(clean_paths, clean_images):
            save_image(path, image)
            image_hashes.update(bytes.fromhex(sha256(path)))
        clean_paths = [path.relative_to(root).as_posix() for path in clean_paths]
        actual_paths = ["blank.png", "blank.png"] if masked else clean_paths
        frame_rows.append([stamp, *actual_paths, int(masked), *clean_paths])
        truth_rows.append([stamp, *(center + room_offset(flight.room)).tolist(), *quaternion(yaw)])

        # At these low yaw angles the rear face cannot enter either camera's field of view.
        if stamp >= resume_ns and first_visual_cue_ns is None and abs(yaw) > 45.0:
            other_left, other_right, _ = stereo_views(scene, 1 - flight.room, center, rotation)
            if not (np.array_equal(left, other_left) and np.array_equal(right, other_right)):
                first_visual_cue_ns = stamp

    imu_rows = []
    for stamp in range(origin_ns, stop_ns + 1, IMU_PERIOD_NS):
        _, acceleration, yaw, yaw_rate = trajectory.pose((stamp - origin_ns) / 1e9, resume_seconds)
        specific_force = yaw_rotation(yaw).T @ (acceleration - GRAVITY)
        imu_rows.append([stamp, 0.0, np.deg2rad(yaw_rate), 0.0, *specific_force.tolist()])

    frames_path = target / "frames.csv"
    imu_path = target / "imu.csv"
    truth_path = root / "ground_truth" / f"{flight.flight_id}.csv"
    write_csv(frames_path, ("timestamp_ns", "left", "right", "masked", "clean_left", "clean_right"), frame_rows)
    write_csv(imu_path, ("timestamp_ns", "wx", "wy", "wz", "ax", "ay", "az"), imu_rows)
    write_csv(truth_path, ("timestamp_ns", *POSE_COLUMNS), truth_rows)
    return {
        "incident_id": f"{flight.flight_id}-outage",
        "flight_id": flight.flight_id,
        "split": flight.split,
        "scenario": flight.scenario,
        "motion_seed": flight.seed,
        "target_region": "AB"[flight.room],
        "map_version": MAP_VERSION,
        "map_sha256": map_hash,
        "mask_start_ns": start_ns,
        "first_unmasked_ns": resume_ns,
        "deadline_ns": stop_ns,
        "first_visual_cue_ns": first_visual_cue_ns,
        "frames": indexed_file(root, frames_path),
        "imu": indexed_file(root, imu_path),
        "ground_truth": indexed_file(root, truth_path),
        "clean_images_sha256": image_hashes.hexdigest(),
    }


def build(root, jobs):
    root.mkdir(parents=True, exist_ok=False)
    (root / "ground_truth").mkdir()
    cv.setNumThreads(1)
    blank = root / "blank.png"
    save_image(blank, np.zeros((HEIGHT, WIDTH), dtype=np.uint8))
    mapping = build_map(root)
    tasks = [(root, mapping["sha256"], flight) for flight in flight_plan()]
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        incidents = list(executor.map(build_flight, tasks))
    write_json(root / "manifest.json", {
        "schema_version": 1,
        "map_version": MAP_VERSION,
        "calibration": {
            "camera_intrinsics": K.tolist(), "image_size": [WIDTH, HEIGHT],
            "right_from_left_translation_m": [-BASELINE_M, 0, 0],
            "gravity_world_m_s2": GRAVITY.tolist(),
            "camera_period_ns": CAMERA_PERIOD_NS, "imu_period_ns": IMU_PERIOD_NS,
            "imu_body_frame": "left_camera",
        },
        "map": mapping,
        "blank_image": indexed_file(root, blank),
        "observation_window_ns": OBSERVATION_NS,
        "incidents": incidents,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new directory for the benchmark")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            parser.error(f"output already exists: {args.output}")
        with TemporaryDirectory(prefix=".relocate-benchmark-", dir=args.output.parent) as staging:
            built = Path(staging) / "dataset"
            build(built, args.jobs)
            if args.output.exists():
                parser.error(f"output already exists: {args.output}")
            built.rename(args.output)
    except OSError as error:
        parser.exit(1, f"Benchmark build failed: {error}\n")
    print(args.output / "manifest.json")


if __name__ == "__main__":
    main()

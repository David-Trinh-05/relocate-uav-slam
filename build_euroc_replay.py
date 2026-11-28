"""Build a fixed-map relocalization replay from a recorded EuRoC flight."""

import argparse
import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from benchmark import POSE_COLUMNS, indexed_file, write_csv, write_json
from inspect_replays import Reference
from relocalizer import read_gray
from check_trajectory import read_timestamps


MAP_VERSION = "euroc-vicon-room-1-v1"
MAPPING_FRAMES = tuple(range(60, 1500, 15))
OUTAGES = tuple(zip(range(1630, 2720, 180), (10, 20, 40, 10, 20, 40, 10), strict=True))
OBSERVATION_FRAMES = 100
SOURCE_FIRST_TIMESTAMP_NS = 1_403_715_273_262_142_976


class StereoRig:
    def __init__(self, sensors):
        parameters = [yaml.safe_load((sensor / "sensor.yaml").read_text()) for sensor in sensors]
        if any(row["camera_model"] != "pinhole" or row["distortion_model"] != "radial-tangential"
               for row in parameters):
            raise ValueError("expected EuRoC pinhole radial-tangential stereo calibration")
        self.size = tuple(parameters[0]["resolution"])
        if tuple(parameters[1]["resolution"]) != self.size:
            raise ValueError("camera resolutions differ")
        matrices, distortions, extrinsics = [], [], []
        for row in parameters:
            fx, fy, cx, cy = row["intrinsics"]
            matrices.append(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]))
            distortions.append(np.array(row["distortion_coefficients"]))
            extrinsics.append(np.array(row["T_BS"]["data"]).reshape(4, 4))
        right_from_left = np.linalg.inv(extrinsics[1]) @ extrinsics[0]
        r0, r1, p0, p1, *_ = cv.stereoRectify(
            matrices[0], distortions[0], matrices[1], distortions[1], self.size,
            right_from_left[:3, :3].copy(), right_from_left[:3, 3:].copy(), flags=cv.CALIB_ZERO_DISPARITY, alpha=0,
        )
        self.intrinsics = p0[:, :3]
        self.baseline = -p1[0, 3] / p1[0, 0]
        if self.baseline <= 0 or p1[1, 3] != 0:
            raise ValueError("expected rectified horizontal stereo with positive baseline")
        self.body_from_camera = extrinsics[0][:3, :3] @ r0.T
        self.camera_in_body = extrinsics[0][:3, 3]
        self.sensors = sensors
        self.remap = [cv.initUndistortRectifyMap(k, d, r, p, self.size, cv.CV_16SC2)
                      for k, d, r, p in zip(matrices, distortions, (r0, r1), (p0, p1), strict=True)]

    def images(self, stamp):
        return [cv.remap(read_gray(sensor / "data" / f"{stamp}.png"), *maps, cv.INTER_LINEAR)
                for sensor, maps in zip(self.sensors, self.remap, strict=True)]

    def camera_truth(self, reference, timestamps):
        positions, orientations = reference.evaluate([{"timestamp_ns": stamp} for stamp in timestamps])
        return (positions + orientations.apply(self.camera_in_body),
                (orientations * Rotation.from_matrix(self.body_from_camera)).as_quat())


def save_images(root, directory, rig, stamp):
    images = rig.images(stamp)
    paths = [directory / f"{stamp}-{eye}.png" for eye in ("left", "right")]
    for path, image in zip(paths, images, strict=True):
        if not cv.imwrite(str(path), image, [cv.IMWRITE_PNG_COMPRESSION, 1]):
            raise OSError(f"cannot write {path}")
    return [path.relative_to(root).as_posix() for path in paths], images


def build_reference(root, rig, timestamps, truth):
    mapping = root / "map"
    mapping.mkdir()
    positions, rotations = rig.camera_truth(truth, [timestamps[frame] for frame in MAPPING_FRAMES])
    region_boundary = float(np.median(positions[:, 1]))
    matcher = cv.StereoSGBM_create(minDisparity=0, numDisparities=96, blockSize=5, P1=200, P2=800,
                                    disp12MaxDiff=1, uniquenessRatio=12, speckleWindowSize=60, speckleRange=2)
    keyframes = []
    for index, source_frame in enumerate(MAPPING_FRAMES):
        stamp = timestamps[source_frame]
        (left, right), images = save_images(root, mapping, rig, stamp)
        disparity = matcher.compute(*images).astype(np.float32) / 16
        depth = np.zeros_like(disparity)
        valid = disparity > 0
        depth[valid] = rig.intrinsics[0, 0] * rig.baseline / disparity[valid]
        depth[(depth < 0.25) | (depth > 15)] = 0
        depth_path = mapping / f"{stamp}-depth.npy"
        np.save(depth_path, depth, allow_pickle=False)
        keyframes.append({
            "keyframe_id": f"euroc-{source_frame}", "region": "lower-y" if positions[index, 1] < region_boundary else "upper-y",
            "source_frame": source_frame, "timestamp_ns": stamp, "world_position_m": positions[index].tolist(),
            "world_from_body_xyzw": rotations[index].tolist(), "left": indexed_file(root, root / left),
            "right": indexed_file(root, root / right), "left_optical_z_m": indexed_file(root, depth_path),
        })
    index = mapping / "index.json"
    write_json(index, {"version": MAP_VERSION, "keyframes": keyframes})
    return indexed_file(root, index)


def export_query(root, rig, timestamps, truth, imu, mapping, start_index, masked_frames):
    resume_index = start_index + masked_frames
    end_index = resume_index + OBSERVATION_FRAMES
    start, resume, end = (timestamps[index] for index in (start_index, resume_index, end_index))
    query_id = f"V1_01_easy-frame-{start_index}"
    directory = root / "queries" / query_id
    directory.mkdir(parents=True)
    query_times = timestamps[start_index:end_index + 1]
    frame_rows = []
    for stamp in query_times:
        (left, right), _ = save_images(root, directory, rig, stamp)
        masked = stamp < resume
        actual = ("blank.png", "blank.png") if masked else (left, right)
        frame_rows.append((stamp, *actual, int(masked), left, right))
    frames_path = directory / "frames.csv"
    write_csv(frames_path, ("timestamp_ns", "left", "right", "masked", "clean_left", "clean_right"), frame_rows)

    positions, quaternions = rig.camera_truth(truth, query_times)
    gt_path = root / "ground_truth" / f"{query_id}.csv"
    write_csv(gt_path, ("timestamp_ns", *POSE_COLUMNS), (
        (stamp, *position, *quaternion) for stamp, position, quaternion in zip(query_times, positions, quaternions, strict=True)
    ))

    imu_path = directory / "imu.csv"
    write_csv(imu_path, ("timestamp_ns", "wx", "wy", "wz"), (
        (stamp, *(rig.body_from_camera.T @ omega)) for stamp, omega in imu if start <= stamp <= end
    ))
    return {
        "incident_id": f"{query_id}-outage", "flight_id": "V1_01_easy", "split": "held_out", "scenario": "recorded_replay",
        "target_region": None, "map_version": MAP_VERSION, "map_sha256": mapping["sha256"], "start_frame": start_index,
        "mask_start_ns": start, "first_unmasked_ns": resume, "deadline_ns": end, "first_visual_cue_ns": None,
        "frames": indexed_file(root, frames_path), "imu": indexed_file(root, imu_path),
        "ground_truth": indexed_file(root, gt_path),
    }


def build(root, sequence):
    mav = sequence / "mav0"
    sensors = [mav / "cam0", mav / "cam1"]
    timelines = [read_timestamps(sensor / "data.csv", ",") for sensor in sensors]
    timestamps = timelines[0]
    if timestamps != timelines[1] or len(timestamps) != 2912 or timestamps[0] != SOURCE_FIRST_TIMESTAMP_NS:
        raise ValueError("expected the complete synchronized ASL EuRoC V1_01_easy dataset")
    truth_path = mav / "state_groundtruth_estimate0/data.csv"
    imu_path = mav / "imu0/data.csv"
    source_truth = Reference(truth_path)
    with imu_path.open(newline="") as stream:
        imu = [(int(row[0]), np.array(row[1:4], dtype=float)) for row in csv.reader(stream)
               if row and not row[0].startswith("#")]
    if not set(timestamps).issubset({stamp for stamp, _ in imu}):
        raise ValueError("expected a synchronized IMU sample at each stereo timestamp")
    rig = StereoRig(sensors)
    root.mkdir()
    (root / "ground_truth").mkdir()
    blank = root / "blank.png"
    if not cv.imwrite(str(blank), np.zeros(rig.size[::-1], dtype=np.uint8)):
        raise OSError("cannot write blank stereo frame")
    mapping = build_reference(root, rig, timestamps, source_truth)
    incidents = [export_query(root, rig, timestamps, source_truth, imu, mapping, *outage) for outage in OUTAGES]
    write_json(root / "manifest.json", {
        "schema_version": 1, "source_sequence": "EuRoC MAV V1_01_easy", "map_version": MAP_VERSION,
        "source_files": {"camera0_calibration": indexed_file(sequence, sensors[0] / "sensor.yaml"),
                         "camera1_calibration": indexed_file(sequence, sensors[1] / "sensor.yaml"),
                         "imu": indexed_file(sequence, imu_path), "ground_truth": indexed_file(sequence, truth_path)},
        "mapping_source_frames": list(MAPPING_FRAMES), "map": mapping,
        "calibration": {"camera_intrinsics": rig.intrinsics.tolist(), "image_size": list(rig.size),
                        "right_from_left_translation_m": [-rig.baseline, 0, 0], "camera_period_ns": 50_000_000,
                        "imu_period_ns": 5_000_000, "imu_body_frame": "rectified_left_camera"},
        "blank_image": indexed_file(root, blank), "incidents": incidents,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    cv.setNumThreads(4)
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise ValueError(f"output already exists: {args.output}")
        with TemporaryDirectory(dir=args.output.parent, prefix=".euroc-replay-") as staging:
            archive = Path(staging) / "dataset"
            build(archive, args.sequence.resolve())
            archive.rename(args.output)
    except (KeyError, OSError, ValueError, cv.error, yaml.YAMLError) as error:
        parser.exit(1, f"EuRoC replay build failed: {error}\n")
    print(args.output / "manifest.json")


if __name__ == "__main__":
    main()

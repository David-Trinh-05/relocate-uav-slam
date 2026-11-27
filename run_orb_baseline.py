"""Run the upstream ORB-SLAM3 relocalization routine on the frozen corridor snapshot."""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import subprocess

import cv2 as cv
import numpy as np

from audit_benchmark import artifact, csv_rows
from benchmark import sha256, write_json
from run_relocalization import read_rows, score


def settings_text(calibration):
    k = calibration["camera_intrinsics"]
    width, height = calibration["image_size"]
    baseline = -calibration["right_from_left_translation_m"][0]
    fps = 1_000_000_000 // calibration["camera_period_ns"]
    imu_hz = 1_000_000_000 // calibration["imu_period_ns"]
    return f'''%YAML:1.0
File.version: "1.0"
Camera.type: "Rectified"
Camera1.fx: {float(k[0][0])}
Camera1.fy: {float(k[1][1])}
Camera1.cx: {float(k[0][2])}
Camera1.cy: {float(k[1][2])}
Camera.width: {width}
Camera.height: {height}
Camera.fps: {fps}
Camera.RGB: 0
Stereo.b: {baseline}
Stereo.ThDepth: 60.0
IMU.T_b_c1: !!opencv-matrix
  rows: 4
  cols: 4
  dt: f
  data: [1.,0.,0.,0., 0.,1.,0.,0., 0.,0.,1.,0., 0.,0.,0.,1.]
IMU.NoiseGyro: 1.7e-04
IMU.NoiseAcc: 2.0e-03
IMU.GyroWalk: 1.9393e-05
IMU.AccWalk: 3.0e-03
IMU.Frequency: {imu_hz}.0
ORBextractor.nFeatures: 1200
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0
'''


def query_window(root, incident):
    start, stop = incident["first_unmasked_ns"], incident["deadline_ns"]
    rows = [row for row in read_rows(root, incident["frames"]) if start <= int(row["timestamp_ns"]) <= stop]
    if not rows or int(rows[0]["timestamp_ns"]) != start or int(rows[-1]["timestamp_ns"]) != stop or any(
        row["masked"] != "0" for row in rows
    ):
        raise ValueError(f"invalid query window in {incident['incident_id']}")
    return rows


def prepare(root, output, manifest, incidents):
    fixed_map = json.loads(artifact(root, manifest["map"]).read_text())
    keyframes = []
    for index, key in enumerate(fixed_map["keyframes"]):
        depth = np.load(artifact(root, key["left_optical_z_m"]), allow_pickle=False)
        path = output / f"depth-{index:02}.tiff"
        if not cv.imwrite(str(path), depth):
            raise OSError(f"cannot export {path}")
        decoded = cv.imread(str(path), cv.IMREAD_UNCHANGED)
        if decoded is None or decoded.dtype != depth.dtype or not np.array_equal(decoded, depth):
            raise ValueError(f"depth round trip changed {key['keyframe_id']}")
        keyframes.append({
            "name": key["keyframe_id"], "region": key["region"], "depth": str(path),
            "position": key["world_position_m"], "quaternion": key["world_from_body_xyzw"],
            "left": str(artifact(root, key["left"])), "right": str(artifact(root, key["right"])),
        })
    requests = []
    for incident in incidents:
        frames = []
        for row in query_window(root, incident):
            frames.append({"timestamp_ns": row["timestamp_ns"], "left": str(root / row["left"]),
                           "right": str(root / row["right"])})
        requests.append({"id": incident["incident_id"], "frames": frames})
    write_json(output / "request.json", {"keyframes": keyframes, "incidents": requests})
    (output / "settings.yaml").write_text(settings_text(manifest["calibration"]))
    return {entry["id"]: entry["frames"] for entry in requests}, keyframes


def classify(root, incident, request, frames, events, keyframes):
    incident_id = incident["incident_id"]
    if not frames or len(frames) > len(request) or any(
        row["timestamp_ns"] != expected["timestamp_ns"] or int(row["frame"]) != index
        for index, (row, expected) in enumerate(zip(frames, request))
    ):
        raise ValueError(f"native output diverged from the input timeline in {incident_id}")
    start, stop = int(frames[0]["timestamp_ns"]), int(frames[-1]["timestamp_ns"])
    episode_events = [event for event in events if start <= int(event["timestamp_ns"]) <= stop]
    publications = [row for row in frames if row["published"] == "1"]
    accepts = [event for event in episode_events if event["event"] == "relocalization_accept"]
    attempts = [event for event in episode_events if event["event"] == "relocalization_attempt"]
    if len(attempts) != len(frames) or len(publications) != len(accepts) or len(publications) > 1:
        raise ValueError(f"missing or inconsistent native relocalization events in {incident_id}")
    candidates = [event for event in episode_events if event["event"] == "relocalization_candidate"]
    target = incident["target_region"]
    first_target = next((int(row["timestamp_ns"]) for row in candidates
                         if keyframes[row["keyframe_id"]]["region"] == target), None)
    decision = None
    diagnostic = {"attempts": len(attempts), "keyframe_candidates": len(candidates),
                  "first_target_candidate_ms": None if first_target is None else (first_target - start) / 1e6}
    if publications:
        if publications[0] is not frames[-1] or int(accepts[0]["timestamp_ns"]) != stop:
            raise ValueError(f"misaligned native publication in {incident_id}")
        winner = keyframes[accepts[0]["keyframe_id"]]
        region = winner["region"]
        pose = {"position_m": [float(frames[-1][field]) for field in ("px", "py", "pz")],
                "quaternion_xyzw": [float(frames[-1][field]) for field in ("qx", "qy", "qz", "qw")]}
        decision = {"frame": int(frames[-1]["frame"]), "timestamp_ns": stop, "published_region": region,
                    "candidates": {region: {"pose": pose}}}
        at_publication = [row for row in candidates if int(row["timestamp_ns"]) == stop]
        diagnostic.update(
            accepted_keyframe=winner["name"], accepted_inliers=int(accepts[0]["count"]),
            candidates_at_publication=[{"keyframe": keyframes[row["keyframe_id"]]["name"],
                                        "region": keyframes[row["keyframe_id"]]["region"],
                                        "bow_matches": int(row["count"])} for row in at_publication],
            target_retrieved_at_publication=any(keyframes[row["keyframe_id"]]["region"] == target
                                                for row in at_publication),
        )
    elif len(frames) != len(request):
        raise ValueError(f"incomplete native timeout in {incident_id}")
    return dict(score(root, incident, decision), native=diagnostic)


def run(root, output, executable, vocabulary, split, incident_id, realtime=False):
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    incidents = [row for row in manifest["incidents"] if row["split"] == split and
                 (incident_id is None or row["incident_id"] == incident_id)]
    if not incidents or any(row["map_sha256"] != manifest["map"]["sha256"] for row in incidents):
        raise ValueError("no matching fixed-map incidents")
    output.mkdir(parents=True, exist_ok=False)
    requests, expected_keys = prepare(root, output, manifest, incidents)
    command = [executable, vocabulary, "settings.yaml", "request.json"]
    if realtime:
        command.append("--realtime")
    with (output / "native.log").open("w") as log:
        process = subprocess.run(command, cwd=output,
                                 env=dict(os.environ, RELOCATE_TRACE_REPLAY="1"), stdout=log,
                                 stderr=subprocess.STDOUT, check=False)
    if process.returncode:
        raise ValueError(f"baseline exited with status {process.returncode}. See {output / 'native.log'}")
    keyframes = {row["keyframe_id"]: row for row in csv_rows(output / "atlas.csv")}
    if [(row["name"], row["region"]) for row in keyframes.values()] != [
        (row["name"], row["region"]) for row in expected_keys
    ]:
        raise ValueError("native Atlas keyframe inventory differs from the fixed map")
    frames = defaultdict(list)
    for row in csv_rows(output / "native-frames.csv"):
        frames[row["incident_id"]].append(row)
    events = csv_rows(output / "events.csv")
    results = [classify(root, incident, requests[incident["incident_id"]], frames[incident["incident_id"]],
                        events, keyframes) for incident in incidents]
    write_json(output / "summary.json", {
        "method": "ORB-SLAM3 Tracking::Relocalization", "split": split, "manifest_sha256": sha256(manifest_path),
        "map_sha256": manifest["map"]["sha256"], "vocabulary_sha256": sha256(vocabulary),
        "atlas_keyframes": len(keyframes), "atlas_landmarks": sum(int(row["landmarks"]) for row in keyframes.values()),
        "outcomes": dict(Counter(case["outcome"] for case in results)), "cases": results,
    })
    print(output / "summary.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("executable", type=Path)
    parser.add_argument("vocabulary", type=Path)
    parser.add_argument("--split", choices=("development", "held_out"), default="development")
    parser.add_argument("--incident")
    parser.add_argument("--realtime", action="store_true", help="pace frames at sensor time and record timing.csv")
    args = parser.parse_args()
    try:
        run(args.benchmark.resolve(), args.output.resolve(), args.executable.resolve(), args.vocabulary.resolve(),
            args.split, args.incident, args.realtime)
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"Native baseline failed: {error}\n")


if __name__ == "__main__":
    main()

"""Package the measured demo frames and pose evidence as a portable replay."""

import argparse
import json
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile, ZipInfo

import cv2 as cv
import numpy as np

from audit_benchmark import csv_rows
from benchmark import Motion, flight_plan, setup_scene, sha256, stereo_views, write_json, yaw_rotation
from run_relocalization import read_rows


EXAMPLES = {
    "delayed": "held_out-delayed-03-outage",
    "unresolved": "held_out-unresolved-02-outage",
}


def xz(values):
    return [round(float(values[0]), 5), round(float(values[2]), 5)]


def capture(root, native, multi, output):
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    digest = sha256(manifest_path)
    summaries = [json.loads((path / "summary.json").read_text()) for path in (native, multi)]
    if any((summary["manifest_sha256"], summary["map_sha256"], summary["split"]) !=
           (digest, manifest["map"]["sha256"], "held_out") for summary in summaries):
        raise ValueError("camera and pose runs must use the same held-out manifest and map")
    reference = {row["incident_id"]: row for row in summaries[0]["cases"]}
    proposal = {row["incident_id"]: row for row in summaries[1]["cases"]}
    publications = {row["incident_id"]: row for row in csv_rows(native / "native-frames.csv") if row["published"] == "1"}
    incident_by_id = {row["incident_id"]: row for row in manifest["incidents"]}
    output.mkdir(parents=True, exist_ok=False)
    archive_path = output / "observations.zip"
    clips = {}
    with ZipFile(archive_path, "w") as archive:
        for example, incident_id in EXAMPLES.items():
            incident = incident_by_id[incident_id]
            start, cue = incident["first_unmasked_ns"], incident["first_visual_cue_ns"]
            frames = {int(row["timestamp_ns"]): row for row in read_rows(root, incident["frames"])}
            truth = {int(row["timestamp_ns"]): row for row in read_rows(root, incident["ground_truth"])}
            publication = publications[incident_id]
            with (multi / f"{incident_id}.jsonl").open() as stream:
                trace = [json.loads(line) for line in stream]
            flight = next(entry for entry in flight_plan() if entry.flight_id == incident["flight_id"])
            motion, scene = Motion(flight), setup_scene()
            origin = flight.seed * 20_000_000_000
            samples = []
            for row in trace:
                frame_index, stamp = row["frame"], row["timestamp_ns"]
                if frames[stamp]["masked"] != "0" or frame_index != len(samples):
                    raise ValueError(f"incorrect frame trace for {incident_id}")
                alternate = None
                if cue is not None and stamp >= cue:
                    center, _, yaw, _ = motion.pose((stamp - origin) / 1e9, (start - origin) / 1e9)
                    alternate = stereo_views(scene, 1 - flight.room, center, yaw_rotation(yaw))
                images, regions = {}, {}
                for index, eye in enumerate(("left", "right")):
                    image = root / frames[stamp][eye]
                    path = f"{example}/{frame_index:03d}-{eye}.png"
                    archive.writestr(ZipInfo(path), image.read_bytes(), ZIP_STORED)
                    images[eye] = path
                    if alternate is not None:
                        yy, xx = np.nonzero(cv.imread(str(image), cv.IMREAD_GRAYSCALE) != alternate[index])
                        if len(xx):
                            regions[eye] = [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())]
                samples.append({
                    "time_ms": (stamp - start) / 1e6, "images": images, "cue_boxes": regions,
                    "truth_xz": xz([truth[stamp][key] for key in ("px", "py", "pz")]),
                    "candidates": {label: [candidate["score"], *xz(candidate["pose"]["position_m"])] if candidate else None
                                   for label, candidate in row["candidates"].items()},
                    "streak": row["confirmation_streak"], "published": row["published_region"],
                })
            clips[example] = {
                "id": incident_id, "true_region": incident["target_region"],
                "cue_ms": None if cue is None else (cue - start) / 1e6,
                "blackout_ms": (start - incident["mask_start_ns"]) / 1e6,
                "native": {
                    "time_ms": (int(publication["timestamp_ns"]) - start) / 1e6,
                    "region": reference[incident_id]["published_region"],
                    "outcome": reference[incident_id]["outcome"],
                    "position_error_m": reference[incident_id]["position_error_m"],
                    "inliers": reference[incident_id]["native"]["accepted_inliers"],
                    "xz": xz([publication[key] for key in ("px", "py", "pz")]),
                },
                "proposal_outcome": proposal[incident_id]["outcome"], "samples": samples,
            }
    write_json(output / "replay.json", {"manifest_sha256": digest, "map_sha256": manifest["map"]["sha256"],
                                        "images_sha256": sha256(archive_path), "clips": clips})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("benchmark", "native", "multi", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    capture(args.benchmark.resolve(), args.native, args.multi, args.output)


if __name__ == "__main__":
    main()

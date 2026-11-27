"""Summarize 1× replay timing and diagnose first-pose decisions from the same traces."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import platform
import subprocess

import cv2 as cv
import numpy as np
import scipy

from audit_benchmark import csv_rows
from benchmark import sha256
from compare_relocalization import require


def milliseconds(nanoseconds):
    return round(float(nanoseconds) / 1e6, 4)


def percentiles(samples):
    return {key: milliseconds(np.percentile(samples, p)) for key, p in (("p50_ms", 50), ("p95_ms", 95))}


def machine():
    info = Path("/proc/cpuinfo")
    model = next((line.split(":", 1)[1].strip() for line in info.read_text().splitlines()
                  if line.startswith("model name")), platform.processor()) if info.exists() else platform.processor()
    governor = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    try:
        native_opencv = subprocess.check_output(["pkg-config", "--modversion", "opencv4"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        native_opencv = None
    return {
        "cpu": model, "logical_cpus": os.cpu_count(), "kernel": platform.release(),
        "cpu0_governor": governor.read_text().strip() if governor.exists() else None,
        "python": platform.python_version(), "opencv_python": cv.__version__, "opencv_system": native_opencv,
        "numpy": np.__version__, "scipy": scipy.__version__,
        "openblas_num_threads": os.getenv("OPENBLAS_NUM_THREADS"), "omp_num_threads": os.getenv("OMP_NUM_THREADS"),
    }


def timing_rows(directory):
    by_incident = defaultdict(list)
    for entry in csv_rows(directory / "timing.csv"):
        incident = entry.pop("incident_id")
        by_incident[incident].append({key: int(value) for key, value in entry.items()})
    return by_incident


def frame_samples(rows, incident, case, period):
    start, deadline = incident["first_unmasked_ns"], incident["deadline_ns"]
    last_index = (deadline - start) // period
    expected_last = last_index if case["outcome"] == "timeout" else case["publication_frame"]
    require(len(rows) == expected_last + 1 and expected_last <= last_index,
            f"timing length mismatch for {case['incident_id']}")
    samples = []
    previous_finish = 0
    for index, row in enumerate(rows):
        due = index * period
        begin, end, cpu = row["started_ns"], row["finished_ns"], row["cpu_ns"]
        require((row["frame"], row["timestamp_ns"], row["scheduled_ns"]) == (index, start + due, due) and
                begin >= max(due, previous_finish) and end >= begin and cpu >= 0,
                f"invalid FIFO timing in {case['incident_id']} frame {index}")
        # Count subsequent frames that have reached their sensor release time during this service.
        backlog = min(last_index, end // period) - index
        samples.append({"service": end - begin, "cpu": cpu, "queue": begin - due,
                        "capture_to_ready": end - due, "backlog": backlog})
        previous_finish = end
    return samples


def timing_stats(samples, period):
    return {
        "processed_frames": len(samples),
        "service_wall": percentiles([row["service"] for row in samples]),
        "process_cpu": percentiles([row["cpu"] for row in samples]),
        "queue_delay": percentiles([row["queue"] for row in samples]),
        "capture_to_ready": percentiles([row["capture_to_ready"] for row in samples]),
        "service_over_frame_period": sum(row["service"] > period for row in samples),
        "ready_after_frame_period": sum(row["capture_to_ready"] > period for row in samples),
        "frames_with_waiting_queue_after_processing": sum(row["backlog"] > 0 for row in samples),
        "max_waiting_frames": max(row["backlog"] for row in samples),
    }


def read_trace(directory, incident_id):
    with (directory / f"{incident_id}.jsonl").open() as stream:
        return [json.loads(line) for line in stream]


def native_failures(directory, summary, timings):
    keyframes = {row["keyframe_id"]: row["region"] for row in csv_rows(directory / "atlas.csv")}
    candidates = defaultdict(set)
    for row in csv_rows(directory / "events.csv"):
        if row["event"] == "relocalization_candidate":
            candidates[int(row["timestamp_ns"])].add(keyframes[row["keyframe_id"]])
    missing = wrong = 0
    wrong_examples = []
    for case in summary["cases"]:
        trace = timings[case["incident_id"]]
        target = case["target_region"]
        missing += sum(target not in candidates[row["timestamp_ns"]] for row in trace)
        if case["outcome"] == "false":
            found = target in candidates[trace[-1]["timestamp_ns"]]
            wrong += int(found)
            wrong_examples.append({key: case[key] for key in (
                "incident_id", "target_region", "published_region", "publication_frame", "position_error_m",
            )} | {"target_region_retrieved": found, "accepted_inliers": case["native"]["accepted_inliers"]})
    return {"frames_without_target_region_in_retrieval": missing,
            "false_poses_with_target_region_retrieved": wrong, "false_publications": wrong_examples}


def policy_failures(directory, summary, timings):
    verified_missing = 0
    pruning = []
    unresolved_tied_frames = 0
    for case in summary["cases"]:
        incident_id, target = case["incident_id"], case["target_region"]
        trace = read_trace(directory, incident_id)
        require([row["timestamp_ns"] for row in trace] == [row["timestamp_ns"] for row in timings[incident_id]],
                f"timing and policy trace are out of sync in {incident_id}")
        verified_missing += sum(row["candidates"][target] is None for row in trace)
        for previous, row in zip(trace, trace[1:]):
            if target in previous["active_hypotheses"] and target not in row["active_hypotheses"] and row["candidates"][target]:
                candidates = row["candidates"]
                next_frame = next((later["frame"] for later in trace[row["frame"] + 1:]
                                   if target in later["active_hypotheses"]), None)
                pruning.append({"incident_id": incident_id, "target_region": target, "pruned_at_frame": row["frame"],
                                "active_regions": row["active_hypotheses"], "reacquired_at_frame": next_frame,
                                "scores": {key: item["score"] if item else None for key, item in candidates.items()}})
        if case["scenario"] == "unresolved":
            for row in trace:
                a, b = row["candidates"].values()
                if a and b and a["score"] == b["score"] and row["published_region"] is None:
                    unresolved_tied_frames += 1
    return {"frames_without_verified_target_pose": verified_missing,
            "target_pruning_with_current_geometric_support": pruning,
            "unresolved_tied_frames_without_publication": unresolved_tied_frames}


def describe(directory, manifest_hash, map_hash, incidents, period):
    summary = json.loads((directory / "summary.json").read_text())
    require((summary["manifest_sha256"], summary["map_sha256"], summary["split"]) ==
            (manifest_hash, map_hash, "held_out"), f"different inputs in {directory}")
    cases = summary["cases"]
    timings = timing_rows(directory)
    require([case["incident_id"] for case in cases] == list(incidents) == list(timings),
            f"missing, extra or reordered incidents in {directory}")
    all_samples, episodes = [], []
    for case in cases:
        name = case["incident_id"]
        rows = timings[name]
        samples = frame_samples(rows, incidents[name], case, period)
        all_samples.extend(samples)
        printed_pose = case["outcome"] != "timeout"
        cue = incidents[name]["first_visual_cue_ns"]
        after_cue = None if cue is None or not printed_pose else rows[-1]["timestamp_ns"] - cue
        episodes.append({
            "incident_id": name, "outcome": case["outcome"],
            "frames": len(samples), "max_waiting_frames": max(row["backlog"] for row in samples),
            "sensor_wait_ms": milliseconds(rows[-1]["scheduled_ns"]),
            "final_frame_queue_ms": milliseconds(samples[-1]["queue"]),
            "final_frame_service_ms": milliseconds(samples[-1]["service"]),
            "wall_until_decision_ms": milliseconds(rows[-1]["finished_ns"]),
            "sensor_time_from_cue_to_pose_ms": None if after_cue is None else milliseconds(after_cue),
            "publication_after_wall_deadline": bool(printed_pose and rows[-1]["finished_ns"] >
                                                    incidents[name]["deadline_ns"] - incidents[name]["first_unmasked_ns"]),
        })
    native = "policy" not in summary
    failures = native_failures(directory, summary, timings) if native else policy_failures(directory, summary, timings)
    return {
        "run": directory.name, "method": summary.get("policy", summary["method"]),
        "timing": timing_stats(all_samples, period),
        "publications_after_wall_deadline": sum(case["publication_after_wall_deadline"] for case in episodes),
        "episodes": episodes, "failure_analysis": failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    try:
        path = args.benchmark / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest_hash, map_hash = sha256(path), manifest["map"]["sha256"]
        period = manifest["calibration"]["camera_period_ns"]
        incidents = {row["incident_id"]: row for row in manifest["incidents"] if row["split"] == "held_out"}
        require(period > 0 and incidents, "benchmark requires a positive camera period and held-out incidents")
        runs = [describe(directory, manifest_hash, map_hash, incidents, period) for directory in args.runs]
        print(json.dumps({"manifest_sha256": manifest_hash, "map_sha256": map_hash, "split": "held_out",
                          "camera_frame_period_ms": milliseconds(period), "host": machine(), "runs": runs}, indent=2))
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"Replay profiling failed: {error}\n")


if __name__ == "__main__":
    main()

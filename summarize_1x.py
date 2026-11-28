"""Summarize paced relocalization using the exact camera timeline in a replay manifest."""

import argparse
from bisect import bisect_right
from collections import Counter
import json
from pathlib import Path

from benchmark import sha256
from compare_relocalization import require
from profile_replay import machine, milliseconds, read_trace, timing_rows, timing_stats
from run_relocalization import read_rows


def summarize(root, output):
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    summary = json.loads((output / "summary.json").read_text())
    digest, map_hash = sha256(manifest_path), manifest["map"]["sha256"]
    require((summary["manifest_sha256"], summary["map_sha256"]) == (digest, map_hash), "mixed replay inputs")
    incidents = {row["incident_id"]: row for row in manifest["incidents"] if row["split"] == summary["split"]}
    timings = timing_rows(output)
    require(list(timings) == [case["incident_id"] for case in summary["cases"]], "incomplete replay timing")
    period = manifest["calibration"]["camera_period_ns"]
    samples, cases = [], []
    total_comparisons = total_calls = 0
    for case in summary["cases"]:
        name = case["incident_id"]
        incident = incidents[name]
        start, deadline = incident["first_unmasked_ns"], incident["deadline_ns"]
        releases = [int(row["timestamp_ns"]) - start for row in read_rows(root, incident["frames"])
                    if start <= int(row["timestamp_ns"]) <= deadline]
        trace = read_trace(output, name)
        rows = timings[name]
        expected = len(releases) if case["outcome"] == "timeout" else case["publication_frame"] + 1
        require(len(rows) == len(trace) == expected, f"incomplete trace in {name}")
        previous_end = 0
        queue = []
        for index, (row, decision) in enumerate(zip(rows, trace, strict=True)):
            due = releases[index]
            begin, end, cpu = row["started_ns"], row["finished_ns"], row["cpu_ns"]
            require(row["frame"] == decision["frame"] == index and row["scheduled_ns"] == due and
                    row["timestamp_ns"] == decision["timestamp_ns"] == start + due and
                    end >= begin >= max(due, previous_end) and cpu >= 0, f"invalid paced FIFO in {name}")
            waiting = bisect_right(releases, end) - index - 1
            queue.append(waiting)
            samples.append({"service": end - begin, "cpu": cpu, "queue": begin - due,
                            "capture_to_ready": end - due, "backlog": waiting})
            total_comparisons += sum(item["keyframes_compared"] for item in decision["region_search"].values())
            total_calls += sum(item["descriptor_match_calls"] for item in decision["region_search"].values())
            previous_end = end
        details = dict(case, frames=len(rows), max_waiting_frames=max(queue),
                       wall_until_decision_ms=milliseconds(rows[-1]["finished_ns"]))
        if case["outcome"] == "timeout":
            candidates = [(decision["frame"], candidate) for decision in trace
                          for candidate in decision["candidates"].values() if candidate]
            if candidates:
                frame, candidate = max(candidates, key=lambda pair: pair[1]["score"])
                details["strongest_unpublished_candidate"] = {"frame": frame} | {
                    field: candidate[field] for field in ("keyframe", "inliers", "stereo_inliers", "coverage_cells", "score")
                }
        cases.append(details)
    return {
        "benchmark": manifest.get("source_sequence", "synthetic corridors"), "manifest_sha256": digest,
        "map_sha256": map_hash, "split": summary["split"], "runtime": summary.get("runtime"), "host": machine(),
        "timing": timing_stats(samples, period), "outcomes": dict(Counter(case["outcome"] for case in cases)),
        "keyframe_comparisons": total_comparisons, "descriptor_match_calls": total_calls, "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(summarize(args.benchmark.resolve(), args.run), indent=2))
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"1× summary failed: {error}\n")


if __name__ == "__main__":
    main()

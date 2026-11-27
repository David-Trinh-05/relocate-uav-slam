"""Compare complete relocalization runs on the same frozen benchmark split."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from benchmark import sha256
from relocalizer import POLICIES


OUTCOMES = ("correct", "false", "timeout")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def evaluate_run(directory, manifest_hash, map_hash, split, cases, camera_period_ns):
    summary = json.loads((directory / "summary.json").read_text())
    policy = summary["policy"]
    require(policy in POLICIES, f"unknown policy in {directory}")
    require((summary["manifest_sha256"], summary["map_sha256"], summary["split"]) ==
            (manifest_hash, map_hash, split), f"manifest, map or split mismatch in {directory}")
    results = {case["incident_id"]: case for case in summary["cases"]}
    require(list(results) == list(cases) and len(results) == len(summary["cases"]),
            f"incomplete, reordered or duplicate incident results in {directory}")
    totals = Counter()
    by_scenario = {}
    latencies = []
    per_case = []

    for incident_id, incident in cases.items():
        result = results[incident_id]
        scenario = incident["scenario"]
        outcome = result["outcome"]
        require(outcome in OUTCOMES and (result["scenario"], result["target_region"]) ==
                (scenario, incident["target_region"]), f"unexpected scoring metadata for {incident_id}")
        with (directory / f"{incident_id}.jsonl").open() as stream:
            trace = [json.loads(line) for line in stream]
        start, deadline = incident["first_unmasked_ns"], incident["deadline_ns"]
        require(trace and trace[0]["timestamp_ns"] == start, f"missing initial observation in {incident_id}")
        require(all(row["frame"] == index and row["timestamp_ns"] == start + index * camera_period_ns and
                    len(row["active_hypotheses"]) <= POLICIES[policy].retained_regions
                    for index, row in enumerate(trace)), f"invalid frame or retained region count in {incident_id}")
        publications = [row for row in trace if row["published_region"] is not None]
        end = trace[-1]["timestamp_ns"]
        if outcome == "timeout":
            require(not publications and end == deadline, f"incomplete timeout in {incident_id}")
        else:
            require(len(publications) == 1 and publications[0] is trace[-1] and start <= end <= deadline and
                    trace[-1]["published_region"] == result["published_region"] and
                    trace[-1]["confirmation_streak"] == POLICIES[policy].confirmation_frames and
                    (end - start) / 1e6 == result["recovery_ms"], f"inconsistent publication in {incident_id}")
            if outcome == "correct":
                latencies.append(result["recovery_ms"])
        comparisons = sum(region["keyframes_compared"] for row in trace for region in row["region_search"].values())
        totals.update({outcome: 1, "observed_frames": len(trace), "keyframe_comparisons": comparisons})
        by_scenario.setdefault(scenario, Counter())[outcome] += 1
        per_case.append({
            "incident_id": incident_id, "scenario": scenario, "outcome": outcome,
            "published_region": result.get("published_region"), "recovery_ms": result.get("recovery_ms"),
            "observed_frames": len(trace), "keyframe_comparisons": comparisons,
        })

    report = {
        "correct": totals["correct"], "false": totals["false"], "timeout": totals["timeout"],
        "correct_latency_p50_ms": float(np.median(latencies)) if latencies else None,
        "correct_latency_p95_ms": float(np.percentile(latencies, 95)) if latencies else None,
        "observed_frames": totals["observed_frames"], "keyframe_comparisons": totals["keyframe_comparisons"],
        "keyframe_comparisons_per_frame": round(totals["keyframe_comparisons"] / totals["observed_frames"], 3),
        "by_scenario": {scenario: {outcome: counts[outcome] for outcome in OUTCOMES}
                        for scenario, counts in by_scenario.items()},
        "cases": per_case,
    }
    return policy, report


def compare(benchmark, split, directories):
    manifest_path = benchmark / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_hash, map_hash = sha256(manifest_path), manifest["map"]["sha256"]
    cases = {case["incident_id"]: case for case in manifest["incidents"] if case["split"] == split}
    require(cases and all(case["map_sha256"] == map_hash for case in cases.values()), "invalid fixed-map split")
    period = manifest["calibration"]["camera_period_ns"]
    evaluated = [evaluate_run(directory, manifest_hash, map_hash, split, cases, period) for directory in directories]
    reports = dict(evaluated)
    require(len(evaluated) == len(reports) and set(reports) == set(POLICIES), "supply one complete run for each policy")
    single, multi = reports["single-temporal"], reports["multi-temporal"]
    paired = zip(single["cases"], multi["cases"], strict=True)
    changes = []
    for a, b in paired:
        if (a["outcome"], a["recovery_ms"]) != (b["outcome"], b["recovery_ms"]):
            changes.append({
                "incident_id": a["incident_id"], "single_outcome": a["outcome"], "multi_outcome": b["outcome"],
                "single_publication_ms": a["recovery_ms"], "multi_publication_ms": b["recovery_ms"],
            })
    return {
        "split": split, "incident_count": len(cases), "manifest_sha256": manifest_hash, "map_sha256": map_hash,
        "policies": {policy: reports[policy] for policy in POLICIES}, "temporal_policy_changes": changes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("runs", type=Path, nargs=3)
    parser.add_argument("--split", choices=("development", "held_out"), default="held_out")
    args = parser.parse_args()
    try:
        print(json.dumps(compare(args.benchmark, args.split, args.runs), indent=2))
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"Comparison failed: {error}\n")


if __name__ == "__main__":
    main()

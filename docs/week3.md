# Week 3 fixed-map benchmark

The benchmark supplies a frozen map of the two synthetic corridors, stereo and IMU inputs, and ground truth. Each of 18 distinct trajectories produces one outage. The development and held-out splits each contain nine flights with a distinct trajectory seed for every flight. Within each split and scenario, the blackout durations are 0.5, 1, and 2 seconds. The observation window includes the first unmasked frame and ends exactly five seconds later.

| Scenario | Development | Held out | Observed first visual difference between A and B |
| --- | ---: | ---: | --- |
| Immediate | 3 | 3 | First unmasked stereo pair |
| Delayed | 3 | 3 | 1.40–2.10 seconds after images return |
| Unresolved | 3 | 3 | Identical observations through the five-second deadline |

The audited dataset contains 2,778 stereo pairs, 27,618 IMU samples, and one map with 42 keyframes. The [frozen manifest](../results/week3-benchmark-manifest.json) records the exact incident IDs, split, ground-truth region and pose file, blackout interval, first visual cue, and map digest for every case. The [audit report](../results/week3-benchmark-audit.json) records counts and per-flight checks.

## Generate and audit

From the repository root, use Python 3.11+ with NumPy and OpenCV. The recorded hashes were generated with OpenCV 5.0.0 and NumPy 2.5.3. PNG encoder differences across environments can change file hashes. Each run must validate against its own manifest, and methods being compared must use the same generated snapshot and manifest.

```bash
python3 benchmark.py runs/week3-benchmark --jobs 4
python3 audit_benchmark.py runs/week3-benchmark
```

The audit verifies the map and data hashes, nonoverlapping flight seeds, exact 20 Hz stereo and 200 Hz IMU timelines, both-image masking, ground-truth coverage, and cue onset using a counterfactual rendering of the other corridor. It compares IMU angular speed and acceleration against finite differences of the recorded ground-truth motion. The supplied IMU samples are ideal simulated measurements in the left camera's body frame with gravity included in the specific-force calculation.

## Use the same inputs across methods

- `manifest.json` is the common index for the runner and scorer. The harness selects cases by `incident_id` and verifies `map_sha256`. Use `development` for tuning and leave `held_out` for evaluation.
- `map/index.json` holds 42 separately rendered reference viewpoints, with stereo images, left optical depth in meters, and body poses in the common map/world frame. The keyframes come from a dedicated set of reference viewpoints, separate from the query trajectories.
- `queries/<flight_id>/frames.csv` supplies the timestamp and stereo paths for each frame. The `left` and `right` fields point to a shared black image during the outage. The `clean_left` and `clean_right` fields support a control replay using the same `imu.csv`. Pass the method only the selected camera views, sensor timestamps, IMU, calibration, and fixed map. Keep scenario labels, clean views during outages, target region, cue onset, and ground-truth poses in the evaluation harness.
- `ground_truth/<flight_id>.csv` supplies the world position and quaternion in `xyzw` order at every camera timestamp. The synthetic map and poses already use the same frame. The evaluator uses the ground truth to classify the first published relocalization using the [fixed-map protocol](evaluation.md).

This fixture models global relocalization with an unknown global pose when images return. A method that carries a known global pose through the gap can resolve the translated-corridor symmetry from that prior. Such a run measures tracking recovery, as in week 2. For the global benchmark, start each method at the first unmasked pair with the same prior information and the same relative sensor motion.

The map is a synthetic reference snapshot with exact rendered depth. A native ORB-SLAM3 Atlas adapter and the three-method comparison remain future work. The benchmark currently measures visibility and sensor consistency on the synthetic scenes. Field performance requires separate data.

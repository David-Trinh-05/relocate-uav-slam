# Week 6 native ORB-SLAM3 relocalization baseline

The adapter imports all 42 reference views from the frozen corridor map into a single ORB-SLAM3 Atlas. It computes native ORB features and creates 45,239 landmarks by projecting the exact snapshot depth into the map coordinate frame. Each landmark belongs to its reference keyframe. The two-region snapshot supplies known poses and depth, while ORB-SLAM3 supplies the feature extractor, vocabulary, BoW retrieval, MLPnP, pose optimization and acceptance rule.

The harness calls the upstream `Tracking::Relocalization()` routine directly from the first unmasked stereo pair, with a fixed seed per incident. It tests the visual relocalization stage in isolation. The Week 2 trace patch records candidates and accepted pose evidence without changing the decision rule. ReLocate uses the same reference images, depth, query stereo pairs and incident window, plus relative gyro rotation for temporal confirmation. The Atlas is populated from exact synthetic depth and independent reference views. Accuracy with a map built by upstream SLAM and the behavior of the full closed-loop tracker are separate questions.

## Held-out outcomes

| Method | Correct first pose | False first pose | No pose by 5 s |
| --- | ---: | ---: | ---: |
| ORB-SLAM3 `Tracking::Relocalization()` | 6 | 3 | 0 |
| ReLocate, one-frame confirmation | 6 | 0 | 3 |
| ReLocate, single-temporal | 6 | 0 | 3 |
| ReLocate, multi-temporal | 6 | 0 | 3 |

All three baseline errors were wrong-region publications at the first unmasked pair, with position errors of 5.99–6.02 m. The correct region had a retrieved candidate at that frame in every case. The two delayed-cue errors became correct publications with ReLocate at 2.15 and 2.35 s. The third had visually identical corridors throughout the five-second window, and ReLocate withheld a pose. ReLocate also withheld poses for the two unresolved cases in which the baseline's first choice happened to agree with ground truth. [Paired results](../results/week6-comparison.json)

In `held_out-delayed-03-outage`, the native database returned only `A-1-000` and `B-1-000`, with 58 BoW matches each. The native routine accepted A with 52 inliers while the true pose was in B. The first visual difference appeared at 2.10 s. The multi-temporal policy published B at 2.35 s, 250 ms after the cue. [Full native candidate evidence](../results/week6-orb-heldout.json)

A diagnostic reran those three errors after importing region B's keyframes first. All three first-frame publications switched from A to B and met the pose thresholds. The visual evidence and seed were identical in both imports. This verifies sensitivity to Atlas insertion order on the tied observations. [Order diagnostic](../results/week6-order-check.json)

## Reproduce

Build the pinned ORB-SLAM3 checkout in Release with the replay patch. Build the adapter with the same Release compiler settings so Eigen layouts agree across the shared-library boundary. Generate the frozen synthetic benchmark and run the upstream routine. [Build instructions](build.md) · [Benchmark generation](week3.md)

```bash
cmake -S native -B runs/native-build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DORB_SLAM3_ROOT="$PWD/third_party/ORB_SLAM3" \
    -DPangolin_DIR="$PWD/third_party/Pangolin/build"
cmake --build runs/native-build
python3 run_orb_baseline.py runs/week3-benchmark runs/week6-orb-heldout \
    runs/native-build/orb_fixed_map third_party/ORB_SLAM3/Vocabulary/ORBvoc.txt --split held_out
```

Run the [week 5 policies](week5.md#reproduce) on the same benchmark directory, then include the native summary in the paired comparison.

```bash
python3 compare_relocalization.py runs/week3-benchmark \
    runs/week5-single-frame runs/week5-single-temporal runs/week5-multi-temporal \
    --native runs/week6-orb-heldout/summary.json
```

The adapter exports the Atlas inventory, tracker frames and native candidate events to the run directory. Two held-out runs reproduced the same result JSON and candidate event trace with ORB-SLAM3 revision `4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4` and OpenCV 4.6.0 for the native binary.

For the order diagnostic, copy the generated `request.json` and settings to another output directory, sort the request's `keyframes` by `region` descending while retaining order within each region, and run `orb_fixed_map` there with `RELOCATE_TRACE_REPLAY=1`. The depth and image paths in the request point to the original data. Compare `relocalization_accept` in each `events.csv` using the corresponding keyframe IDs in `atlas.csv`.

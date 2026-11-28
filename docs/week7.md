# Week 7 replay profiling and failure analysis

These measurements describe the original Python/OpenCV matcher. [Optimized replay measurements](recorded-replay.md)

The nine held-out clips were replayed at their original 20 Hz on an Intel Core i7-13700H with 20 logical CPUs, Ubuntu 24.04.4, Python 3.12.3, OpenCV 5.0.0 for Python and OpenCV 4.6.0 for native ORB-SLAM3. The runs executed sequentially with the OpenCV worker limit and OpenBLAS and OMP limits set to one. ORB-SLAM3's stereo frame constructor still launches separate left and right extraction threads. Map initialization takes place before the per-incident clock begins.

Each method consumes an in-order FIFO without dropping frames. The replay clock releases each observation at its recorded offset from the first unmasked pair. A frame can wait after release while the previous frame is processed. The reported queue depth counts unprocessed frames that have reached their release time. Processing time includes stereo image decoding and estimation, plus gyro integration for the Python policies. Process CPU time counts work on all threads in the process. Wall time to a pose is the sensor offset of its decisive frame plus that frame's queue delay and processing time.

## Measured 1× replay

| Method | Frames processed | Wall processing p50 | Wall processing p95 | Queue delay p95 | Max frames waiting |
| --- | ---: | ---: | ---: | ---: | ---: |
| ORB-SLAM3 `Tracking::Relocalization()` | 13 | 22.24 ms | 32.51 ms | 0.14 ms | 0 |
| ReLocate, one carried hypothesis | 450 | 245.58 ms | 247.82 ms | 18.33 s | 81 |
| ReLocate, two carried hypotheses | 450 | 135.94 ms | 139.64 ms | 8.27 s | 66 |

The two Python policies published at the same sensor timestamps and processed the same 450 frames. Carrying both hypotheses cut p95 processing time by 43.7%. The prototype exceeded its 50 ms frame budget on every measured frame. A separate repeat gave 138.97 ms p95 and 65 waiting frames at peak. The native routine often published on frame zero, so its 13-frame sample describes the work performed through its first decisions. [Timing results](../results/week7-profile.json)

In `held_out-delayed-03-outage`, ReLocate needed 2.350 s of sensor history for confirmation. Its decisive frame then had 3.546 s of queue delay and 0.126 s of processing. The pose was ready **6.022 s** after the first unmasked frame arrived. All three delayed correct poses from each Python policy arrived after five seconds of wall time on this host. The sensor-time metrics from week 6 still describe the information required to make those decisions.

A separate untimed `cProfile` run on the 101-frame development incident `development-unresolved-01-outage` spent 12.24 of 15.53 seconds, about 79%, inside OpenCV's descriptor matcher. ORB extraction accounted for 1.18 s of self time and PnP RANSAC for 0.20 s. Descriptor matching is the largest processing cost in this sample. [Profile breakdown](../results/week7-bottleneck.json)

## Failure traces

- **Selection error in the native baseline.** Every native query frame retrieved the true region. In the three false publications, the native rule accepted A while the true region B was already in the same candidate list. In `held_out-delayed-03-outage`, both candidates had 58 BoW matches and the accepted A pose had 52 inliers.
- **Pruning an available true pose.** The single-hypothesis policy discarded the true region three times across two incidents despite a geometrically verified pose in that frame. In `held_out-delayed-03-outage`, B scored 116 at frame 26 and A scored 122.5, so the policy retained A. B returned to the active set at frame 42. The multi-hypothesis policy carried B throughout that episode. Both policies ultimately published the correct B pose at frame 47.
- **Persistent ambiguity.** Both regions had equal scores in all 303 unresolved frames of the multi-hypothesis run. The policy withheld a pose throughout all three incidents. Every observed multi-hypothesis frame had a geometrically verified pose in the true region. [Per-incident diagnostics](../results/week7-profile.json)

## Reproduce

Use the [week 6 native build](week6.md#reproduce) and the same benchmark directory. Run one timed process at a time to measure the current machine.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cmake --build runs/native-build
python3 run_orb_baseline.py runs/week3-benchmark runs/week7-native \
    runs/native-build/orb_fixed_map third_party/ORB_SLAM3/Vocabulary/ORBvoc.txt --split held_out --realtime
python3 run_relocalization.py runs/week3-benchmark runs/week7-single-temporal \
    --split held_out --method single-temporal --realtime --matcher opencv --opencv-threads 1 --no-match-cache
python3 run_relocalization.py runs/week3-benchmark runs/week7-multi-temporal \
    --split held_out --method multi-temporal --realtime --matcher opencv --opencv-threads 1 --no-match-cache
python3 run_relocalization.py runs/week3-benchmark runs/week7-multi-repeat \
    --split held_out --method multi-temporal --realtime --matcher opencv --opencv-threads 1 --no-match-cache
python3 profile_replay.py runs/week3-benchmark runs/week7-native runs/week7-single-temporal \
    runs/week7-multi-temporal runs/week7-multi-repeat
```

Each timed run writes `timing.csv` alongside its pose and candidate traces. The report validates the manifest and map digests, incident order, frame schedule, capture-to-ready timing and trace alignment before calculating percentiles.

```bash
python3 -m cProfile -s tottime run_relocalization.py runs/week3-benchmark runs/week7-multi-dev-profile \
    --method multi-temporal --incident development-unresolved-01-outage \
    --matcher opencv --opencv-threads 1 --no-match-cache
```

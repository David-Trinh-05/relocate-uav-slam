# Week 5 held-out comparison

All three policies share the fixed-map stereo PnP and gyro implementation. Thresholds, camera inputs, IMU, and map stay the same. The policies were committed in `998c355` before running the nine held-out incidents from the frozen manifest.

| Policy | Regions with a carried pose | Consecutive supporting frames for publication |
| --- | ---: | ---: |
| `single-frame` | 1 | 1 |
| `single-temporal` | 1 | 3 |
| `multi-temporal` | 2 | 3 |

The single-region policies retain the best region's pose after each frame. The other region can still challenge through fresh retrieval on the next frame. This keeps the same relative score gate in all variants. The comparison between `single-temporal` and `multi-temporal` isolates the effect of retaining the second pose while holding temporal confirmation fixed.

## Held-out results

| Policy | Correct | False | Timeout | Correct latency p50 | Correct latency p95 | Keyframe comparisons per frame |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `single-frame` | 6 | 0 | 3 | 0.625 s | 2.125 s | 27.323 |
| `single-temporal` | 6 | 0 | 3 | 1.125 s | 2.325 s | 27.300 |
| `multi-temporal` | 6 | 0 | 3 | 1.125 s | 2.325 s | 12.813 |

Both three-frame policies published the same outcomes at identical timestamps on all nine incidents. They processed the same 450 frames, with 12,285 keyframe comparisons for the single-region policy and 5,766 for the two-region policy. Keeping both gyro-predicted poses allows a shortlist in each region, which reduced descriptor-matching work by **53.1%** in this fixture. The measurement counts descriptor comparisons and leaves wall-clock performance for profiling.

An exploratory visibility check found that the one-frame policy published region B at 1.25 s in `held_out-delayed-03-outage`, before the first visual difference at 2.10 s. Ground truth labels that pose correct. The stereo observations were still symmetric, so the publication timing exposes an unsupported early choice. Both three-frame policies waited until 2.35 s in that incident. [Held-out results](../results/week5-heldout.json) · [Development results](../results/week5-development.json)

## Reproduce

Install NumPy, OpenCV, and SciPy, then [generate and audit the benchmark](week3.md). Run the three policies from the repository root.

```bash
for policy in single-frame single-temporal multi-temporal; do
    python3 run_relocalization.py runs/week3-benchmark "runs/week5-$policy" \
        --split held_out --method "$policy" || exit
done
python3 compare_relocalization.py runs/week3-benchmark \
    runs/week5-single-frame runs/week5-single-temporal runs/week5-multi-temporal
```

The comparison checks manifest and map digests, the complete ordered case list, the frame timeline, the retained-region limit, and the publication rule. A keyframe comparison is one descriptor match against one reference keyframe, including any fallback search. Latencies use sensor timestamps and percentiles include only correct first poses.

The policies run on the generated reference snapshot and use idealized synthetic sensing. ORB-SLAM3's EuRoC mapping-mode replay is documented in [week 2](week2.md). A native Atlas comparison on the corridor benchmark and real-time profiling remain open.

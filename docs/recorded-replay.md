# Real-time replay and recorded UAV data

ReLocate uses a C++ Hamming matcher to compute both nearest-neighbor directions in one traversal. The matcher stops evaluating a descriptor pair once its partial distance exceeds the acceptance threshold. It preserves OpenCV's match order, Hamming distances and first-index tie behavior for all accepted pairs. The Python frontend reuses matches within each observation when reference descriptor arrays are identical, including repeated comparisons in a fallback search.

For a map region with at least 32 keyframes, a coarse ORB-word histogram ranks the views during global search. The six best views advance to exact descriptor matching and geometric verification. The synthetic benchmark has 21 keyframes per region and uses all of them for global search. The recorded map has 48 per region and uses the index. Confirmation and pose thresholds are the same in both experiments.

## Measured throughput

All timings use 1× replay at 20 Hz on an i7-13700H. The 50 ms frame budget corresponds to the camera period. Processing includes image decoding, gyroscope integration, feature extraction and pose estimation. Capture-to-ready time also includes FIFO queue delay. The native matcher ran with one OpenCV worker.

| Input and backend | Frames | Processing p50 | Processing p95 | Capture-to-ready p95 | Max waiting frames |
| --- | ---: | ---: | ---: | ---: | ---: |
| Synthetic, original Python/OpenCV | 450 | 135.94 ms | 139.64 ms | 8.402 s | 66 |
| Synthetic, native matcher | 450 | 16.81 ms | 22.91 ms | 22.91 ms | 1 |
| EuRoC, native matcher | 320 | 33.45 ms | 45.62 ms | 46.00 ms | 1 |
| EuRoC, repeat | 320 | 33.22 ms | 44.89 ms | 45.11 ms | 0 |

On the synthetic benchmark, p95 processing fell by 83.6% and all 450 per-frame candidate poses, scores and decisions matched the original run. Caching executed 3,116 descriptor matching calls for 5,766 keyframe comparisons. In the delayed case shown in the demo, the same 2.35 s sensor-time publication moved from 6.02 s to 2.38 s wall time. [Original timings](../results/week7-profile.json) · [Optimized synthetic replay](../results/optimized-synthetic-1x.json)

## Recorded evaluation

The recorded test uses rectified stereo images and 200 Hz gyro samples from `V1_01_easy`. The reference snapshot contains 96 keyframes sampled from frames 60 through 1485 of that flight. Reference poses come from the dataset's provided ground-truth trajectory, transformed into the rectified left camera frame. Map depths come from stereo disparity. This is a controlled relocalization test across disjoint segments of one flight. [EuRoC MAV Dataset](https://projects.asl.ethz.ch/datasets/euroc-mav/)

The seven query segments start at frames 1630, 1810, 1990, 2170, 2350, 2530 and 2710. Both images are masked for the preset interval while camera and IMU timestamps remain intact. Each episode starts pose estimation at the first unmasked frame with an unknown global pose. The evaluator uses query ground truth to score the first published pose with thresholds of 0.30 m and 10°. The same snapshot and query periods were used for the exhaustive and indexed runs.

Both optimized runs produced **4 correct first poses, 0 false first poses and 3 timeouts**. The correct poses were ready after 135–283 ms wall time in the primary run, with position errors of 5.1–17.4 cm. One timeout reached 29 of the required 30 left-image inliers. Another found 139 left-image inliers concentrated in three image cells, below the four-cell coverage gate. These outcomes establish performance on this recorded sequence and map coverage. [Recorded run](../results/euroc-v1-01-replay.json) · [Repeat](../results/euroc-v1-01-repeat.json)

## Reproduce

Build the matcher with a C++17 compiler. The unit tests compare its mutual nearest-neighbor output to OpenCV, including ties and the distance cutoff. Rebuild the library on each target machine to select its CPU instructions.

```bash
./build-native-matcher.sh
python3 -m unittest discover -s tests
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

# Generate runs/week3-benchmark first using benchmark.py.
python3 run_relocalization.py runs/week3-benchmark runs/synthetic-optimized \
    --split held_out --realtime --matcher native
python3 summarize_1x.py runs/week3-benchmark runs/synthetic-optimized
```

Use the ASL-format sequence with `mav0/` under `data/V1_01_easy`. Install PyYAML for the camera calibration parser. The builder writes rectified images and a fresh manifest to the output directory. [Dataset setup](build.md#run-the-euroc-smoke-test)

```bash
python3 -m pip install PyYAML
python3 build_euroc_replay.py data/V1_01_easy runs/euroc-fixed-map
python3 run_relocalization.py runs/euroc-fixed-map runs/euroc-optimized \
    --split held_out --realtime --matcher native
python3 summarize_1x.py runs/euroc-fixed-map runs/euroc-optimized
```

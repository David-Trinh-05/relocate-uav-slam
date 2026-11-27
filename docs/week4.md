# Week 4 multi-hypothesis relocalization

The prototype starts at the first unmasked stereo pair with an unknown global pose. The week 3 snapshot supplies two labeled map regions. The runner passes images and relative gyroscope rotations to the estimator and reads ground truth when scoring the resulting decision.

1. **Retrieve views in each region.** Extract ORB descriptors and rank reference keyframes by cross-checked Hamming matches. Keep the best geometrically verified pose per region so several nearby keyframes cannot consume both hypotheses. Subsequent frames search near the gyro-predicted orientation and fall back to a region-wide search if the shortlist produces no pose.
2. **Verify each pose.** Use map depth and left-image matches for PnP RANSAC, then count right-image reprojection support from stereo matches. Solve PnP in the reference keyframe's optical frame. Geometrically identical reference views translated in world coordinates therefore receive the same numerical PnP inputs.
3. **Carry both hypotheses forward.** Integrate the gyroscope between camera timestamps and gate candidates against the predicted rotation. Stereo geometry updates position. After eight consecutive frames with weak or missing geometry, discard that region's stale motion prior so retrieval can initialize it again.
4. **Confirm over time.** Score each region as its left-image inliers plus half its right-image inliers. A leader needs at least 30 left inliers, 10 right inliers, and support in four image cells. Its score must exceed the runner-up by at least 24 points and a factor of 1.3 in three consecutive frames. Ties and weak geometry reset confirmation. The runner stops after the first published pose or the five-second deadline.

## Verified development results

| Scenario | Cases | Correct first pose | False first pose | Timeout | Sensor time after the first unmasked pair |
| --- | ---: | ---: | ---: | ---: | --- |
| Immediate | 3 | 3 | 0 | 0 | 100 ms |
| Delayed | 3 | 3 | 0 | 0 | 1.60–2.05 s |
| Unresolved | 3 | 0 | 0 | 3 | No pose published in 5 s |

The three delayed cases published 200 ms after the first visible difference between the corridors. In the example below, frame 0 is the first unmasked pair. The cue becomes visible at frame 28. The region scores separate enough to start confirmation at frame 30, and the estimator publishes region A at frame 32. [Example trace](../results/week4-delayed-trace.json)

| Frame | Time | A score | B score | Decision |
| ---: | ---: | ---: | ---: | --- |
| 0 | 0 ms | 144.5 | 144.5 | Keep both |
| 28 | 1,400 ms | 248.5 | 248.5 | Keep both |
| 30 | 1,500 ms | 394.0 | 267.0 | Confirming A, frame 1 of 3 |
| 31 | 1,550 ms | 289.5 | 172.5 | Confirming A, frame 2 of 3 |
| 32 | 1,600 ms | 300.5 | 116.5 | Publish A |

The largest position and orientation errors across the six published poses were 5.4 cm and 1.22°. All 303 frames across the three unresolved incidents retained both hypotheses with tied geometry scores. The measurements come from the development split on idealized synthetic observations. [Summary](../results/week4-development.json)

## Reproduce

Generate the [week 3 benchmark](week3.md) and use Python with NumPy, OpenCV and SciPy.

```bash
python3 run_relocalization.py runs/week3-benchmark runs/week4-development
```

Each run writes `summary.json` and one JSONL trace per incident. The traces include both candidate poses, descriptor retrieval status, inliers, median reprojection error, gyro residuals, and confirmation state. The estimator operates on the standalone snapshot with exact synthetic depth, predefined regions, and gyro rotation. ORB-SLAM3 Atlas integration, comparison with the single-hypothesis confirmation variant, and measurement on the held-out split remain separate experiments.

# ReLocate portfolio

ReLocate maintains alternative 6-DoF pose hypotheses after a UAV camera outage. It checks both regions with stereo reprojection, carries them forward using gyro rotation, and confirms a region only after it leads for three frames. The benchmark uses a fixed map of two synthetic corridors six meters apart.

## Demo video

The 76-second, 720p MP4 includes English captions. The red cross marks the first ORB-SLAM3 pose, the blue circles show ReLocate's candidates, and the larger green ring indicates ground truth. The clip uses captured traces from the 20 Hz experiment. The video slows the sensor timeline so viewers can read the decisions. [Watch the MP4](https://raw.githubusercontent.com/David-Trinh-05/relocate-uav-slam/main/demo/relocate.mp4)

| Video chapter | What to look for |
| --- | --- |
| 00:00 | Repeated corridor geometry and the camera blackout |
| 00:09 | ORB-SLAM3 publishes A at the first unmasked pair. Both ReLocate poses remain active. |
| 00:35 | The first distinctive pixels enter the right camera at 2.10 s sensor time. |
| 00:47 | ReLocate publishes B at 2.35 s sensor time. Processing and queueing bring measured wall time to 6.02 s. |
| 00:54 | In the second clip both regions stay plausible through the five-second deadline. |
| 01:08 | Nine held-out results and the measured processing budget |

## Run locally

From the repo root, use Python 3.11+, Pillow, FFmpeg with `libx264`, and DejaVu Sans to render the captured demonstration. The camera bundle and pose trace are checked against the archived benchmark evidence. The renderer writes `runs/relocate-demo.mp4`.

```bash
python3 -m demo.render
```

To rerun the experiments, install NumPy, OpenCV, and SciPy and build the pinned ORB-SLAM3 dependencies with the [week 2 trace patch and week 6 setup](week6.md#reproduce). The benchmark command generates the synthetic data if needed, audits the manifest and map, builds the native adapter, and reports all four methods. The final `comparison.json` and traces are written under a fresh `runs/portfolio-benchmark.*` directory.

```bash
./run-portfolio-benchmark.sh
```

The main comparison measures 6 correct, 3 false and 0 withheld poses for the native ORB-SLAM3 relocalization routine. ReLocate records 6 correct, 0 false and 3 withheld. The prototype's 20 Hz replay exceeded its 50 ms/frame budget, with 139.64 ms p95 on the tested i7-13700H. The result concerns synthetic fixed-map relocalization. [Paired comparison](../results/week6-comparison.json) · [Wall-clock profile](../results/week7-profile.json)

## CV bullet

> Built a stereo–inertial multi-hypothesis UAV relocalizer that reduced false first poses from **3/9 to 0/9** versus ORB-SLAM3's native relocalization on a held-out synthetic fixed-map benchmark, with **6 correct recoveries and 3 abstentions**.

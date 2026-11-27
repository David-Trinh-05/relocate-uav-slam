# ReLocate

Multi-hypothesis relocalization for UAV visual–inertial SLAM. The evaluation asks whether keeping several plausible map locations reduces false relocalizations when two places look alike.

## Week 1 baseline

The baseline runs the upstream stereo–inertial EuRoC example on `V1_01_easy`. It checks the input timeline and trajectory exports so the next experiment can focus on relocalization events. The [evaluation protocol](docs/evaluation.md) defines how those events will be scored on a fixed map.

The [first verified run](results/week1-v1-01-easy.json) processed 2,912 stereo pairs and exported 2,809 frame poses and 107 keyframe poses. Every exported timestamp matched the camera timeline. Ground truth was within 20 ms for 2,790 frame poses and 105 keyframe poses. These measurements validate the clean replay and its exports.

### Build on Ubuntu

Use a C++17 compiler, CMake, Ninja, Python 3, and development packages for OpenCV 4.4+, Eigen3, Boost Serialization, OpenSSL, OpenGL, EGL, and X11. See the upstream dependency notes for [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) and [Pangolin](https://github.com/stevenlovegrove/Pangolin). From the repository root, use these upstream revisions and limit parallel compilation to fit available memory.

```bash
mkdir -p third_party
git clone --branch v0.8 --depth 1 https://github.com/stevenlovegrove/Pangolin.git third_party/Pangolin
git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git third_party/ORB_SLAM3
git -C third_party/ORB_SLAM3 switch --detach 4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4
git -C third_party/ORB_SLAM3 apply "$PWD/patches/orb-slam3-cxx17.patch"

pangolin="$PWD/third_party/Pangolin"
orb="$PWD/third_party/ORB_SLAM3"
cmake -S "$pangolin" -B "$pangolin/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS='-include cstdint' -DBUILD_PANGOLIN_PYTHON=OFF \
    -DBUILD_EXAMPLES=OFF -DBUILD_TOOLS=OFF -DBUILD_TESTS=OFF \
    -DBUILD_PANGOLIN_FFMPEG=OFF -DBUILD_PANGOLIN_OPENNI=OFF -DBUILD_PANGOLIN_OPENNI2=OFF
cmake --build "$pangolin/build" --parallel 4

cmake -S "$orb/Thirdparty/DBoW2" -B "$orb/Thirdparty/DBoW2/build" \
    -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build "$orb/Thirdparty/DBoW2/build" --parallel 2
tar -xf "$orb/Vocabulary/ORBvoc.txt.tar.gz" -C "$orb/Vocabulary"

cmake -S "$orb" -B "$orb/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=17 -DPangolin_DIR="$pangolin/build"
cmake --build "$orb/build" --target stereo_inertial_euroc --parallel 4
```

### Run

Extract `V1_01_easy` in ASL format to `data/V1_01_easy`, with `mav0/` directly inside that directory. The dataset is available from [ETH Zürich](https://projects.asl.ethz.ch/datasets/euroc-mav/) and as a [single-sequence mirror](https://huggingface.co/datasets/pepijn223/euroc-mirror/blob/d05bd6a/V1_01_easy.zip). The mirrored ZIP used here has SHA-256 `a920fe5b5e69a6ad19b32f1cfaf90ac2f59d45dfd1ca18cc9722e07684ba45fa`.

```bash
./run-euroc.sh third_party/ORB_SLAM3 data/V1_01_easy
```

Each run writes `baseline.log`, `f_baseline.txt`, `kf_baseline.txt`, the upstream revision, input and build hashes, and `summary.json` to a new directory under `runs/`. The example paces its replay using the dataset timestamps. The audit checks finite poses, quaternion norms, camera timestamps and association with ground-truth timestamps. Its counts describe trajectory exports after SLAM optimization.

The build patch replaces the three legacy post-increments of a `bool` with assignment to `true`. This preserves the C++11 effect of those statements under C++17.

## Week 2 replay and visual ambiguity

Mask both camera images while replaying the same camera timestamps and IMU, then inspect the live tracker decisions. The [week 2 report](docs/week2.md) includes the reproduction commands, the measured EuRoC outcomes, and an independently rendered two-corridor ambiguity case.

## Week 3 fixed-map benchmark

The [benchmark guide](docs/week3.md) defines 18 synthetic stereo–inertial outages across three visibility scenarios. Each method uses the same incident manifest and 42-keyframe map. Separate development and held-out flights keep threshold tuning out of the final comparison.

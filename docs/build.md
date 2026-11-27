# Build on Ubuntu

Use a C++17 compiler, CMake, Ninja, Python 3, and development packages for OpenCV 4.4+, Eigen3, Boost Serialization, OpenSSL, OpenGL, EGL, and X11. Build ORB-SLAM3 from the repository root. Use lower parallelism if compilation runs out of memory. [ORB-SLAM3 dependencies](https://github.com/UZ-SLAMLab/ORB_SLAM3) · [Pangolin dependencies](https://github.com/stevenlovegrove/Pangolin)

## Build ORB-SLAM3

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

## Enable relocalization traces

The full benchmark and blackout replay require the instrumentation patch. Apply it once and rebuild the library.

```bash
git -C third_party/ORB_SLAM3 apply "$PWD/patches/orb-slam3-replay.patch"
cmake --build third_party/ORB_SLAM3/build --target stereo_inertial_euroc --parallel 2
```

You can now run `./run-portfolio-benchmark.sh` from the repository root. The script builds the native adapter in Release using the same CPU flags as the ORB-SLAM3 build.

## Run the EuRoC smoke test

Extract `V1_01_easy` in ASL format to `data/V1_01_easy`, with `mav0/` directly inside that directory. The mirrored ZIP used for the recorded run has SHA-256 `a920fe5b5e69a6ad19b32f1cfaf90ac2f59d45dfd1ca18cc9722e07684ba45fa`. [ETH Zürich data](https://projects.asl.ethz.ch/datasets/euroc-mav/) · [Single-sequence mirror](https://huggingface.co/datasets/pepijn223/euroc-mirror/blob/d05bd6a/V1_01_easy.zip)

```bash
./run-euroc.sh third_party/ORB_SLAM3 data/V1_01_easy
```

Each replay saves the native log, frame and keyframe trajectories, build hashes, and a timestamp audit in a fresh directory under `runs/`. The recorded baseline exported 2,809 frame poses and 107 keyframe poses from 2,912 stereo pairs. [Recorded smoke-test result](../results/week1-v1-01-easy.json)

"""Replay the EuRoC stereo–inertial baseline with an optional image outage."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from check_trajectory import read_timestamps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("orb_slam3", type=Path)
    parser.add_argument("sequence", type=Path)
    parser.add_argument("--start-frame", type=int, required=True, help="zero-based index")
    parser.add_argument("--mask-frames", type=int, required=True, help="use 0 for the control")
    parser.add_argument("--limit-frames", type=int, help="replay this many pairs from the beginning")
    args = parser.parse_args()

    orb = args.orb_slam3.resolve()
    sequence = args.sequence.resolve()
    example = orb / "Examples/Stereo-Inertial"
    executable = example / "stereo_inertial_euroc"
    vocabulary = orb / "Vocabulary/ORBvoc.txt"
    settings = example / "EuRoC.yaml"
    imu = sequence / "mav0/imu0/data.csv"
    try:
        timestamps = read_timestamps(example / "EuRoC_TimeStamps/V101.txt")
        for path in (settings, vocabulary, executable, imu):
            if not path.is_file():
                raise ValueError(f"missing required input {path}")
        for camera in ("cam0", "cam1"):
            sensor = sequence / "mav0" / camera
            if read_timestamps(sensor / "data.csv", ",") != timestamps:
                raise ValueError(f"{camera} timeline must match the upstream V1_01_easy timeline")
            if not (sensor / "data").is_dir():
                raise ValueError(f"missing image directory {sensor / 'data'}")
    except (OSError, ValueError) as error:
        parser.error(str(error))

    limit = len(timestamps) if args.limit_frames is None else args.limit_frames
    start = args.start_frame
    resume = start + args.mask_frames
    if not 0 < start <= resume < limit <= len(timestamps):
        parser.error("expected 0 < start-frame <= first-unmasked-frame < limit-frames <= sequence length")
    if timestamps[limit - 1] < timestamps[resume] + 5_000_000_000:
        parser.error("include at least five seconds of replay after the first unmasked frame")

    root = Path(__file__).resolve().parent / "runs"
    root.mkdir(exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=f"replay-{start}-{args.mask_frames}.", dir=root))
    prefix = run / "timestamps.txt"
    prefix.write_text("".join(f"{stamp}\n" for stamp in timestamps[:limit]))
    with imu.open("rb") as stream:
        imu_checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    metadata = {
        "sequence": "V1_01_easy",
        "mode": "stereo-inertial mapping",
        "frames": limit,
        "start_frame": start,
        "masked_frames": args.mask_frames,
        "resume_frame": resume,
        "imu_sha256": imu_checksum,
        "orb_slam3_commit": subprocess.check_output(
            ["git", "-C", str(orb), "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    (run / "replay.json").write_text(json.dumps(metadata, indent=2) + "\n")
    environment = dict(
        os.environ,
        RELOCATE_TRACE_REPLAY="1",
        RELOCATE_MASK_FIRST=str(start),
        RELOCATE_MASK_STOP=str(resume),
    )
    print(run, flush=True)
    with (run / "baseline.log").open("w") as log:
        completed = subprocess.run(
            [executable, vocabulary, settings, sequence, prefix, "replay"],
            cwd=run, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False,
        )
    if completed.returncode:
        parser.exit(1, f"ORB-SLAM3 exited with status {completed.returncode}. See {run / 'baseline.log'}.\n")
    if not (run / "frames.csv").is_file():
        parser.exit(1, "Missing frames.csv. Apply the replay instrumentation patch and rebuild ORB-SLAM3.\n")


if __name__ == "__main__":
    main()

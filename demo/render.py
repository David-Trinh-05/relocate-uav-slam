"""Render the captioned ReLocate demo from the recorded stereo / pose traces."""

import argparse
from functools import lru_cache
import hashlib
from io import BytesIO
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SIZE = (1280, 720)
FPS = 24
BACKGROUND = "#101824"
PANEL = "#192536"
LINE = "#34445A"
TEXT = "#F1F5FA"
MUTED = "#A7B7CD"
BLUE = "#79CBFF"
GREEN = "#8FE0AE"
YELLOW = "#FFD483"
RED = "#FF9A95"
RESAMPLE = Image.Resampling.LANCZOS


# Times in this storyboard are video time. The visible clock uses sensor timestamps.
STORY = (
    (0, 7, "intro", 0, 0, ""),
    (7, 9, "blackout", 0, 0, ""),
    (9, 17, "delayed", 0, 0, "At camera return, ORB-SLAM3 publishes corridor A. Ground truth is in B, about 6 m away."),
    (17, 35, "delayed", 0, 42, "ReLocate carries both poses while the stereo views provide similar support for A and B."),
    (35, 41, "delayed", 42, 42, "At 2.10 s, the first distinguishing wall pixels appear at the right edge of the right camera."),
    (41, 47, "delayed", 42, 47, "As the wall marker enters view, B gains enough support for three consecutive frames."),
    (47, 54, "delayed", 47, 47, "ReLocate publishes B at sensor time 2.35 s. On the timed replay, the pose was ready at 6.02 s."),
    (54, 64, "unresolved", 0, 100, "In this incident, both regions explain every stereo pair in the five-second window."),
    (64, 68, "unresolved", 100, 100, "ReLocate withholds a pose at the deadline. ORB-SLAM3's first pose was in the wrong corridor."),
    (68, 76, "summary", 0, 0, ""),
)


@lru_cache
def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(name, size)


def label(draw, x, y, value, size=19, color=TEXT, bold=False, anchor="lt"):
    draw.text((x, y), value, font=font(size, bold), fill=color, anchor=anchor)


def paragraph(draw, x, y, value, width, size=21, color=MUTED):
    line = ""
    for word in value.split():
        candidate = f"{line} {word}".strip()
        if line and draw.textlength(candidate, font=font(size)) > width:
            label(draw, x, y, line, size, color)
            y += size + 9
            line = word
        else:
            line = candidate
    if line:
        label(draw, x, y, line, size, color)


def card(draw, box):
    draw.rounded_rectangle(box, radius=14, fill=PANEL, outline=LINE, width=1)


class Demo:
    def __init__(self, archive):
        replay = json.loads((HERE / "replay.json").read_text())
        expected = json.loads((ROOT / "results/week6-comparison.json").read_text())
        profile = json.loads((ROOT / "results/week7-profile.json").read_text())
        with (HERE / "observations.zip").open("rb") as stream:
            image_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        if image_hash != replay["images_sha256"] or any(
            replay[key] != report[key] for report in (expected, profile) for key in ("manifest_sha256", "map_sha256")
        ):
            raise ValueError("demo bundle does not match the benchmark evidence")
        self.clips = replay["clips"]
        self.archive = archive
        self.baseline = expected["orb_slam3_relocalization"]
        self.proposal = expected["policies"]["multi-temporal"]
        self.service_p95 = next(run["timing"]["service_wall"]["p95_ms"] for run in profile["runs"]
                                if run["run"] == "week7-multi-temporal")

    def image(self, sample, eye):
        return Image.open(BytesIO(self.archive.read(sample["images"][eye]))).convert("RGB")

    def intro(self, canvas, draw, blackout=False):
        sample = self.clips["delayed"]["samples"][0]
        title = "Camera blackout" if blackout else "ReLocate"
        label(draw, 48, 101, title, 50 if blackout else 63, TEXT, True)
        if blackout:
            seconds = self.clips["delayed"]["blackout_ms"] / 1000
            paragraph(draw, 50, 209, f"Both camera views are masked for {seconds:g} seconds. The comparison begins at the first unmasked pair.", 480, 27)
        else:
            label(draw, 50, 192, "Stereo–inertial UAV relocalization", 26, BLUE, True)
            paragraph(draw, 50, 271, "Two corridors 6 m apart can produce identical stereo observations. ReLocate waits for evidence that distinguishes them.", 490, 26)
            label(draw, 50, 468, "Held-out synthetic benchmark", 20, MUTED)
        card(draw, (585, 107, 1231, 594))
        canvas.paste(self.image(sample, "right").resize((620, 413), RESAMPLE)
                     if not blackout else Image.new("RGB", (620, 413), "black"), (598, 153))
        label(draw, 599, 124, "RIGHT CAMERA" if not blackout else "MASKED RIGHT CAMERA", 15, MUTED, True)
        label(draw, 49, 649, "ORB-SLAM3 and ReLocate use the same fixed corridor snapshot", 19, MUTED)

    def camera_panel(self, canvas, draw, clip, sample, index):
        card(draw, (48, 124, 619, 548))
        for x, eye in ((59, "left"), (338, "right")):
            image = self.image(sample, eye)
            label(draw, x, 140, f"{eye.upper()} CAMERA", 14, MUTED, True)
            canvas.paste(image.resize((270, 180), RESAMPLE), (x, 164))
            if eye in sample["cue_boxes"]:
                left, top, right, bottom = sample["cue_boxes"][eye]
                draw.rectangle((x + left * 0.5625 - 2, 164 + top * 0.5625 - 2,
                                x + right * 0.5625 + 1, 164 + bottom * 0.5625 + 2), outline=YELLOW, width=2)
        if "right" in sample["cue_boxes"]:
            left, top, right, bottom = sample["cue_boxes"]["right"]
            origin_x, origin_y = min(315, left), 67
            crop = self.image(sample, "right").crop((origin_x, origin_y, 480, 267))
            scale = 151 / 200
            canvas.paste(crop.resize((round(crop.width * scale), 151), RESAMPLE), (64, 373))
            draw.rectangle((64 + (left - origin_x) * scale, 373 + (top - origin_y) * scale,
                            64 + (right - origin_x) * scale, 373 + (bottom - origin_y) * scale), outline=YELLOW, width=2)
            label(draw, 264, 390, "Right-eye detail", 23, YELLOW, True)
            paragraph(draw, 264, 434, "Yellow marks the pixels that differ from the corresponding view in corridor A.", 318, 18)
        else:
            label(draw, 65, 383, "Stereo observations", 24, TEXT, True)
            message = "The two candidate locations still produce identical views." if clip["cue_ms"] is None else "The distinguishing rear-wall marker is still out of view."
            paragraph(draw, 65, 432, message, 490, 19)
        label(draw, 605, 524, f"FRAME {index:03d}", 13, MUTED, anchor="rt")

    def method_cards(self, draw, clip, sample):
        native = clip["native"]
        card(draw, (648, 124, 927, 271))
        card(draw, (943, 124, 1232, 271))
        label(draw, 662, 140, "ORB-SLAM3", 24, TEXT, True)
        label(draw, 662, 182, f"WRONG POSE · {native['region']}", 18, RED, True)
        label(draw, 662, 218, f"{native['inliers']} pose inliers at {native['time_ms'] / 1000:.2f} s", 16, MUTED)
        label(draw, 662, 241, f"{native['position_error_m']:.2f} m position error", 16, MUTED)
        label(draw, 957, 140, "ReLocate", 24, TEXT, True)
        if sample["published"]:
            status, color = f"CORRECT POSE · {sample['published']}", GREEN
        elif clip["cue_ms"] is None and sample is clip["samples"][-1]:
            status, color = "POSE WITHHELD", YELLOW
        elif sample["streak"]:
            status, color = f"CONFIRMING · {sample['streak']}/3", YELLOW
        else:
            status, color = "WAITING · A AND B", YELLOW
        label(draw, 957, 182, status, 18, color, True)
        scores = "     ".join(f"{region}  {value[0]:g}" if value else f"{region}  —"
                             for region, value in sample["candidates"].items())
        label(draw, 957, 218, scores, 17, BLUE)
        label(draw, 957, 241, "Stereo geometry scores", 15, MUTED)

    def map_panel(self, draw, clip, sample):
        card(draw, (648, 287, 1232, 548))
        label(draw, 664, 302, "MAP CROP · TOP-DOWN", 15, MUTED, True)

        def project(x, z):
            return 824 + 38 * x, 496 - 38 * z

        for x, region in ((0, "A"), (6, "B")):
            upper, lower = project(x - 1.65, 4.0), project(x + 1.65, 0)
            draw.rounded_rectangle((*upper, *lower), radius=9, fill=BACKGROUND, outline=LINE, width=2)
            label(draw, 824 + x * 38, 365, f"Corridor {region}", 17, MUTED, anchor="mt")
        a, b = project(0, 0), project(6, 0)
        draw.line((a[0], a[1], b[0], b[1]), fill=LINE, width=1)
        label(draw, (a[0] + b[0]) / 2, a[1] - 18, "6 m apart", 15, MUTED, anchor="mt")
        x, y = project(*sample["truth_xz"])
        draw.ellipse((x - 14, y - 14, x + 14, y + 14), outline=GREEN, width=3)
        for region, candidate in sample["candidates"].items():
            if candidate:
                x, y = project(*candidate[1:])
                color = GREEN if sample["published"] == region else BLUE
                draw.ellipse((x - 8, y - 8, x + 8, y + 8), outline=color, width=3)
        x, y = project(*clip["native"]["xz"])
        draw.line((x - 5, y - 5, x + 5, y + 5), fill=RED, width=3)
        draw.line((x - 5, y + 5, x + 5, y - 5), fill=RED, width=3)
        label(draw, 666, 520, "×  ORB-SLAM3 first pose", 14, RED)
        label(draw, 865, 520, "○  ReLocate candidates", 14, BLUE)
        label(draw, 1060, 520, "◯  Ground truth", 14, GREEN)

    def experiment(self, canvas, draw, example, index, caption):
        clip = self.clips[example]
        sample = clip["samples"][index]
        title = "A visual cue appears late" if example == "delayed" else "The corridors remain indistinguishable"
        label(draw, 48, 34, title, 34, TEXT, True)
        label(draw, 49, 83, "Shared camera input and fixed map · held-out replay", 19, MUTED)
        label(draw, 1232, 87, f"{sample['time_ms'] / 1000:.2f} s sensor time", 20, TEXT, True, anchor="rt")
        self.camera_panel(canvas, draw, clip, sample, index)
        self.method_cards(draw, clip, sample)
        self.map_panel(draw, clip, sample)

        left, right, y = 49, 1231, 577
        end_ms = clip["samples"][-1]["time_ms"]
        draw.line((left, y, right, y), fill=LINE, width=4)
        current = left + (right - left) * sample["time_ms"] / end_ms
        draw.line((left, y, current, y), fill=BLUE, width=4)
        draw.ellipse((current - 5, y - 5, current + 5, y + 5), fill=BLUE)
        if clip["cue_ms"] is not None:
            cue = left + (right - left) * clip["cue_ms"] / end_ms
            draw.line((cue, y - 8, cue, y + 8), fill=YELLOW, width=3)
        label(draw, left, 593, "0.00 s · image return", 15, MUTED)
        end_label = f"{end_ms / 1000:.2f} s"
        if sample["time_ms"] == end_ms:
            end_label += " · pose confirmed" if sample["published"] else " · no pose at deadline"
        label(draw, right, 593, end_label, 15, MUTED, anchor="rt")
        paragraph(draw, 49, 633, caption, 1180, 23, TEXT)

    def summary(self, draw):
        label(draw, 48, 59, "Nine held-out camera outages", 50, TEXT, True)
        label(draw, 51, 128, "Synthetic corridor benchmark · outcomes scored on sensor timestamps", 23, MUTED)
        card(draw, (50, 191, 1230, 467))
        for x, text in ((633, "CORRECT"), (828, "WRONG"), (1034, "NO POSE IN 5 S")):
            label(draw, x, 218, text, 17, MUTED, True, anchor="mt")
        for y, title, value in ((282, "ORB-SLAM3 first pose", self.baseline), (379, "ReLocate, two hypotheses", self.proposal)):
            label(draw, 78, y + 7, title, 26, TEXT, True)
            for x, key, color in ((633, "correct", GREEN), (828, "false", RED), (1034, "timeout", YELLOW)):
                label(draw, x, y, str(value[key]), 43, color, True, anchor="mt")
        draw.line((74, 353, 1206, 353), fill=LINE)
        label(draw, 52, 506, "Measured processing budget", 24, TEXT, True)
        label(draw, 52, 550, f"{self.service_p95:.1f} ms/frame p95 on i7-13700H · 50 ms available at 20 Hz", 22, YELLOW)
        label(draw, 52, 636, "Code, benchmark and traces", 20, MUTED)
        label(draw, 52, 671, "github.com/David-Trinh-05/relocate-uav-slam", 20, BLUE)

    def render(self, mode, index, caption):
        canvas = Image.new("RGB", SIZE, BACKGROUND)
        draw = ImageDraw.Draw(canvas)
        if mode in ("intro", "blackout"):
            self.intro(canvas, draw, blackout=mode == "blackout")
        elif mode == "summary":
            self.summary(draw)
        else:
            self.experiment(canvas, draw, mode, index, caption)
        return canvas


def write_video(output, demo):
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".relocate-video-") as directory:
        video = Path(directory) / "demo.mp4"
        command = ["ffmpeg", "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
                   "-video_size", f"{SIZE[0]}x{SIZE[1]}", "-framerate", str(FPS), "-i", "pipe:0",
                   "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
                   "-movflags", "+faststart", str(video)]
        with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
            for start, stop, mode, first, last, caption in STORY:
                cached = None
                frames = (stop - start) * FPS
                for frame in range(frames):
                    index = first + round((last - first) * frame / max(1, frames - 1))
                    if cached != index:
                        pixels = demo.render(mode, index, caption).tobytes()
                        cached = index
                    encoder.stdin.write(pixels)
            encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError("FFmpeg failed to encode the demo")
        video.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/relocate-demo.mp4")
    args = parser.parse_args()
    with ZipFile(HERE / "observations.zip") as archive:
        write_video(args.output, Demo(archive))
    print(args.output)


if __name__ == "__main__":
    main()

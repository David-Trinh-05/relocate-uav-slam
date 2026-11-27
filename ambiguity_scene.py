"""Render two translated corridors and test geometric ambiguity with stereo views and PnP."""

import argparse
import json
from pathlib import Path

import cv2 as cv
import numpy as np


WIDTH, HEIGHT = 480, 320
FOCAL = 345.0
BASELINE_M = 0.11
ROOM_OFFSET_M = 6.0
K = np.array([[FOCAL, 0, (WIDTH - 1) / 2], [0, FOCAL, (HEIGHT - 1) / 2], [0, 0, 1]])
BOUNDS = np.array([[-1.65, 1.65], [-1.25, 1.25], [0, 8]])
YAWS = (0, 0, 0, 25, 55, 85, 115, 145, 173)


def make_textures():
    rng = np.random.default_rng(241)
    textures = {}
    for axis in range(3):
        for side in range(2):
            image = np.full((512, 512), 140, dtype=np.uint8)
            for index in range(0, 512, 64):
                cv.line(image, (index, 0), (index, 511), 65, 2)
                cv.line(image, (0, index), (511, index), 65, 2)
            for _ in range(580):
                x, y = rng.integers(0, 512, size=2).tolist()
                size = int(rng.integers(2, 14))
                intensity = int(rng.integers(30, 225))
                cv.rectangle(image, (x, y), (x + size, y + size), intensity, -1)
            textures[axis, side] = image

    back_panels = []
    for room, label in enumerate(("A", "B")):
        panel = textures[2, 0].copy()
        cv.rectangle(panel, (112, 125), (400, 370), 235, -1)
        rng = np.random.default_rng(100 + room)
        for y in range(7):
            for x in range(11):
                if rng.integers(0, 2):
                    cx, cy = 140 + 21 * x, 145 + 19 * y
                    cv.rectangle(panel, (cx, cy), (cx + 19, cy + 17), 25, -1)
        cv.putText(panel, label, (212, 348), cv.FONT_HERSHEY_SIMPLEX, 2.2, 20, 5, cv.LINE_AA)
        # The rear wall's world x axis points left when viewed from inside the corridor.
        back_panels.append(cv.flip(panel, 1))
    return textures, back_panels


def yaw_rotation(degrees):
    sine, cosine = np.sin(np.deg2rad(degrees)), np.cos(np.deg2rad(degrees))
    return np.array([[cosine, 0, sine], [0, 1, 0], [-sine, 0, cosine]])


def render(textures, panels, room, local_origin, rotation, camera_rays):
    directions = camera_rays @ rotation.T
    depth = np.full((HEIGHT, WIDTH), np.inf)
    image = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for axis in range(3):
        uv = [dimension for dimension in range(3) if dimension != axis]
        for side, coord in enumerate(BOUNDS[axis]):
            ray = directions[:, :, axis]
            safe = np.where(np.abs(ray) > 1e-8, ray, np.nan)
            distance = (coord - local_origin[axis]) / safe
            hit = (distance > 0) & (distance < depth)
            # Rays from inside a closed corridor hit the nearest face first.
            u, v = [local_origin[i] + distance[hit] * directions[:, :, i][hit] for i in uv]
            texture = panels[room] if (axis, side) == (2, 0) else textures[axis, side]
            x = np.clip((u - BOUNDS[uv[0], 0]) / np.ptp(BOUNDS[uv[0]]) * 511, 0, 511).astype(int)
            y = np.clip((v - BOUNDS[uv[1], 0]) / np.ptp(BOUNDS[uv[1]]) * 511, 0, 511).astype(int)
            image[hit] = texture[y, x]
            depth[hit] = distance[hit]
    if not np.isfinite(depth).all():
        raise ValueError("scene must enclose every camera ray")
    return image, depth


def pnp_candidate(reference, depth, reference_center, rotation, query, detector, matcher):
    reference_keys, reference_descriptors = detector.detectAndCompute(reference, None)
    query_keys, query_descriptors = query
    if reference_descriptors is None or query_descriptors is None:
        return {"matches": 0, "inliers": 0}
    matches = [match for match in matcher.match(reference_descriptors, query_descriptors) if match.distance <= 40]
    if len(matches) < 6:
        return {"matches": len(matches), "inliers": 0}
    pixels = np.array([reference_keys[match.queryIdx].pt for match in matches])
    ix, iy = pixels.round().astype(int).T
    image_rays = (pixels - K[:2, 2]) / FOCAL
    camera_points = np.column_stack([image_rays, np.ones(len(pixels))]) * depth[iy, ix, None]
    world_points = camera_points @ rotation.T + reference_center
    image_points = np.array([query_keys[match.trainIdx].pt for match in matches])
    cv.setRNGSeed(42)
    success, rvec, tvec, inliers = cv.solvePnPRansac(
        world_points, image_points, K, None, reprojectionError=2, iterationsCount=150,
    )
    result = {"matches": len(matches), "inliers": int(inliers.size) if success else 0}
    if success:
        world_to_camera, _ = cv.Rodrigues(rvec)
        result["estimated_camera_center_m"] = (-world_to_camera.T @ tvec).ravel().round(4).tolist()
    return result


def save_image(path, image):
    if not cv.imwrite(str(path), image):
        raise OSError(f"could not write {path}")


def generate(output):
    output.mkdir(parents=True, exist_ok=False)
    textures, panels = make_textures()
    detector = cv.ORB_create(nfeatures=1400)
    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    y, x = np.indices((HEIGHT, WIDTH))
    rays = np.stack([(x - K[0, 2]) / FOCAL, (y - K[1, 2]) / FOCAL, np.ones_like(x)], axis=-1)
    frames = []
    preview = []

    for frame_index, yaw in enumerate(YAWS):
        rotation = yaw_rotation(yaw)
        center = np.array([0.05, 0.02, 2.0 + 0.10 * frame_index])
        stereo = []
        for room, label in enumerate(("a", "b")):
            views = []
            for eye, disparity in (("left", 0), ("right", BASELINE_M)):
                origin = center + disparity * rotation[:, 0]
                image, _ = render(textures, panels, room, origin, rotation, rays)
                save_image(output / f"{frame_index:02d}-{label}-{eye}.png", image)
                views.append(image)
            stereo.append(views)

        query = detector.detectAndCompute(stereo[0][0], None)
        reference_origin = center + [0, 0, -0.08]
        candidates = []
        for room, label in enumerate(("A", "B")):
            reference, depth = render(textures, panels, room, reference_origin, rotation, rays)
            offset = np.array([ROOM_OFFSET_M * room, 0, 0])
            result = pnp_candidate(reference, depth, reference_origin + offset, rotation, query, detector, matcher)
            candidates.append(dict(room=label, ground_truth_center_m=(center + offset).round(4).tolist(), **result))

        entry = {
            "frame": frame_index,
            "yaw_deg": yaw,
            "different_pixels_left": int(np.count_nonzero(stereo[0][0] != stereo[1][0])),
            "different_pixels_right": int(np.count_nonzero(stereo[0][1] != stereo[1][1])),
            "candidates": candidates,
        }
        frames.append(entry)
        if frame_index in (0, len(YAWS) - 1):
            preview_row = []
            for room, image in zip(("A", "B"), stereo):
                labeled = cv.copyMakeBorder(image[0], 30, 0, 0, 0, cv.BORDER_CONSTANT, value=24)
                cv.putText(labeled, f"Frame {frame_index} | Corridor {room}", (10, 20),
                           cv.FONT_HERSHEY_SIMPLEX, 0.55, 245, 1, cv.LINE_AA)
                preview_row.append(labeled)
            preview.append(np.hstack(preview_row))

    identical_prefix = next((item["frame"] for item in frames if item["different_pixels_left"] or
                             item["different_pixels_right"]), len(frames))
    if identical_prefix < 3 or identical_prefix == len(frames):
        raise ValueError("expected an identical initial stereo sequence and a later visual cue")
    for candidate in frames[0]["candidates"]:
        if candidate["inliers"] < 30 or np.linalg.norm(
            np.array(candidate["estimated_camera_center_m"]) - candidate["ground_truth_center_m"]
        ) > 0.15:
            raise ValueError("both initial place hypotheses must support a geometric pose")

    save_image(output / "preview.png", np.vstack(preview))
    report = {
        "scene": "two translated synthetic corridors",
        "query_room": "A",
        "corridor_separation_m": ROOM_OFFSET_M,
        "stereo_baseline_m": BASELINE_M,
        "identical_stereo_prefix_frames": identical_prefix,
        "frames": frames,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new directory for rendered stereo pairs and evidence")
    args = parser.parse_args()
    generate(args.output)
    print(args.output / "summary.json")


if __name__ == "__main__":
    main()

"""Fixed-map stereo relocalization with one live pose hypothesis per region."""

from dataclasses import dataclass
import json

import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation

from benchmark import sha256


MAX_UNOBSERVED_FRAMES = 8
CONFIRMATION_FRAMES = 3


@dataclass(frozen=True)
class Policy:
    retained_regions: int
    confirmation_frames: int


POLICIES = {
    "single-frame": Policy(retained_regions=1, confirmation_frames=1),
    "single-temporal": Policy(retained_regions=1, confirmation_frames=CONFIRMATION_FRAMES),
    "multi-temporal": Policy(retained_regions=2, confirmation_frames=CONFIRMATION_FRAMES),
}


@dataclass
class Pose:
    position: np.ndarray
    rotation: np.ndarray

    def as_dict(self):
        return {
            "position_m": self.position.tolist(),
            "quaternion_xyzw": Rotation.from_matrix(self.rotation).as_quat().tolist(),
        }


@dataclass
class Keyframe:
    name: str
    pose: Pose
    descriptors: np.ndarray
    optical_points: np.ndarray


@dataclass
class Hypothesis:
    pose: Pose
    unobserved_frames: int = 0


def features(detector, image):
    keys, descriptors = detector.detectAndCompute(image, None)
    return np.array([key.pt for key in keys], dtype=np.float32).reshape(-1, 2), descriptors


def angle_between(first, second):
    return np.rad2deg(Rotation.from_matrix(first.T @ second).magnitude())


def read_gray(path):
    image = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"cannot read {path}")
    return image


class FixedMap:
    def __init__(self, root, map_file, calibration, detector):
        self.intrinsics = np.array(calibration["camera_intrinsics"])
        self.image_size = tuple(calibration["image_size"])
        self.baseline = -calibration["right_from_left_translation_m"][0]
        inverse_intrinsics = np.linalg.inv(self.intrinsics)
        self.regions = {}
        for record in json.loads(self.artifact(root, map_file).read_text())["keyframes"]:
            image = read_gray(self.artifact(root, record["left"]))
            depth = np.load(self.artifact(root, record["left_optical_z_m"]), allow_pickle=False)
            pixels, descriptors = features(detector, image)
            if descriptors is None:
                continue
            x, y = pixels.round().astype(int).T
            rays = np.column_stack((pixels, np.ones(len(pixels)))) @ inverse_intrinsics.T
            pose = Pose(np.array(record["world_position_m"]),
                        Rotation.from_quat(record["world_from_body_xyzw"]).as_matrix())
            entry = Keyframe(record["keyframe_id"], pose, descriptors, rays * depth[y, x, None])
            self.regions.setdefault(record["region"], []).append(entry)
        if len(self.regions) != 2:
            raise ValueError("the corridor benchmark requires two feature-bearing map regions")

    @staticmethod
    def artifact(root, record):
        path = (root / record["path"]).resolve()
        if not path.is_relative_to(root) or sha256(path) != record["sha256"]:
            raise ValueError(f"invalid map artifact {record['path']}")
        return path


@dataclass
class Evidence:
    keyframe: str
    pose: Pose
    matches: int
    inliers: int
    stereo_inliers: int
    coverage_cells: int
    median_reprojection_px: float
    gyro_error_deg: float | None

    @property
    def score(self):
        return self.inliers + 0.5 * self.stereo_inliers

    def log(self):
        return {
            "keyframe": self.keyframe, "pose": self.pose.as_dict(), "matches": self.matches,
            "inliers": self.inliers, "stereo_inliers": self.stereo_inliers,
            "coverage_cells": self.coverage_cells, "score": self.score,
            "median_reprojection_px": self.median_reprojection_px,
            "gyro_error_deg": self.gyro_error_deg,
        }


@dataclass
class SearchResult:
    evidence: Evidence | None
    best_match_count: int
    keyframes_compared: int


class Relocalizer:
    def __init__(self, mapping, detector, policy=POLICIES["multi-temporal"]):
        self.mapping = mapping
        self.detector = detector
        self.policy = policy
        self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        self.hypotheses = {}
        self.pending_region = None
        self.streak = 0

    def stereo_matches(self, pixels, descriptors, right):
        right_pixels, right_descriptors = features(self.detector, right)
        paired = np.full((len(pixels), 2), np.nan)
        if right_descriptors is None:
            return paired
        for match in self.matcher.match(descriptors, right_descriptors):
            left_point, right_point = pixels[match.queryIdx], right_pixels[match.trainIdx]
            if match.distance <= 40 and abs(left_point[1] - right_point[1]) <= 2 and left_point[0] > right_point[0]:
                paired[match.queryIdx] = right_point
        return paired

    def geometry(self, keyframe, matches, pixels, stereo, predicted):
        if len(matches) < 12:
            return None
        object_points = keyframe.optical_points[[match.queryIdx for match in matches]]
        query_indices = [match.trainIdx for match in matches]
        image_points = pixels[query_indices]
        intrinsics = self.mapping.intrinsics
        # Solve in keyframe coordinates so a translated copy gets identical numerical input.
        cv.setRNGSeed(42)
        success, rvec, tvec, inliers = cv.solvePnPRansac(
            object_points, image_points, intrinsics, None, iterationsCount=90, reprojectionError=2.5,
        )
        if not success or inliers is None or len(inliers) < 10:
            return None
        inliers = inliers.ravel()
        key_to_query, _ = cv.Rodrigues(rvec)
        query_to_key = key_to_query.T
        rotation = keyframe.pose.rotation @ query_to_key
        center = keyframe.pose.position - rotation @ tvec.ravel()
        gyro_error = None if predicted is None else float(angle_between(predicted.rotation, rotation))
        if gyro_error is not None and gyro_error > 25:
            return None

        points = object_points[inliers] @ key_to_query.T + tvec.ravel()
        if np.any(points[:, 2] <= 0):
            return None
        projected = points @ intrinsics.T
        projected = projected[:, :2] / projected[:, 2, None]
        observed = image_points[inliers]
        errors = np.linalg.norm(projected - observed, axis=1)
        right_projected = projected.copy()
        right_projected[:, 0] -= intrinsics[0, 0] * self.mapping.baseline / points[:, 2]
        right_errors = np.linalg.norm(right_projected - stereo[np.array(query_indices)[inliers]], axis=1)
        cells = np.floor(observed / self.mapping.image_size * [4, 3]).astype(int)
        return Evidence(
            keyframe.name, Pose(center, rotation), len(matches), len(inliers), int(np.sum(right_errors <= 3)),
            len(np.unique(cells, axis=0)), float(np.median(errors)), gyro_error,
        )

    def region_evidence(self, references, pixels, descriptors, stereo, predicted):
        def candidates(keyframes):
            ranked = []
            for keyframe in keyframes:
                matches = [match for match in self.matcher.match(keyframe.descriptors, descriptors) if match.distance <= 40]
                ranked.append((keyframe, matches))
            ranked.sort(key=lambda item: len(item[1]), reverse=True)
            poses = [self.geometry(keyframe, matches, pixels, stereo, predicted) for keyframe, matches in ranked[:4]]
            best = max((pose for pose in poses if pose is not None), key=lambda pose: pose.score, default=None)
            return SearchResult(best, len(ranked[0][1]), len(ranked))

        if predicted is not None:
            shortlist = sorted(references, key=lambda keyframe: (
                angle_between(predicted.rotation, keyframe.pose.rotation)
                + 5 * np.linalg.norm(keyframe.pose.position - predicted.position)
            ))[:6]
            result = candidates(shortlist)
            if result.evidence is not None:
                return result
            fallback = candidates(references)
            fallback.keyframes_compared += result.keyframes_compared
            return fallback
        return candidates(references)

    def update(self, left, right, gyro_step):
        pixels, descriptors = features(self.detector, left)
        stereo = self.stereo_matches(pixels, descriptors, right) if descriptors is not None else None
        evidence = {}
        search_log = {}
        for region, references in self.mapping.regions.items():
            track = self.hypotheses.get(region)
            predicted = None
            if track is not None:
                predicted = Pose(track.pose.position, track.pose.rotation @ gyro_step)
            search = SearchResult(None, 0, 0) if descriptors is None else self.region_evidence(
                references, pixels, descriptors, stereo, predicted,
            )
            candidate = search.evidence
            evidence[region] = candidate
            if descriptors is None:
                status = "no_query_features"
            elif candidate is not None:
                status = "geometry_verified"
            elif search.best_match_count < 12:
                status = "insufficient_descriptor_matches"
            else:
                status = "geometry_rejected"
            search_log[region] = {
                "status": status, "best_descriptor_matches": search.best_match_count,
                "keyframes_compared": search.keyframes_compared,
            }
            if candidate is not None and candidate.inliers >= 20 and candidate.coverage_cells >= 3:
                self.hypotheses[region] = Hypothesis(candidate.pose)
            elif track is not None:
                misses = track.unobserved_frames + 1
                if misses < MAX_UNOBSERVED_FRAMES:
                    self.hypotheses[region] = Hypothesis(predicted, misses)
                else:
                    del self.hypotheses[region]

        ranked = sorted(evidence, key=lambda region: evidence[region].score if evidence[region] else 0, reverse=True)
        # A discarded region can still challenge on the next frame without a motion prior.
        live = [region for region in ranked if region in self.hypotheses]
        for region in live[self.policy.retained_regions:]:
            del self.hypotheses[region]
        leader, alternative = ranked
        best, other = evidence[leader], evidence[alternative]
        runner_score = other.score if other is not None else 0
        ready = best is not None and best.inliers >= 30 and best.stereo_inliers >= 10 and best.coverage_cells >= 4
        distinct = ready and best.score - runner_score >= 24 and best.score >= 1.3 * runner_score
        if distinct:
            self.streak = self.streak + 1 if self.pending_region == leader else 1
            self.pending_region = leader
        else:
            self.pending_region = None
            self.streak = 0
        accepted = leader if self.streak >= self.policy.confirmation_frames else None
        reason = "accepted" if accepted else "confirming" if distinct else "ambiguous" if ready else "insufficient_geometry"
        return accepted, {
            "reason": reason, "confirmation_streak": self.streak,
            "candidates": {region: pose.log() if pose else None for region, pose in evidence.items()},
            "active_hypotheses": sorted(self.hypotheses),
            "region_search": search_log,
        }

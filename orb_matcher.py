"""Optional native backend with the same ORB matches as OpenCV cross-check."""

import ctypes
from pathlib import Path

import cv2 as cv
import numpy as np


LIBRARY = Path(__file__).resolve().parent / "runs/native-matcher/liborb_match.so"


class NativeMatcher:
    def __init__(self, library):
        self.library = ctypes.CDLL(str(library))
        self.match_orb = self.library.relocate_match_orb32
        descriptors = np.ctypeslib.ndpointer(dtype=np.uint8, ndim=2, flags="C_CONTIGUOUS")
        output = np.ctypeslib.ndpointer(dtype=np.intc, ndim=2, flags="C_CONTIGUOUS")
        self.match_orb.argtypes = [descriptors, ctypes.c_int, descriptors, ctypes.c_int, ctypes.c_int, output]
        self.match_orb.restype = ctypes.c_int

    def match(self, left, right, max_distance=256):
        if left.dtype != np.uint8 or right.dtype != np.uint8 or left.shape[1:] != (32,) or right.shape[1:] != (32,):
            raise ValueError("expected arrays of 32-byte ORB descriptors")
        if not len(left) or not len(right):
            return []
        output = np.empty((min(len(left), len(right)), 3), dtype=np.intc)
        count = self.match_orb(np.ascontiguousarray(left), len(left), np.ascontiguousarray(right), len(right),
                               max_distance, output)
        return [cv.DMatch(int(query), int(train), float(distance)) for query, train, distance in output[:count]]


def make_matcher(mode):
    if mode == "native" or (mode == "auto" and LIBRARY.is_file()):
        if not LIBRARY.is_file():
            raise OSError("native matcher library missing, build it with ./build-native-matcher.sh")
        return NativeMatcher(LIBRARY)
    return cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)


def bounded_matches(matcher, left, right, max_distance):
    if isinstance(matcher, NativeMatcher):
        return matcher.match(left, right, max_distance)
    return [match for match in matcher.match(left, right) if match.distance <= max_distance]

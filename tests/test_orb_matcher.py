import unittest

import cv2 as cv
import numpy as np

from orb_matcher import LIBRARY, make_matcher


@unittest.skipUnless(LIBRARY.is_file(), "build the native library with ./build-native-matcher.sh")
class NativeMatcherTest(unittest.TestCase):
    def setUp(self):
        cv.setNumThreads(1)
        self.reference = make_matcher("opencv")
        self.native = make_matcher("native")

    def assert_same_matches(self, left, right):
        def triples(matches):
            return [(match.queryIdx, match.trainIdx, match.distance) for match in matches]

        expected = self.reference.match(left, right)
        self.assertEqual(triples(self.native.match(left, right)), triples(expected))
        self.assertEqual(triples(self.native.match(left, right, 40)),
                         triples(match for match in expected if match.distance <= 40))

    def test_random_descriptors_and_mixed_sizes(self):
        rng = np.random.default_rng(1729)
        for left_size, right_size in ((1, 1), (80, 43), (900, 900)):
            left = rng.integers(0, 256, (left_size, 32), dtype=np.uint8)
            right = rng.integers(0, 256, (right_size, 32), dtype=np.uint8)
            with self.subTest(left_size=left_size, right_size=right_size):
                self.assert_same_matches(left, right)

    def test_equal_distances_and_noncontiguous_inputs(self):
        left = np.zeros((5, 64), dtype=np.uint8)[:, ::2]
        right = left[::-1]
        self.assert_same_matches(left, right)

    def test_empty_descriptor_set(self):
        empty = np.empty((0, 32), dtype=np.uint8)
        single = np.zeros((1, 32), dtype=np.uint8)
        self.assertEqual(self.native.match(single, empty), [])
        self.assertEqual(self.native.match(empty, single), [])

    def test_exact_limit(self):
        left = np.zeros((1, 32), dtype=np.uint8)
        right = left.copy()
        right[0, :5] = 255
        self.assertEqual(len(self.native.match(left, right, 40)), 1)
        self.assertEqual(self.native.match(left, right, 39), [])


if __name__ == "__main__":
    unittest.main()

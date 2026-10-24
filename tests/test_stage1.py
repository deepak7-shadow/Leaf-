"""Unit tests for Stage 1 fast candidate detector and vegetation indices."""

import unittest
import numpy as np
import cv2

from core.vegetation_indices import compute_exg, compute_exg_mask, compute_normalized_rgb
from core.stage1_detector import Stage1CandidateDetector
from utils.synthetic_generator import SyntheticSceneGenerator


class TestStage1Detector(unittest.TestCase):
    """Test suite for Stage 1 vegetation candidate detection."""

    def setUp(self):
        self.detector = Stage1CandidateDetector()
        self.generator = SyntheticSceneGenerator(width=640, height=480)

    def test_vegetation_indices_on_green_patch(self):
        """Verify ExG is strongly positive on green pixels and negative on red/blue."""
        # Pure Green BGR: (0, 255, 0)
        green_img = np.zeros((50, 50, 3), dtype=np.uint8)
        green_img[:, :, 1] = 255
        exg_green = compute_exg(green_img)
        self.assertTrue(np.all(exg_green > 1.5), "Pure green must have high ExG")

        # Pure Red BGR: (0, 0, 255)
        red_img = np.zeros((50, 50, 3), dtype=np.uint8)
        red_img[:, :, 2] = 255
        exg_red = compute_exg(red_img)
        self.assertTrue(np.all(exg_red < 0.0), "Pure red must have negative ExG")

        # Pure Blue BGR: (255, 0, 0)
        blue_img = np.zeros((50, 50, 3), dtype=np.uint8)
        blue_img[:, :, 0] = 255
        exg_blue = compute_exg(blue_img)
        self.assertTrue(np.all(exg_blue < 0.0), "Pure blue must have negative ExG")

    def test_early_exit_on_soil_background(self):
        """Verify zero-inference early exit when no vegetation is present."""
        soil_frame = self.generator.generate_soil_background()
        # Warmup call
        self.detector.detect(soil_frame)
        candidates, mask, latency_ms = self.detector.detect(soil_frame)

        self.assertEqual(len(candidates), 0, "Soil background should produce 0 candidates")
        self.assertLess(latency_ms, 20.0, "Stage 1 early exit should take < 20 ms")



    def test_candidate_detection_on_synthetic_leaf(self):
        """Verify Stage 1 correctly isolates a synthetic leaf ROI."""
        frame, gt = self.generator.generate_scene(include_real=True, include_fake=False)
        candidates, mask, latency_ms = self.detector.detect(frame)

        self.assertGreaterEqual(len(candidates), 1, "Must detect at least one candidate ROI")
        cand = candidates[0]
        x, y, w, h = cand.bbox
        self.assertGreater(w, 20)
        self.assertGreater(h, 20)
        self.assertGreater(cand.area, 400.0)
        self.assertGreaterEqual(cand.solidity, 0.40)


if __name__ == "__main__":
    unittest.main()

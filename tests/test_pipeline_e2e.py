"""End-to-end integration tests for AI Nozzle pipeline."""

import unittest
import numpy as np

from core.pipeline import AINozzlePipeline, PipelineResult
from utils.synthetic_generator import SyntheticSceneGenerator


class TestPipelineE2E(unittest.TestCase):
    """End-to-end integration test suite."""

    def setUp(self):
        self.pipeline = AINozzlePipeline()
        self.generator = SyntheticSceneGenerator(width=640, height=480)

    def tearDown(self):
        self.pipeline.close()

    def test_pipeline_on_empty_soil(self):
        """Empty background should skip Stage 2 and leave nozzle idle."""
        soil = self.generator.generate_soil_background()
        # Warmup frame to avoid cold-start memory allocation distortion
        self.pipeline.process_frame(soil)
        result: PipelineResult = self.pipeline.process_frame(soil)

        self.assertEqual(result.candidates_count, 0)
        self.assertEqual(result.stage2_ms, 0.0, "Stage 2 should be skipped (0 ms) on empty frame")
        self.assertFalse(result.is_spraying)
        self.assertFalse(result.spray_triggered_this_frame)
        self.assertLess(result.total_ms, 25.0)


    def test_pipeline_on_scene_with_candidates(self):
        """Scene with leaves should trigger Stage 1 and Stage 2 classification."""
        scene, _ = self.generator.generate_scene(include_real=True, include_fake=True)
        result: PipelineResult = self.pipeline.process_frame(scene)

        self.assertGreaterEqual(result.candidates_count, 1)
        self.assertGreater(result.stage2_ms, 0.0, "Stage 2 must execute on candidate ROIs")
        self.assertEqual(len(result.detections), result.candidates_count)

        for det in result.detections:
            self.assertIn(det["class_name"], ["REAL_LIVING_LEAF", "FAKE_PRINTED_ARTIFICIAL"])
            self.assertGreater(det["inference_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()

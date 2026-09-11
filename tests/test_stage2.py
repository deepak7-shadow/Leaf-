"""Unit tests for Stage 2 MobileNetV3-Small ONNX classifier."""

import unittest
import numpy as np

from core.stage2_classifier import Stage2Classifier, ClassificationResult


class TestStage2Classifier(unittest.TestCase):
    """Test suite for Stage 2 MobileNetV3-Small classifier."""

    def setUp(self):
        self.classifier = Stage2Classifier()

    def test_model_session_initialized(self):
        """Verify ONNX model session is loaded."""
        self.assertIsNotNone(self.classifier.session, "ONNX session must be loaded")

    def test_single_roi_classification(self):
        """Verify inference returns valid class, confidence, and latency."""
        # Simulated leaf crop (224x224x3)
        sample_patch = np.random.randint(30, 180, (224, 224, 3), dtype=np.uint8)
        result = self.classifier.classify_roi(sample_patch)

        self.assertIsInstance(result, ClassificationResult)
        self.assertIn(result.class_id, [0, 1])
        self.assertIn(result.class_name, ["REAL_LIVING_LEAF", "FAKE_PRINTED_ARTIFICIAL"])
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
        self.assertGreater(result.inference_ms, 0.0)
        # Verify latency is fast (typically < 15 ms on CPU)
        self.assertLess(result.inference_ms, 30.0)

    def test_batched_classification(self):
        """Verify batched inference handles multiple ROIs efficiently."""
        patches = [
            np.random.randint(20, 150, (180, 180, 3), dtype=np.uint8)
            for _ in range(3)
        ]
        results = self.classifier.classify_batch(patches)

        self.assertEqual(len(results), 3)
        for res in results:
            self.assertIn(res.class_id, [0, 1])
            self.assertGreaterEqual(res.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()

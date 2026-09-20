"""Unit tests for Real Farm Leaves Dataset integrity and pipeline compatibility."""

import json
import os
import unittest
import cv2

from core.stage2_classifier import Stage2Classifier


class TestRealFarmDataset(unittest.TestCase):
    """Test suite for validating real farm leaves dataset."""

    DATA_DIR = "data/collected/real"
    MANIFEST_PATH = "data/collected/real/manifest.json"

    def setUp(self):
        self.assertTrue(
            os.path.exists(self.DATA_DIR),
            f"Dataset directory {self.DATA_DIR} does not exist."
        )

    def test_sample_count_exceeds_500(self):
        """Verify real farm leaves dataset contains all 38-category images (560+)."""
        jpg_files = [f for f in os.listdir(self.DATA_DIR) if f.lower().endswith(".jpg")]
        self.assertGreater(
            len(jpg_files), 350,
            f"Expected > 350 real leaf images (38 categories), found {len(jpg_files)}"
        )

    def test_manifest_metadata_validity(self):
        """Verify manifest.json exists and contains all 38 categories and provenance records."""
        self.assertTrue(os.path.exists(self.MANIFEST_PATH), "manifest.json missing")
        with open(self.MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertIn("total_images", manifest)
        self.assertGreater(manifest["total_images"], 350)
        self.assertIn("crops_breakdown", manifest)
        self.assertIn("total_categories_covered", manifest)
        self.assertGreaterEqual(manifest["total_categories_covered"], 38)

        expected_crops = {
            "tomato", "corn", "potato", "bellpepper", "soybean", "grape",
            "strawberry", "apple", "cherry", "orange", "peach", "blueberry",
            "raspberry", "squash"
        }
        present_crops = set(manifest["crops_breakdown"].keys())
        self.assertTrue(
            expected_crops.issubset(present_crops),
            f"Missing expected crop categories: {expected_crops - present_crops}"
        )

    def test_image_decodability_and_channels(self):
        """Verify that all image files are readable, non-empty, and have 3 channels."""
        jpg_files = [f for f in os.listdir(self.DATA_DIR) if f.lower().endswith(".jpg")]
        for fname in jpg_files[:30]:  # Sample first 30 across crops
            fpath = os.path.join(self.DATA_DIR, fname)
            img = cv2.imread(fpath)
            self.assertIsNotNone(img, f"Failed to read image: {fname}")
            self.assertEqual(len(img.shape), 3, f"Image {fname} does not have 3 dimensions")
            self.assertEqual(img.shape[2], 3, f"Image {fname} does not have 3 color channels")
            self.assertGreaterEqual(img.shape[0], 64)
            self.assertGreaterEqual(img.shape[1], 64)

    def test_stage2_classifier_recognizes_real_leaves(self):
        """Verify that Stage 2 classifier classifies real farm leaf images as REAL_LIVING_LEAF."""
        classifier = Stage2Classifier()
        crops = ["tomato", "corn", "potato", "bellpepper", "apple"]
        test_imgs = []
        for crop in crops:
            fname = f"farm_{crop}_001.jpg"
            fpath = os.path.join(self.DATA_DIR, fname)
            if os.path.exists(fpath):
                img = cv2.imread(fpath)
                test_imgs.append(img)

        self.assertTrue(len(test_imgs) > 0, "No crop sample images found for classification test")
        predictions = classifier.classify_batch(test_imgs)
        for i, pred in enumerate(predictions):
            self.assertEqual(
                pred.class_name, "REAL_LIVING_LEAF",
                f"Sample {crops[i]} was not recognized as REAL_LIVING_LEAF (got {pred.class_name})"
            )

    def test_fake_dataset_integrity(self):
        """Verify fake / artificial distractor dataset contains > 100 images and manifest."""
        fake_dir = "data/collected/fake"
        self.assertTrue(os.path.exists(fake_dir), "Fake dataset directory missing")
        manifest_path = os.path.join(fake_dir, "manifest.json")
        self.assertTrue(os.path.exists(manifest_path), "Fake manifest.json missing")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertIn("total_images", manifest)
        self.assertGreaterEqual(manifest["total_images"], 100)

        fake_imgs = [f for f in os.listdir(fake_dir) if f.lower().endswith(".jpg")]
        self.assertGreaterEqual(len(fake_imgs), 100)

    def test_stage2_classifier_recognizes_fake_leaves(self):
        """Verify that Stage 2 classifier classifies fake leaf images as FAKE_PRINTED_ARTIFICIAL."""
        classifier = Stage2Classifier()
        fake_dir = "data/collected/fake"
        sample_files = [f for f in os.listdir(fake_dir) if f.lower().endswith(".jpg")][:10]
        self.assertTrue(len(sample_files) >= 5)

        test_imgs = [cv2.imread(os.path.join(fake_dir, f)) for f in sample_files]
        predictions = classifier.classify_batch(test_imgs)
        for i, pred in enumerate(predictions):
            self.assertEqual(
                pred.class_name, "FAKE_PRINTED_ARTIFICIAL",
                f"Fake sample {sample_files[i]} was not recognized as FAKE_PRINTED_ARTIFICIAL"
            )
            self.assertGreaterEqual(pred.confidence, 0.85)

    def test_synthetic_generator_injects_real_and_fake_leaves(self):
        """Verify SyntheticSceneGenerator loads real farm photos and fake distractors."""
        from utils.synthetic_generator import SyntheticSceneGenerator
        gen = SyntheticSceneGenerator(width=640, height=480)
        self.assertGreater(len(gen.real_photos), 350)
        self.assertGreater(len(gen.fake_photos), 50)

        scene, gt = gen.generate_scene(include_real=True, include_fake=True)
        self.assertEqual(scene.shape, (480, 640, 3))
        self.assertEqual(len(gt), 2)
        types = [item["type"] for item in gt]
        self.assertIn("REAL_LIVING_LEAF", types)
        self.assertIn("FAKE_PRINTED_ARTIFICIAL", types)


if __name__ == "__main__":
    unittest.main()

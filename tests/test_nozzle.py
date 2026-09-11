"""Unit tests for Nozzle Controller logic, safety, and debounce timing."""

import time
import unittest

from core.nozzle_controller import NozzleController, SprayCommand


class TestNozzleController(unittest.TestCase):
    """Test suite for Nozzle actuation decisions and hardware drivers."""

    def setUp(self):
        self.controller = NozzleController({
            "nozzle_controller": {
                "mode": "simulation",
                "spray_pulse_ms": 100,
                "cooldown_ms": 200,
                "safety_lockout": False
            },
            "stage2_classifier": {
                "confidence_threshold": 0.80
            }
        })

    def test_spray_on_high_confidence_real_leaf(self):
        """Verify spray triggers for REAL_LIVING_LEAF above confidence threshold."""
        cmd: SprayCommand = self.controller.evaluate_and_actuate(
            predicted_class="REAL_LIVING_LEAF",
            confidence=0.92,
            bbox=(100, 100, 80, 80)
        )
        self.assertTrue(cmd.should_spray)
        self.assertTrue(self.controller.is_spraying)

    def test_reject_fake_printed_leaf(self):
        """Verify spray inhibits for FAKE_PRINTED_ARTIFICIAL regardless of high confidence."""
        cmd: SprayCommand = self.controller.evaluate_and_actuate(
            predicted_class="FAKE_PRINTED_ARTIFICIAL",
            confidence=0.98,
            bbox=(200, 200, 60, 60)
        )
        self.assertFalse(cmd.should_spray)
        self.assertIn("Rejected non-living", cmd.reason)

    def test_reject_low_confidence(self):
        """Verify spray inhibits when confidence is below cutoff."""
        cmd: SprayCommand = self.controller.evaluate_and_actuate(
            predicted_class="REAL_LIVING_LEAF",
            confidence=0.65,
            bbox=(150, 150, 70, 70)
        )
        self.assertFalse(cmd.should_spray)
        self.assertIn("below threshold", cmd.reason)

    def test_cooldown_debounce_lockout(self):
        """Verify that immediate consecutive spray triggers are debounced."""
        # 1. Trigger spray
        cmd1 = self.controller.evaluate_and_actuate(
            predicted_class="REAL_LIVING_LEAF",
            confidence=0.95,
            bbox=(100, 100, 50, 50)
        )
        self.assertTrue(cmd1.should_spray)

        # Let pulse finish
        time.sleep(0.12)
        self.controller.update()
        self.assertFalse(self.controller.is_spraying)

        # 2. Attempt immediate second spray (within 200ms cooldown)
        cmd2 = self.controller.evaluate_and_actuate(
            predicted_class="REAL_LIVING_LEAF",
            confidence=0.95,
            bbox=(100, 100, 50, 50)
        )
        self.assertFalse(cmd2.should_spray)
        self.assertIn("Cooldown active", cmd2.reason)

    def test_safety_lockout(self):
        """Verify master safety lockout overrides all triggers."""
        self.controller.toggle_safety()
        self.assertTrue(self.controller.safety_lockout)

        cmd = self.controller.evaluate_and_actuate(
            predicted_class="REAL_LIVING_LEAF",
            confidence=0.99,
            bbox=(100, 100, 50, 50)
        )
        self.assertFalse(cmd.should_spray)
        self.assertIn("Safety lockout", cmd.reason)


if __name__ == "__main__":
    unittest.main()

"""Nozzle Controller module for automated precision pesticide spraying.

Controls solenoid spray valves based on classification results, confidence thresholds,
pulse timing, debounce intervals, and hardware interfaces.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import logging
import time

logger = logging.getLogger("NozzleController")


@dataclass
class SprayCommand:
    """Decision output for the spray actuation system."""
    should_spray: bool
    reason: str
    target_class: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    nozzle_index: int = 0
    pulse_duration_ms: float = 0.0


class BaseNozzleDriver(ABC):
    """Abstract driver interface for physical or simulated spray valves."""

    @abstractmethod
    def set_nozzle_state(self, nozzle_index: int, active: bool) -> None:
        """Set physical valve state."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Release hardware resources."""
        pass


class SimulatedNozzleDriver(BaseNozzleDriver):
    """Simulated valve driver for development, testing, and GUI demonstration."""

    def __init__(self):
        self.state = False
        self.trigger_count = 0
        self.last_actuation_time = 0.0

    def set_nozzle_state(self, nozzle_index: int, active: bool) -> None:
        if active and not self.state:
            self.trigger_count += 1
            self.last_actuation_time = time.time()
            logger.info(f"[SIMULATED NOZZLE] Solenoid valve {nozzle_index} OPEN (Actuation #{self.trigger_count})")
        elif not active and self.state:
            logger.debug(f"[SIMULATED NOZZLE] Solenoid valve {nozzle_index} CLOSED")
        self.state = active

    def close(self) -> None:
        self.state = False


class SerialNozzleDriver(BaseNozzleDriver):
    """Serial driver for Arduino/ESP32 relay microcontrollers."""

    def __init__(self, port: str = "COM3", baud_rate: int = 115200):
        self.port = port
        self.baud_rate = baud_rate
        self.serial_conn = None
        try:
            import serial
            self.serial_conn = serial.Serial(self.port, self.baud_rate, timeout=0.1)
            logger.info(f"Connected to spray microcontroller on {self.port} at {self.baud_rate} baud.")
        except Exception as e:
            logger.warning(f"Failed to open serial port {self.port}: {e}. Falling back to simulation mode.")

    def set_nozzle_state(self, nozzle_index: int, active: bool) -> None:
        if self.serial_conn and self.serial_conn.is_open:
            cmd = f"VALVE:{nozzle_index}:{'ON' if active else 'OFF'}\n".encode()
            try:
                self.serial_conn.write(cmd)
            except Exception as e:
                logger.error(f"Serial write error: {e}")

    def close(self) -> None:
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()


class NozzleController:
    """High-level pesticide spray controller managing safety, cooldown, and timing."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        n_cfg = cfg.get("nozzle_controller", {})
        
        self.mode = n_cfg.get("mode", "simulation").lower()
        self.pulse_duration_s = float(n_cfg.get("spray_pulse_ms", 120)) / 1000.0
        self.cooldown_s = float(n_cfg.get("cooldown_ms", 250)) / 1000.0
        self.confidence_threshold = float(
            cfg.get("stage2_classifier", {}).get("confidence_threshold", 0.80)
        )
        self.safety_lockout = bool(n_cfg.get("safety_lockout", False))

        # Hardware Driver instantiation
        if self.mode == "serial":
            port = n_cfg.get("serial_port", "COM3")
            baud = int(n_cfg.get("baud_rate", 115200))
            self.driver: BaseNozzleDriver = SerialNozzleDriver(port, baud)
        else:
            self.driver = SimulatedNozzleDriver()

        # State tracking
        self.is_spraying = False
        self.spray_start_time = 0.0
        self.last_spray_completion_time = 0.0
        self.total_sprays = 0

    def evaluate_and_actuate(
        self,
        predicted_class: str,
        confidence: float,
        bbox: Tuple[int, int, int, int],
        frame_width: int = 640
    ) -> SprayCommand:
        """Evaluate classification results and determine whether to actuate spray.
        
        Args:
            predicted_class: Output from Stage 2 ("REAL_LIVING_LEAF" or "FAKE_PRINTED_ARTIFICIAL").
            confidence: Confidence probability in range [0.0, 1.0].
            bbox: (x, y, w, h) bounding box of candidate leaf.
            frame_width: Frame width for multi-nozzle spatial assignment.
            
        Returns:
            SprayCommand object detailing the action taken.
        """
        now = time.perf_counter()

        # Check if currently active spray pulse has expired
        if self.is_spraying:
            if (now - self.spray_start_time) >= self.pulse_duration_s:
                self.is_spraying = False
                self.driver.set_nozzle_state(0, False)
                self.last_spray_completion_time = now

        # Master Safety Override Check
        if self.safety_lockout:
            return SprayCommand(
                should_spray=False,
                reason="Safety lockout active",
                target_class=predicted_class,
                confidence=confidence,
                bbox=bbox
            )

        # Debounce / Cooldown Check
        if (now - self.last_spray_completion_time) < self.cooldown_s:
            return SprayCommand(
                should_spray=False,
                reason=f"Cooldown active ({int((self.cooldown_s - (now - self.last_spray_completion_time)) * 1000)}ms remaining)",
                target_class=predicted_class,
                confidence=confidence,
                bbox=bbox
            )

        # Target Class Verification: Only spray REAL_LIVING_LEAF
        if predicted_class != "REAL_LIVING_LEAF":
            return SprayCommand(
                should_spray=False,
                reason=f"Rejected non-living/fake target: {predicted_class}",
                target_class=predicted_class,
                confidence=confidence,
                bbox=bbox
            )

        # Confidence Threshold Verification
        if confidence < self.confidence_threshold:
            return SprayCommand(
                should_spray=False,
                reason=f"Confidence {confidence*100:.1f}% below threshold {self.confidence_threshold*100:.1f}%",
                target_class=predicted_class,
                confidence=confidence,
                bbox=bbox
            )

        # Spatial nozzle indexing (e.g. 3-zone boom: Left, Center, Right)
        x_center = bbox[0] + (bbox[2] / 2)
        if x_center < frame_width / 3:
            nozzle_idx = 0  # Left
        elif x_center < 2 * frame_width / 3:
            nozzle_idx = 1  # Center
        else:
            nozzle_idx = 2  # Right

        # ACTUATE SPRAY!
        self.is_spraying = True
        self.spray_start_time = now
        self.total_sprays += 1
        self.driver.set_nozzle_state(nozzle_idx, True)

        return SprayCommand(
            should_spray=True,
            reason="Confirmed REAL_LIVING_LEAF with high confidence",
            target_class=predicted_class,
            confidence=confidence,
            bbox=bbox,
            nozzle_index=nozzle_idx,
            pulse_duration_ms=self.pulse_duration_s * 1000.0
        )

    def update(self) -> None:
        """Periodic update to shut off valve if spray pulse duration elapsed."""
        if self.is_spraying:
            now = time.perf_counter()
            if (now - self.spray_start_time) >= self.pulse_duration_s:
                self.is_spraying = False
                self.driver.set_nozzle_state(0, False)
                self.last_spray_completion_time = now

    def toggle_safety(self) -> bool:
        """Toggle master safety lockout switch."""
        self.safety_lockout = not self.safety_lockout
        if self.safety_lockout and self.is_spraying:
            self.is_spraying = False
            self.driver.set_nozzle_state(0, False)
        return self.safety_lockout

    def close(self) -> None:
        """Shutdown driver and close valves."""
        if self.is_spraying:
            self.driver.set_nozzle_state(0, False)
        self.driver.close()

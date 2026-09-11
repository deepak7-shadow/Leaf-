"""End-to-End AI Nozzle Computer Vision Pipeline.

Orchestrates:
  Camera Frame -> Stage 1 (Fast OpenCV Candidate Detector)
               -> Zero-Inference Early Exit if no vegetation
               -> Stage 2 (MobileNetV3-Small ONNX ROI Classifier)
               -> Nozzle Actuation Controller (Spray / Don't Spray)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import time
import cv2
import numpy as np

from core.stage1_detector import CandidateROI, Stage1CandidateDetector
from core.stage2_classifier import ClassificationResult, Stage2Classifier
from core.nozzle_controller import NozzleController, SprayCommand


@dataclass
class PipelineResult:
    """Full telemetry and decision packet for a single video frame."""
    frame: np.ndarray
    detections: List[Dict[str, Any]]
    vegetation_mask: np.ndarray
    stage1_ms: float
    stage2_ms: float
    total_ms: float
    fps: float
    candidates_count: int
    is_spraying: bool
    spray_triggered_this_frame: bool
    cooldown_remaining_ms: float
    safety_locked: bool


class AINozzlePipeline:
    """Integrated two-stage computer vision pipeline for precision pesticide spraying."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize pipeline with components.
        
        Args:
            config: Complete configuration dictionary.
        """
        self.config = config or {}
        self.stage1 = Stage1CandidateDetector(self.config)
        self.stage2 = Stage2Classifier(self.config)
        self.nozzle = NozzleController(self.config)

        # Telemetry & FPS tracking
        self.last_frame_time = time.perf_counter()
        self.fps_smoothed = 0.0
        self.fps_alpha = 0.1  # Exponential moving average factor

    def process_frame(self, frame: np.ndarray) -> PipelineResult:
        """Process a single camera frame through the two-stage pipeline.
        
        Args:
            frame: Input BGR frame.
            
        Returns:
            PipelineResult containing detections, telemetry, and nozzle actions.
        """
        t_start = time.perf_counter()

        # Step 1: Stage 1 Fast Candidate Detection
        candidates, mask, s1_ms = self.stage1.detect(frame)
        s2_ms = 0.0
        detections: List[Dict[str, Any]] = []
        spray_triggered = False

        frame_h, frame_w = frame.shape[:2]

        # Step 2: Zero-Inference Early Exit Check
        if not candidates:
            # No vegetation in frame! Skip CNN inference completely.
            self.nozzle.update()
        else:
            # Step 3: Stage 2 Deep Learning ROI Classification
            # Crop candidate ROIs and classify only the target regions
            roi_images = [cand.roi_image for cand in candidates]
            s2_start = time.perf_counter()
            class_results: List[ClassificationResult] = self.stage2.classify_batch(roi_images)
            s2_ms = (time.perf_counter() - s2_start) * 1000.0

            # Step 4: Evaluate Spray Actuation for each candidate
            for cand, res in zip(candidates, class_results):
                spray_cmd: SprayCommand = self.nozzle.evaluate_and_actuate(
                    predicted_class=res.class_name,
                    confidence=res.confidence,
                    bbox=cand.bbox,
                    frame_width=frame_w
                )

                if spray_cmd.should_spray:
                    spray_triggered = True

                detections.append({
                    "bbox": cand.bbox,
                    "class_id": res.class_id,
                    "class_name": res.class_name,
                    "confidence": res.confidence,
                    "probabilities": res.probabilities,
                    "inference_ms": res.inference_ms,
                    "should_spray": spray_cmd.should_spray,
                    "spray_reason": spray_cmd.reason,
                    "nozzle_index": spray_cmd.nozzle_index,
                    "area": cand.area,
                    "solidity": cand.solidity
                })

        # Calculate Total Pipeline Latency & FPS
        t_end = time.perf_counter()
        total_ms = (t_end - t_start) * 1000.0

        instant_fps = 1.0 / max(1e-5, (t_end - self.last_frame_time))
        self.last_frame_time = t_end
        if self.fps_smoothed == 0.0:
            self.fps_smoothed = instant_fps
        else:
            self.fps_smoothed = (self.fps_alpha * instant_fps) + ((1.0 - self.fps_alpha) * self.fps_smoothed)

        # Calculate remaining cooldown time
        now = time.perf_counter()
        elapsed_since_spray = now - self.nozzle.last_spray_completion_time
        remaining_cd_ms = max(0.0, (self.nozzle.cooldown_s - elapsed_since_spray) * 1000.0)

        return PipelineResult(
            frame=frame,
            detections=detections,
            vegetation_mask=mask,
            stage1_ms=s1_ms,
            stage2_ms=s2_ms,
            total_ms=total_ms,
            fps=self.fps_smoothed,
            candidates_count=len(candidates),
            is_spraying=self.nozzle.is_spraying,
            spray_triggered_this_frame=spray_triggered,
            cooldown_remaining_ms=remaining_cd_ms,
            safety_locked=self.nozzle.safety_lockout
        )

    def toggle_safety(self) -> bool:
        """Toggle master spray lockout."""
        return self.nozzle.toggle_safety()

    def close(self) -> None:
        """Release nozzle hardware and resources."""
        self.nozzle.close()

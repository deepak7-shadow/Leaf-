"""Stage 2: Real vs Fake Leaf Classification using MobileNetV3-Small ONNX Runtime.

Receives cropped candidate ROIs from Stage 1, resizes to 224x224, normalizes,
and executes inference with millisecond latency logging.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import os
import time
import cv2
import numpy as np
import onnxruntime as ort


@dataclass
class ClassificationResult:
    """Result of Stage 2 MobileNetV3-Small classification for a single ROI."""
    class_id: int                     # 0 or 1
    class_name: str                   # "REAL_LIVING_LEAF" or "FAKE_PRINTED_ARTIFICIAL"
    confidence: float                 # Softmax probability (0.0 to 1.0)
    probabilities: List[float]        # [p_real, p_fake]
    inference_ms: float               # Execution time in milliseconds


class Stage2Classifier:
    """Lightweight MobileNetV3-Small classifier powered by ONNX Runtime."""

    CLASS_MAPPING = {
        0: "REAL_LIVING_LEAF",
        1: "FAKE_PRINTED_ARTIFICIAL"
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize MobileNetV3-Small ONNX classifier.
        
        Args:
            config: Configuration dictionary with model path, input size, and thresholds.
        """
        cfg = config or {}
        st2_cfg = cfg.get("stage2_classifier", {})

        self.model_path = st2_cfg.get("model_path", "models/mobilenet_v3_small.onnx")
        self.input_size = int(st2_cfg.get("input_size", 224))
        
        # ImageNet normalization standards (RGB)
        self.mean = np.array(st2_cfg.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32).reshape((1, 1, 3))
        self.std = np.array(st2_cfg.get("std", [0.229, 0.224, 0.225]), dtype=np.float32).reshape((1, 1, 3))
        
        self.confidence_threshold = float(st2_cfg.get("confidence_threshold", 0.80))
        num_threads = int(st2_cfg.get("intra_op_num_threads", 2))

        # Session options for edge CPU optimization
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session: Optional[ort.InferenceSession] = None
        self.input_name: str = ""
        self.output_name: str = ""

        if os.path.exists(self.model_path):
            self.load_model(self.model_path, opts)

    def load_model(self, model_path: str, options: Optional[ort.SessionOptions] = None) -> None:
        """Load ONNX model into runtime session."""
        self.session = ort.InferenceSession(
            model_path,
            sess_options=options or ort.SessionOptions(),
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, roi_bgr: np.ndarray) -> np.ndarray:
        """Preprocess cropped ROI for MobileNetV3-Small input.
        
        Args:
            roi_bgr: Cropped candidate image patch in BGR format.
            
        Returns:
            Preprocessed tensor shaped (1, 3, 224, 224) as float32.
        """
        # Resize to input dimensions (224x224)
        resized = cv2.resize(roi_bgr, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        # BGR -> RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # Normalize (RGB - mean) / std
        normalized = (rgb - self.mean) / self.std
        # Transpose HWC (224, 224, 3) to CHW (3, 224, 224)
        chw = np.transpose(normalized, (2, 0, 1))
        # Add batch dimension -> (1, 3, 224, 224)
        return np.expand_dims(chw, axis=0).astype(np.float32)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Compute numerically stable softmax probabilities."""
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    def classify_roi(self, roi_bgr: np.ndarray) -> ClassificationResult:
        """Perform classification on a single cropped candidate ROI.
        
        Args:
            roi_bgr: Cropped candidate image patch.
            
        Returns:
            ClassificationResult containing class, confidence score, and latency.
        """
        t0 = time.perf_counter()

        if self.session is None:
            # Fallback heuristic if ONNX model is missing/uninitialized
            return self._heuristic_classify(roi_bgr, t0)

        # Preprocess input ROI
        input_tensor = self.preprocess(roi_bgr)

        # Run ONNX inference
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        logits = outputs[0][0]  # Shape [2]

        probs = self._softmax(logits)
        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return ClassificationResult(
            class_id=class_id,
            class_name=self.CLASS_MAPPING.get(class_id, "UNKNOWN"),
            confidence=confidence,
            probabilities=[float(p) for p in probs],
            inference_ms=latency_ms
        )

    def classify_batch(self, roi_list: List[np.ndarray]) -> List[ClassificationResult]:
        """Perform batched inference on multiple candidate ROIs for optimal throughput.
        
        Args:
            roi_list: List of cropped BGR candidate images.
            
        Returns:
            List of ClassificationResult objects.
        """
        if not roi_list:
            return []

        if self.session is None:
            return [self._heuristic_classify(roi, time.perf_counter()) for roi in roi_list]

        t0 = time.perf_counter()
        tensors = [self.preprocess(roi)[0] for roi in roi_list]
        batch_tensor = np.stack(tensors, axis=0).astype(np.float32)

        outputs = self.session.run([self.output_name], {self.input_name: batch_tensor})
        logits = outputs[0]  # Shape [N, 2]
        probs = self._softmax(logits)

        total_latency_ms = (time.perf_counter() - t0) * 1000.0
        avg_ms_per_roi = total_latency_ms / len(roi_list)

        results = []
        for p in probs:
            cid = int(np.argmax(p))
            results.append(ClassificationResult(
                class_id=cid,
                class_name=self.CLASS_MAPPING.get(cid, "UNKNOWN"),
                confidence=float(p[cid]),
                probabilities=[float(x) for x in p],
                inference_ms=avg_ms_per_roi
            ))
        return results

    def _heuristic_classify(self, roi_bgr: np.ndarray, start_time: float) -> ClassificationResult:
        """Texture-based baseline classifier used if neural weights are not yet trained."""
        # Analyze chromatic variance and gradient sharpness (real leaves have organic vein gradients)
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Check for unnatural flat paper background around borders
        h, w = gray.shape
        border_mask = np.ones((h, w), dtype=bool)
        border_mask[int(h * 0.15):int(h * 0.85), int(w * 0.15):int(w * 0.85)] = False
        border_std = np.std(gray[border_mask]) if np.any(border_mask) else 0.0

        # Printed paper typically has high white/uniform border and lower micro-texture variance
        is_real = (laplacian_var > 60.0 and border_std < 40.0)
        cid = 0 if is_real else 1
        conf = 0.88 if is_real else 0.82

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        probs = [conf, 1.0 - conf] if cid == 0 else [1.0 - conf, conf]

        return ClassificationResult(
            class_id=cid,
            class_name=self.CLASS_MAPPING.get(cid, "UNKNOWN"),
            confidence=conf,
            probabilities=probs,
            inference_ms=latency_ms
        )

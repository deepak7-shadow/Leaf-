"""Stage 1: Fast Candidate Detection using OpenCV, HSV, ExG, and Contour Geometry.

Designed for ultra-low latency (<3 ms) execution to screen out non-vegetation background
before triggering costly neural-network inference.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import time
import cv2
import numpy as np

from core.vegetation_indices import compute_exg_mask


@dataclass
class CandidateROI:
    """Represents a potential leaf region isolated by Stage 1."""
    roi_image: np.ndarray             # Cropped BGR candidate image
    bbox: Tuple[int, int, int, int]   # (x, y, w, h) in processing frame coordinates
    contour: np.ndarray               # Filtered contour points
    area: float                       # Contour area (px)
    aspect_ratio: float               # Width / Height
    solidity: float                   # Area / ConvexHullArea
    extent: float                     # Area / (w * h)


class Stage1CandidateDetector:
    """Stage 1 fast candidate detector combining HSV green thresholding, ExG, and geometry."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Stage 1 detector parameters.
        
        Args:
            config: Optional configuration dictionary. Defaults to standard settings.
        """
        cfg = config or {}
        st1_cfg = cfg.get("stage1_candidate_detector", {})
        
        # Processing Resolution
        sys_cfg = cfg.get("system", {})
        proc_res = sys_cfg.get("processing_resolution", {})
        self.target_width = proc_res.get("width", 640)
        self.target_height = proc_res.get("height", 480)

        # HSV Thresholds
        hsv_cfg = st1_cfg.get("hsv", {})
        self.h_min = int(hsv_cfg.get("hue_min", 25))
        self.h_max = int(hsv_cfg.get("hue_max", 88))
        self.s_min = int(hsv_cfg.get("sat_min", 35))
        self.s_max = int(hsv_cfg.get("sat_max", 255))
        self.v_min = int(hsv_cfg.get("val_min", 30))
        self.v_max = int(hsv_cfg.get("val_max", 255))

        # Vegetation Index (ExG)
        veg_cfg = st1_cfg.get("vegetation_index", {})
        self.exg_enabled = bool(veg_cfg.get("enabled", True))
        self.exg_threshold = float(veg_cfg.get("exg_threshold", 0.05))
        self.use_otsu = bool(veg_cfg.get("use_otsu", False))

        # Morphology Kernels
        morph_cfg = st1_cfg.get("morphology", {})
        open_k = int(morph_cfg.get("open_kernel_size", 3))
        close_k = int(morph_cfg.get("close_kernel_size", 7))
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))

        # Contour Geometry Filters
        filt_cfg = st1_cfg.get("contour_filters", {})
        self.min_area = float(filt_cfg.get("min_area", 450.0))
        self.max_area = float(filt_cfg.get("max_area", 180000.0))
        self.min_aspect_ratio = float(filt_cfg.get("min_aspect_ratio", 0.18))
        self.max_aspect_ratio = float(filt_cfg.get("max_aspect_ratio", 5.50))
        self.min_solidity = float(filt_cfg.get("min_solidity", 0.40))
        self.min_extent = float(filt_cfg.get("min_extent", 0.15))
        self.margin_pct = float(filt_cfg.get("margin_expansion_pct", 0.08))

    def detect(self, frame: np.ndarray) -> Tuple[List[CandidateROI], np.ndarray, float]:
        """Execute Stage 1 candidate detection on a frame.
        
        Args:
            frame: Input BGR image from camera or video stream.
            
        Returns:
            Tuple of:
              - List of CandidateROI objects (empty if no candidates)
              - Final cleaned binary vegetation mask (for visualization)
              - Execution latency in milliseconds
        """
        t0 = time.perf_counter()

        # Step 1: Resize to processing resolution if needed
        h, w = frame.shape[:2]
        if w != self.target_width or h != self.target_height:
            proc_frame = cv2.resize(frame, (self.target_width, self.target_height), interpolation=cv2.INTER_LINEAR)
        else:
            proc_frame = frame

        # Step 2: HSV Green Mask
        hsv = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([self.h_min, self.s_min, self.v_min], dtype=np.uint8)
        upper_green = np.array([self.h_max, self.s_max, self.v_max], dtype=np.uint8)
        mask_hsv = cv2.inRange(hsv, lower_green, upper_green)

        # Fast Rejection: If green pixel count is below minimal leaf area, skip all remaining ops
        if cv2.countNonZero(mask_hsv) < self.min_area:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return [], mask_hsv, latency_ms

        # Step 3: Excess Green (ExG) Mask
        if self.exg_enabled:
            mask_exg = compute_exg_mask(proc_frame, threshold=self.exg_threshold, use_otsu=self.use_otsu)
            # Step 4: Combine Masks (bitwise AND to retain confident vegetation regions)
            combined_mask = cv2.bitwise_and(mask_hsv, mask_exg)
        else:
            combined_mask = mask_hsv

        # Step 5: Morphological Filtering (Remove salt-and-pepper noise, fill venation gaps)
        if self.kernel_open is not None:
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, self.kernel_open)
        if self.kernel_close is not None:
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, self.kernel_close)

        # Step 6: Find External Contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates: List[CandidateROI] = []
        frame_h, frame_w = proc_frame.shape[:2]

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter 1: Area bounds
            if area < self.min_area or area > self.max_area:
                continue

            x, y, w_box, h_box = cv2.boundingRect(cnt)
            if w_box <= 0 or h_box <= 0:
                continue

            # Filter 2: Aspect Ratio
            aspect_ratio = float(w_box) / float(h_box)
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                continue

            # Filter 3: Solidity (Contour Area / Convex Hull Area)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0.0
            if solidity < self.min_solidity:
                continue

            # Filter 4: Extent (Contour Area / Bounding Box Area)
            rect_area = float(w_box * h_box)
            extent = float(area) / rect_area if rect_area > 0 else 0.0
            if extent < self.min_extent:
                continue

            # Expand bounding box slightly for CNN context margin
            pad_x = int(w_box * self.margin_pct)
            pad_y = int(h_box * self.margin_pct)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(frame_w, x + w_box + pad_x)
            y2 = min(frame_h, y + h_box + pad_y)

            crop_w = x2 - x1
            crop_h = y2 - y1
            if crop_w < 10 or crop_h < 10:
                continue

            roi_patch = proc_frame[y1:y2, x1:x2].copy()

            candidates.append(CandidateROI(
                roi_image=roi_patch,
                bbox=(x1, y1, crop_w, crop_h),
                contour=cnt,
                area=area,
                aspect_ratio=aspect_ratio,
                solidity=solidity,
                extent=extent
            ))

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return candidates, combined_mask, latency_ms

"""Vegetation index calculation module optimized for real-time agricultural vision.

Implements high-speed vectorized index calculations:
- Excess Green Index (ExG): 2g - r - b
- Normalized Green-Red Difference Index (NGRDI): (g - r) / (g + r)
"""

from typing import Tuple
import cv2
import numpy as np


def compute_normalized_rgb(bgr_image: np.ndarray, epsilon: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute normalized chromaticity coordinates (r, g, b).
    
    Args:
        bgr_image: Input image in OpenCV BGR format (H, W, 3) as uint8.
        epsilon: Small constant to avoid zero-division.
        
    Returns:
        Tuple of (r, g, b) normalized float32 planes.
    """
    b = bgr_image[:, :, 0].astype(np.float32)
    g = bgr_image[:, :, 1].astype(np.float32)
    r = bgr_image[:, :, 2].astype(np.float32)

    total = r + g + b + epsilon
    return r / total, g / total, b / total


def compute_exg(bgr_image: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Compute normalized Excess Green Index (ExG = 2g - r - b).
    
    ExG highlights chlorophyll-rich vegetation against soil, plastic,
    and inorganic backgrounds under varied daylight conditions.
    
    Args:
        bgr_image: BGR image (H, W, 3) uint8.
        epsilon: Numerical stability term.
        
    Returns:
        ExG 2D float32 array in range approx [-2.0, 2.0].
    """
    r, g, b = compute_normalized_rgb(bgr_image, epsilon=epsilon)
    return (2.0 * g) - r - b


def compute_ngrdi(bgr_image: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Compute Normalized Green-Red Difference Index (NGRDI = (G - R) / (G + R)).
    
    Args:
        bgr_image: BGR image (H, W, 3) uint8.
        epsilon: Numerical stability term.
        
    Returns:
        NGRDI 2D float32 array in range [-1.0, 1.0].
    """
    g = bgr_image[:, :, 1].astype(np.float32)
    r = bgr_image[:, :, 2].astype(np.float32)
    denom = g + r + epsilon
    return (g - r) / denom


def compute_exg_mask(
    bgr_image: np.ndarray,
    threshold: float = 0.05,
    use_otsu: bool = False
) -> np.ndarray:
    """Compute binary vegetation mask using ExG.
    
    Args:
        bgr_image: BGR image (H, W, 3) uint8.
        threshold: Normalized ExG cutoff (default 0.05).
        use_otsu: If True, uses Otsu's thresholding dynamically.
        
    Returns:
        Binary mask (H, W) uint8 where 255 represents green vegetation.
    """
    exg = compute_exg(bgr_image)
    
    if use_otsu:
        # Scale ExG [-1.0, 1.0] to [0, 255] uint8 for Otsu
        clipped = np.clip(exg, -0.5, 1.0)
        norm_uint8 = ((clipped + 0.5) / 1.5 * 255.0).astype(np.uint8)
        _, mask = cv2.threshold(norm_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        mask = np.where(exg > threshold, 255, 0).astype(np.uint8)
        
    return mask

"""Real-time Heads-Up Display (HUD) and visual analytics for AI Nozzle.

Renders bounding boxes, classification badges, millisecond latency metrics,
solenoid actuation indicators, and picture-in-picture vegetation masks.
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np

# Aesthetic Color Palette (BGR)
COLOR_REAL_LEAF = (46, 204, 113)     # Vibrant Emerald Green
COLOR_FAKE_LEAF = (60, 60, 231)      # Crimson Red
COLOR_WARNING   = (0, 165, 255)      # Amber / Orange
COLOR_TEXT_BG   = (20, 20, 20)       # Dark charcoal backdrop
COLOR_TEXT_FG   = (255, 255, 255)    # Crisp White
COLOR_SPRAY_ON  = (0, 255, 127)      # Spring Green Neon
COLOR_SPRAY_OFF = (100, 100, 100)    # Muted Gray
COLOR_COOLDOWN  = (235, 150, 50)     # Sky Blue / Cyan


def draw_hud(
    frame: np.ndarray,
    detections: List[dict],
    stage1_ms: float,
    stage2_ms: float,
    fps: float,
    is_spraying: bool,
    cooldown_remaining_ms: float = 0.0,
    safety_locked: bool = False,
    pip_mask: Optional[np.ndarray] = None,
    show_pip: bool = True
) -> np.ndarray:
    """Render comprehensive HUD overlay on video frame.
    
    Args:
        frame: BGR image from camera or stream (H, W, 3).
        detections: List of detection result dictionaries containing:
            - bbox: (x, y, w, h)
            - class_name: "REAL_LIVING_LEAF" | "FAKE_PRINTED_ARTIFICIAL"
            - confidence: float [0, 1]
            - inference_ms: float
            - should_spray: bool
        stage1_ms: Execution time of Stage 1 in ms.
        stage2_ms: Cumulative execution time of Stage 2 in ms.
        fps: Current pipeline frames per second.
        is_spraying: Whether the spray nozzle is currently actuated.
        cooldown_remaining_ms: Milliseconds of debounce remaining.
        safety_locked: Master safety status.
        pip_mask: Binary mask from Stage 1 for picture-in-picture view.
        show_pip: Whether to draw the PiP mask.
        
    Returns:
        Frame with HUD elements rendered.
    """
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    # 1. Top Telemetry Ribbon
    ribbon_height = 42
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, ribbon_height), (15, 15, 18), -1)
    cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0, canvas)
    cv2.line(canvas, (0, ribbon_height), (w, ribbon_height), (60, 60, 65), 1)

    # Telemetry Text: FPS and Latency
    total_ms = stage1_ms + stage2_ms
    telemetry_str = f"FPS: {fps:4.1f} | S1: {stage1_ms:3.1f}ms | S2: {stage2_ms:3.1f}ms | Total: {total_ms:3.1f}ms"
    cv2.putText(
        canvas, telemetry_str, (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_TEXT_FG, 1, cv2.LINE_AA
    )

    # 2. Nozzle Actuation Status Badge
    badge_w = 175
    badge_x = w - badge_w - 12
    badge_h = 28
    badge_y = 7


    if safety_locked:
        badge_color = (0, 0, 200)
        badge_text = "[SAFETY LOCKED]"
    elif is_spraying:
        badge_color = COLOR_SPRAY_ON
        badge_text = ">>> SPRAYING <<<"
    elif cooldown_remaining_ms > 0:
        badge_color = COLOR_COOLDOWN
        badge_text = f"COOLDOWN ({int(cooldown_remaining_ms)}ms)"
    else:
        badge_color = COLOR_SPRAY_OFF
        badge_text = "NOZZLE: READY"

    cv2.rectangle(
        canvas, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h),
        badge_color, -1, cv2.LINE_AA
    )
    text_color = (0, 0, 0) if (is_spraying or cooldown_remaining_ms > 0) else (255, 255, 255)
    cv2.putText(
        canvas, badge_text, (badge_x + 12, badge_y + 19),
        cv2.FONT_HERSHEY_SIMPLEX, 0.50, text_color, 2, cv2.LINE_AA
    )

    # 3. Draw Bounding Boxes and Candidate Badges
    for det in detections:
        bbox = det.get("bbox", (0, 0, 0, 0))
        x, y, bw, bh = bbox
        cls_name = det.get("class_name", "UNKNOWN")
        conf = det.get("confidence", 0.0)
        inf_ms = det.get("inference_ms", 0.0)
        spray = det.get("should_spray", False)

        is_real = (cls_name == "REAL_LIVING_LEAF")
        box_color = COLOR_REAL_LEAF if is_real else COLOR_FAKE_LEAF

        # Draw smooth rounded-corner style box
        cv2.rectangle(canvas, (x, y), (x + bw, y + bh), box_color, 2, cv2.LINE_AA)

        # Subtle corner brackets for high-tech aesthetic
        corner_len = min(15, bw // 4, bh // 4)
        if corner_len > 3:
            # Top-left
            cv2.line(canvas, (x, y), (x + corner_len, y), box_color, 4)
            cv2.line(canvas, (x, y), (x, y + corner_len), box_color, 4)
            # Top-right
            cv2.line(canvas, (x + bw, y), (x + bw - corner_len, y), box_color, 4)
            cv2.line(canvas, (x + bw, y), (x + bw, y + corner_len), box_color, 4)
            # Bottom-left
            cv2.line(canvas, (x, y + bh), (x + corner_len, y + bh), box_color, 4)
            cv2.line(canvas, (x, y + bh), (x, y + bh - corner_len), box_color, 4)
            # Bottom-right
            cv2.line(canvas, (x + bw, y + bh), (x + bw - corner_len, y + bh), box_color, 4)
            cv2.line(canvas, (x + bw, y + bh), (x + bw, y + bh - corner_len), box_color, 4)

        # Label Pill: Multi-line badge above or below ROI
        line1 = f"{cls_name} {'[SPRAY]' if spray else '[NO SPRAY]'}"
        line2 = f"Conf: {conf * 100.0:4.1f}% | Inf: {inf_ms:3.1f} ms"

        label_h = 36
        label_w = max(230, bw)
        label_y = y - label_h - 6 if (y - label_h - 6) > ribbon_height else y + bh + 6

        # Backdrop
        cv2.rectangle(
            canvas, (x, label_y), (x + label_w, label_y + label_h),
            COLOR_TEXT_BG, -1
        )
        # Accent left border
        cv2.rectangle(
            canvas, (x, label_y), (x + 4, label_y + label_h),
            box_color, -1
        )

        cv2.putText(
            canvas, line1, (x + 8, label_y + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, box_color, 1, cv2.LINE_AA
        )
        cv2.putText(
            canvas, line2, (x + 8, label_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA
        )

    # 4. Picture-in-Picture (PiP) Vegetation Mask View
    if show_pip and pip_mask is not None:
        pip_w = int(w * 0.22)
        pip_h = int(pip_w * (h / w))
        pip_x = w - pip_w - 12
        pip_y = h - pip_h - 12

        # Convert 1-channel mask to 3-channel green tint
        mask_small = cv2.resize(pip_mask, (pip_w, pip_h), interpolation=cv2.INTER_NEAREST)
        pip_bgr = np.zeros((pip_h, pip_w, 3), dtype=np.uint8)
        pip_bgr[:, :, 1] = mask_small  # Green channel

        # Border and header
        cv2.rectangle(canvas, (pip_x - 2, pip_y - 18), (pip_x + pip_w + 2, pip_y + pip_h + 2), (40, 40, 45), -1)
        cv2.rectangle(canvas, (pip_x, pip_y), (pip_x + pip_w, pip_y + pip_h), (80, 80, 85), 1)
        canvas[pip_y:pip_y + pip_h, pip_x:pip_x + pip_w] = pip_bgr

        cv2.putText(
            canvas, "STAGE 1 MASK (HSV+ExG)", (pip_x + 4, pip_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1, cv2.LINE_AA
        )

    return canvas

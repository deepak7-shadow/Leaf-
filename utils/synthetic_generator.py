"""Synthetic sample and frame generator for AI Nozzle validation and testing.

Generates realistic simulated scenes containing:
- Natural living leaves with venation and organic gradients
- Printed leaves on paper, t-shirts, and posters with halftone / moire artifacts
- Plastic/artificial leaves with specular glare
- Non-leaf distractors (soil, stones, dry debris)
"""

from typing import List, Tuple
import cv2
import numpy as np


class SyntheticSceneGenerator:
    """Generates synthetic camera frames containing real and fake leaf candidates."""

    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height

    def generate_soil_background(self) -> np.ndarray:
        """Create agricultural field background with soil and mulch textures."""
        # Base earthy soil brown (BGR)
        bg = np.full((self.height, self.width, 3), [35, 45, 60], dtype=np.uint8)
        # Add random soil speckles and noise
        noise = np.random.randint(-15, 15, (self.height, self.width, 3), dtype=np.int16)
        bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        # Add soft lighting gradient (sunlight from top-left)
        x_grad = np.linspace(1.1, 0.9, self.width, dtype=np.float32)
        y_grad = np.linspace(1.1, 0.9, self.height, dtype=np.float32)
        light_map = np.outer(y_grad, x_grad)
        for c in range(3):
            bg[:, :, c] = np.clip(bg[:, :, c] * light_map, 0, 255).astype(np.uint8)
        return bg

    def draw_real_leaf(
        self,
        canvas: np.ndarray,
        center: Tuple[int, int],
        axes: Tuple[int, int],
        angle: float
    ) -> Tuple[int, int, int, int]:
        """Draw a simulated living natural leaf with chlorophyll gradient and venation."""
        cx, cy = center
        ax, ay = axes
        
        # Leaf contour (ovate / lanceolate polygon)
        pts = []
        num_pts = 36
        for i in range(num_pts):
            theta = 2.0 * np.pi * i / num_pts
            # Modulate radius to form leaf taper at tip and petiole
            r_scale = 1.0 - 0.3 * np.cos(theta)
            px = ax * np.cos(theta) * r_scale
            py = ay * np.sin(theta)
            # Rotate by angle
            rad = np.radians(angle)
            rx = px * np.cos(rad) - py * np.sin(rad) + cx
            ry = px * np.sin(rad) + py * np.cos(rad) + cy
            pts.append([int(rx), int(ry)])

        pts_arr = np.array([pts], dtype=np.int32)

        # 1. Fill leaf body with rich organic chlorophyll green (BGR: ~35, 155, 65)
        leaf_color = (38, 160, 68)
        cv2.fillPoly(canvas, pts_arr, leaf_color)

        # 2. Add organic color shading (darker green margin, lighter interior)
        hull = cv2.convexHull(pts_arr)
        x, y, w, h = cv2.boundingRect(hull)
        
        # 3. Draw central midrib and lateral veins
        rad = np.radians(angle)
        tip_x = int(cx + (ax * 1.2) * np.cos(rad))
        tip_y = int(cy + (ax * 1.2) * np.sin(rad))
        base_x = int(cx - (ax * 0.8) * np.cos(rad))
        base_y = int(cy - (ax * 0.8) * np.sin(rad))
        
        vein_color = (65, 205, 105)  # Light yellow-green vein
        cv2.line(canvas, (base_x, base_y), (tip_x, tip_y), vein_color, 2, cv2.LINE_AA)

        # Secondary side veins
        for t in np.linspace(0.2, 0.8, 5):
            vx = int(base_x + t * (tip_x - base_x))
            vy = int(base_y + t * (tip_y - base_y))
            # Perpendicular vectors
            nx = int(-np.sin(rad) * (ay * 0.5))
            ny = int(np.cos(rad) * (ay * 0.5))
            cv2.line(canvas, (vx, vy), (vx + nx, vy + ny), vein_color, 1, cv2.LINE_AA)
            cv2.line(canvas, (vx, vy), (vx - nx, vy - ny), vein_color, 1, cv2.LINE_AA)

        # Subtle organic surface noise
        mask = np.zeros((canvas.shape[0], canvas.shape[1]), dtype=np.uint8)
        cv2.fillPoly(mask, pts_arr, 255)
        leaf_noise = np.random.normal(0, 6, canvas.shape).astype(np.int16)
        canvas_int = canvas.astype(np.int16)
        np.copyto(canvas_int, canvas_int + leaf_noise, where=(mask[:, :, None] == 255))
        canvas[:] = np.clip(canvas_int, 0, 255).astype(np.uint8)

        return (x, y, w, h)

    def draw_fake_printed_leaf(
        self,
        canvas: np.ndarray,
        center: Tuple[int, int],
        axes: Tuple[int, int],
        substrate: str = "paper"
    ) -> Tuple[int, int, int, int]:
        """Draw a fake/printed leaf with halftone artifacts or flat paper substrate."""
        cx, cy = center
        ax, ay = axes
        
        # Substrate background (e.g. white/gray paper rectangle or poster border)
        pad = int(max(ax, ay) * 0.4)
        x1 = max(0, cx - ax - pad)
        y1 = max(0, cy - ay - pad)
        x2 = min(canvas.shape[1], cx + ax + pad)
        y2 = min(canvas.shape[0], cy + ay + pad)

        if substrate == "paper":
            # Paper color: off-white
            paper_color = (230, 235, 238)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), paper_color, -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (180, 185, 190), 1)  # Paper edge
        elif substrate == "fabric":
            # Blue fabric / shirt background
            fabric_color = (180, 100, 60)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), fabric_color, -1)

        # Draw printed leaf shape
        fake_pts = []
        for i in range(24):
            theta = 2.0 * np.pi * i / 24
            px = int(cx + ax * np.cos(theta))
            py = int(cy + ay * np.sin(theta))
            fake_pts.append([px, py])
        fake_arr = np.array([fake_pts], dtype=np.int32)

        # Printed ink color: unnatural flat oversaturated green
        ink_color = (25, 210, 50)
        cv2.fillPoly(canvas, fake_arr, ink_color)

        # Halftone / CMYK raster dots simulation
        mask = np.zeros((canvas.shape[0], canvas.shape[1]), dtype=np.uint8)
        cv2.fillPoly(mask, fake_arr, 255)
        
        # High-frequency grid artifact (printer raster / screen pixel grid)
        for gy in range(y1, y2, 4):
            for gx in range(x1, x2, 4):
                if mask[gy, gx] == 255:
                    canvas[gy, gx] = [15, 120, 25]  # dot pattern

        return (x1, y1, x2 - x1, y2 - y1)

    def generate_scene(
        self,
        include_real: bool = True,
        include_fake: bool = True
    ) -> Tuple[np.ndarray, List[dict]]:
        """Generate a complete frame with specified targets."""
        frame = self.generate_soil_background()
        ground_truth = []

        if include_real:
            # Place real leaf in left or center zone
            cx = int(self.width * 0.32)
            cy = int(self.height * 0.50)
            bbox = self.draw_real_leaf(frame, (cx, cy), (75, 45), angle=-25)
            ground_truth.append({
                "type": "REAL_LIVING_LEAF",
                "bbox": bbox
            })

        if include_fake:
            # Place fake printed leaf in right zone
            cx = int(self.width * 0.72)
            cy = int(self.height * 0.52)
            bbox = self.draw_fake_printed_leaf(frame, (cx, cy), (65, 40), substrate="paper")
            ground_truth.append({
                "type": "FAKE_PRINTED_ARTIFICIAL",
                "bbox": bbox
            })

        return frame, ground_truth

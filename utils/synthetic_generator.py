"""Synthetic sample and frame generator for AI Nozzle validation and testing.

Generates realistic simulated scenes containing:
- Natural living leaves with rich chlorophyll gradients, multi-layer venation,
  surface micro-texture, and organic irregular edges
- Printed leaves on paper/fabric substrates with halftone CMYK raster dots,
  flat ink color, moiré lines, and a clearly visible paper/cloth border
- Non-leaf distractors (soil, stones, dry debris)
"""

import os
from typing import List, Optional, Tuple
import cv2
import numpy as np


class SyntheticSceneGenerator:
    """Generates synthetic camera frames containing real and fake leaf candidates.

    Supports blending authentic farm leaf photographs from data/collected/real/
    and artificial/printed leaf images from data/collected/fake/ onto simulated
    agricultural soil backgrounds, with graceful fallback to procedural generation.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        real_dir: str = "data/collected/real",
        fake_dir: str = "data/collected/fake",
        use_real_photos: bool = True,
        use_fake_photos: bool = True
    ):
        self.width = width
        self.height = height
        self.real_dir = real_dir
        self.fake_dir = fake_dir
        self.use_real_photos = use_real_photos
        self.use_fake_photos = use_fake_photos

        # Discover real farm leaf photos
        self.real_photos = []
        if os.path.isdir(self.real_dir):
            self.real_photos = [
                os.path.join(self.real_dir, f)
                for f in os.listdir(self.real_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

        # Discover fake / artificial leaf photos
        self.fake_photos = []
        if os.path.isdir(self.fake_dir):
            self.fake_photos = [
                os.path.join(self.fake_dir, f)
                for f in os.listdir(self.fake_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        if not self.fake_photos and os.path.isdir("data/synthetic/fake"):
            self.fake_photos = [
                os.path.join("data/synthetic/fake", f)
                for f in os.listdir("data/synthetic/fake")
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

    # ------------------------------------------------------------------
    # Background helpers
    # ------------------------------------------------------------------

    def generate_soil_background(self) -> np.ndarray:
        """Create agricultural field background with rich soil and mulch textures."""
        bg = np.full((self.height, self.width, 3), [32, 42, 58], dtype=np.uint8)

        # Coarse soil clumps
        for _ in range(120):
            rx = np.random.randint(0, self.width)
            ry = np.random.randint(0, self.height)
            rr = np.random.randint(2, 10)
            rc = np.random.randint(25, 80)
            cv2.circle(bg, (rx, ry), rr, (rc, rc - 5, rc - 10), -1)

        # Fine grain noise
        noise = np.random.randint(-18, 18, (self.height, self.width, 3), dtype=np.int16)
        bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Sun lighting gradient (top-left brighter)
        x_grad = np.linspace(1.12, 0.88, self.width, dtype=np.float32)
        y_grad = np.linspace(1.12, 0.88, self.height, dtype=np.float32)
        light_map = np.outer(y_grad, x_grad)
        for c in range(3):
            bg[:, :, c] = np.clip(bg[:, :, c] * light_map, 0, 255).astype(np.uint8)

        # Scatter a few small pebbles
        for _ in range(8):
            px = np.random.randint(20, self.width - 20)
            py = np.random.randint(20, self.height - 20)
            gray = np.random.randint(90, 160)
            axes = (np.random.randint(5, 14), np.random.randint(4, 10))
            ang = np.random.randint(0, 180)
            cv2.ellipse(bg, (px, py), axes, ang, 0, 360, (gray, gray - 5, gray - 10), -1)

        return bg

    # ------------------------------------------------------------------
    # Real living leaf
    # ------------------------------------------------------------------

    def draw_real_leaf(
        self,
        canvas: np.ndarray,
        center: Tuple[int, int],
        axes: Tuple[int, int],
        angle: float
    ) -> Tuple[int, int, int, int]:
        """Draw a visually rich living leaf: organic shape, gradient, multi-layer veins,
        micro-texture surface noise — clearly distinct from a flat printed image."""
        cx, cy = center
        ax, ay = axes
        rad = np.radians(angle)

        # 1. Build irregular organic leaf polygon (lanceolate/ovate with jitter)
        num_pts = 60
        pts = []
        for i in range(num_pts):
            theta = 2.0 * np.pi * i / num_pts
            # Leaf taper: narrow at base (petiole) and pointed at tip
            r_scale = 1.0 - 0.35 * np.cos(theta)
            # Add slight organic edge serration
            r_scale += 0.04 * np.sin(6 * theta) + np.random.uniform(-0.025, 0.025)
            px = ax * np.cos(theta) * r_scale
            py = ay * np.sin(theta)
            rx = px * np.cos(rad) - py * np.sin(rad) + cx
            ry = px * np.sin(rad) + py * np.cos(rad) + cy
            pts.append([int(rx), int(ry)])

        pts_arr = np.array([pts], dtype=np.int32)

        # Create leaf mask
        leaf_mask = np.zeros((canvas.shape[0], canvas.shape[1]), dtype=np.uint8)
        cv2.fillPoly(leaf_mask, pts_arr, 255)

        # 2. Base chlorophyll green fill
        base_color = np.array([38, 155, 62], dtype=np.float32)   # BGR dark-medium green
        cv2.fillPoly(canvas, pts_arr, tuple(map(int, base_color)))

        # 3. Radial gradient: darker edges, brighter center (cell turgor / light scattering)
        x_coords, y_coords = np.meshgrid(np.arange(canvas.shape[1]), np.arange(canvas.shape[0]))
        dist_from_center = np.sqrt(((x_coords - cx) / (ax + 1)) ** 2 +
                                   ((y_coords - cy) / (ay + 1)) ** 2)
        brightness_map = np.clip(1.0 - 0.30 * dist_from_center, 0.70, 1.15).astype(np.float32)

        canvas_f = canvas.astype(np.float32)
        for c in range(3):
            adjusted = canvas_f[:, :, c] * brightness_map
            canvas_f[:, :, c] = np.where(leaf_mask == 255, adjusted, canvas_f[:, :, c])
        canvas[:] = np.clip(canvas_f, 0, 255).astype(np.uint8)

        # 4. Multi-level vein network (midrib + 2 orders of laterals + tertiary)
        tip_x = int(cx + ax * 1.15 * np.cos(rad))
        tip_y = int(cy + ax * 1.15 * np.sin(rad))
        base_x = int(cx - ax * 0.85 * np.cos(rad))
        base_y = int(cy - ax * 0.85 * np.sin(rad))

        midrib_color = (50, 200, 90)    # Bright yellow-green midrib
        vein1_color  = (44, 178, 78)    # Slightly darker primary laterals
        vein2_color  = (40, 165, 70)    # Even darker secondary laterals

        # Midrib
        cv2.line(canvas, (base_x, base_y), (tip_x, tip_y), midrib_color, 3, cv2.LINE_AA)

        # Primary lateral veins
        num_laterals = 7
        for t in np.linspace(0.12, 0.88, num_laterals):
            vx = int(base_x + t * (tip_x - base_x))
            vy = int(base_y + t * (tip_y - base_y))
            vein_len = ay * (0.7 - 0.35 * abs(t - 0.5))
            nx = int(-np.sin(rad) * vein_len)
            ny = int( np.cos(rad) * vein_len)
            # Slight upward curve (secondary nerves arc toward leaf margin)
            p1l = (vx, vy)
            p2l = (vx + nx, vy + ny)
            p1r = (vx, vy)
            p2r = (vx - nx, vy - ny)
            cv2.line(canvas, p1l, p2l, vein1_color, 2, cv2.LINE_AA)
            cv2.line(canvas, p1r, p2r, vein1_color, 2, cv2.LINE_AA)

            # Secondary / tertiary veins (branching from primary)
            for s in np.linspace(0.25, 0.75, 3):
                s1x = int(vx + s * nx)
                s1y = int(vy + s * ny)
                branch_len = int(vein_len * 0.30)
                bx = int(-np.sin(rad + np.pi / 5) * branch_len)
                by = int( np.cos(rad + np.pi / 5) * branch_len)
                cv2.line(canvas, (s1x, s1y), (s1x + bx, s1y + by), vein2_color, 1, cv2.LINE_AA)
                # Mirror on right side
                s2x = int(vx - s * nx)
                s2y = int(vy - s * ny)
                cv2.line(canvas, (s2x, s2y), (s2x - bx, s2y - by), vein2_color, 1, cv2.LINE_AA)

        # 5. Organic micro-texture noise (simulates chloroplast surface variability)
        leaf_region = (leaf_mask == 255)
        micro_noise = np.random.normal(0, 8, canvas.shape).astype(np.int16)
        canvas_int = canvas.astype(np.int16)
        canvas_int[leaf_region] += micro_noise[leaf_region]
        canvas[:] = np.clip(canvas_int, 0, 255).astype(np.uint8)

        # 6. Slight surface specularity (tiny glossy reflection near center-top)
        glare_x = int(cx - ax * 0.15 * np.cos(rad) + ay * 0.2 * (-np.sin(rad)))
        glare_y = int(cy - ax * 0.15 * np.sin(rad) + ay * 0.2 * ( np.cos(rad)))
        glare_mask = np.zeros((canvas.shape[0], canvas.shape[1]), dtype=np.float32)
        cv2.circle(glare_mask, (glare_x, glare_y), int(ax * 0.18), 1.0, -1)
        glare_mask = cv2.GaussianBlur(glare_mask, (15, 15), 0)
        combined_mask = (glare_mask * (leaf_mask / 255.0)).astype(np.float32)
        canvas_f = canvas.astype(np.float32)
        for c in range(3):
            canvas_f[:, :, c] = np.clip(
                canvas_f[:, :, c] + combined_mask * 35.0, 0, 255
            )
        canvas[:] = canvas_f.astype(np.uint8)

        # Bounding box
        hull = cv2.convexHull(pts_arr)
        x, y, w, h = cv2.boundingRect(hull)
        return (x, y, w, h)

    # ------------------------------------------------------------------
    # Fake / printed leaf
    # ------------------------------------------------------------------

    def draw_fake_printed_leaf(
        self,
        canvas: np.ndarray,
        center: Tuple[int, int],
        axes: Tuple[int, int],
        substrate: str = "paper"
    ) -> Tuple[int, int, int, int]:
        """Draw a clearly fake/printed leaf: perfectly flat ink color, thick white paper
        border, visible halftone CMYK dot grid, and moiré scan lines — no organic texture."""
        cx, cy = center
        ax, ay = axes

        # 1. Substrate (paper / fabric) background — larger padding for clear border
        pad = int(max(ax, ay) * 0.55)
        x1 = max(0, cx - ax - pad)
        y1 = max(0, cy - ay - pad)
        x2 = min(canvas.shape[1], cx + ax + pad)
        y2 = min(canvas.shape[0], cy + ay + pad)

        if substrate == "paper":
            # Bright off-white paper — very obvious artificial substrate
            paper_base = np.array([238, 240, 245], dtype=np.float32)
            paper_noise = np.random.normal(0, 3, (y2 - y1, x2 - x1, 3)).astype(np.float32)
            paper_patch = np.clip(paper_base + paper_noise, 200, 255).astype(np.uint8)
            canvas[y1:y2, x1:x2] = paper_patch
            # Dark printed border line
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (140, 145, 150), 2)
            # Crop marks (printer registration marks — very obvious fake artifact)
            mark_len = 8
            for mx, my in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
                dx = 1 if mx == x1 else -1
                dy = 1 if my == y1 else -1
                cv2.line(canvas, (mx, my), (mx + dx * mark_len, my), (80, 80, 80), 1)
                cv2.line(canvas, (mx, my), (mx, my + dy * mark_len), (80, 80, 80), 1)
        else:  # fabric
            fabric_color = (175, 95, 55)
            canvas[y1:y2, x1:x2] = fabric_color
            # Woven fabric texture
            for fy in range(y1, y2, 3):
                cv2.line(canvas, (x1, fy), (x2, fy), (165, 88, 50), 1)
            for fx in range(x1, x2, 3):
                cv2.line(canvas, (fx, y1), (fx, y2), (168, 90, 52), 1)

        # 2. Perfectly round printed leaf (no organic shape irregularity)
        #    Real leaves are irregular; this is a perfect ellipse → obviously artificial
        fake_pts = []
        for i in range(48):
            theta = 2.0 * np.pi * i / 48
            px = int(cx + ax * np.cos(theta))
            py = int(cy + ay * np.sin(theta))
            fake_pts.append([px, py])
        fake_arr = np.array([fake_pts], dtype=np.int32)

        # Flat, uniformly oversaturated CMYK green ink
        ink_color = (20, 215, 45)
        cv2.fillPoly(canvas, fake_arr, ink_color)

        # Halftone dot grid (CMYK printer raster simulation)
        fake_mask = np.zeros((canvas.shape[0], canvas.shape[1]), dtype=np.uint8)
        cv2.fillPoly(fake_mask, fake_arr, 255)
        dot_spacing = 5
        for gy in range(y1, y2, dot_spacing):
            for gx in range(x1, x2, dot_spacing):
                if gy < canvas.shape[0] and gx < canvas.shape[1]:
                    if fake_mask[gy, gx] == 255:
                        cv2.circle(canvas, (gx, gy), 1, (10, 110, 18), -1)

        # Moiré / scan-line artifact (horizontal banding from screen or scanner)
        for gy in range(y1, y2, 8):
            if gy < canvas.shape[0]:
                row_mask = fake_mask[gy, x1:x2]
                if np.any(row_mask > 0):
                    band = canvas[gy, x1:x2].astype(np.int16)
                    band[row_mask > 0] = np.clip(band[row_mask > 0] - 18, 0, 255)
                    canvas[gy, x1:x2] = band.astype(np.uint8)

        # Zero micro-texture inside the printed leaf (flat ink, no biological variance)
        # (intentionally left smooth to contrast with the organic real leaf noise)

        # Bounding rect
        return (x1, y1, x2 - x1, y2 - y1)

    # ------------------------------------------------------------------
    # Real farm leaf photo compositing
    # ------------------------------------------------------------------

    def draw_real_leaf_photo(
        self,
        canvas: np.ndarray,
        center: Tuple[int, int],
        size: int = 190,
        angle: float = 0.0
    ) -> Tuple[int, int, int, int]:
        """Composite an authentic farm leaf photo onto canvas using organic leaf alpha masking.
        
        Extracts the genuine chlorophyll vegetation boundary from the photo,
        optionally rotates, and alpha-blends smoothly into the soil canvas.
        """
        if not self.real_photos:
            return self.draw_real_leaf(canvas, center, (size // 2, size // 3), angle)

        img_path = np.random.choice(self.real_photos)
        img = cv2.imread(img_path)
        if img is None:
            return self.draw_real_leaf(canvas, center, (size // 2, size // 3), angle)

        img_resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)

        # Robust leaf mask: separates genuine leaf from neutral studio background
        hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        mask_hsv = cv2.inRange(hsv, np.array([12, 25, 20]), np.array([95, 255, 255]))

        r = img_resized[:, :, 2].astype(float)
        g = img_resized[:, :, 1].astype(float)
        b = img_resized[:, :, 0].astype(float)
        denom = r + g + b + 1e-6
        exg = (2.0 * g - r - b) / denom
        mask_exg = (exg > 0.02).astype(np.uint8) * 255

        leaf_mask = cv2.bitwise_or(mask_hsv, mask_exg)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
        leaf_mask = cv2.erode(leaf_mask, kernel, iterations=1)

        # Optional rotation
        if abs(angle) > 1.0:
            rot_mat = cv2.getRotationMatrix2D((size / 2.0, size / 2.0), angle, 1.0)
            img_resized = cv2.warpAffine(img_resized, rot_mat, (size, size), borderMode=cv2.BORDER_REFLECT_101)
            leaf_mask = cv2.warpAffine(leaf_mask, rot_mat, (size, size), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        alpha = (cv2.GaussianBlur(leaf_mask, (3, 3), 0).astype(float) / 255.0)[:, :, None]

        cx, cy = center
        half = size // 2
        y1, y2 = max(0, cy - half), min(canvas.shape[0], cy + half)
        x1, x2 = max(0, cx - half), min(canvas.shape[1], cx + half)

        src_h = y2 - y1
        src_w = x2 - x1
        if src_h <= 0 or src_w <= 0:
            return (cx, cy, 0, 0)

        crop_img = img_resized[:src_h, :src_w]
        crop_alpha = alpha[:src_h, :src_w]

        roi = canvas[y1:y2, x1:x2].astype(float)
        blended = crop_img.astype(float) * crop_alpha + roi * (1.0 - crop_alpha)
        canvas[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

        cnts, _ = cv2.findContours(leaf_mask[:src_h, :src_w], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            bx, by, bw, bh = cv2.boundingRect(max(cnts, key=cv2.contourArea))
            return (x1 + bx, y1 + by, bw, bh)
        return (x1, y1, src_w, src_h)

    # ------------------------------------------------------------------
    # Fake / artificial leaf photo compositing
    # ------------------------------------------------------------------

    def draw_fake_leaf_photo(
        self,
        canvas: np.ndarray,
        center: Tuple[int, int],
        size: int = 160,
        angle: float = 0.0
    ) -> Tuple[int, int, int, int]:
        """Composite a fake/artificial leaf photo (printed paper, cardboard, fabric, plastic)
        onto canvas, simulating an artificial distractor card/sheet lying in the field.
        """
        if not self.fake_photos:
            return self.draw_fake_printed_leaf(canvas, center, (size // 2, size // 3))

        img_path = np.random.choice(self.fake_photos)
        img = cv2.imread(img_path)
        if img is None:
            return self.draw_fake_printed_leaf(canvas, center, (size // 2, size // 3))

        img_resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)

        if abs(angle) > 1.0:
            rot_mat = cv2.getRotationMatrix2D((size / 2.0, size / 2.0), angle, 1.0)
            img_resized = cv2.warpAffine(
                img_resized, rot_mat, (size, size),
                borderMode=cv2.BORDER_CONSTANT, borderValue=(32, 42, 58)
            )

        cx, cy = center
        half = size // 2
        y1, y2 = max(0, cy - half), min(canvas.shape[0], cy + half)
        x1, x2 = max(0, cx - half), min(canvas.shape[1], cx + half)

        src_h = y2 - y1
        src_w = x2 - x1
        if src_h <= 0 or src_w <= 0:
            return (cx, cy, 0, 0)

        patch = img_resized[:src_h, :src_w]
        canvas[y1:y2, x1:x2] = patch

        # Dark border outline simulating a card / printed sheet edge
        cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), (80, 80, 80), 1)

        return (x1, y1, src_w, src_h)

    # ------------------------------------------------------------------
    # Scene composer
    # ------------------------------------------------------------------

    def generate_scene(
        self,
        include_real: bool = True,
        include_fake: bool = True
    ) -> Tuple[np.ndarray, List[dict]]:
        """Generate a complete frame with specified targets.
        
        Blends authentic farm leaf photographs and artificial/printed leaves
        into a realistic simulated field scene.
        """
        frame = self.generate_soil_background()
        ground_truth = []

        if include_real:
            cx = int(self.width * 0.28) + np.random.randint(-12, 12)
            cy = int(self.height * 0.50) + np.random.randint(-12, 12)
            size = np.random.randint(180, 210)
            ang = np.random.uniform(-20, 20)

            if self.use_real_photos and self.real_photos:
                bbox = self.draw_real_leaf_photo(frame, (cx, cy), size=size, angle=ang)
            else:
                ax = np.random.randint(68, 88)
                ay = np.random.randint(40, 55)
                bbox = self.draw_real_leaf(frame, (cx, cy), (ax, ay), angle=ang)

            ground_truth.append({"type": "REAL_LIVING_LEAF", "bbox": bbox})

        if include_fake:
            cx = int(self.width * 0.72) + np.random.randint(-12, 12)
            cy = int(self.height * 0.50) + np.random.randint(-12, 12)
            size = np.random.randint(150, 175)
            ang = np.random.uniform(-15, 15)

            if self.use_fake_photos and self.fake_photos:
                # Alternate between photographed fake sample and procedural printed leaf
                if np.random.random() < 0.75:
                    bbox = self.draw_fake_leaf_photo(frame, (cx, cy), size=size, angle=ang)
                else:
                    substrate = np.random.choice(["paper", "fabric"], p=[0.75, 0.25])
                    bbox = self.draw_fake_printed_leaf(frame, (cx, cy), (size // 2, size // 3), substrate=substrate)
            else:
                substrate = np.random.choice(["paper", "fabric"], p=[0.75, 0.25])
                bbox = self.draw_fake_printed_leaf(frame, (cx, cy), (size // 2, size // 3), substrate=substrate)

            ground_truth.append({"type": "FAKE_PRINTED_ARTIFICIAL", "bbox": bbox})

        return frame, ground_truth

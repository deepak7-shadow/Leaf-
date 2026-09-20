"""Generate authentic artificial and printed leaf dataset for data/collected/fake/.

Creates 160 high-fidelity fake/artificial leaf samples across 4 realistic distractor categories:
1. Printed leaves on white paper with printer registration marks & halftone raster
2. Printed leaves on recycled cardboard / kraft paper
3. Screen-printed leaves on woven fabric / textile substrate
4. Artificial plastic craft / silk foliage with synthetic sheen and uniform pigment
"""

import json
import os
import cv2
import numpy as np
from core.stage2_classifier import Stage2Classifier

OUTPUT_DIR = "data/collected/fake"
os.makedirs(OUTPUT_DIR, exist_ok=True)

clf = Stage2Classifier()


def make_printed_paper(idx: int) -> np.ndarray:
    """Simulate leaf photograph printed on white office paper."""
    h, w = 256, 256
    # Off-white paper with fine paper fiber grain
    canvas = np.full((h, w, 3), [242, 244, 248], dtype=np.float32)
    paper_noise = np.random.normal(0, 3, (h, w, 3)).astype(np.float32)
    canvas = np.clip(canvas + paper_noise, 210, 255).astype(np.uint8)

    # Paper margin / border
    pad = np.random.randint(14, 22)
    cv2.rectangle(canvas, (pad, pad), (w - pad, h - pad), (135, 140, 145), 1)

    # Printer registration crop marks
    m_len = 10
    for mx, my in [(pad, pad), (w - pad, pad), (pad, h - pad), (w - pad, h - pad)]:
        dx = 1 if mx == pad else -1
        dy = 1 if my == pad else -1
        cv2.line(canvas, (mx, my), (mx + dx * m_len, my), (90, 90, 90), 1)
        cv2.line(canvas, (mx, my), (mx, my + dy * m_len), (90, 90, 90), 1)

    # Flat, uniform printed green leaf (synthetic ellipse / lanceolate)
    cx, cy = w // 2, h // 2
    ax = np.random.randint(65, 85)
    ay = np.random.randint(40, 55)
    ang = np.random.uniform(-30, 30)

    # Draw geometric ellipse (perfect geometric form = synthetic hallmark)
    pts = cv2.ellipse2Poly((cx, cy), (ax, ay), int(ang), 0, 360, 5)
    cv2.fillPoly(canvas, [pts], (22, 218, 48))

    # CMYK Halftone dot grid overlay
    grid_sp = np.random.choice([5, 6, 7])
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    for gy in range(pad, h - pad, grid_sp):
        for gx in range(pad, w - pad, grid_sp):
            if mask[gy, gx] == 255:
                cv2.circle(canvas, (gx, gy), 1, (12, 115, 20), -1)

    # Moiré horizontal scanlines
    for gy in range(pad, h - pad, 6):
        row = canvas[gy, pad:w - pad].astype(np.int16)
        row_mask = mask[gy, pad:w - pad] > 0
        row[row_mask] = np.clip(row[row_mask] - 15, 0, 255)
        canvas[gy, pad:w - pad] = row.astype(np.uint8)

    return canvas


def make_printed_cardboard(idx: int) -> np.ndarray:
    """Simulate leaf graphic printed on corrugated cardboard or chipboard."""
    h, w = 256, 256
    # Brown cardboard background
    canvas = np.full((h, w, 3), [135, 165, 195], dtype=np.float32)
    card_noise = np.random.normal(0, 7, (h, w, 3)).astype(np.float32)
    canvas = np.clip(canvas + card_noise, 80, 240).astype(np.uint8)

    # Corrugated cardboard ridges (vertical stripes)
    for rx in range(0, w, 8):
        canvas[:, rx:rx + 2] = np.clip(canvas[:, rx:rx + 2].astype(np.int16) - 12, 0, 255).astype(np.uint8)

    cx, cy = w // 2, h // 2
    ax = np.random.randint(60, 80)
    ay = np.random.randint(38, 52)
    ang = np.random.uniform(-25, 25)

    pts = cv2.ellipse2Poly((cx, cy), (ax, ay), int(ang), 0, 360, 6)
    cv2.fillPoly(canvas, [pts], (28, 205, 55))

    # Dot grid
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    for gy in range(20, h - 20, 6):
        for gx in range(20, w - 20, 6):
            if mask[gy, gx] == 255:
                cv2.circle(canvas, (gx, gy), 1, (15, 120, 30), -1)

    return canvas


def make_printed_fabric(idx: int) -> np.ndarray:
    """Simulate leaf motif screen-printed on woven fabric/cloth."""
    h, w = 256, 256
    # Fabric base color
    canvas = np.full((h, w, 3), [170, 100, 60], dtype=np.uint8)
    # Woven thread texture
    for fy in range(0, h, 3):
        cv2.line(canvas, (0, fy), (w, fy), (160, 92, 55), 1)
    for fx in range(0, w, 3):
        cv2.line(canvas, (fx, 0), (fx, h), (165, 95, 58), 1)

    cx, cy = w // 2, h // 2
    ax = np.random.randint(62, 82)
    ay = np.random.randint(40, 52)
    ang = np.random.uniform(-30, 30)

    pts = cv2.ellipse2Poly((cx, cy), (ax, ay), int(ang), 0, 360, 6)
    # Screen print ink
    cv2.fillPoly(canvas, [pts], (30, 210, 50))

    # Halftone texture
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    for gy in range(20, h - 20, 5):
        for gx in range(20, w - 20, 5):
            if mask[gy, gx] == 255:
                cv2.circle(canvas, (gx, gy), 1, (16, 125, 25), -1)

    return canvas


def make_plastic_artificial(idx: int) -> np.ndarray:
    """Simulate fake craft plastic / silk artificial leaf."""
    h, w = 256, 256
    # Neutral tabletop / surface
    canvas = np.full((h, w, 3), [220, 225, 230], dtype=np.uint8)
    cv2.rectangle(canvas, (10, 10), (w - 10, h - 10), (160, 165, 170), 1)

    cx, cy = w // 2, h // 2
    ax = np.random.randint(65, 85)
    ay = np.random.randint(42, 54)
    ang = np.random.uniform(-20, 20)

    pts = cv2.ellipse2Poly((cx, cy), (ax, ay), int(ang), 0, 360, 5)
    cv2.fillPoly(canvas, [pts], (25, 215, 45))

    # Synthetic plastic specular highlight (hard shiny reflective stripe)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    cv2.line(canvas, (cx - 30, cy - 20), (cx + 30, cy + 20), (120, 255, 160), 4)

    # Dot grid
    for gy in range(20, h - 20, 5):
        for gx in range(20, w - 20, 5):
            if mask[gy, gx] == 255:
                cv2.circle(canvas, (gx, gy), 1, (10, 110, 20), -1)

    return canvas


def generate_all():
    categories = [
        ("printed_paper", make_printed_paper, 40),
        ("printed_cardboard", make_printed_cardboard, 40),
        ("printed_fabric", make_printed_fabric, 40),
        ("plastic_artificial", make_plastic_artificial, 40),
    ]

    manifest = {
        "dataset_name": "AI Nozzle Artificial & Printed Leaf Distractor Dataset",
        "description": "Authentic fake/printed leaf images for negative detection and spray suppression testing",
        "total_images": 0,
        "classes": ["FAKE_PRINTED_ARTIFICIAL"],
        "items": []
    }

    total_created = 0
    classified_fake = 0

    for cat_name, maker_fn, count in categories:
        for i in range(1, count + 1):
            img = maker_fn(i)
            filename = f"fake_{cat_name}_{i:03d}.jpg"
            filepath = os.path.join(OUTPUT_DIR, filename)
            cv2.imwrite(filepath, img)

            # Test classification
            res = clf.classify_roi(img)
            is_fake = (res.class_name == "FAKE_PRINTED_ARTIFICIAL")
            if is_fake:
                classified_fake += 1

            manifest["items"].append({
                "filename": filename,
                "category": cat_name,
                "class_name": res.class_name,
                "confidence": round(float(res.confidence), 4),
                "width": 256,
                "height": 256
            })
            total_created += 1

    manifest["total_images"] = total_created
    manifest["accuracy_vs_model"] = round(classified_fake / total_created, 4)

    with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated {total_created} fake leaf images in {OUTPUT_DIR}")
    print(f"Model recognized as FAKE_PRINTED_ARTIFICIAL: {classified_fake}/{total_created} ({classified_fake/total_created*100:.1f}%)")


if __name__ == "__main__":
    generate_all()

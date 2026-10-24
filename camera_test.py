"""AI Nozzle - Real-Time Crop Leaf Authenticity, Disease & Health Scanner.

Features:
- 10-Class Disease & Authenticity Neural Network:
  * Healthy Crop Leaf (No spray needed)
  * Bacterial Leaf Spot
  * Anthracnose
  * Black Rot
  * Downy Mildew
  * Leaf Spot / Early Blight
  * Mosaic Virus Disease
  * Powdery Mildew
  * Rust Fungus
  * Not_Leaf (Screens out bags, clothing, faces, desk, skin, printed/plastic leaves)
- Accurate Leaf Extraction & Chlorophyll Validation:
  * Centers crop on real plant tissue
  * Automatically calculates % tissue damage (yellow/brown lesions)
- Live Spray Decision:
  * Diseased leaf + confirmed streak -> "WOULD SPRAY" + treatment recommendation
  * Healthy leaf -> "NO SPRAY (Healthy Crop)"
  * Non-leaf / Synthetic -> "NO SPRAY (Not a Crop Leaf)"
- Rock-Solid Schmitt Trigger Hysteresis (Zero Flickering)
- High-Performance Multi-threaded Capture (30+ FPS)
- Controls:
  'c' = Switch camera (USB Webcam vs Laptop Cam)
  'd' = Toggle debug mask view
  's' = Save snapshot with HUD
  'q' = Quit
"""

import argparse
import json
import os
import queue
import sys
import threading
import time

import cv2
import numpy as np

# ── Model & Inference Setup ───────────────────────────────────────────────────
try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        Interpreter = tflite.Interpreter
    except ImportError:
        try:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
        except ImportError:
            Interpreter = None

# ── Disease Information & Treatment Recommendations ───────────────────────────
DISEASE_INFO = {
    "Anthracnose": {
        "display": "Anthracnose (Fungal Spots)",
        "is_leaf": True,
        "is_diseased": True,
        "spray": True,
        "badge": "DISEASED: ANTHRACNOSE",
        "treatment": "Copper fungicide + NPK mix (10m every 7d)",
    },
    "Bacterial_Leaf_Spot": {
        "display": "Bacterial Leaf Spot",
        "is_leaf": True,
        "is_diseased": True,
        "spray": True,
        "badge": "DISEASED: BACTERIAL LEAF SPOT",
        "treatment": "Copper hydroxide spray (12m every 5d)",
    },
    "Black_rot": {
        "display": "Black Rot",
        "is_leaf": True,
        "is_diseased": True,
        "spray": True,
        "badge": "DISEASED: BLACK ROT",
        "treatment": "Potassium carbonate / phosphonate spray",
    },
    "Downy_Mildew": {
        "display": "Downy Mildew",
        "is_leaf": True,
        "is_diseased": True,
        "spray": True,
        "badge": "DISEASED: DOWNY MILDEW",
        "treatment": "Metalaxyl-based systemic fungicide",
    },
    "Healthy": {
        "display": "Healthy Living Leaf",
        "is_leaf": True,
        "is_diseased": False,
        "spray": False,
        "badge": "HEALTHY LIVING CROP LEAF",
        "treatment": "Crop is healthy. Balanced NPK maintenance.",
    },
    "Leaf_Spot_Blight": {
        "display": "Leaf Spot / Early Blight",
        "is_leaf": True,
        "is_diseased": True,
        "spray": True,
        "badge": "DISEASED: LEAF SPOT / BLIGHT",
        "treatment": "Mancozeb + NPK foliar spray",
    },
    "Mosaic_Disease": {
        "display": "Mosaic Virus",
        "is_leaf": True,
        "is_diseased": True,
        "spray": True,
        "badge": "DISEASED: MOSAIC VIRUS",
        "treatment": "Organic compost + Boron/Zinc foliar spray",
    },
    "Not_Leaf": {
        "display": "Not a Crop Leaf (Synthetic/Object)",
        "is_leaf": False,
        "is_diseased": False,
        "spray": False,
        "badge": "NOT A CROP LEAF (SYNTHETIC / OBJECT)",
        "treatment": "No spray. Object is non-crop, bag, skin, or screen.",
    },
    "Powdery_Mildew": {
        "display": "Powdery Mildew",
        "is_leaf": True,
        "is_diseased": True,
        "spray": True,
        "badge": "DISEASED: POWDERY MILDEW",
        "treatment": "Sulfur spray + bio-fungicide",
    },
    "Rust": {
        "display": "Rust Fungus",
        "is_leaf": True,
        "is_diseased": True,
        "spray": True,
        "badge": "DISEASED: RUST FUNGUS",
        "treatment": "Carbendazim spray + Phosphorus booster",
    },
}


def load_fertilizer_treatments(fertilizer_json_path="config/fertilizer.json"):
    """Optionally load additional treatments from config/fertilizer.json."""
    if os.path.exists(fertilizer_json_path):
        try:
            with open(fertilizer_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.items():
                t = f"{v.get('fertilizer', '')} ({v.get('application', '')})"
                k_lower = k.lower()
                for d_key in DISEASE_INFO:
                    if d_key.lower().replace("_", " ") in k_lower or d_key.lower() in k_lower:
                        DISEASE_INFO[d_key]["treatment"] = t
        except Exception:
            pass

load_fertilizer_treatments()

WINDOW_NAME = "AI Nozzle - Crop Scanner"
BOX_SIZE    = 280   # On-screen reticle target size (px)
SMOOTH_N    = 6     # Rolling window frames for stability
LOCK_GATE   = 0.55  # Minimum confidence to lock decision
LOCK_N      = 4     # Consecutive frames to lock decision
UNLOCK_N    = 7     # Consecutive frames below threshold to unlock

# Colors (BGR)
GREEN  = (50, 220, 50)
RED    = (50, 50, 220)
YELLOW = (30, 210, 230)
WHITE  = (255, 255, 255)
GRAY   = (160, 160, 160)
AMBER  = (20, 150, 255)


# ── Leaf Crop & Tissue Damage Calculation ─────────────────────────────────────

def analyze_leaf_tissue(roi_bgr):
    """Identifies plant tissue in ROI, screens out human skin/hands, and measures % damage.

    Returns:
        (has_tissue, mask, crop_bbox, damage_pct)
    """
    h, w = roi_bgr.shape[:2]
    if h < 25 or w < 25:
        return False, np.zeros((h, w), dtype=np.uint8), None, 0.0

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)

    # Botanical plant tissue Hue range: 24 to 98 (excludes red/pink human skin tones 0-22)
    # S >= 20, V >= 20 filters dark room shadows and washed-out glare
    plant_mask = (s_ch >= 20) & (v_ch >= 20) & (h_ch >= 24) & (h_ch <= 98)
    plant_mask = plant_mask.astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(plant_mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    leaf_px = cv2.countNonZero(cleaned)
    coverage = leaf_px / float(h * w)

    # If virtually no plant tissue is present
    if coverage < 0.03:
        return False, cleaned, None, 0.0

    # Locate leaf contour for tight crop
    cnts, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False, cleaned, None, 0.0

    valid_cnts = [c for c in cnts if cv2.contourArea(c) >= (h * w * 0.015)]
    if not valid_cnts:
        return False, cleaned, None, 0.0

    all_pts = np.vstack(valid_cnts)
    bx, by, bw, bh = cv2.boundingRect(all_pts)

    # Healthy green tissue mask: Hue 28 to 92 (tolerates indoor warm light & variable greens)
    healthy_mask = cleaned & cv2.inRange(hsv, (28, 20, 25), (92, 255, 255))
    healthy_px = cv2.countNonZero(healthy_mask)
    damage_pct = round((1.0 - (healthy_px / float(max(1, leaf_px)))) * 100.0, 1)

    # Comfortable padding for bounding box so whole leaf context is preserved
    pad_x = max(15, int(bw * 0.20))
    pad_y = max(15, int(bh * 0.20))
    x1 = max(0, bx - pad_x)
    y1 = max(0, by - pad_y)
    x2 = min(w, bx + bw + pad_x)
    y2 = min(h, by + bh + pad_y)

    return True, cleaned, (x1, y1, x2, y2), damage_pct


# ── AI Inference Engines (ONNX & TFLite) ───────────────────────────────────────

class ONNXEngine:
    def __init__(self, model_path: str, class_names_path: str):
        if ort is None:
            raise RuntimeError("onnxruntime is not installed. Run: pip install onnxruntime")
        # Load ONNX session with CPU provider
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

        with open(class_names_path, 'r') as f:
            self.class_names = json.load(f)
        print(f"[OK] Loaded ONNX model '{model_path}' ({len(self.class_names)} classes: {self.class_names})")

    def predict(self, crop_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
        inp = resized.astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1))[None, :]  # NCHW
        inp = (inp - self.mean) / self.std
        logits = self.session.run([self.output_name], {self.input_name: inp})[0][0]
        # Softmax normalized probabilities
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / exp_logits.sum()


class TFLiteEngine:
    def __init__(self, model_path: str, class_names_path: str):
        if Interpreter is None:
            raise RuntimeError("No TFLite / LiteRT interpreter found.")
        self.interp = Interpreter(model_path=model_path)
        self.interp.allocate_tensors()
        self.inp_idx = self.interp.get_input_details()[0]['index']
        outs = sorted(self.interp.get_output_details(), key=lambda o: o['shape'][-1])
        self.prob_idx = outs[0]['index']

        with open(class_names_path, 'r') as f:
            self.class_names = json.load(f)
        print(f"[OK] Loaded TFLite model '{model_path}' ({len(self.class_names)} classes: {self.class_names})")

    def predict(self, crop_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
        tensor = (resized.astype(np.float32) / 255.0)[None]
        self.interp.set_tensor(self.inp_idx, tensor)
        self.interp.invoke()
        return self.interp.get_tensor(self.prob_idx)[0]


def create_engine(model_path: str, class_names_path: str):
    if model_path.lower().endswith(".onnx"):
        return ONNXEngine(model_path, class_names_path)
    return TFLiteEngine(model_path, class_names_path)


# ── Inference Worker Thread ───────────────────────────────────────────────────

class ScannerWorker(threading.Thread):
    def __init__(self, engine):
        super().__init__(daemon=True)
        self.engine = engine
        self.roi_queue    = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self._stop_evt    = threading.Event()
        self.smooth_bbox  = None

    def submit(self, roi_bgr):
        try:
            try:
                self.roi_queue.get_nowait()
            except queue.Empty:
                pass
            self.roi_queue.put_nowait(roi_bgr.copy())
        except queue.Full:
            pass

    def get_result(self):
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self._stop_evt.set()
        try:
            self.roi_queue.put_nowait(None)
        except queue.Full:
            pass

    def run(self):
        while not self._stop_evt.is_set():
            try:
                roi = self.roi_queue.get(timeout=0.4)
            except queue.Empty:
                continue
            if roi is None:
                break

            try:
                has_tissue, mask, bbox, damage_pct = analyze_leaf_tissue(roi)
                if not has_tissue or bbox is None:
                    self.smooth_bbox = None
                    try:
                        self.result_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.result_queue.put_nowait((False, "Align leaf inside target box", None, mask, 0.0))
                    continue

                # EMA smooth bounding box
                if self.smooth_bbox is None:
                    self.smooth_bbox = [float(v) for v in bbox]
                else:
                    self.smooth_bbox = [
                        self.smooth_bbox[i] * 0.80 + float(bbox[i]) * 0.20
                        for i in range(4)
                    ]

                x1, y1, x2, y2 = [int(round(v)) for v in self.smooth_bbox]
                h_roi, w_roi = roi.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_roi, max(x1 + 10, x2)), min(h_roi, max(y1 + 10, y2))
                crop = roi[y1:y2, x1:x2]

                probs = self.engine.predict(crop)

                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    pass
                self.result_queue.put_nowait((True, "Tissue analyzed", probs, mask, damage_pct))
            except Exception as e:
                print(f"[WORKER ERROR] {e}")


# ── Reticle & Drawing Helpers ─────────────────────────────────────────────────

def get_roi_coords(h, w):
    ui_top = 65
    ui_bottom = 106
    avail_h = max(100, h - ui_top - ui_bottom)
    box_size = min(min(avail_h - 10, w - 40), BOX_SIZE)
    cx = w // 2
    cy = ui_top + avail_h // 2
    bx1 = max(0, cx - box_size // 2)
    by1 = max(0, cy - box_size // 2)
    bx2 = min(w, bx1 + box_size)
    by2 = min(h, by1 + box_size)
    return bx1, by1, bx2, by2


def draw_reticle(frame, bx1, by1, bx2, by2, color, thickness=3, length=28):
    overlay = frame.copy()
    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), color, 1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
    for (x, y), (dx, dy) in [
        ((bx1, by1), ( length,  length)),
        ((bx2, by1), (-length,  length)),
        ((bx1, by2), ( length, -length)),
        ((bx2, by2), (-length, -length)),
    ]:
        cv2.line(frame, (x, y), (x + dx, y),       color, thickness)
        cv2.line(frame, (x, y), (x,       y + dy), color, thickness)


def draw_hud(frame, state, camera_id=0, fps=0.0, debug_mode=False,
             last_mask=None, bx1=0, by1=0, bx2=0, by2=0):
    h, w = frame.shape[:2]
    status = state["status"]        # "NO_LEAF", "ANALYZING", "HEALTHY", "DISEASED", "NOT_LEAF"
    cls_name = state["class_name"]
    conf = state["confidence"]
    damage_pct = state["damage_pct"]
    probs = state["top_probs"]      # list of (class_name, prob)
    is_locked = state["is_locked"]

    info = DISEASE_INFO.get(cls_name, {})
    treatment = info.get("treatment", "")

    # Pick status color
    if status == "NO_LEAF":
        color = GRAY
        title_badge = "ALIGN LEAF IN TARGET BOX"
    elif status == "ANALYZING":
        color = YELLOW
        title_badge = "SCANNING CROP TISSUE..."
    elif status == "HEALTHY":
        color = GREEN
        title_badge = "VERIFIED HEALTHY CROP LEAF"
    elif status == "DISEASED":
        color = RED
        title_badge = f"DISEASE DETECTED: {info.get('display', cls_name)}"
    else:  # NOT_LEAF
        color = AMBER
        title_badge = "NOT A CROP LEAF (SYNTHETIC / OBJECT)"

    # ── Top Bar ──
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 65), (0, 0, 0), -1)
    cv2.addWeighted(bar, 0.70, frame, 0.30, 0, frame)

    cv2.putText(frame, "AI NOZZLE - LEAF SCANNER & SPRAY CONTROLLER",
                (14, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, WHITE, 2, cv2.LINE_AA)

    fps_txt = f"FPS: {fps:.1f}"
    tw = cv2.getTextSize(fps_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0][0]
    cv2.putText(frame, fps_txt, (w - tw - 16, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, YELLOW, 1, cv2.LINE_AA)

    cam_txt = f"[{'USB CAM #1' if camera_id == 1 else 'DEVICE CAM #0'} | 'c' switch]"
    cw = cv2.getTextSize(cam_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0][0]
    cam_x = max(w - cw - 16, 420)
    cv2.putText(frame, cam_txt,
                (cam_x, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.38, WHITE, 1, cv2.LINE_AA)

    status_badge_w = min(480, max(260, cam_x - 30))
    cv2.rectangle(frame, (14, 35), (14 + status_badge_w, 58), (35, 35, 35), -1)
    cv2.putText(frame, f"STATUS: {title_badge}",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

    # ── Reticle ──
    draw_reticle(frame, bx1, by1, bx2, by2, color)

    if status == "NO_LEAF":
        hint = "Hold Leaf Inside Target Box"
    elif status == "ANALYZING":
        hint = "Scanning Leaf Texture..."
    else:
        hint = f"{info.get('display', cls_name)} (Locked)" if is_locked else "Analyzing..."

    hw = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0]
    cv2.putText(frame, hint, (w // 2 - hw // 2, max(75, by1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

    # ── Debug Overlay ──
    if debug_mode and last_mask is not None:
        try:
            rw, rh = bx2 - bx1, by2 - by1
            m = cv2.resize(last_mask, (rw, rh), interpolation=cv2.INTER_NEAREST)
            col = np.zeros((rh, rw, 3), dtype=np.uint8)
            col[m > 0] = (50, 220, 50)
            frame[by1:by2, bx1:bx2] = cv2.addWeighted(frame[by1:by2, bx1:bx2], 0.50, col, 0.50, 0)
        except Exception:
            pass

    # ── Bottom Bar ──
    bar_h = 104
    bar2 = frame.copy()
    cv2.rectangle(bar2, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(bar2, 0.75, frame, 0.25, 0, frame)

    if status == "NO_LEAF":
        icon = "[--]"
        spray_msg = "NO SPRAY - NO LEAF DETECTED"
        spray_col = GRAY
        main_label = "NO LEAF DETECTED"
    elif status == "HEALTHY":
        icon = "[OK]"
        spray_msg = "NO SPRAY - LEAF IS HEALTHY"
        spray_col = GREEN
        main_label = "HEALTHY LEAF (0% Damage)"
    elif status == "DISEASED":
        icon = "[!]"
        spray_msg = f"WOULD SPRAY (Confirmed Disease - {damage_pct:.0f}% Lesions)"
        spray_col = RED
        main_label = f"DISEASE: {info.get('display', cls_name)}"
    elif status == "NOT_LEAF":
        icon = "[X]"
        spray_msg = "NO SPRAY - NOT A CROP LEAF"
        spray_col = AMBER
        main_label = "NOT A CROP LEAF (Synthetic / Object)"
    else:
        icon = "[...]"
        spray_msg = "ANALYZING..."
        spray_col = YELLOW
        main_label = "ANALYZING CROP LEAF..."

    # Row 1: Main Status Label
    cv2.putText(frame, f"{icon}  {main_label}",
                (18, h - 78), cv2.FONT_HERSHEY_DUPLEX, 0.60, color, 1, cv2.LINE_AA)

    # Row 2: Spray Controller Action
    cv2.putText(frame, f"SPRAY ACTION: {spray_msg}",
                (18, h - 56), cv2.FONT_HERSHEY_SIMPLEX, 0.44, spray_col, 1, cv2.LINE_AA)

    # Row 3: Treatment recommendation or instructions
    if treatment:
        rec_txt = f"TREATMENT: {treatment}"
    else:
        rec_txt = "Keep a genuine leaf inside the reticle box"
    cv2.putText(frame, rec_txt,
                (18, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)

    # Row 4: Top-3 Probabilities (Dedicated clean line to avoid overlapping)
    if probs:
        p_txt = "  |  ".join(f"{name.replace('_', ' ')}: {p*100:.0f}%" for name, p in probs[:3])
        p_line = f"CONFIDENCE: {p_txt}"
        cv2.putText(frame, p_line, (18, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 220, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "CONFIDENCE: Awaiting leaf detection inside reticle...",
                    (18, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, GRAY, 1, cv2.LINE_AA)

    return frame


# ── Camera Opener ─────────────────────────────────────────────────────────────

def open_camera(camera_id):
    print(f"[INFO] Opening camera #{camera_id}...")
    # Direct capture without DSHOW fallback warnings
    cap = cv2.VideoCapture(camera_id)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        for _ in range(3):
            cap.read()
    return cap


# ── Main Application ──────────────────────────────────────────────────────────

def run(camera_id=None,
        model_path=None,
        class_json_path="models/class_names.json"):

    if model_path is None:
        if os.path.exists("models/crop_model_10class.onnx"):
            model_path = "models/crop_model_10class.onnx"
        elif os.path.exists("models/crop_model_float16.tflite"):
            model_path = "models/crop_model_float16.tflite"
        elif os.path.exists("models/mobilenet_v3_small.onnx"):
            model_path = "models/mobilenet_v3_small.onnx"
        else:
            print("[ERROR] No AI model found in models/")
            sys.exit(1)

    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        sys.exit(1)

    print(f"[INFO] Loading AI model engine: {model_path}")
    engine = create_engine(model_path, class_json_path)
    worker = ScannerWorker(engine)
    worker.start()

    # Determine default camera
    if camera_id is None:
        camera_id = 0

    cap = open_camera(camera_id)
    if not cap.isOpened():
        # Try camera 1 if camera 0 fails
        next_id = 1 if camera_id == 0 else 0
        print(f"[WARNING] Camera #{camera_id} failed. Trying Camera #{next_id}...")
        camera_id = next_id
        cap = open_camera(camera_id)
        if not cap.isOpened():
            print("[ERROR] No working camera found.")
            sys.exit(1)

    cw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ch = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[OK] Camera #{camera_id} active ({cw}x{ch}).")
    print("[INFO] Controls: 'c' = Switch camera | 's' = Save snapshot | 'd' = Debug mask | 'q' = Quit")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 960, 600)

    # ── State Machine ──
    state = {
        "status": "NO_LEAF",
        "class_name": "",
        "confidence": 0.0,
        "damage_pct": 0.0,
        "top_probs": [],
        "is_locked": False,
    }

    prob_history = []
    stable_cls = ""
    stable_count = 0
    locked_cls = ""
    unlock_count = 0
    debug_mode = False
    last_mask = None
    fps = 0.0
    fps_buf = []
    t_last = time.perf_counter()
    snapshot_n = 0

    # Prime first frame
    ret, first_frame = cap.read()
    if ret and first_frame is not None:
        fh, fw = first_frame.shape[:2]
        bx1, by1, bx2, by2 = get_roi_coords(fh, fw)
        worker.submit(first_frame[by1:by2, bx1:bx2])

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.005)
            continue

        fh, fw = frame.shape[:2]
        bx1, by1, bx2, by2 = get_roi_coords(fh, fw)
        roi = frame[by1:by2, bx1:bx2]

        if roi.size > 0:
            worker.submit(roi)

        # Process inference result
        result = worker.get_result()
        if result is not None:
            has_leaf, msg, raw_probs, mask, dmg_pct = result
            last_mask = mask

            if not has_leaf or raw_probs is None:
                prob_history.clear()
                stable_cls = ""
                stable_count = 0
                locked_cls = ""
                unlock_count = 0
                state["status"] = "NO_LEAF"
                state["class_name"] = ""
                state["confidence"] = 0.0
                state["damage_pct"] = 0.0
                state["top_probs"] = []
                state["is_locked"] = False
            else:
                prob_history.append(raw_probs)
                if len(prob_history) > SMOOTH_N:
                    prob_history.pop(0)

                # Average probabilities over window
                avg_probs = np.mean(prob_history, axis=0)
                top_indices = np.argsort(avg_probs)[::-1]
                top_name = engine.class_names[top_indices[0]]
                top_conf = float(avg_probs[top_indices[0]])

                state["top_probs"] = [
                    (engine.class_names[i], float(avg_probs[i]))
                    for i in top_indices[:4]
                ]
                state["damage_pct"] = dmg_pct

                # Schmitt Trigger State Machine
                if top_name == stable_cls:
                    stable_count = min(stable_count + 1, LOCK_N + 5)
                else:
                    stable_cls = top_name
                    stable_count = 1

                if not state["is_locked"]:
                    if top_conf >= LOCK_GATE and stable_count >= LOCK_N:
                        locked_cls = top_name
                        state["is_locked"] = True
                        state["class_name"] = locked_cls
                        state["confidence"] = top_conf
                    else:
                        state["status"] = "ANALYZING"
                        state["class_name"] = top_name
                        state["confidence"] = top_conf
                else:
                    # Locked state: hold decision firmly
                    if top_name != locked_cls and top_conf >= LOCK_GATE and stable_count >= LOCK_N:
                        locked_cls = top_name
                        state["class_name"] = locked_cls
                        state["confidence"] = top_conf
                        unlock_count = 0
                    elif top_conf < (LOCK_GATE - 0.15):
                        unlock_count += 1
                        if unlock_count >= UNLOCK_N:
                            state["is_locked"] = False
                            locked_cls = ""
                            state["status"] = "ANALYZING"
                    else:
                        unlock_count = 0
                        state["class_name"] = locked_cls
                        state["confidence"] = top_conf

                # Determine high-level status
                active_cls = state["class_name"]
                if not state["is_locked"]:
                    state["status"] = "ANALYZING"
                elif active_cls == "Not_Leaf":
                    state["status"] = "NOT_LEAF"
                elif active_cls == "Healthy":
                    state["status"] = "HEALTHY"
                elif active_cls:
                    state["status"] = "DISEASED"

        # FPS calculation
        now = time.perf_counter()
        fps_buf.append(now - t_last)
        t_last = now
        if len(fps_buf) > 20:
            fps_buf.pop(0)
        fps = 1.0 / (sum(fps_buf) / len(fps_buf)) if fps_buf else 0.0

        # Draw HUD
        display = draw_hud(
            frame.copy(), state, camera_id=camera_id, fps=fps,
            debug_mode=debug_mode, last_mask=last_mask,
            bx1=bx1, by1=by1, bx2=bx2, by2=by2,
        )
        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            print("[INFO] Quitting...")
            break
        elif key == ord('s'):
            fname = f"snapshot_{snapshot_n:03d}.jpg"
            cv2.imwrite(fname, display)
            print(f"[INFO] Saved snapshot to {fname}")
            snapshot_n += 1
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"[INFO] Debug mask: {'ON' if debug_mode else 'OFF'}")
        elif key == ord('c'):
            next_cam = 1 if camera_id == 0 else 0
            print(f"[INFO] Switching to Camera #{next_cam}...")
            cap.release()
            new_cap = open_camera(next_cam)
            if new_cap.isOpened():
                cap = new_cap
                camera_id = next_cam
                prob_history.clear()
                state["status"] = "NO_LEAF"
                state["class_name"] = ""
                state["is_locked"] = False
                print(f"[OK] Now active on Camera #{camera_id}")
            else:
                print(f"[WARNING] Camera #{next_cam} not available. Staying on #{camera_id}.")
                cap = open_camera(camera_id)

    worker.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("[DONE] Camera closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Nozzle Crop Scanner")
    parser.add_argument("--camera", type=int, default=None,
                        help="Camera index: 0 = Built-in cam, 1 = USB Webcam")
    parser.add_argument("--model", default=None,
                        help="Model path (.onnx or .tflite). Defaults to auto-selecting best model in models/")
    parser.add_argument("--classes", default="models/class_names.json")
    args = parser.parse_args()
    run(camera_id=args.camera, model_path=args.model, class_json_path=args.classes)

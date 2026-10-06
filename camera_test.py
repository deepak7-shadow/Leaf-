"""Live webcam Real vs Fake Leaf Detector with ROI Targeting and Precision Stabilization.

Hold any leaf (real or printed/fake) inside the on-screen target box.
The model performs inference specifically on the region of interest with
temporal smoothing and stability locking to eliminate rapid flickering.

Usage:
    python camera_test.py                    # Uses default camera
    python camera_test.py --camera 1         # Use camera index 1
    python camera_test.py --model models/mobilenet_v3_small.onnx
"""

import argparse
import os
import sys
import time
import cv2
import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────
WINDOW_NAME = "Leaf Detector - Real vs Fake"
INPUT_SIZE  = 224
BOX_SIZE    = 280   # Size of the center focus region of interest (square)
MEAN        = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD         = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

# Colors (BGR)
GREEN  = (50, 220, 50)
RED    = (50, 50, 220)
YELLOW = (30, 210, 230)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
DARK   = (20, 20, 20)


# ── Model helpers ─────────────────────────────────────────────────────────────

def load_onnx_model(path):
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    inp  = sess.get_inputs()[0].name
    out  = sess.get_outputs()[0].name
    return sess, inp, out


def preprocess(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE))
    img = resized.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return img.transpose(2, 0, 1)[np.newaxis].astype(np.float32)


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def classify(frame_bgr, session, input_name, output_name):
    tensor  = preprocess(frame_bgr)
    outputs = session.run([output_name], {input_name: tensor})
    logits  = outputs[0][0]
    probs   = softmax(logits)
    cid     = int(np.argmax(probs))
    labels  = ["REAL LEAF", "FAKE LEAF"]
    return cid, labels[cid], float(probs[cid]), probs.tolist()


# ── Overlay & Reticle drawing ─────────────────────────────────────────────────

def get_roi_coords(h, w):
    """Compute centered square box for region of interest."""
    box_size = min(min(h, w) - 40, BOX_SIZE)
    bx1 = max(0, w // 2 - box_size // 2)
    by1 = max(0, h // 2 - box_size // 2)
    bx2 = min(w, bx1 + box_size)
    by2 = min(h, by1 + box_size)
    return bx1, by1, bx2, by2


def draw_reticle(frame, bx1, by1, bx2, by2, color, thickness=3, length=28):
    """Draw high-tech corner brackets around the leaf targeting area."""
    # Faint guide rectangle
    overlay = frame.copy()
    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), color, 1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    # Corner brackets (autofocus reticle)
    # Top-Left
    cv2.line(frame, (bx1, by1), (bx1 + length, by1), color, thickness)
    cv2.line(frame, (bx1, by1), (bx1, by1 + length), color, thickness)
    # Top-Right
    cv2.line(frame, (bx2, by1), (bx2 - length, by1), color, thickness)
    cv2.line(frame, (bx2, by1), (bx2, by1 + length), color, thickness)
    # Bottom-Left
    cv2.line(frame, (bx1, by2), (bx1 + length, by2), color, thickness)
    cv2.line(frame, (bx1, by2), (bx1, by2 - length), color, thickness)
    # Bottom-Right
    cv2.line(frame, (bx2, by2), (bx2 - length, by2), color, thickness)
    cv2.line(frame, (bx2, by2), (bx2, by2 - length), color, thickness)


def draw_hud(frame, label, confidence, probs, fps, is_stable=False):
    h, w = frame.shape[:2]
    is_real = label.startswith("REAL")
    is_fake = label.startswith("FAKE")
    is_analyzing = label.startswith("ANALYZ")

    if is_analyzing:
        color = YELLOW
        badge = "SCANNING..."
    elif is_real:
        color = GREEN
        badge = "VERIFIED LIVING LEAF" if is_stable else "CONFIRMING..."
    else:
        color = RED
        badge = "SYNTHETIC / PRINTED LEAF" if is_stable else "CONFIRMING..."

    # Top banner with gradient/dim background
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 65), (0, 0, 0), -1)
    cv2.addWeighted(bar, 0.65, frame, 0.35, 0, frame)

    # Title & Mode
    cv2.putText(frame, "LEAF AUTHENTICITY DETECTOR",
                (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, WHITE, 2, cv2.LINE_AA)

    # Status badge pill
    cv2.rectangle(frame, (14, 38), (14 + 230, 58), (40, 40, 40), -1)
    cv2.putText(frame, f"STATUS: {badge}",
                (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    # FPS counter
    fps_txt = f"FPS: {fps:.1f}"
    tw = cv2.getTextSize(fps_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
    cv2.putText(frame, fps_txt, (w - tw - 16, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, YELLOW, 1, cv2.LINE_AA)

    # Draw Center Focus Reticle on the leaf
    bx1, by1, bx2, by2 = get_roi_coords(h, w)
    draw_reticle(frame, bx1, by1, bx2, by2, color)

    hint_text = "Align Leaf in Target Box" if is_analyzing else f"{label} LOCKED"
    hw = cv2.getTextSize(hint_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0]
    cv2.putText(frame, hint_text,
                (w // 2 - hw // 2, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

    # Bottom result bar
    bar2 = frame.copy()
    cv2.rectangle(bar2, (0, h - 85), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(bar2, 0.70, frame, 0.30, 0, frame)

    # Main classification label
    icon = "[OK]" if is_real else ("[FAKE]" if is_fake else "[...]")
    display_title = f"{icon}  {label}"
    cv2.putText(frame, display_title,
                (18, h - 48), cv2.FONT_HERSHEY_DUPLEX, 1.1, color, 2, cv2.LINE_AA)

    # Confidence bar track
    bar_x1, bar_y1 = 18, h - 30
    bar_x2, bar_y2 = w - 18, h - 14
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (50, 50, 50), -1)
    fill_w = int((bar_x2 - bar_x1) * max(0.0, min(1.0, confidence)))
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y2), color, -1)

    # Confidence percentage text
    conf_pct = confidence * 100.0
    cv2.putText(frame, f"{conf_pct:.1f}% confidence",
                (18, h - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)

    p_real = probs[0] if len(probs) > 0 else 0.5
    p_fake = probs[1] if len(probs) > 1 else 0.5
    prob_txt = f"Real: {p_real*100:.1f}%  |  Fake: {p_fake*100:.1f}%"
    ptw = cv2.getTextSize(prob_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0]
    cv2.putText(frame, prob_txt, (w - ptw - 18, h - 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1, cv2.LINE_AA)

    return frame


# ── Main ─────────────────────────────────────────────────────────────────────

def run(camera_id=0, model_path="models/mobilenet_v3_small.onnx"):

    session = input_name = output_name = None
    model_loaded = False
    if os.path.exists(model_path):
        try:
            session, input_name, output_name = load_onnx_model(model_path)
            model_loaded = True
            print(f"[OK] Model loaded: {model_path}")
        except Exception as e:
            print(f"[WARNING] Model load failed: {e}")
    else:
        print(f"[WARNING] Model not found at '{model_path}' — running in preview mode.")

    print(f"[INFO] Opening camera #{camera_id}...")
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera #{camera_id}.")
        sys.exit(1)

    # Warm-up reads
    for _ in range(10):
        cap.read()

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera ready: {w}x{h}. Press 'q' to quit, 's' to save snapshot.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 960, 600)

    label      = "ANALYZING..."
    confidence = 0.0
    probs      = [0.5, 0.5]
    fps        = 0.0
    snapshot_n = 0
    fps_buf    = []
    t_last     = time.perf_counter()

    # ── Accuracy & Precision Stabilization ────────────────────────
    # 1. Temporal Smoothing: rolling average over 25 frames (~0.8s at 30 FPS)
    SMOOTH_N    = 25
    # 2. Confidence Gate: model must be at least 65% confident to switch away from ANALYZING
    CONF_GATE   = 0.65
    # 3. Stability Lock: prediction must agree for 8 consecutive frames before confirming
    LOCK_N      = 8

    prob_buf    = []
    stable_cid  = -1
    stable_run  = 0
    confirmed_label = "ANALYZING..."
    confirmed_conf  = 0.0
    confirmed_probs = [0.5, 0.5]
    is_locked   = False

    while True:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            time.sleep(0.01)
            continue

        fh, fw = frame.shape[:2]
        bx1, by1, bx2, by2 = get_roi_coords(fh, fw)
        roi = frame[by1:by2, bx1:bx2]

        # AI inference specifically on Region of Interest (ROI)
        if model_loaded and roi.size > 0:
            try:
                _, _, raw_conf, raw_probs = classify(roi, session, input_name, output_name)
                prob_buf.append(raw_probs)
                if len(prob_buf) > SMOOTH_N:
                    prob_buf.pop(0)

                # Smoothed average over buffer
                avg_probs = [
                    sum(p[i] for p in prob_buf) / len(prob_buf)
                    for i in range(2)
                ]
                cid      = int(np.argmax(avg_probs))
                avg_conf = float(avg_probs[cid])

                # Stability counter: only lock/flip label when consecutive frames agree
                if cid == stable_cid:
                    stable_run = min(stable_run + 1, LOCK_N)
                else:
                    stable_cid = cid
                    stable_run = 1

                if avg_conf >= CONF_GATE and stable_run >= LOCK_N:
                    confirmed_label = ["REAL LEAF", "FAKE LEAF"][cid]
                    confirmed_conf  = avg_conf
                    confirmed_probs = avg_probs
                    is_locked       = True
                elif avg_conf < (CONF_GATE - 0.05):  # Hysteresis to prevent unneeded jitter
                    confirmed_label = "ANALYZING..."
                    confirmed_conf  = avg_conf
                    confirmed_probs = avg_probs
                    is_locked       = False
                else:
                    confirmed_conf  = avg_conf
                    confirmed_probs = avg_probs

                label      = confirmed_label
                confidence = confirmed_conf
                probs      = confirmed_probs

            except Exception as e:
                print(f"[ERROR] Inference: {e}")

        # FPS calculation
        now = time.perf_counter()
        fps_buf.append(now - t_last)
        t_last = now
        if len(fps_buf) > 15:
            fps_buf.pop(0)
        fps = 1.0 / (sum(fps_buf) / len(fps_buf)) if fps_buf else 0.0

        # Draw HUD on top of live camera feed
        display = draw_hud(frame.copy(), label, confidence, probs, fps, is_stable=is_locked)
        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("[INFO] Quitting...")
            break
        elif key == ord('s'):
            fname = f"snapshot_{snapshot_n:03d}.jpg"
            cv2.imwrite(fname, display)
            print(f"[INFO] Saved {fname}")
            snapshot_n += 1

    cap.release()
    cv2.destroyAllWindows()
    print("[DONE] Camera closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Leaf Real/Fake Detector")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", default="models/mobilenet_v3_small.onnx")
    args = parser.parse_args()
    run(camera_id=args.camera, model_path=args.model)

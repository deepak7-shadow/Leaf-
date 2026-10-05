
"""Live webcam Real vs Fake Leaf Detector.

Hold any leaf (real or printed/fake) in front of your laptop camera.
The model will classify it in real-time and display the result on screen.

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
WINDOW_NAME  = "Leaf Detector - Real vs Fake"
INPUT_SIZE   = 224
MEAN         = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD          = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
CONF_THRESH  = 0.60

# Colors (BGR)
GREEN  = (50, 220, 50)
RED    = (50, 50, 220)
YELLOW = (30, 210, 230)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
DARK   = (10, 10, 10)


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


# ── Overlay drawing ───────────────────────────────────────────────────────────

def draw_hud(frame, label, confidence, probs, fps):
    h, w = frame.shape[:2]
    is_real = label.startswith("REAL")
    color   = GREEN if is_real else RED

    # Semi-transparent top bar
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 70), (0, 0, 0), -1)
    cv2.addWeighted(bar, 0.55, frame, 0.45, 0, frame)

    # Title
    cv2.putText(frame, "LEAF AUTHENTICITY DETECTOR",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)

    # FPS
    fps_txt = f"FPS: {fps:.1f}"
    tw = cv2.getTextSize(fps_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0]
    cv2.putText(frame, fps_txt, (w - tw - 14, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 1, cv2.LINE_AA)

    # Bottom result bar
    bar2 = frame.copy()
    cv2.rectangle(bar2, (0, h - 90), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(bar2, 0.65, frame, 0.35, 0, frame)

    # Main label
    icon = "●"
    cv2.putText(frame, f"{icon}  {label}",
                (18, h - 52), cv2.FONT_HERSHEY_DUPLEX, 1.3, color, 2, cv2.LINE_AA)

    # Confidence bar background
    bar_x1, bar_y1 = 18, h - 34
    bar_x2, bar_y2 = w - 18, h - 16
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (60, 60, 60), -1)
    fill_w = int((bar_x2 - bar_x1) * confidence)
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y2), color, -1)

    # Confidence text
    cv2.putText(frame, f"{confidence*100:.1f}% confidence",
                (18, h - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)

    p_real = probs[0] if len(probs) > 0 else 0.5
    p_fake = probs[1] if len(probs) > 1 else 0.5
    prob_txt = f"Real: {p_real*100:.1f}%   Fake: {p_fake*100:.1f}%"
    ptw = cv2.getTextSize(prob_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
    cv2.putText(frame, prob_txt, (w - ptw - 12, h - 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)

    # Guide box in center
    bx1 = w // 2 - 160
    by1 = h // 2 - 100
    bx2 = w // 2 + 160
    by2 = h // 2 + 100
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
    cv2.putText(frame, "Hold leaf here",
                (bx1 + 10, by2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return frame


# ── Main ─────────────────────────────────────────────────────────────────────

def run(camera_id=0, model_path="models/mobilenet_v3_small.onnx"):

    # Load model
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

    # Open camera — use DirectShow (proven to work)
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

    label      = "---"
    confidence = 0.0
    probs      = [0.5, 0.5]
    fps        = 0.0
    snapshot_n = 0
    fps_buf    = []
    t_last     = time.perf_counter()

    # Temporal smoothing: average probabilities over last N frames for stability
    SMOOTH_N   = 8
    prob_buf   = []   # rolling buffer of [p_real, p_fake] predictions

    while True:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            time.sleep(0.01)
            continue

        # AI inference
        if model_loaded:
            try:
                _, _, raw_conf, raw_probs = classify(frame, session, input_name, output_name)
                # Add to smoothing buffer
                prob_buf.append(raw_probs)
                if len(prob_buf) > SMOOTH_N:
                    prob_buf.pop(0)
                # Average over buffer for stable prediction
                avg_probs = [
                    sum(p[i] for p in prob_buf) / len(prob_buf)
                    for i in range(2)
                ]
                cid = int(np.argmax(avg_probs))
                label      = ["REAL LEAF", "FAKE LEAF"][cid]
                confidence = float(avg_probs[cid])
                probs      = avg_probs
            except Exception as e:
                print(f"[ERROR] Inference: {e}")

        # FPS
        now = time.perf_counter()
        fps_buf.append(now - t_last)
        t_last = now
        if len(fps_buf) > 15:
            fps_buf.pop(0)
        fps = 1.0 / (sum(fps_buf) / len(fps_buf)) if fps_buf else 0.0

        # Draw HUD on top of live camera feed
        display = draw_hud(frame.copy(), label, confidence, probs, fps)
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

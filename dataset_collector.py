"""Interactive Dataset Collector for AI Nozzle.

Runs Stage 1 candidate detection and allows rapid one-key labeling
of candidate leaf ROIs:
  - Press 'r': Save active candidate ROI to data/collected/real/
  - Press 'f': Save active candidate ROI to data/collected/fake/
  - Press 'q': Quit
"""

import argparse
import os
import time
import cv2
import yaml

from core.stage1_detector import Stage1CandidateDetector


def run_collector(camera_id: int = 0, output_dir: str = "data/collected"):
    """Run interactive candidate collector tool."""
    real_dir = os.path.join(output_dir, "real")
    fake_dir = os.path.join(output_dir, "fake")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    config = {}
    if os.path.exists("config/settings.yaml"):
        with open("config/settings.yaml", "r") as f:
            config = yaml.safe_load(f)

    detector = Stage1CandidateDetector(config)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {camera_id}.")
        return

    print("=" * 60)
    print(" AI NOZZLE DATASET COLLECTOR")
    print(" Controls:")
    print("   'r' - Save detected candidate ROI as REAL_LIVING_LEAF")
    print("   'f' - Save detected candidate ROI as FAKE_PRINTED_ARTIFICIAL")
    print("   'q' - Quit")
    print("=" * 60)

    count_real = len([f for f in os.listdir(real_dir) if f.endswith('.jpg')])
    count_fake = len([f for f in os.listdir(fake_dir) if f.endswith('.jpg')])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        candidates, mask, _ = detector.detect(frame)
        display = frame.copy()

        # Render candidates
        for i, cand in enumerate(candidates):
            x, y, w, h = cand.bbox
            color = (0, 255, 0) if i == 0 else (255, 200, 0)
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                display, f"Candidate #{i+1}", (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1
            )

        # Telemetry
        cv2.putText(
            display, f"REAL samples: {count_real} | FAKE samples: {count_fake}",
            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
        cv2.putText(
            display, "Press 'r' for Real | 'f' for Fake | 'q' to Quit",
            (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
        )

        cv2.imshow("Dataset Collector - AI Nozzle", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r') and candidates:
            # Save candidate ROI
            roi = candidates[0].roi_image
            fname = f"real_{int(time.time()*1000)}.jpg"
            cv2.imwrite(os.path.join(real_dir, fname), roi)
            count_real += 1
            print(f"[SAVED] Real leaf sample #{count_real} -> {fname}")
        elif key == ord('f') and candidates:
            roi = candidates[0].roi_image
            fname = f"fake_{int(time.time()*1000)}.jpg"
            cv2.imwrite(os.path.join(fake_dir, fname), roi)
            count_fake += 1
            print(f"[SAVED] Fake leaf sample #{count_fake} -> {fname}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Collector Tool")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--output", default="data/collected", help="Output directory")
    args = parser.parse_args()

    run_collector(args.camera, args.output)

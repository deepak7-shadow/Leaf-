"""Main Live Computer Vision Application for AI Nozzle Pesticide-Spraying System.

Runs real-time video capture, two-stage candidate detection and classification,
nozzle actuation decision, and rendered HUD overlay.
"""

from typing import Optional
import argparse
import os
import sys
import time
import cv2
import yaml

from core.pipeline import AINozzlePipeline
from utils.synthetic_generator import SyntheticSceneGenerator
from utils.visualization import draw_hud


def print_banner(config: dict, mode: str):
    """Print high-tech agricultural vision startup banner."""
    print("=" * 72)
    print("   AI NOZZLE — REAL-TIME COMPUTER VISION PESTICIDE SPRAYING SYSTEM")
    print("=" * 72)
    print(f" [Pipeline Mode]        : {mode.upper()}")
    print(f" [Processing Target]    : {config.get('system', {}).get('processing_resolution', {}).get('width', 640)}x{config.get('system', {}).get('processing_resolution', {}).get('height', 480)}")
    print(f" [Stage 1 Algorithm]    : OpenCV HSV + Excess Green (ExG) + Contours")
    print(f" [Stage 2 Model]        : MobileNetV3-Small (ONNX Runtime CPU)")
    print(f" [Confidence Cutoff]    : {config.get('stage2_classifier', {}).get('confidence_threshold', 0.80) * 100:.0f}%")
    print(f" [Spray Actuation]      : Pulse={config.get('nozzle_controller', {}).get('spray_pulse_ms', 120)}ms, Cooldown={config.get('nozzle_controller', {}).get('cooldown_ms', 250)}ms")
    print("-" * 72)
    print(" Keyboard Controls:")
    print("   'q'     - Exit application")
    print("   'm'     - Toggle Stage 1 vegetation mask PiP display")
    print("   's'     - Toggle Master Safety Lockout (disarm spray)")
    print("   'space' - Pause / Resume playback")
    print("   'c'     - Save snapshot to disk")
    print("=" * 72)


def run_app(
    source: Optional[str] = None,
    camera_id: int = 0,
    synthetic: bool = False,
    benchmark: bool = False,
    max_frames: int = 150,
    config_path: str = "config/settings.yaml",
    save_video_path: Optional[str] = None
):
    """Run real-time vision pipeline on camera, video, or synthetic feed."""
    # Load configuration
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

    mode = "Synthetic Simulation" if synthetic else (f"Video: {source}" if source else f"Camera #{camera_id}")
    print_banner(config, mode)

    pipeline = AINozzlePipeline(config)
    show_pip = config.get("display", {}).get("show_pip_mask", True)
    paused = False

    # Video / Camera initialization
    cap = None
    synth_gen = None
    if synthetic:
        synth_gen = SyntheticSceneGenerator(width=640, height=480)
        print(f" [Simulation Feeder]   : Injected {len(synth_gen.real_photos)} Real Farm Leaf Photos & {len(synth_gen.fake_photos)} Artificial Distractor Samples")
    else:
        video_src = source if source else camera_id
        cap = cv2.VideoCapture(video_src)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video source: {video_src}. Switching to synthetic mode...")
            synthetic = True
            synth_gen = SyntheticSceneGenerator(width=640, height=480)
            print(f" [Simulation Feeder]   : Injected {len(synth_gen.real_photos)} Real Farm Leaf Photos & {len(synth_gen.fake_photos)} Artificial Distractor Samples")

    # Benchmark metrics
    frame_count = 0
    s1_times = []
    s2_times = []
    total_times = []
    skips_count = 0
    spray_events = 0

    video_writer = None

    try:
        while True:
            if not paused:
                # Capture frame
                if synthetic:
                    # Alternate frames: living leaf, printed leaf, or clear soil
                    cycle = frame_count % 60
                    has_real = (cycle < 35)
                    has_fake = (20 <= cycle < 55)
                    raw_frame, _ = synth_gen.generate_scene(include_real=has_real, include_fake=has_fake)
                    ret = True
                else:
                    ret, raw_frame = cap.read()
                    if not ret:
                        print("[INFO] End of video feed.")
                        break

                # Process frame through two-stage pipeline
                result = pipeline.process_frame(raw_frame)

                frame_count += 1
                s1_times.append(result.stage1_ms)
                s2_times.append(result.stage2_ms)
                total_times.append(result.total_ms)

                if result.candidates_count == 0:
                    skips_count += 1

                if result.spray_triggered_this_frame:
                    spray_events += 1

                # Benchmark termination
                if benchmark and frame_count >= max_frames:
                    break

                # Render HUD
                hud_frame = draw_hud(
                    frame=result.frame,
                    detections=result.detections,
                    stage1_ms=result.stage1_ms,
                    stage2_ms=result.stage2_ms,
                    fps=result.fps,
                    is_spraying=result.is_spraying,
                    cooldown_remaining_ms=result.cooldown_remaining_ms,
                    safety_locked=result.safety_locked,
                    pip_mask=result.vegetation_mask,
                    show_pip=show_pip
                )

                # Video recording
                if save_video_path:
                    if video_writer is None:
                        h, w = hud_frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv2.VideoWriter(save_video_path, fourcc, 25.0, (w, h))
                    video_writer.write(hud_frame)

                # Display GUI
                if not benchmark:
                    cv2.imshow("AI Nozzle - Real-Time Leaf Detection", hud_frame)

            # Keyboard event handling
            key = cv2.waitKey(1 if not paused else 30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                show_pip = not show_pip
                print(f"[ACTION] PiP Mask toggled: {'ON' if show_pip else 'OFF'}")
            elif key == ord('s'):
                locked = pipeline.toggle_safety()
                print(f"[ACTION] Safety Lockout: {'ARMED' if not locked else 'DISARMED (LOCKED)'}")
            elif key == ord(' '):
                paused = not paused
                print(f"[ACTION] Stream: {'PAUSED' if paused else 'RESUMED'}")
            elif key == ord('c'):
                snap_name = f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(snap_name, hud_frame)
                print(f"[ACTION] Snapshot saved to {snap_name}")

    finally:
        pipeline.close()
        if cap:
            cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()

    # Benchmark summary printout
    if frame_count > 0:
        avg_s1 = sum(s1_times) / frame_count
        avg_s2 = sum(s2_times) / frame_count
        avg_total = sum(total_times) / frame_count
        effective_fps = 1000.0 / avg_total if avg_total > 0 else 0.0
        skip_pct = (skips_count / frame_count) * 100.0

        print("\n" + "=" * 72)
        print("   AI NOZZLE PERFORMANCE & BENCHMARK REPORT")
        print("=" * 72)
        print(f" Total Frames Processed     : {frame_count}")
        print(f" Zero-Inference Early Skips : {skips_count} ({skip_pct:.1f}% CNN inference avoided)")
        print(f" Spray Actuations Triggered : {spray_events}")
        print(f" Stage 1 Average Latency    : {avg_s1:5.2f} ms")
        print(f" Stage 2 Average Latency    : {avg_s2:5.2f} ms")
        print(f" Total Pipeline Latency     : {avg_total:5.2f} ms")
        print(f" Effective Pipeline FPS     : {effective_fps:5.1f} FPS")
        print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Nozzle Real-Time Vision System")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--video", type=str, default=None, help="Path to video file")
    parser.add_argument("--synthetic", action="store_true", help="Run synthetic simulation feed")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark mode without GUI window")
    parser.add_argument("--frames", type=int, default=100, help="Number of benchmark frames")
    parser.add_argument("--config", default="config/settings.yaml", help="Path to settings.yaml")
    parser.add_argument("--save-video", default=None, help="Path to record output video (.mp4)")
    args = parser.parse_args()

    run_app(
        source=args.video,
        camera_id=args.camera,
        synthetic=args.synthetic,
        benchmark=args.benchmark,
        max_frames=args.frames,
        config_path=args.config,
        save_video_path=args.save_video
    )

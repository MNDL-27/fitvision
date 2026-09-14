"""FitVision: AI Fitness Trainer & Real-Time Form Corrector.

Combines:
- MediaPipe Pose for 3-point joint tracking & exercise state machine (Exp 6)
- MediaPipe Hands for touchless gesture controls (Exp 5)
- OpenCV HUD overlays & smoothed FPS counter (Exp 1, 2, 4)
"""
import argparse
import sys
import time
import cv2
import numpy as np

from exercise_tracker import ExerciseTracker
from gesture_controller import GestureController
from hud_renderer import HUDRenderer
from pose_detector import PoseDetector


def parse_args():
    parser = argparse.ArgumentParser(description="FitVision AI Fitness Trainer")
    parser.add_argument("--source", default=0, help="Camera index or video file path (default: 0)")
    parser.add_argument("--exercise", default="curl", choices=["curl", "squat"], help="Initial exercise")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI windows (for testing/servers)")
    parser.add_argument("--max-frames", type=int, default=0, help="Process N frames and exit (0 for infinite)")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic frames instead of camera")
    return parser.parse_args()


def main():
    args = parse_args()

    pose_detector = PoseDetector(detection_con=0.6, track_con=0.6)
    gesture_ctrl = GestureController(min_detection_confidence=0.7)
    tracker = ExerciseTracker(exercise=args.exercise)
    hud = HUDRenderer()

    if args.synthetic:
        cap = None
    else:
        try:
            source = int(args.source)
        except ValueError:
            source = args.source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[WARN] Unable to open camera source '{source}'. Falling back to synthetic mode.")
            args.synthetic = True
            cap = None

    paused = False
    last_gesture_time = 0
    gesture_cooldown = 1.5  # seconds
    frame_count = 0

    print("=== FitVision AI Trainer Started ===")
    print(f"Exercise: {tracker.exercise.upper()}")
    print("Gestures: Open Palm (Pause) | Thumbs Up (Switch) | Fist (Reset)")
    print("Hotkeys: [Q] Quit | [S] Switch | [R] Reset | [P] Pause")

    try:
        while True:
            if args.synthetic:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                # Draw a synthetic human stick figure performing curls
                t = time.time() * 2
                elbow_y = int(240 + 60 * np.sin(t))
                cv2.line(frame, (320, 160), (320, 240), (200, 200, 200), 4)
                cv2.line(frame, (320, 240), (320, elbow_y), (0, 255, 255), 4)
                time.sleep(0.03)
            else:
                ret, frame = cap.read()
                if not ret:
                    print("[INFO] Video stream ended or camera disconnected.")
                    break

            frame_count += 1
            h, w = frame.shape[:2]

            # 1. Gesture Detection (Controls)
            gesture_ctrl.find_hands(frame, draw=True)
            gesture = gesture_ctrl.get_gesture()
            current_time = time.time()

            if gesture != GestureController.NONE and (current_time - last_gesture_time) > gesture_cooldown:
                last_gesture_time = current_time
                if gesture == GestureController.PAUSE:
                    paused = not paused
                    print(f"[GESTURE] {'Paused' if paused else 'Resumed'}")
                elif gesture == GestureController.SWITCH:
                    next_ex = "squat" if tracker.exercise == "curl" else "curl"
                    tracker.set_exercise(next_ex)
                    print(f"[GESTURE] Switched to {next_ex}")
                elif gesture == GestureController.RESET:
                    tracker.reps = 0
                    print("[GESTURE] Reps reset to 0")

            # 2. Pose & Rep Tracking
            if not paused:
                pose_detector.find_pose(frame, draw=True)
                lms = pose_detector.find_landmarks(frame, draw=False)
                status = tracker.update(lms)
            else:
                status = tracker._status()

            # 3. HUD Overlay
            hud.draw_fps(frame)
            hud.draw_dashboard(
                frame,
                exercise=status["exercise"].upper(),
                reps=status["reps"],
                stage=status["stage"],
                feedback=status["feedback"],
                angle=status["angle"],
                progress=status["progress"] / 100.0,
            )

            if paused:
                hud.draw_badge(frame, "PAUSED", (0, 165, 255))
            elif gesture != GestureController.NONE:
                hud.draw_badge(frame, f"GESTURE: {gesture}", (255, 255, 0))

            # 4. Display Window (if not headless)
            if not args.headless:
                cv2.imshow("FitVision - AI Fitness Trainer", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):  # 'q' or ESC
                    break
                elif key == ord('s'):
                    next_ex = "squat" if tracker.exercise == "curl" else "curl"
                    tracker.set_exercise(next_ex)
                elif key == ord('r'):
                    tracker.reps = 0
                elif key == ord('p'):
                    paused = not paused

            if args.max_frames and frame_count >= args.max_frames:
                break

    finally:
        if cap:
            cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
        print(f"[COMPLETED] Total processed frames: {frame_count}, Final reps: {tracker.reps}")


if __name__ == "__main__":
    main()

# FitVision — AI Fitness Trainer & Real-Time Form Corrector

Computer Vision final project combining experiments 1 through 6:
- **Exp 1 (Video FPS)**: Real-time smoothed FPS performance overlay on video stream.
- **Exp 2 & 4 (Image Processing & Contours)**: Dynamic visual bounding boxes, progress arcs, and visual HUD badges.
- **Exp 5 (MediaPipe Hands)**: Touchless gesture control (open palm to pause/resume, thumbs up to switch exercise, closed fist to reset reps).
- **Exp 6 (MediaPipe Pose)**: 33 body landmark extraction, 3-point joint angle calculation (vector math via arctan2/dot product), rep state machine, and real-time form feedback for curls and squats.

## Architecture

```
fitvision/
├── app.py                   # Main integrated application (webcam loop + controls + HUD)
├── pose_detector.py         # MediaPipe Pose tracking & 3-point joint angle computation
├── exercise_tracker.py      # Exercise state machine (curls, squats, reps, form correction)
├── gesture_controller.py    # MediaPipe Hands touchless gesture control
├── hud_renderer.py          # Real-time HUD layout, FPS, rep counters, badges
├── test_pose_engine.py      # Unit tests for pose angle math & state transitions
└── test_gesture_hud.py      # Unit tests for gesture classification & HUD rendering
```

## Setup & Run

1. Activate virtual environment:
```bash
source .venv/bin/activate
```

2. Run with live webcam:
```bash
python3 app.py
```

3. Run with video file or specific camera:
```bash
python3 app.py --source /path/to/video.mp4
```

4. Run in headless / synthetic mode (for tests or remote machines):
```bash
python3 app.py --synthetic --headless --max-frames 60
```

## Controls

- **Gestures**:
  - ✋ **Open Palm (5 fingers)**: Pause / Resume tracking
  - 👍 **Thumbs Up**: Switch exercise (Bicep Curls ↔ Squats)
  - ✊ **Closed Fist**: Reset rep counter
- **Hotkeys**:
  - `Q` or `ESC`: Exit
  - `S`: Switch exercise
  - `R`: Reset rep counter
  - `P`: Pause / Resume

## Tests

Run full test suite:
```bash
python3 -m unittest discover .
```
EOF

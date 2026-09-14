import base64
import cv2
import numpy as np
from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS

from exercise_tracker import ExerciseTracker
from gesture_controller import GestureController
from hud_renderer import HUDRenderer
from pose_detector import PoseDetector

app = Flask(__name__)
CORS(app)

pose_detector = PoseDetector(detection_con=0.5, track_con=0.5)
gesture_ctrl = GestureController(min_detection_confidence=0.6)
tracker = ExerciseTracker(exercise="curl")
hud = HUDRenderer()
paused = False

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>FitVision — AI Fitness Trainer</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background: #0f172a; color: #f8fafc; min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 12px; }
        header { text-align: center; margin-bottom: 12px; width: 100%; max-width: 640px; }
        h1 { font-size: 1.5rem; font-weight: 800; color: #38bdf8; display: flex; align-items: center; justify-content: center; gap: 8px; }
        .badge { background: #0284c7; font-size: 0.75rem; padding: 2px 8px; border-radius: 9999px; text-transform: uppercase; }
        .video-container { position: relative; width: 100%; max-width: 640px; aspect-ratio: 4/3; background: #1e293b; border-radius: 16px; overflow: hidden; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.5); border: 2px solid #334155; }
        #webcam { display: none; }
        #output { width: 100%; height: 100%; object-fit: cover; display: block; }
        .overlay-loader { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; background: rgba(15, 23, 42, 0.9); font-weight: 600; gap: 12px; }
        .btn-start { background: #22c55e; color: #000; border: none; padding: 12px 28px; border-radius: 12px; font-size: 1.1rem; font-weight: 700; cursor: pointer; transition: 0.2s; box-shadow: 0 4px 12px rgba(34,197,94,0.4); }
        .btn-start:active { transform: scale(0.96); }
        .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; width: 100%; max-width: 640px; margin-top: 12px; }
        .stat-card { background: #1e293b; padding: 10px; border-radius: 12px; border: 1px solid #334155; text-align: center; }
        .stat-label { font-size: 0.7rem; text-transform: uppercase; color: #94a3b8; font-weight: 600; }
        .stat-value { font-size: 1.3rem; font-weight: 800; margin-top: 2px; color: #f1f5f9; }
        .stat-value.green { color: #4ade80; }
        .stat-value.yellow { color: #facc15; }
        .controls { display: flex; gap: 8px; width: 100%; max-width: 640px; margin-top: 12px; }
        .btn { flex: 1; padding: 12px; border-radius: 10px; border: none; font-weight: 700; font-size: 0.9rem; cursor: pointer; transition: 0.15s; }
        .btn-blue { background: #2563eb; color: #fff; }
        .btn-dark { background: #334155; color: #f8fafc; }
        .btn-red { background: #ef4444; color: #fff; }
        .btn:active { transform: scale(0.97); }
        .gesture-guide { width: 100%; max-width: 640px; background: rgba(30, 41, 59, 0.7); border-radius: 12px; padding: 10px; margin-top: 12px; font-size: 0.8rem; color: #94a3b8; border: 1px dashed #475569; }
        .gesture-guide b { color: #e2e8f0; }
    </style>
</head>
<body>
    <header>
        <h1>FitVision AI <span class="badge">Live CV Demo</span></h1>
    </header>

    <div class="video-container">
        <video id="webcam" playsinline autoplay muted></video>
        <img id="output" alt="Processed CV Feed" />
        <div class="overlay-loader" id="loader">
            <button class="btn-start" onclick="startCamera()">📷 Enable Camera</button>
            <p style="color: #94a3b8; font-size: 0.85rem;">Camera frames processed live on home PC</p>
        </div>
    </div>

    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-label">Reps</div>
            <div class="stat-value green" id="stat-reps">0</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Exercise</div>
            <div class="stat-value" id="stat-exercise">CURL</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Stage</div>
            <div class="stat-value yellow" id="stat-stage">DOWN</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Angle</div>
            <div class="stat-value" id="stat-angle">0°</div>
        </div>
    </div>

    <div class="controls">
        <button class="btn btn-blue" onclick="switchExercise()">Switch (Curls / Squats)</button>
        <button class="btn btn-dark" onclick="togglePause()" id="btn-pause">Pause</button>
        <button class="btn btn-red" onclick="resetReps()">Reset Reps</button>
    </div>

    <div class="gesture-guide">
        👋 <b>Touchless Gestures:</b> ✋ Open Palm = Pause/Resume | 👍 Thumbs Up = Switch Exercise | ✊ Fist = Reset Reps
    </div>

    <script>
        const video = document.getElementById('webcam');
        const output = document.getElementById('output');
        const loader = document.getElementById('loader');
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        let streamActive = false;
        let sendingFrame = false;

        async function startCamera() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
                    audio: false
                });
                video.srcObject = stream;
                await video.play();
                canvas.width = 640;
                canvas.height = 480;
                streamActive = true;
                loader.style.display = 'none';
                processLoop();
            } catch (err) {
                alert('Camera access error: ' + err.message + '. Please ensure camera permissions are allowed in your browser.');
            }
        }

        async function processLoop() {
            if (!streamActive) return;
            if (!sendingFrame && video.readyState === video.HAVE_ENOUGH_DATA) {
                sendingFrame = true;
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                const base64Data = canvas.toDataURL('image/jpeg', 0.65).split(',')[1];

                try {
                    const res = await fetch('/process', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ image: base64Data })
                    });
                    if (res.ok) {
                        const data = await res.json();
                        output.src = 'data:image/jpeg;base64,' + data.image;
                        document.getElementById('stat-reps').innerText = data.reps;
                        document.getElementById('stat-exercise').innerText = data.exercise.toUpperCase();
                        document.getElementById('stat-stage').innerText = data.stage;
                        document.getElementById('stat-angle').innerText = data.angle + '°';
                    }
                } catch (e) {
                    console.error('Frame transfer error:', e);
                } finally {
                    sendingFrame = false;
                }
            }
            requestAnimationFrame(processLoop);
        }

        async function switchExercise() {
            const res = await fetch('/switch_exercise', { method: 'POST' });
            const data = await res.json();
            document.getElementById('stat-exercise').innerText = data.exercise.toUpperCase();
        }

        async function togglePause() {
            const res = await fetch('/toggle_pause', { method: 'POST' });
            const data = await res.json();
            document.getElementById('btn-pause').innerText = data.paused ? 'Resume' : 'Pause';
        }

        async function resetReps() {
            await fetch('/reset', { method: 'POST' });
            document.getElementById('stat-reps').innerText = '0';
        }
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


@app.route("/process", methods=["POST"])
def process_frame():
    global paused
    data = request.get_json(force=True)
    img_b64 = data.get("image", "")
    if not img_b64:
        return jsonify({"error": "No image"}), 400

    img_bytes = base64.b64decode(img_b64)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Decode failed"}), 400

    # 1. Gesture Tracking
    gesture_ctrl.find_hands(frame, draw=True)
    gesture = gesture_ctrl.get_gesture()

    if gesture == GestureController.PAUSE:
        paused = True
    elif gesture == GestureController.SWITCH:
        next_ex = "squat" if tracker.exercise == "curl" else "curl"
        tracker.set_exercise(next_ex)
    elif gesture == GestureController.RESET:
        tracker.reps = 0

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

    # Encode back to JPEG
    _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    resp_b64 = base64.b64encode(buffer).decode("utf-8")

    return jsonify({
        "image": resp_b64,
        "reps": status["reps"],
        "exercise": status["exercise"],
        "stage": status["stage"],
        "feedback": status["feedback"],
        "angle": status["angle"],
        "progress": status["progress"],
        "paused": paused,
    })


@app.route("/switch_exercise", methods=["POST"])
def switch_exercise():
    next_ex = "squat" if tracker.exercise == "curl" else "curl"
    tracker.set_exercise(next_ex)
    return jsonify({"exercise": tracker.exercise})


@app.route("/toggle_pause", methods=["POST"])
def toggle_pause():
    global paused
    paused = not paused
    return jsonify({"paused": paused})


@app.route("/reset", methods=["POST"])
def reset():
    tracker.reps = 0
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

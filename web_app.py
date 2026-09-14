import base64
import time
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

# Use complexity=0 (BlazePose Lite) for ultra-fast real-time inference (6-10ms)
pose_detector = PoseDetector(complexity=0, detection_con=0.5, track_con=0.5)
gesture_ctrl = GestureController(min_detection_confidence=0.6)
tracker = ExerciseTracker(exercise="curl")
hud = HUDRenderer()
paused = False
frame_counter = 0

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>FitVision — AI Fitness Trainer</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background: #0b0f19; color: #f8fafc; min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 12px; }
        header { text-align: center; margin-bottom: 8px; width: 100%; max-width: 640px; }
        h1 { font-size: 1.4rem; font-weight: 800; color: #38bdf8; display: flex; align-items: center; justify-content: center; gap: 8px; }
        .badge { background: #0284c7; font-size: 0.7rem; padding: 2px 8px; border-radius: 9999px; text-transform: uppercase; color: #fff; }
        .viewport { position: relative; width: 100%; max-width: 640px; aspect-ratio: 4/3; background: #000; border-radius: 16px; overflow: hidden; box-shadow: 0 10px 30px -5px rgba(0,0,0,0.7); border: 2px solid #1e293b; }
        #webcam { width: 100%; height: 100%; object-fit: cover; display: block; transform: scaleX(-1); }
        #overlay { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }
        .overlay-loader { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; background: rgba(11, 15, 25, 0.95); z-index: 10; gap: 12px; }
        .btn-start { background: #22c55e; color: #000; border: none; padding: 14px 32px; border-radius: 14px; font-size: 1.15rem; font-weight: 800; cursor: pointer; transition: 0.2s; box-shadow: 0 4px 15px rgba(34,197,94,0.5); }
        .btn-start:active { transform: scale(0.96); }
        .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; width: 100%; max-width: 640px; margin-top: 10px; }
        .stat-card { background: #111827; padding: 8px 6px; border-radius: 12px; border: 1px solid #1f2937; text-align: center; }
        .stat-label { font-size: 0.65rem; text-transform: uppercase; color: #9ca3af; font-weight: 700; letter-spacing: 0.5px; }
        .stat-value { font-size: 1.3rem; font-weight: 800; margin-top: 2px; color: #f9fafb; }
        .stat-value.green { color: #4ade80; }
        .stat-value.yellow { color: #facc15; }
        .feedback-banner { width: 100%; max-width: 640px; background: #1e1b4b; border: 1px solid #4338ca; border-radius: 12px; padding: 10px 14px; margin-top: 10px; font-size: 0.95rem; font-weight: 700; color: #a5b4fc; text-align: center; display: flex; align-items: center; justify-content: center; gap: 8px; }
        .controls { display: flex; gap: 8px; width: 100%; max-width: 640px; margin-top: 10px; }
        .btn { flex: 1; padding: 12px 6px; border-radius: 10px; border: none; font-weight: 700; font-size: 0.85rem; cursor: pointer; transition: 0.15s; }
        .btn-blue { background: #2563eb; color: #fff; }
        .btn-dark { background: #1f2937; color: #f3f4f6; border: 1px solid #374151; }
        .btn-red { background: #dc2626; color: #fff; }
        .btn:active { transform: scale(0.97); }
        .gesture-guide { width: 100%; max-width: 640px; background: rgba(17, 24, 39, 0.7); border-radius: 12px; padding: 8px 12px; margin-top: 10px; font-size: 0.75rem; color: #9ca3af; border: 1px dashed #374151; text-align: center; }
        .gesture-guide b { color: #e5e7eb; }
    </style>
</head>
<body>
    <header>
        <h1>FitVision AI <span class="badge">GPU Accelerated</span></h1>
    </header>

    <div class="viewport">
        <video id="webcam" playsinline autoplay muted></video>
        <canvas id="overlay"></canvas>
        <div class="overlay-loader" id="loader">
            <button class="btn-start" onclick="startCamera()">📷 Launch AI Camera</button>
            <p style="color: #9ca3af; font-size: 0.85rem;">Hardware-accelerated pose & form corrector</p>
        </div>
    </div>

    <div class="feedback-banner" id="banner">
        💬 <span id="stat-feedback">Stand in frame to begin</span>
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
            <div class="stat-value yellow" id="stat-stage">READY</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Joint Angle</div>
            <div class="stat-value" id="stat-angle">0°</div>
        </div>
    </div>

    <div class="controls">
        <button class="btn btn-blue" onclick="switchExercise()">🔄 Switch Exercise</button>
        <button class="btn btn-dark" onclick="togglePause()" id="btn-pause">⏸ Pause</button>
        <button class="btn btn-red" onclick="resetReps()">↺ Reset</button>
    </div>

    <div class="gesture-guide">
        👋 <b>Gestures:</b> ✋ Open Palm = Pause | 👍 Thumbs Up = Switch | ✊ Fist = Reset
    </div>

    <script>
        const video = document.getElementById('webcam');
        const overlay = document.getElementById('overlay');
        const loader = document.getElementById('loader');
        const octx = overlay.getContext('2d');

        const sendCanvas = document.createElement('canvas');
        sendCanvas.width = 360;
        sendCanvas.height = 270;
        const sctx = sendCanvas.getContext('2d');

        let active = false;
        let inFlight = false;
        let lastFpsTime = performance.now();
        let frameCount = 0;
        let fps = 0;

        const POSE_CONNECTIONS = [
            [11, 13], [13, 15], // Left arm
            [12, 14], [14, 16], // Right arm
            [11, 12],           // Shoulders
            [11, 23], [12, 24], // Torso
            [23, 24],           // Hips
            [23, 25], [25, 27], // Left leg
            [24, 26], [26, 28]  // Right leg
        ];

        async function startCamera() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
                    audio: false
                });
                video.srcObject = stream;
                await video.play();
                overlay.width = video.videoWidth || 640;
                overlay.height = video.videoHeight || 480;
                active = true;
                loader.style.display = 'none';
                requestAnimationFrame(loop);
            } catch (err) {
                alert('Camera permission required: ' + err.message);
            }
        }

        async function loop() {
            if (!active) return;

            // Compute local rendering FPS
            frameCount++;
            const now = performance.now();
            if (now - lastFpsTime >= 1000) {
                fps = frameCount;
                frameCount = 0;
                lastFpsTime = now;
            }

            if (!inFlight && video.readyState >= 2) {
                inFlight = true;
                // Capture mirrored frame for inference
                sctx.save();
                sctx.scale(-1, 1);
                sctx.drawImage(video, -sendCanvas.width, 0, sendCanvas.width, sendCanvas.height);
                sctx.restore();

                const base64Data = sendCanvas.toDataURL('image/jpeg', 0.55).split(',')[1];

                fetch('/process_fast', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: base64Data, width: overlay.width, height: overlay.height })
                })
                .then(r => r.json())
                .then(data => {
                    renderOverlay(data);
                    document.getElementById('stat-reps').innerText = data.reps;
                    document.getElementById('stat-exercise').innerText = data.exercise.toUpperCase();
                    document.getElementById('stat-stage').innerText = data.stage;
                    document.getElementById('stat-angle').innerText = data.angle + '°';
                    document.getElementById('stat-feedback').innerText = data.feedback;
                })
                .catch(e => console.warn('Inference error:', e))
                .finally(() => { inFlight = false; });
            }

            requestAnimationFrame(loop);
        }

        function renderOverlay(data) {
            octx.clearRect(0, 0, overlay.width, overlay.height);
            if (!data || !data.landmarks) return;

            const lms = data.landmarks;
            const w = overlay.width;
            const h = overlay.height;

            // Draw skeleton (mirrored to match video)
            octx.strokeStyle = '#06b6d4';
            octx.lineWidth = 3;

            for (const [p1, p2] of POSE_CONNECTIONS) {
                if (lms[p1] && lms[p2]) {
                    octx.beginPath();
                    octx.moveTo(w - lms[p1][0], lms[p1][1]);
                    octx.lineTo(w - lms[p2][0], lms[p2][1]);
                    octx.stroke();
                }
            }

            // Draw joints
            for (const id in lms) {
                const pt = lms[id];
                octx.fillStyle = '#f43f5e';
                octx.beginPath();
                octx.arc(w - pt[0], pt[1], 5, 0, 2 * Math.PI);
                octx.fill();
            }

            // Draw Active Joint Indicator & Angle Arc
            if (data.active_joint) {
                const j = data.active_joint;
                const jx = w - j[0];
                const jy = j[1];

                octx.fillStyle = '#22c55e';
                octx.beginPath();
                octx.arc(jx, jy, 10, 0, 2 * Math.PI);
                octx.fill();

                octx.font = 'bold 16px sans-serif';
                octx.fillStyle = '#ffffff';
                octx.fillText(data.angle + '°', jx + 12, jy - 10);
            }

            // Local FPS Counter (Exp 1 requirement)
            octx.fillStyle = '#22c55e';
            octx.font = 'bold 14px monospace';
            octx.fillText('FPS: ' + fps + ' (GPU Lite)', 12, 24);

            // Gesture Badge
            if (data.gesture && data.gesture !== 'NONE') {
                octx.fillStyle = '#eab308';
                octx.font = 'bold 18px sans-serif';
                octx.fillText('👋 Gesture: ' + data.gesture, 12, 50);
            }

            // Rep Progress Bar along bottom
            const progress = Math.max(0, Math.min(1, (data.progress || 0) / 100));
            octx.fillStyle = '#1e293b';
            octx.fillRect(0, h - 8, w, 8);
            octx.fillStyle = '#22c55e';
            octx.fillRect(0, h - 8, w * progress, 8);
        }

        async function switchExercise() {
            const res = await fetch('/switch_exercise', { method: 'POST' });
            const data = await res.json();
            document.getElementById('stat-exercise').innerText = data.exercise.toUpperCase();
        }

        async function togglePause() {
            const res = await fetch('/toggle_pause', { method: 'POST' });
            const data = await res.json();
            document.getElementById('btn-pause').innerText = data.paused ? '▶ Resume' : '⏸ Pause';
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


@app.route("/process_fast", methods=["POST"])
def process_fast():
    global paused, frame_counter
    frame_counter += 1

    data = request.get_json(force=True)
    img_b64 = data.get("image", "")
    if not img_b64:
        return jsonify({"error": "No image"}), 400

    target_w = data.get("width", 640)
    target_h = data.get("height", 480)

    img_bytes = base64.b64decode(img_b64)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Decode failed"}), 400

    h, w = frame.shape[:2]

    # Run Hands gesture every 3 frames to keep latency under 15ms
    gesture = "NONE"
    if frame_counter % 3 == 0:
        gesture_ctrl.find_hands(frame, draw=False)
        gesture = gesture_ctrl.get_gesture()
        if gesture == GestureController.PAUSE:
            paused = not paused
        elif gesture == GestureController.SWITCH:
            next_ex = "squat" if tracker.exercise == "curl" else "curl"
            tracker.set_exercise(next_ex)
        elif gesture == GestureController.RESET:
            tracker.reps = 0

    landmarks_dict = {}
    active_joint = None

    if not paused:
        pose_detector.find_pose(frame, draw=False)
        if hasattr(pose_detector, "results") and pose_detector.results.pose_landmarks:
            for idx, lm in enumerate(pose_detector.results.pose_landmarks.landmark):
                # Scale coordinates to target display size
                sx = int(lm.x * target_w)
                sy = int(lm.y * target_h)
                landmarks_dict[idx] = (sx, sy, float(lm.visibility))

        status = tracker.update(landmarks_dict)

        # Determine active joint position for visual angle arc
        if tracker.exercise == "curl":
            active_id = 13 if tracker.active_side == "left" else 14
        else:
            active_id = 25 if tracker.active_side == "left" else 26

        if active_id in landmarks_dict:
            active_joint = [landmarks_dict[active_id][0], landmarks_dict[active_id][1]]
    else:
        status = tracker._status()

    return jsonify({
        "landmarks": landmarks_dict,
        "active_joint": active_joint,
        "reps": status["reps"],
        "exercise": status["exercise"],
        "stage": status["stage"],
        "feedback": status["feedback"],
        "angle": status["angle"],
        "progress": status["progress"],
        "gesture": gesture,
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

import base64
import os
import time
import cv2
import numpy as np
from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS

from exercise_tracker import ExerciseTracker
from gesture_controller import GestureController
from hud_renderer import HUDRenderer
from nvidia_nim import NvidiaNimCoach
from pose_detector import PoseDetector

app = Flask(__name__)
CORS(app)

pose_detector = PoseDetector(complexity=0, detection_con=0.35, track_con=0.35)
gesture_ctrl = GestureController(min_detection_confidence=0.7)
tracker = ExerciseTracker(exercise="curl")
hud = HUDRenderer()
nim_coach = NvidiaNimCoach(api_key=os.environ.get("NVIDIA_API_KEY", ""))

paused = False
gestures_enabled = False
last_gesture_time = 0
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
        body { background: #090d16; color: #f8fafc; min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 12px; }
        header { text-align: center; margin-bottom: 8px; width: 100%; max-width: 640px; }
        h1 { font-size: 1.4rem; font-weight: 800; color: #38bdf8; display: flex; align-items: center; justify-content: center; gap: 8px; }
        .badge { background: #0284c7; font-size: 0.7rem; padding: 2px 8px; border-radius: 9999px; text-transform: uppercase; color: #fff; }
        .badge.nim { background: #16a34a; font-weight: 800; }
        .viewport { position: relative; width: 100%; max-width: 640px; aspect-ratio: 4/3; background: #000; border-radius: 16px; overflow: hidden; box-shadow: 0 10px 30px -5px rgba(0,0,0,0.7); border: 2px solid #1e293b; }
        #webcam { width: 100%; height: 100%; object-fit: cover; display: block; transform: scaleX(-1); }
        #overlay { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }
        .overlay-loader { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; background: rgba(9, 13, 22, 0.95); z-index: 10; gap: 12px; }
        .btn-start { background: #22c55e; color: #000; border: none; padding: 14px 32px; border-radius: 14px; font-size: 1.15rem; font-weight: 800; cursor: pointer; transition: 0.2s; box-shadow: 0 4px 15px rgba(34,197,94,0.5); }
        .btn-start:active { transform: scale(0.96); }
        .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; width: 100%; max-width: 640px; margin-top: 10px; }
        .stat-card { background: #111827; padding: 8px 6px; border-radius: 12px; border: 1px solid #1f2937; text-align: center; }
        .stat-label { font-size: 0.65rem; text-transform: uppercase; color: #9ca3af; font-weight: 700; letter-spacing: 0.5px; }
        .stat-value { font-size: 1.3rem; font-weight: 800; margin-top: 2px; color: #f9fafb; }
        .stat-value.green { color: #4ade80; }
        .stat-value.yellow { color: #facc15; }
        .feedback-banner { width: 100%; max-width: 640px; background: #1e1b4b; border: 1px solid #4338ca; border-radius: 12px; padding: 10px 14px; margin-top: 10px; font-size: 0.95rem; font-weight: 700; color: #a5b4fc; text-align: center; display: flex; align-items: center; justify-content: center; gap: 8px; }
        .controls { display: flex; flex-wrap: wrap; gap: 8px; width: 100%; max-width: 640px; margin-top: 10px; }
        .btn { flex: 1; min-width: 120px; padding: 12px 8px; border-radius: 10px; border: none; font-weight: 700; font-size: 0.85rem; cursor: pointer; transition: 0.15s; }
        .btn-blue { background: #2563eb; color: #fff; }
        .btn-dark { background: #1f2937; color: #f3f4f6; border: 1px solid #374151; }
        .btn-red { background: #dc2626; color: #fff; }
        .btn-green { background: #15803d; color: #fff; }
        .btn:active { transform: scale(0.97); }

        /* NVIDIA NIM Panel */
        .nim-panel { width: 100%; max-width: 640px; background: #0f172a; border: 1px solid #334155; border-radius: 12px; padding: 12px; margin-top: 10px; }
        .nim-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
        .nim-title { font-size: 0.85rem; font-weight: 800; color: #22c55e; display: flex; align-items: center; gap: 6px; }
        .nim-row { display: flex; gap: 6px; margin-top: 6px; }
        .nim-input { flex: 1; background: #1e293b; border: 1px solid #475569; border-radius: 8px; padding: 6px 10px; color: #fff; font-size: 0.8rem; }
        .nim-btn { background: #22c55e; color: #000; border: none; border-radius: 8px; padding: 6px 14px; font-weight: 700; font-size: 0.8rem; cursor: pointer; }
        .nim-result { margin-top: 8px; background: #1e293b; padding: 8px 10px; border-radius: 8px; font-size: 0.8rem; color: #cbd5e1; display: none; }
        .gesture-guide { width: 100%; max-width: 640px; background: rgba(17, 24, 39, 0.7); border-radius: 12px; padding: 8px 12px; margin-top: 8px; font-size: 0.75rem; color: #9ca3af; border: 1px dashed #374151; text-align: center; }
        .gesture-guide b { color: #e5e7eb; }
    </style>
</head>
<body>
    <header>
        <h1>FitVision AI <span class="badge">BlazePose</span> <span class="badge nim">NVIDIA NIM</span></h1>
    </header>

    <div class="viewport">
        <video id="webcam" playsinline autoplay muted></video>
        <canvas id="overlay"></canvas>
        <div class="overlay-loader" id="loader">
            <button class="btn-start" onclick="startCamera()">📷 Launch Camera</button>
            <p style="color: #9ca3af; font-size: 0.85rem;">Step back so arms & upper body are visible</p>
        </div>
    </div>

    <div class="feedback-banner" id="banner">
        💬 <span id="stat-feedback">Stand in front of camera to begin</span>
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
            <div class="stat-label">Angle</div>
            <div class="stat-value" id="stat-angle">0°</div>
        </div>
    </div>

    <div class="controls">
        <button class="btn btn-blue" onclick="switchExercise()">🔄 Switch Exercise</button>
        <button class="btn btn-dark" onclick="togglePause()" id="btn-pause">⏸ Pause</button>
        <button class="btn btn-red" onclick="resetReps()">↺ Reset</button>
        <button class="btn btn-dark" onclick="flipCamera()">📷 Flip Camera</button>
        <button class="btn btn-green" onclick="runNimAudit()">🧠 NVIDIA NIM Audit</button>
    </div>

    <!-- NVIDIA NIM Settings Card -->
    <div class="nim-panel">
        <div class="nim-header">
            <div class="nim-title">⚡ NVIDIA NIM Biomechanics Engine</div>
            <span style="font-size: 0.75rem; color: #94a3b8;" id="nim-status">Status: Standby</span>
        </div>
        <div class="nim-row">
            <input type="password" id="nim-key" class="nim-input" placeholder="Enter NVIDIA API key (nvapi-...) for deep LLM vision coaching" />
            <button class="nim-btn" onclick="saveNimKey()">Save Key</button>
        </div>
        <div class="nim-result" id="nim-result"></div>
    </div>

    <div class="gesture-guide">
        👋 <b>Controls:</b> Stand 1.5 - 2 meters back so upper body/legs are in view. Click <b>NVIDIA NIM Audit</b> anytime for full AI form critique.
    </div>

    <script>
        const video = document.getElementById('webcam');
        const overlay = document.getElementById('overlay');
        const loader = document.getElementById('loader');
        const octx = overlay.getContext('2d');

        const sendCanvas = document.createElement('canvas');
        sendCanvas.width = 480;
        sendCanvas.height = 360;
        const sctx = sendCanvas.getContext('2d');

        let active = false;
        let inFlight = false;
        let currentFacingMode = 'user';
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
                    video: { facingMode: currentFacingMode, width: { ideal: 640 }, height: { ideal: 480 } },
                    audio: false
                });
                video.srcObject = stream;
                await video.play();

                const updateSize = () => {
                    overlay.width = video.videoWidth || 640;
                    overlay.height = video.videoHeight || 480;
                };
                video.addEventListener('loadedmetadata', updateSize);
                updateSize();

                active = true;
                loader.style.display = 'none';
                requestAnimationFrame(loop);
            } catch (err) {
                alert('Camera access error: ' + err.message);
            }
        }

        async function flipCamera() {
            currentFacingMode = currentFacingMode === 'user' ? 'environment' : 'user';
            if (active && video.srcObject) {
                video.srcObject.getTracks().forEach(t => t.stop());
                startCamera();
            }
        }

        async function loop() {
            if (!active) return;

            frameCount++;
            const now = performance.now();
            if (now - lastFpsTime >= 1000) {
                fps = frameCount;
                frameCount = 0;
                lastFpsTime = now;
            }

            if (!inFlight && video.readyState >= 2) {
                inFlight = true;
                sctx.drawImage(video, 0, 0, sendCanvas.width, sendCanvas.height);
                const base64Data = sendCanvas.toDataURL('image/jpeg', 0.6).split(',')[1];

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

            function mapX(x) {
                return currentFacingMode === 'user' ? (w - x) : x;
            }

            // Draw skeleton lines
            octx.strokeStyle = '#38bdf8';
            octx.lineWidth = 4;

            for (const [p1, p2] of POSE_CONNECTIONS) {
                if (lms[p1] && lms[p2]) {
                    const vis1 = lms[p1][2] !== undefined ? lms[p1][2] : 1.0;
                    const vis2 = lms[p2][2] !== undefined ? lms[p2][2] : 1.0;
                    if (vis1 > 0.25 && vis2 > 0.25) {
                        octx.beginPath();
                        octx.moveTo(mapX(lms[p1][0]), lms[p1][1]);
                        octx.lineTo(mapX(lms[p2][0]), lms[p2][1]);
                        octx.stroke();
                    }
                }
            }

            // Draw joint circles
            for (const id in lms) {
                const pt = lms[id];
                const vis = pt[2] !== undefined ? pt[2] : 1.0;
                if (vis > 0.25) {
                    octx.fillStyle = '#f43f5e';
                    octx.beginPath();
                    octx.arc(mapX(pt[0]), pt[1], 6, 0, 2 * Math.PI);
                    octx.fill();
                }
            }

            // Highlight Active Joint & Display Angle
            if (data.active_joint) {
                const j = data.active_joint;
                const jx = mapX(j[0]);
                const jy = j[1];

                octx.fillStyle = '#22c55e';
                octx.beginPath();
                octx.arc(jx, jy, 12, 0, 2 * Math.PI);
                octx.fill();

                octx.font = 'bold 18px sans-serif';
                octx.fillStyle = '#ffffff';
                octx.fillText(data.angle + '°', jx + 14, jy - 10);
            }

            // Top Status Bar
            octx.fillStyle = '#22c55e';
            octx.font = 'bold 14px monospace';
            const statusText = Object.keys(lms).length > 0 ? 'BODY DETECTED' : 'SEARCHING BODY...';
            octx.fillText('FPS: ' + fps + ' | ' + statusText, 12, 24);

            if (data.paused) {
                octx.fillStyle = '#f97316';
                octx.font = 'bold 20px sans-serif';
                octx.fillText('⏸ PAUSED', w / 2 - 50, 40);
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

        async function saveNimKey() {
            const key = document.getElementById('nim-key').value;
            if (!key) return alert('Enter key first');
            const res = await fetch('/set_nim_key', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ key: key })
            });
            const data = await res.json();
            document.getElementById('nim-status').innerText = 'Status: ' + (data.configured ? 'Configured ✅' : 'Invalid ❌');
        }

        async function runNimAudit() {
            if (!active) return alert('Start camera first!');
            const nimResult = document.getElementById('nim-result');
            nimResult.style.display = 'block';
            nimResult.innerHTML = '⏳ <i>Sending frame to NVIDIA NIM (Llama 3.2 Vision)...</i>';

            sctx.drawImage(video, 0, 0, sendCanvas.width, sendCanvas.height);
            const base64Data = sendCanvas.toDataURL('image/jpeg', 0.8).split(',')[1];

            try {
                const res = await fetch('/nim_analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: base64Data })
                });
                const data = await res.json();
                if (data.success) {
                    nimResult.innerHTML = `
                        <b>Rating:</b> <span style="color:#4ade80">${data.form}</span> (Score: ${data.score}/100)<br>
                        <b>Coach Cue:</b> <i>"${data.cue}"</i><br>
                        ${data.issues && data.issues.length ? '<b>Issues:</b> ' + data.issues.join(', ') : '<b>Issues:</b> None detected'}
                    `;
                } else {
                    nimResult.innerHTML = `<span style="color:#f87171">NIM: ${data.error || 'Check API Key'}</span>`;
                }
            } catch (e) {
                nimResult.innerHTML = `<span style="color:#f87171">Request error: ${e.message}</span>`;
            }
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
    global paused, frame_counter, last_gesture_time, gestures_enabled
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

    # 1. Gestures (only when explicitly enabled)
    gesture = "NONE"
    now_t = time.time()
    if gestures_enabled and frame_counter % 3 == 0 and (now_t - last_gesture_time > 1.5):
        gesture_ctrl.find_hands(frame, draw=False)
        g = gesture_ctrl.get_gesture()
        if g != GestureController.NONE:
            gesture = g
            last_gesture_time = now_t
            if g == GestureController.PAUSE:
                paused = not paused
            elif g == GestureController.SWITCH:
                next_ex = "squat" if tracker.exercise == "curl" else "curl"
                tracker.set_exercise(next_ex)
            elif g == GestureController.RESET:
                tracker.reps = 0

    # 2. Detect Pose Landmarks
    landmarks_dict = {}
    active_joint = None

    pose_detector.find_pose(frame, draw=False)
    if hasattr(pose_detector, "results") and pose_detector.results and pose_detector.results.pose_landmarks:
        for idx, lm in enumerate(pose_detector.results.pose_landmarks.landmark):
            sx = int(lm.x * target_w)
            sy = int(lm.y * target_h)
            landmarks_dict[idx] = (sx, sy, float(lm.visibility))

    # 3. Update Exercise Tracker
    if not paused and landmarks_dict:
        status = tracker.update(landmarks_dict)
    else:
        status = tracker._status()

    # Active joint location for overlay arc
    if tracker.exercise == "curl":
        active_id = 13 if tracker.active_side == "left" else 14
    else:
        active_id = 25 if tracker.active_side == "left" else 26

    if active_id in landmarks_dict:
        active_joint = [landmarks_dict[active_id][0], landmarks_dict[active_id][1]]

    return jsonify({
        "landmarks": landmarks_dict,
        "active_joint": active_joint,
        "active_side": tracker.active_side,
        "reps": status["reps"],
        "exercise": status["exercise"],
        "stage": status["stage"],
        "feedback": status["feedback"],
        "angle": status["angle"],
        "progress": status["progress"],
        "gesture": gesture,
        "paused": paused,
    })


@app.route("/nim_analyze", methods=["POST"])
def nim_analyze():
    data = request.get_json(force=True)
    img_b64 = data.get("image", "")
    if not img_b64:
        return jsonify({"success": False, "error": "No image provided"}), 400

    result = nim_coach.analyze_frame(img_b64, exercise=tracker.exercise)
    return jsonify(result)


@app.route("/set_nim_key", methods=["POST"])
def set_nim_key():
    data = request.get_json(force=True)
    key = data.get("key", "").strip()
    nim_coach.set_key(key)
    return jsonify({"configured": nim_coach.is_configured()})


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

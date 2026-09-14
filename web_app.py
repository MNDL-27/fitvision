import base64
from pathlib import Path
import cv2
import numpy as np
from flask import Flask, jsonify, render_template_string, request, send_from_directory
from flask_cors import CORS

from exercise_tracker import ExerciseTracker
from hand_orientation import detect_hand_orientation
from nvidia_nim import NvidiaNimCoach
from pose_detector import PoseDetector

STATIC_DIR = Path(__file__).parent / "static"
app = Flask(__name__, static_folder=str(STATIC_DIR))
CORS(app)

pose_detector = PoseDetector(complexity=0, detection_con=0.35, track_con=0.35)
tracker = ExerciseTracker(exercise="curl")
nim_coach = NvidiaNimCoach()

latest_nim = {
    "score": 100,
    "form": "GOOD",
    "cue": "Stand in frame or select a demo video to begin.",
    "details": "NVIDIA NIM Cloud Biomechanics ready."
}

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>FitVision — AI Fitness Trainer & Biomechanics</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; -webkit-tap-highlight-color: transparent; }
        body { background: #050811; color: #f8fafc; min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 10px; }
        header { text-align: center; margin-bottom: 6px; width: 100%; max-width: 640px; }
        h1 { font-size: 1.3rem; font-weight: 800; color: #38bdf8; display: flex; align-items: center; justify-content: center; gap: 8px; }
        .badge { font-size: 0.65rem; padding: 2px 8px; border-radius: 9999px; text-transform: uppercase; color: #fff; font-weight: 800; }
        .badge.nim { background: #16a34a; box-shadow: 0 0 10px rgba(22,163,74,0.5); }
        .badge.cv { background: #0284c7; }

        /* Toast notification */
        #toast { position: fixed; top: 14px; background: #0284c7; color: #fff; padding: 8px 16px; border-radius: 9999px; font-weight: 700; font-size: 0.82rem; box-shadow: 0 4px 15px rgba(0,0,0,0.5); z-index: 100; opacity: 0; transition: opacity 0.3s; pointer-events: none; }

        /* Mode Selector Tabs */
        .source-bar { display: flex; gap: 6px; width: 100%; max-width: 640px; margin-bottom: 8px; }
        .tab-btn { flex: 1; padding: 9px 4px; background: #1e293b; border: 1px solid #334155; border-radius: 10px; color: #94a3b8; font-size: 0.78rem; font-weight: 700; cursor: pointer; text-align: center; transition: 0.15s; }
        .tab-btn:active { transform: scale(0.96); }
        .tab-btn.active { background: #0284c7; color: #fff; border-color: #38bdf8; box-shadow: 0 0 10px rgba(2,132,199,0.5); }

        .viewport { position: relative; width: 100%; max-width: 640px; aspect-ratio: 4/3; background: #000; border-radius: 16px; overflow: hidden; box-shadow: 0 12px 35px -5px rgba(0,0,0,0.8); border: 2px solid #1e293b; }
        #video-player { width: 100%; height: 100%; object-fit: cover; display: block; }
        #overlay { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }

        .status-pill { position: absolute; top: 10px; left: 10px; background: rgba(15, 23, 42, 0.85); backdrop-filter: blur(8px); border: 1px solid #334155; padding: 5px 12px; border-radius: 9999px; font-size: 0.72rem; font-weight: 700; color: #38bdf8; display: flex; align-items: center; gap: 6px; z-index: 5; }
        .pulse { width: 8px; height: 8px; border-radius: 50%; background: #22c55e; animation: pulse 1.5s infinite; }
        @keyframes pulse { 0% { opacity: 1; transform: scale(1); } 50% { opacity: 0.4; transform: scale(1.3); } 100% { opacity: 1; transform: scale(1); } }

        /* Hand Orientation & Posture Banner */
        .hand-card { width: 100%; max-width: 640px; background: #172554; border: 1px solid #3b82f6; border-radius: 12px; padding: 10px 14px; margin-top: 8px; display: flex; justify-content: space-between; align-items: center; }
        .hand-title { font-size: 0.82rem; font-weight: 700; color: #93c5fd; }
        .hand-badge { background: #1d4ed8; color: #fff; font-size: 0.82rem; font-weight: 800; padding: 4px 10px; border-radius: 8px; }

        /* Live NVIDIA NIM Coach Card */
        .nim-card { width: 100%; max-width: 640px; background: #0f172a; border: 2px solid #22c55e; border-radius: 12px; padding: 12px; margin-top: 8px; box-shadow: 0 4px 20px rgba(34,197,94,0.15); }
        .nim-header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1e293b; padding-bottom: 6px; margin-bottom: 6px; }
        .nim-title { font-size: 0.85rem; font-weight: 800; color: #4ade80; display: flex; align-items: center; gap: 6px; }
        .nim-score { font-size: 0.9rem; font-weight: 800; color: #facc15; }
        .nim-cue { font-size: 1rem; font-weight: 800; color: #f8fafc; margin-bottom: 4px; line-height: 1.3; }
        .nim-details { font-size: 0.78rem; color: #94a3b8; line-height: 1.3; }

        .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; width: 100%; max-width: 640px; margin-top: 8px; }
        .stat-card { background: #111827; padding: 8px 4px; border-radius: 12px; border: 1px solid #1f2937; text-align: center; }
        .stat-label { font-size: 0.62rem; text-transform: uppercase; color: #9ca3af; font-weight: 700; letter-spacing: 0.5px; }
        .stat-value { font-size: 1.25rem; font-weight: 800; margin-top: 2px; color: #f9fafb; }
        .stat-value.green { color: #4ade80; }
        .stat-value.yellow { color: #facc15; }

        .controls { display: flex; flex-wrap: wrap; gap: 6px; width: 100%; max-width: 640px; margin-top: 8px; }
        .btn { flex: 1; min-width: 100px; padding: 12px 8px; border-radius: 10px; border: none; font-weight: 800; font-size: 0.82rem; cursor: pointer; transition: 0.15s; }
        .btn:active { transform: scale(0.95); opacity: 0.85; }
        .btn-green { background: #16a34a; color: #fff; box-shadow: 0 4px 10px rgba(22,163,74,0.4); }
        .btn-blue { background: #2563eb; color: #fff; }
        .btn-dark { background: #1f2937; color: #f3f4f6; border: 1px solid #374151; }
        .btn-red { background: #dc2626; color: #fff; }

        #file-input { display: none; }
        .footer-note { width: 100%; max-width: 640px; text-align: center; margin-top: 8px; font-size: 0.72rem; color: #64748b; }
    </style>
</head>
<body>
    <div id="toast">Message</div>

    <header>
        <h1>FitVision AI <span class="badge cv">CV Pose + Hands</span> <span class="badge nim">NVIDIA NIM</span></h1>
    </header>

    <!-- Source Selector -->
    <div class="source-bar">
        <button class="tab-btn active" id="tab-curl" onclick="loadVideo('/static/curl.mp4', 'curl', this)">🏋️ Curls Demo</button>
        <button class="tab-btn" id="tab-squat" onclick="loadVideo('/static/squat.mp4', 'squat', this)">🏋️ Squats Demo</button>
        <button class="tab-btn" id="tab-cam" onclick="startCamera(this)">📹 Live Camera</button>
        <button class="tab-btn" id="tab-upload" onclick="document.getElementById('file-input').click()">📁 Upload</button>
    </div>
    <input type="file" id="file-input" accept="video/*,image/*" onchange="handleFileUpload(event)">

    <div class="viewport">
        <video id="video-player" src="/static/curl.mp4" playsinline autoplay loop muted preload="auto"></video>
        <canvas id="overlay"></canvas>
        <div class="status-pill" id="status-pill">
            <div class="pulse" id="status-pulse"></div>
            <span id="cv-status-text">CV: Tracking Active</span>
        </div>
    </div>

    <!-- Hand Orientation & Grip Detection Card (Exp 5 Requirement) -->
    <div class="hand-card">
        <div class="hand-title">✋ Hand Grip & Wrist Orientation (Exp 5):</div>
        <div class="hand-badge" id="hand-orientation-badge">SUPINATED (PALM UP)</div>
    </div>

    <!-- Live NVIDIA NIM Coach Card -->
    <div class="nim-card">
        <div class="nim-header">
            <div class="nim-title">🧠 NVIDIA NIM Biomechanical Audit</div>
            <div class="nim-score" id="nim-score">Score: 100/100 (GOOD)</div>
        </div>
        <div class="nim-cue" id="nim-cue">"Analyzing athlete posture with NVIDIA Llama 3.2 Vision..."</div>
        <div class="nim-details" id="nim-details">Connecting to NVIDIA NIM...</div>
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
            <div class="stat-label">Movement Phase</div>
            <div class="stat-value yellow" id="stat-phase">READY</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Joint Angle</div>
            <div class="stat-value" id="stat-angle">0°</div>
        </div>
    </div>

    <div class="controls">
        <button class="btn btn-green" onclick="triggerNimAudit()">⚡ NVIDIA NIM Audit</button>
        <button class="btn btn-blue" onclick="togglePlayPause()" id="btn-play">⏸ Pause Video</button>
        <button class="btn btn-dark" onclick="toggleExercise()">🔄 Switch Exercise</button>
        <button class="btn btn-red" onclick="resetReps()">↺ Reset Reps</button>
    </div>

    <div class="footer-note">
        Computer Vision practical: 33-point Pose Skeleton (Exp 6), Hand Grip Orientation (Exp 5), FPS HUD (Exp 1), and NVIDIA NIM Biomechanical AI.
    </div>

    <script>
        const video = document.getElementById('video-player');
        const overlay = document.getElementById('overlay');
        const octx = overlay.getContext('2d');

        const sendCanvas = document.createElement('canvas');
        sendCanvas.width = 480;
        sendCanvas.height = 360;
        const sctx = sendCanvas.getContext('2d');

        let inFlight = false;
        let isCamera = false;
        let currentExercise = 'curl';
        let lastFpsTime = performance.now();
        let frameCount = 0;
        let fps = 0;
        let prevReps = 0;

        const POSE_CONNECTIONS = [
            [11, 13], [13, 15], // Left arm
            [12, 14], [14, 16], // Right arm
            [11, 12],           // Shoulders
            [11, 23], [12, 24], // Torso
            [23, 24],           // Hips
            [23, 25], [25, 27], // Left leg
            [24, 26], [26, 28]  // Right leg
        ];

        function showToast(msg) {
            const t = document.getElementById('toast');
            t.innerText = msg;
            t.style.opacity = '1';
            setTimeout(() => { t.style.opacity = '0'; }, 2000);
        }

        function setTabs(activeBtn) {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            if (activeBtn) activeBtn.classList.add('active');
        }

        function stopTracks() {
            if (video.srcObject) {
                video.srcObject.getTracks().forEach(t => t.stop());
                video.srcObject = null;
            }
        }

        function updateOverlaySize() {
            overlay.width = video.videoWidth || 640;
            overlay.height = video.videoHeight || 480;
        }
        video.addEventListener('loadedmetadata', updateOverlaySize);

        async function loadVideo(url, ex, btn) {
            showToast('Loading ' + ex.toUpperCase() + ' Demo...');
            setTabs(btn);
            stopTracks();
            isCamera = false;
            video.style.transform = 'none';
            video.src = url;
            video.loop = true;
            video.muted = true;
            video.load();
            try {
                await video.play();
            } catch (e) {
                console.warn('Play error:', e);
            }
            currentExercise = ex;
            document.getElementById('stat-exercise').innerText = ex.toUpperCase();
            document.getElementById('btn-play').innerText = '⏸ Pause Video';

            await fetch('/set_exercise', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ exercise: ex })
            });
        }

        async function startCamera(btn) {
            showToast('Starting Camera...');
            setTabs(btn);
            stopTracks();
            video.removeAttribute('src');
            video.loop = false;
            video.style.transform = 'scaleX(-1)';

            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
                    audio: false
                });
                video.srcObject = stream;
                await video.play();
                isCamera = true;
                showToast('Camera Active');
            } catch (err) {
                alert('Camera error: ' + err.message);
                loadVideo('/static/curl.mp4', 'curl', document.getElementById('tab-curl'));
            }
        }

        function handleFileUpload(e) {
            const file = e.target.files[0];
            if (!file) return;
            showToast('Loaded ' + file.name);
            setTabs(document.getElementById('tab-upload'));
            stopTracks();
            isCamera = false;
            video.style.transform = 'none';
            video.src = URL.createObjectURL(file);
            video.loop = true;
            video.muted = true;
            video.load();
            video.play();
        }

        function togglePlayPause() {
            const btn = document.getElementById('btn-play');
            if (video.paused) {
                video.play();
                btn.innerText = '⏸ Pause Video';
                showToast('Playing');
            } else {
                video.pause();
                btn.innerText = '▶ Play Video';
                showToast('Paused');
            }
        }

        async function toggleExercise() {
            const nextEx = currentExercise === 'curl' ? 'squat' : 'curl';
            showToast('Switching to ' + nextEx.toUpperCase());
            const tab = nextEx === 'curl' ? document.getElementById('tab-curl') : document.getElementById('tab-squat');
            const url = nextEx === 'curl' ? '/static/curl.mp4' : '/static/squat.mp4';
            if (!isCamera) {
                loadVideo(url, nextEx, tab);
            } else {
                currentExercise = nextEx;
                document.getElementById('stat-exercise').innerText = nextEx.toUpperCase();
                await fetch('/set_exercise', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ exercise: nextEx })
                });
            }
        }

        async function resetReps() {
            showToast('Reps Reset to 0');
            document.getElementById('stat-reps').innerText = '0';
            prevReps = 0;
            await fetch('/reset', { method: 'POST' });
        }

        async function triggerNimAudit() {
            showToast('⚡ Asking NVIDIA NIM...');
            sctx.drawImage(video, 0, 0, sendCanvas.width, sendCanvas.height);
            const base64Data = sendCanvas.toDataURL('image/jpeg', 0.7).split(',')[1];
            try {
                const res = await fetch('/nim_analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: base64Data })
                });
                const data = await res.json();
                if (data.success) {
                    document.getElementById('nim-score').innerText = 'Score: ' + data.score + '/100 (' + data.form + ')';
                    document.getElementById('nim-cue').innerText = '"' + data.cue + '"';
                    document.getElementById('nim-details').innerText = data.details || '';
                    showToast('NVIDIA NIM Audit Complete');
                }
            } catch (e) {
                console.warn('NIM error:', e);
            }
        }

        async function sendCvFrame() {
            if (inFlight || video.readyState < 2) return;
            inFlight = true;

            sctx.drawImage(video, 0, 0, sendCanvas.width, sendCanvas.height);
            const base64Data = sendCanvas.toDataURL('image/jpeg', 0.6).split(',')[1];

            try {
                const res = await fetch('/process_frame', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: base64Data, width: overlay.width, height: overlay.height })
                });
                if (res.ok) {
                    const data = await res.json();
                    renderOverlay(data);
                    document.getElementById('stat-reps').innerText = data.reps;
                    document.getElementById('stat-exercise').innerText = data.exercise.toUpperCase();
                    document.getElementById('stat-phase').innerText = data.stage;
                    document.getElementById('stat-angle').innerText = data.angle + '°';

                    if (data.hand_orientation && data.hand_orientation.label) {
                        document.getElementById('hand-orientation-badge').innerText = data.hand_orientation.label.toUpperCase();
                    }

                    // Auto-trigger NVIDIA NIM audit on completed reps!
                    if (data.reps > prevReps) {
                        prevReps = data.reps;
                        triggerNimAudit();
                    }
                }
            } catch (e) {
                console.warn('CV error:', e);
            } finally {
                inFlight = false;
            }
        }

        function renderOverlay(data) {
            octx.clearRect(0, 0, overlay.width, overlay.height);
            if (!data || !data.landmarks) return;

            const lms = data.landmarks;
            const w = overlay.width;
            const h = overlay.height;

            function mapX(x) {
                return isCamera ? (w - x) : x;
            }

            // Draw skeleton lines (Cyan)
            octx.strokeStyle = '#06b6d4';
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

            // Draw joint circles (Pink)
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

            // Highlight Active Joint & Display Angle (Green)
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

            // Draw Hand Orientation Tag above Wrist (Exp 5)
            if (data.hand_orientation && data.hand_orientation.wrist_pos) {
                const wp = data.hand_orientation.wrist_pos;
                const wx = mapX(wp[0]);
                const wy = wp[1];

                octx.fillStyle = '#3b82f6';
                octx.font = 'bold 14px sans-serif';
                octx.fillText('✋ ' + data.hand_orientation.orientation, wx - 30, wy - 18);
            }

            // Top Status Bar: Local FPS Counter (Exp 1 requirement)
            octx.fillStyle = '#22c55e';
            octx.font = 'bold 14px monospace';
            const detected = Object.keys(lms).length > 0;
            octx.fillText('FPS: ' + fps + ' | ' + (detected ? 'POSTURE LOCKED ✅' : 'SEARCHING...'), 12, 24);

            // Rep Progress Bar along bottom
            const progress = Math.max(0, Math.min(1, (data.progress || 0) / 100));
            octx.fillStyle = '#1e293b';
            octx.fillRect(0, h - 8, w, 8);
            octx.fillStyle = '#22c55e';
            octx.fillRect(0, h - 8, w * progress, 8);
        }

        async function continuousCvLoop() {
            // Measure FPS
            frameCount++;
            const now = performance.now();
            if (now - lastFpsTime >= 1000) {
                fps = frameCount;
                frameCount = 0;
                lastFpsTime = now;
            }

            if (!inFlight && !video.paused) {
                await sendCvFrame();
            }
            setTimeout(continuousCvLoop, 60);
        }

        // Auto-start on load
        window.addEventListener('DOMContentLoaded', () => {
            updateOverlaySize();
            video.play().catch(() => {});
            setTimeout(continuousCvLoop, 600);
        });
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(str(STATIC_DIR), filename)


@app.route("/process_frame", methods=["POST"])
def process_frame():
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

    # 1. Pose Landmark Extraction (Exp 6)
    landmarks_dict = {}
    active_joint = None

    pose_detector.find_pose(frame, draw=False)
    if hasattr(pose_detector, "results") and pose_detector.results and pose_detector.results.pose_landmarks:
        for idx, lm in enumerate(pose_detector.results.pose_landmarks.landmark):
            sx = int(lm.x * target_w)
            sy = int(lm.y * target_h)
            landmarks_dict[idx] = (sx, sy, float(lm.visibility))

    # 2. Update Rep Tracker
    if landmarks_dict:
        status = tracker.update(landmarks_dict)
    else:
        status = tracker._status()

    # Active joint coordinates for visual degree indicator
    if tracker.exercise == "curl":
        active_id = 13 if tracker.active_side == "left" else 14
    else:
        active_id = 25 if tracker.active_side == "left" else 26

    if active_id in landmarks_dict:
        active_joint = [landmarks_dict[active_id][0], landmarks_dict[active_id][1]]

    # 3. Hand Orientation & Grip Detection (Exp 5)
    hand_info = detect_hand_orientation(landmarks_dict, tracker.active_side)

    return jsonify({
        "landmarks": landmarks_dict,
        "active_joint": active_joint,
        "active_side": tracker.active_side,
        "hand_orientation": hand_info,
        "reps": status["reps"],
        "exercise": status["exercise"],
        "stage": status["stage"],
        "feedback": status["feedback"],
        "angle": status["angle"],
        "progress": status["progress"],
    })


@app.route("/nim_analyze", methods=["POST"])
def nim_analyze():
    global latest_nim
    data = request.get_json(force=True)
    img_b64 = data.get("image", "")
    if not img_b64:
        return jsonify({"success": False, "error": "No image provided"}), 400

    result = nim_coach.analyze_frame(img_b64, exercise=tracker.exercise)
    if result.get("success"):
        latest_nim = result
    return jsonify(result)


@app.route("/set_exercise", methods=["POST"])
def set_exercise():
    data = request.get_json(force=True)
    ex = data.get("exercise", "curl")
    tracker.set_exercise(ex)
    return jsonify({"exercise": tracker.exercise})


@app.route("/switch_exercise", methods=["POST"])
def switch_exercise():
    next_ex = "squat" if tracker.exercise == "curl" else "curl"
    tracker.set_exercise(next_ex)
    return jsonify({"exercise": tracker.exercise})


@app.route("/reset", methods=["POST"])
def reset():
    tracker.reps = 0
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

import base64
from pathlib import Path
from flask import Flask, jsonify, render_template_string, request, send_from_directory
from flask_cors import CORS

from nim_gesture_engine import NimGestureEngine
from nim_pose_engine import NimPoseEngine

STATIC_DIR = Path(__file__).parent / "static"
app = Flask(__name__, static_folder=str(STATIC_DIR))
CORS(app)

pose_engine = NimPoseEngine(exercise="curl")
gesture_engine = NimGestureEngine()

gestures_enabled = True
frame_counter = 0

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>FitVision — 100% NVIDIA NIM Vision AI</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background: #050811; color: #f8fafc; min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 12px; }
        header { text-align: center; margin-bottom: 8px; width: 100%; max-width: 640px; }
        h1 { font-size: 1.35rem; font-weight: 800; color: #38bdf8; display: flex; align-items: center; justify-content: center; gap: 8px; }
        .badge { font-size: 0.65rem; padding: 3px 8px; border-radius: 9999px; text-transform: uppercase; color: #fff; font-weight: 800; }
        .badge.nim { background: #16a34a; box-shadow: 0 0 12px rgba(22,163,74,0.6); }

        /* Source selector tabs */
        .source-bar { display: flex; gap: 6px; width: 100%; max-width: 640px; margin-bottom: 8px; overflow-x: auto; }
        .tab-btn { flex: 1; min-width: 110px; padding: 8px 10px; background: #1e293b; border: 1px solid #334155; border-radius: 8px; color: #94a3b8; font-size: 0.75rem; font-weight: 700; cursor: pointer; text-align: center; white-space: nowrap; transition: 0.15s; }
        .tab-btn.active { background: #0284c7; color: #fff; border-color: #38bdf8; }

        .viewport { position: relative; width: 100%; max-width: 640px; aspect-ratio: 4/3; background: #000; border-radius: 16px; overflow: hidden; box-shadow: 0 12px 35px -5px rgba(0,0,0,0.8); border: 2px solid #1e293b; }
        #video-player { width: 100%; height: 100%; object-fit: cover; display: block; }
        .overlay-loader { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; background: rgba(5, 8, 17, 0.95); z-index: 10; gap: 12px; }
        .btn-start { background: #22c55e; color: #000; border: none; padding: 14px 32px; border-radius: 14px; font-size: 1.15rem; font-weight: 800; cursor: pointer; transition: 0.2s; box-shadow: 0 4px 15px rgba(34,197,94,0.5); }
        .btn-start:active { transform: scale(0.96); }

        .status-pill { position: absolute; top: 12px; left: 12px; background: rgba(15, 23, 42, 0.85); backdrop-filter: blur(8px); border: 1px solid #334155; padding: 6px 12px; border-radius: 9999px; font-size: 0.75rem; font-weight: 700; color: #38bdf8; display: flex; align-items: center; gap: 6px; }
        .pulse { width: 8px; height: 8px; border-radius: 50%; background: #22c55e; animation: pulse 1.5s infinite; }
        @keyframes pulse { 0% { opacity: 1; transform: scale(1); } 50% { opacity: 0.4; transform: scale(1.3); } 100% { opacity: 1; transform: scale(1); } }

        /* Posture Assessment Card */
        .nim-card { width: 100%; max-width: 640px; background: #0f172a; border: 2px solid #22c55e; border-radius: 14px; padding: 14px; margin-top: 10px; box-shadow: 0 4px 25px rgba(34,197,94,0.15); }
        .nim-header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1e293b; padding-bottom: 8px; margin-bottom: 10px; }
        .nim-title { font-size: 0.9rem; font-weight: 800; color: #4ade80; display: flex; align-items: center; gap: 6px; }
        .nim-score { font-size: 0.95rem; font-weight: 800; color: #facc15; }
        .nim-cue { font-size: 1.1rem; font-weight: 800; color: #f8fafc; margin-bottom: 6px; line-height: 1.3; }
        .nim-breakdown { font-size: 0.85rem; color: #94a3b8; line-height: 1.4; background: #1e293b; padding: 8px 10px; border-radius: 8px; margin-top: 6px; }

        .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; width: 100%; max-width: 640px; margin-top: 10px; }
        .stat-card { background: #111827; padding: 8px 6px; border-radius: 12px; border: 1px solid #1f2937; text-align: center; }
        .stat-label { font-size: 0.65rem; text-transform: uppercase; color: #9ca3af; font-weight: 700; letter-spacing: 0.5px; }
        .stat-value { font-size: 1.3rem; font-weight: 800; margin-top: 2px; color: #f9fafb; }
        .stat-value.green { color: #4ade80; }
        .stat-value.yellow { color: #facc15; }

        .controls { display: flex; flex-wrap: wrap; gap: 8px; width: 100%; max-width: 640px; margin-top: 10px; }
        .btn { flex: 1; min-width: 110px; padding: 12px 8px; border-radius: 10px; border: none; font-weight: 700; font-size: 0.85rem; cursor: pointer; transition: 0.15s; }
        .btn-blue { background: #2563eb; color: #fff; }
        .btn-green { background: #16a34a; color: #fff; font-weight: 800; }
        .btn-dark { background: #1f2937; color: #f3f4f6; border: 1px solid #374151; }
        .btn-red { background: #dc2626; color: #fff; }
        .btn:active { transform: scale(0.97); }

        #file-input { display: none; }
        .footer-note { width: 100%; max-width: 640px; text-align: center; margin-top: 10px; font-size: 0.75rem; color: #64748b; }
    </style>
</head>
<body>
    <header>
        <h1>FitVision AI <span class="badge nim">100% NVIDIA NIM Cloud</span></h1>
    </header>

    <!-- Source Selector (Perfect for classroom demo) -->
    <div class="source-bar">
        <button class="tab-btn active" id="tab-curl" onclick="selectDemoVideo('/static/curl.mp4', 'curl', this)">🏋️ Demo Curls</button>
        <button class="tab-btn" id="tab-squat" onclick="selectDemoVideo('/static/squat.mp4', 'squat', this)">🏋️ Demo Squats</button>
        <button class="tab-btn" id="tab-upload" onclick="triggerFileUpload(this)">📁 Upload Video</button>
        <button class="tab-btn" id="tab-camera" onclick="selectLiveCamera(this)">📹 Live Camera</button>
    </div>
    <input type="file" id="file-input" accept="video/*,image/*" onchange="handleFileUpload(event)">

    <div class="viewport">
        <video id="video-player" playsinline autoplay loop muted></video>
        <div class="status-pill" id="status-pill">
            <div class="pulse" id="status-pulse"></div>
            <span id="nim-status-text">NVIDIA NIM: Initializing...</span>
        </div>
        <div class="overlay-loader" id="loader">
            <button class="btn-start" onclick="initStart()">▶ Start AI Analysis</button>
            <p style="color: #9ca3af; font-size: 0.85rem;">All frames evaluated directly by NVIDIA Llama 3.2 Vision</p>
        </div>
    </div>

    <!-- Live NVIDIA NIM Coach Card -->
    <div class="nim-card">
        <div class="nim-header">
            <div class="nim-title">🧠 NVIDIA NIM Biomechanical Posture Capture</div>
            <div class="nim-score" id="nim-score">Score: 100/100</div>
        </div>
        <div class="nim-cue" id="nim-cue">"Select a demo video or start live camera."</div>
        <div class="nim-breakdown" id="nim-breakdown">Ready for cloud AI inference via NVIDIA NIM API.</div>
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
            <div class="stat-label">Posture Rating</div>
            <div class="stat-value" id="stat-form" style="color: #38bdf8;">GOOD</div>
        </div>
    </div>

    <div class="controls">
        <button class="btn btn-green" onclick="snapScan()">⚡ Scan Posture Now</button>
        <button class="btn btn-blue" onclick="togglePlayPause()" id="btn-play">⏸ Pause Video</button>
        <button class="btn btn-dark" onclick="switchExercise()">🔄 Switch Exercise</button>
        <button class="btn btn-red" onclick="resetReps()">↺ Reset Reps</button>
    </div>

    <div class="footer-note">
        100% Powered by NVIDIA NIM Cloud (<code>meta/llama-3.2-11b-vision-instruct</code>). Both Pose Biomechanics & Touchless Gestures processed via API.
    </div>

    <script>
        const video = document.getElementById('video-player');
        const loader = document.getElementById('loader');

        const sendCanvas = document.createElement('canvas');
        sendCanvas.width = 480;
        sendCanvas.height = 360;
        const sctx = sendCanvas.getContext('2d');

        let active = false;
        let inFlight = false;
        let isLiveCam = false;
        let currentMode = 'demo-curl';

        // Start with Demo Curls by default so classroom evaluation works instantly!
        function initStart() {
            loader.style.display = 'none';
            selectDemoVideo('/static/curl.mp4', 'curl', document.getElementById('tab-curl'));
        }

        function setTabs(activeBtn) {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            if (activeBtn) activeBtn.classList.add('active');
        }

        async function selectDemoVideo(src, ex, btn) {
            setTabs(btn);
            stopCameraTracks();
            isLiveCam = false;
            video.style.transform = 'none';
            video.srcObject = null;
            video.src = src;
            video.loop = true;
            await video.play();
            active = true;

            await fetch('/set_exercise', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ exercise: ex })
            });
            document.getElementById('stat-exercise').innerText = ex.toUpperCase();
            document.getElementById('nim-status-text').innerText = 'NVIDIA NIM: Active (Demo Video)';
            nimLoop();
        }

        function stopCameraTracks() {
            if (video.srcObject) {
                video.srcObject.getTracks().forEach(t => t.stop());
                video.srcObject = null;
            }
        }

        async function selectLiveCamera(btn) {
            setTabs(btn);
            stopCameraTracks();
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
                isLiveCam = true;
                active = true;
                document.getElementById('nim-status-text').innerText = 'NVIDIA NIM: Active (Live Camera)';
                nimLoop();
            } catch (err) {
                alert('Camera access error: ' + err.message);
            }
        }

        function triggerFileUpload(btn) {
            document.getElementById('file-input').click();
        }

        async function handleFileUpload(e) {
            const file = e.target.files[0];
            if (!file) return;
            setTabs(document.getElementById('tab-upload'));
            stopCameraTracks();
            isLiveCam = false;
            video.style.transform = 'none';
            video.srcObject = null;
            video.src = URL.createObjectURL(file);
            video.loop = true;
            await video.play();
            active = true;
            document.getElementById('nim-status-text').innerText = 'NVIDIA NIM: Active (Uploaded File)';
            nimLoop();
        }

        function togglePlayPause() {
            const btn = document.getElementById('btn-play');
            if (video.paused) {
                video.play();
                btn.innerText = '⏸ Pause Video';
            } else {
                video.pause();
                btn.innerText = '▶ Play Video';
            }
        }

        async function snapScan() {
            if (!active) return alert('Start video or camera first!');
            await sendFrameToNim();
        }

        async function sendFrameToNim() {
            if (!active || video.readyState < 2) return;
            inFlight = true;
            document.getElementById('status-pulse').style.background = '#38bdf8';
            document.getElementById('nim-status-text').innerText = 'NVIDIA NIM: Analyzing frame...';

            sctx.drawImage(video, 0, 0, sendCanvas.width, sendCanvas.height);
            const base64Data = sendCanvas.toDataURL('image/jpeg', 0.65).split(',')[1];

            try {
                const res = await fetch('/process_nim', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: base64Data })
                });
                if (res.ok) {
                    const data = await res.json();
                    updateDashboard(data);
                    document.getElementById('status-pulse').style.background = '#22c55e';
                    document.getElementById('nim-status-text').innerText = data.person_detected ? 'NVIDIA NIM: Posture Captured' : 'NVIDIA NIM: Searching Body';
                }
            } catch (e) {
                console.warn('NIM request error:', e);
                document.getElementById('status-pulse').style.background = '#ef4444';
                document.getElementById('nim-status-text').innerText = 'NVIDIA NIM: Retrying...';
            } finally {
                inFlight = false;
            }
        }

        async function nimLoop() {
            if (!active) return;
            if (!inFlight && !video.paused) {
                await sendFrameToNim();
            }
            setTimeout(nimLoop, 300);
        }

        function updateDashboard(data) {
            document.getElementById('stat-reps').innerText = data.reps;
            document.getElementById('stat-exercise').innerText = data.exercise.toUpperCase();
            document.getElementById('stat-phase').innerText = data.phase;
            document.getElementById('stat-form').innerText = data.form;

            const formColor = data.form === 'EXCELLENT' || data.form === 'GOOD' ? '#4ade80' : (data.form === 'POOR' ? '#ef4444' : '#38bdf8');
            document.getElementById('stat-form').style.color = formColor;

            document.getElementById('nim-score').innerText = 'Score: ' + data.score + '/100 (' + data.form + ')';
            document.getElementById('nim-cue').innerText = '"' + data.cue + '"';
            document.getElementById('nim-breakdown').innerText = 'Posture Assessment: ' + (data.breakdown || 'Posture captured');
        }

        async function switchExercise() {
            const res = await fetch('/switch_exercise', { method: 'POST' });
            const data = await res.json();
            document.getElementById('stat-exercise').innerText = data.exercise.toUpperCase();
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


@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(str(STATIC_DIR), filename)


@app.route("/process_nim", methods=["POST"])
def process_nim():
    global frame_counter

    data = request.get_json(force=True)
    img_b64 = data.get("image", "")
    if not img_b64:
        return jsonify({"error": "No image"}), 400

    frame_counter += 1

    # 1. Pose, movement phase, form & rep analysis (100% via NVIDIA NIM)
    pose_result = pose_engine.analyze_frame(img_b64)

    return jsonify({
        "exercise": pose_result["exercise"],
        "reps": pose_result["reps"],
        "phase": pose_result["phase"],
        "form": pose_result["form"],
        "score": pose_result["score"],
        "cue": pose_result["cue"],
        "breakdown": pose_result.get("breakdown", "Posture captured"),
        "person_detected": pose_result.get("person_detected", False),
    })


@app.route("/set_exercise", methods=["POST"])
def set_exercise():
    data = request.get_json(force=True)
    ex = data.get("exercise", "curl")
    pose_engine.set_exercise(ex)
    return jsonify({"exercise": pose_engine.exercise})


@app.route("/switch_exercise", methods=["POST"])
def switch_exercise():
    next_ex = "squat" if pose_engine.exercise == "curl" else "curl"
    pose_engine.set_exercise(next_ex)
    return jsonify({"exercise": pose_engine.exercise})


@app.route("/reset", methods=["POST"])
def reset():
    pose_engine.reset_reps()
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

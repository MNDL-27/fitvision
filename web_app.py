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

frame_counter = 0

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>FitVision — 100% NVIDIA NIM Vision AI</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; -webkit-tap-highlight-color: transparent; }
        body { background: #050811; color: #f8fafc; min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 12px; }
        header { text-align: center; margin-bottom: 8px; width: 100%; max-width: 640px; }
        h1 { font-size: 1.35rem; font-weight: 800; color: #38bdf8; display: flex; align-items: center; justify-content: center; gap: 8px; }
        .badge { font-size: 0.65rem; padding: 3px 8px; border-radius: 9999px; text-transform: uppercase; color: #fff; font-weight: 800; }
        .badge.nim { background: #16a34a; box-shadow: 0 0 12px rgba(22,163,74,0.6); }

        /* Toast notifications for button feedback */
        #toast { position: fixed; top: 16px; background: #0284c7; color: #fff; padding: 8px 16px; border-radius: 9999px; font-weight: 700; font-size: 0.85rem; box-shadow: 0 4px 15px rgba(0,0,0,0.5); z-index: 100; opacity: 0; transition: opacity 0.3s; pointer-events: none; }

        /* Mode Selector Tabs */
        .source-bar { display: flex; gap: 6px; width: 100%; max-width: 640px; margin-bottom: 8px; }
        .tab-btn { flex: 1; padding: 10px 6px; background: #1e293b; border: 1px solid #334155; border-radius: 10px; color: #94a3b8; font-size: 0.8rem; font-weight: 700; cursor: pointer; text-align: center; transition: 0.15s; }
        .tab-btn:active { transform: scale(0.96); }
        .tab-btn.active { background: #0284c7; color: #fff; border-color: #38bdf8; box-shadow: 0 0 12px rgba(2,132,199,0.5); }

        .viewport { position: relative; width: 100%; max-width: 640px; aspect-ratio: 4/3; background: #000; border-radius: 16px; overflow: hidden; box-shadow: 0 12px 35px -5px rgba(0,0,0,0.8); border: 2px solid #1e293b; }
        #video-player { width: 100%; height: 100%; object-fit: cover; display: block; }

        .status-pill { position: absolute; top: 12px; left: 12px; background: rgba(15, 23, 42, 0.85); backdrop-filter: blur(8px); border: 1px solid #334155; padding: 6px 12px; border-radius: 9999px; font-size: 0.75rem; font-weight: 700; color: #38bdf8; display: flex; align-items: center; gap: 6px; z-index: 5; }
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
        .btn { flex: 1; min-width: 110px; padding: 14px 10px; border-radius: 12px; border: none; font-weight: 800; font-size: 0.88rem; cursor: pointer; transition: 0.15s; }
        .btn:active { transform: scale(0.95); opacity: 0.85; }
        .btn-green { background: #16a34a; color: #fff; box-shadow: 0 4px 12px rgba(22,163,74,0.4); }
        .btn-blue { background: #2563eb; color: #fff; }
        .btn-dark { background: #1f2937; color: #f3f4f6; border: 1px solid #374151; }
        .btn-red { background: #dc2626; color: #fff; }

        #file-input { display: none; }
        .footer-note { width: 100%; max-width: 640px; text-align: center; margin-top: 10px; font-size: 0.75rem; color: #64748b; }
    </style>
</head>
<body>
    <div id="toast">Message</div>

    <header>
        <h1>FitVision AI <span class="badge nim">100% NVIDIA NIM Cloud</span></h1>
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
        <!-- Autoplays immediately on page load! -->
        <video id="video-player" src="/static/curl.mp4" playsinline autoplay loop muted preload="auto"></video>
        <div class="status-pill" id="status-pill">
            <div class="pulse" id="status-pulse"></div>
            <span id="nim-status-text">NVIDIA NIM: In Sync</span>
        </div>
    </div>

    <!-- Live NVIDIA NIM Coach Card -->
    <div class="nim-card">
        <div class="nim-header">
            <div class="nim-title">🧠 NVIDIA NIM Biomechanical Posture Capture</div>
            <div class="nim-score" id="nim-score">Score: 100/100</div>
        </div>
        <div class="nim-cue" id="nim-cue">"Analyzing athlete posture with NVIDIA Llama 3.2 Vision..."</div>
        <div class="nim-breakdown" id="nim-breakdown">Connecting to NVIDIA NIM...</div>
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
        <button class="btn btn-dark" onclick="toggleExercise()">🔄 Switch Exercise</button>
        <button class="btn btn-red" onclick="resetReps()">↺ Reset Reps</button>
    </div>

    <div class="footer-note">
        Powered by NVIDIA NIM Cloud (<code>meta/llama-3.2-11b-vision-instruct</code>). Both Pose Biomechanics & Touchless Gestures processed via API.
    </div>

    <script>
        const video = document.getElementById('video-player');
        const sendCanvas = document.createElement('canvas');
        sendCanvas.width = 480;
        sendCanvas.height = 360;
        const sctx = sendCanvas.getContext('2d');

        let inFlight = false;
        let isCamera = false;
        let currentExercise = 'curl';

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

        async function loadVideo(url, ex, btn) {
            showToast('Loading ' + ex.toUpperCase() + ' Demo...');
            setTabs(btn);
            stopTracks();
            isCamera = false;
            video.style.transform = 'none';
            video.src = url;
            video.loop = true;
            video.muted = true;
            video.load(); // CRITICAL for mobile Safari / iOS!
            try {
                await video.play();
            } catch (e) {
                console.warn('Autoplay caught:', e);
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
                alert('Camera access error: ' + err.message);
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
            await fetch('/reset', { method: 'POST' });
        }

        async function snapScan() {
            showToast('⚡ Asking NVIDIA NIM...');
            await sendFrameToNim();
        }

        async function sendFrameToNim() {
            if (inFlight || video.readyState < 2) return;
            inFlight = true;
            document.getElementById('status-pulse').style.background = '#38bdf8';
            document.getElementById('nim-status-text').innerText = 'NVIDIA NIM: Analyzing...';

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
                    document.getElementById('nim-status-text').innerText = data.person_detected ? 'NVIDIA NIM: Posture Captured' : 'NVIDIA NIM: In Sync';
                }
            } catch (e) {
                console.warn('NIM request error:', e);
                document.getElementById('status-pulse').style.background = '#ef4444';
                document.getElementById('nim-status-text').innerText = 'NVIDIA NIM: Retrying...';
            } finally {
                inFlight = false;
            }
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

        async function continuousNimLoop() {
            if (!inFlight && !video.paused) {
                await sendFrameToNim();
            }
            setTimeout(continuousNimLoop, 300);
        }

        // Start continuous loop automatically!
        window.addEventListener('DOMContentLoaded', () => {
            video.play().catch(() => {});
            setTimeout(continuousNimLoop, 800);
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

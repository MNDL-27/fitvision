import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

NIM_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "meta/llama-3.2-11b-vision-instruct"


def get_api_key() -> str:
    key = os.environ.get("NVIDIA_API_KEY", "")
    if key:
        return key.strip()
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("NVIDIA_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


API_KEY = get_api_key()

# Server state
current_exercise = "curl"
reps = 0
last_phase = "READY"
last_cue = "Stand in frame to begin"
last_score = 100
last_form = "READY"
last_flaws = []

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

        .viewport { position: relative; width: 100%; max-width: 640px; aspect-ratio: 4/3; background: #000; border-radius: 16px; overflow: hidden; box-shadow: 0 12px 35px -5px rgba(0,0,0,0.8); border: 2px solid #1e293b; }
        #webcam { width: 100%; height: 100%; object-fit: cover; display: block; transform: scaleX(-1); }
        .overlay-loader { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; background: rgba(5, 8, 17, 0.95); z-index: 10; gap: 12px; }
        .btn-start { background: #22c55e; color: #000; border: none; padding: 14px 32px; border-radius: 14px; font-size: 1.15rem; font-weight: 800; cursor: pointer; transition: 0.2s; box-shadow: 0 4px 15px rgba(34,197,94,0.5); }
        .btn-start:active { transform: scale(0.96); }

        .status-pill { position: absolute; top: 12px; left: 12px; background: rgba(15, 23, 42, 0.85); backdrop-filter: blur(8px); border: 1px solid #334155; padding: 6px 12px; border-radius: 9999px; font-size: 0.75rem; font-weight: 700; color: #38bdf8; display: flex; align-items: center; gap: 6px; }
        .pulse { width: 8px; height: 8px; border-radius: 50%; background: #22c55e; animation: pulse 1.5s infinite; }
        @keyframes pulse { 0% { opacity: 1; transform: scale(1); } 50% { opacity: 0.4; transform: scale(1.3); } 100% { opacity: 1; transform: scale(1); } }

        /* Real-Time NVIDIA Coach Card */
        .nim-card { width: 100%; max-width: 640px; background: #0f172a; border: 1px solid #22c55e; border-radius: 14px; padding: 14px; margin-top: 10px; box-shadow: 0 4px 25px rgba(34,197,94,0.15); }
        .nim-header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1e293b; padding-bottom: 8px; margin-bottom: 10px; }
        .nim-title { font-size: 0.9rem; font-weight: 800; color: #4ade80; display: flex; align-items: center; gap: 6px; }
        .nim-score { font-size: 0.95rem; font-weight: 800; color: #facc15; }
        .nim-cue { font-size: 1.05rem; font-weight: 800; color: #f8fafc; margin-bottom: 6px; line-height: 1.3; }
        .nim-flaws { font-size: 0.8rem; color: #f87171; font-weight: 600; display: flex; gap: 6px; flex-wrap: wrap; }
        .flaw-tag { background: rgba(239, 68, 68, 0.15); border: 1px solid #ef4444; padding: 2px 8px; border-radius: 6px; }

        .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; width: 100%; max-width: 640px; margin-top: 10px; }
        .stat-card { background: #111827; padding: 8px 6px; border-radius: 12px; border: 1px solid #1f2937; text-align: center; }
        .stat-label { font-size: 0.65rem; text-transform: uppercase; color: #9ca3af; font-weight: 700; letter-spacing: 0.5px; }
        .stat-value { font-size: 1.3rem; font-weight: 800; margin-top: 2px; color: #f9fafb; }
        .stat-value.green { color: #4ade80; }
        .stat-value.yellow { color: #facc15; }

        .controls { display: flex; flex-wrap: wrap; gap: 8px; width: 100%; max-width: 640px; margin-top: 10px; }
        .btn { flex: 1; min-width: 120px; padding: 12px 8px; border-radius: 10px; border: none; font-weight: 700; font-size: 0.85rem; cursor: pointer; transition: 0.15s; }
        .btn-blue { background: #2563eb; color: #fff; }
        .btn-dark { background: #1f2937; color: #f3f4f6; border: 1px solid #374151; }
        .btn-red { background: #dc2626; color: #fff; }
        .btn:active { transform: scale(0.97); }

        .footer-note { width: 100%; max-width: 640px; text-align: center; margin-top: 10px; font-size: 0.75rem; color: #64748b; }
    </style>
</head>
<body>
    <header>
        <h1>FitVision AI <span class="badge nim">100% NVIDIA NIM Cloud</span></h1>
    </header>

    <div class="viewport">
        <video id="webcam" playsinline autoplay muted></video>
        <div class="status-pill" id="status-pill">
            <div class="pulse" id="status-pulse"></div>
            <span id="nim-status-text">NVIDIA NIM: Connecting...</span>
        </div>
        <div class="overlay-loader" id="loader">
            <button class="btn-start" onclick="startCamera()">📷 Launch Camera</button>
            <p style="color: #9ca3af; font-size: 0.85rem;">All frames evaluated directly by NVIDIA Llama 3.2 Vision</p>
        </div>
    </div>

    <!-- Live NVIDIA NIM Coach Card -->
    <div class="nim-card">
        <div class="nim-header">
            <div class="nim-title">⚡ NVIDIA Biomechanical Analysis</div>
            <div class="nim-score" id="nim-score">Score: 100/100</div>
        </div>
        <div class="nim-cue" id="nim-cue">"Stand in front of the camera and begin your set."</div>
        <div class="nim-flaws" id="nim-flaws"></div>
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
            <div class="stat-label">AI Phase</div>
            <div class="stat-value yellow" id="stat-phase">READY</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Form Rating</div>
            <div class="stat-value" id="stat-form" style="color: #38bdf8;">GOOD</div>
        </div>
    </div>

    <div class="controls">
        <button class="btn btn-blue" onclick="switchExercise()">🔄 Switch Exercise</button>
        <button class="btn btn-dark" onclick="flipCamera()">📷 Flip Camera</button>
        <button class="btn btn-red" onclick="resetReps()">↺ Reset Reps</button>
    </div>

    <div class="footer-note">
        Powered by NVIDIA NIM Cloud Microservice (<code>meta/llama-3.2-11b-vision-instruct</code>). Pure neural vision without local heuristics.
    </div>

    <script>
        const video = document.getElementById('webcam');
        const loader = document.getElementById('loader');

        const sendCanvas = document.createElement('canvas');
        sendCanvas.width = 480;
        sendCanvas.height = 360;
        const sctx = sendCanvas.getContext('2d');

        let active = false;
        let inFlight = false;
        let currentFacingMode = 'user';

        async function startCamera() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: currentFacingMode, width: { ideal: 640 }, height: { ideal: 480 } },
                    audio: false
                });
                video.srcObject = stream;
                await video.play();
                active = true;
                loader.style.display = 'none';
                document.getElementById('nim-status-text').innerText = 'NVIDIA NIM: Active';
                nimLoop();
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

        async function nimLoop() {
            if (!active) return;

            if (!inFlight && video.readyState >= 2) {
                inFlight = true;
                document.getElementById('status-pulse').style.background = '#38bdf8';
                document.getElementById('nim-status-text').innerText = 'NVIDIA NIM: Analyzing frame...';

                // Grab frame
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
                        document.getElementById('nim-status-text').innerText = 'NVIDIA NIM: In Sync';
                    }
                } catch (e) {
                    console.warn('NIM request error:', e);
                    document.getElementById('status-pulse').style.background = '#ef4444';
                    document.getElementById('nim-status-text').innerText = 'NVIDIA NIM: Retrying...';
                } finally {
                    inFlight = false;
                }
            }

            // Continuous loop
            setTimeout(nimLoop, 200);
        }

        function updateDashboard(data) {
            document.getElementById('stat-reps').innerText = data.reps;
            document.getElementById('stat-exercise').innerText = data.exercise.toUpperCase();
            document.getElementById('stat-phase').innerText = data.phase;
            document.getElementById('stat-form').innerText = data.form;

            const formColor = data.form === 'EXCELLENT' || data.form === 'GOOD' ? '#4ade80' : '#ef4444';
            document.getElementById('stat-form').style.color = formColor;

            document.getElementById('nim-score').innerText = 'Score: ' + data.score + '/100 (' + data.form + ')';
            document.getElementById('nim-cue').innerText = '"' + data.cue + '"';

            const flawsEl = document.getElementById('nim-flaws');
            flawsEl.innerHTML = '';
            if (data.flaws && data.flaws.length) {
                data.flaws.forEach(f => {
                    if (f && f.trim()) {
                        const span = document.createElement('span');
                        span.className = 'flaw-tag';
                        span.innerText = '⚠️ ' + f;
                        flawsEl.appendChild(span);
                    }
                });
            }
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


@app.route("/process_nim", methods=["POST"])
def process_nim():
    global reps, last_phase, last_cue, last_score, last_form, last_flaws

    data = request.get_json(force=True)
    img_b64 = data.get("image", "")
    if not img_b64:
        return jsonify({"error": "No image"}), 400

    prompt = (
        f"You are the primary Computer Vision engine evaluating fitness exercise: {current_exercise.upper()}.\n"
        "Analyze this user's frame.\n"
        "Assess:\n"
        "1. Phase: 'UP', 'DOWN', or 'IN_BETWEEN'\n"
        "2. Form: 'EXCELLENT', 'GOOD', or 'POOR'\n"
        "3. Score: integer 0-100\n"
        "4. Cue: under 12 words direct coaching tip\n"
        "5. Flaws: list any defects (e.g. elbow flare, arched back, knees caving) or None"
    )

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]
            }
        ],
        "temperature": 0.1,
        "max_tokens": 150
    }

    req = urllib.request.Request(
        NIM_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
            "User-Agent": "FitVision-NVIDIA-NIM-Pure/1.0"
        }
    )

    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"].strip()

            phase = "IN_BETWEEN"
            form = "GOOD"
            score = 85
            cue = "Keep posture aligned and steady"
            flaws = []

            # 1. Try regex extraction
            phase_m = re.search(r"Phase\s*\**:\s*\**\s*([A-Za-z_]+)", content, re.I)
            form_m = re.search(r"Form\s*\**:\s*\**\s*(?:The user'?s form is\s+)?(EXCELLENT|GOOD|POOR|FAIR)", content, re.I)
            if not form_m:
                form_m = re.search(r"\b(EXCELLENT|GOOD|POOR|FAIR)\b", content, re.I)
            score_m = re.search(r"Score\s*\**:\s*\**\s*(\d+)", content, re.I)
            cue_m = re.search(r"Cue\s*\**:\s*\**\s*[\"\']?([^\"\n\r\.\;]+)[\"\']?", content, re.I)

            if phase_m:
                phase = phase_m.group(1).upper()
            if form_m:
                form = form_m.group(1).upper()
            if score_m:
                score = int(score_m.group(1))
            if cue_m:
                cue = cue_m.group(1).strip()

            flaws_m = re.search(r"Flaws\s*\**:\s*\**\s*([^\n\r]+)", content, re.I)
            if flaws_m and "none" not in flaws_m.group(1).lower():
                flaws = [f.strip() for f in flaws_m.group(1).split(",") if f.strip()]

            # Rep counting logic based on NVIDIA NIM's phase transitions:
            # Curl: DOWN -> UP -> DOWN = 1 rep
            # Squat: UP -> DOWN -> UP = 1 rep
            if current_exercise == "curl":
                if phase == "UP" and last_phase == "DOWN":
                    last_phase = "UP"
                elif phase == "DOWN" and last_phase == "UP":
                    reps += 1
                    last_phase = "DOWN"
                elif phase in ("UP", "DOWN"):
                    last_phase = phase
            elif current_exercise == "squat":
                if phase == "DOWN" and last_phase == "UP":
                    last_phase = "DOWN"
                elif phase == "UP" and last_phase == "DOWN":
                    reps += 1
                    last_phase = "UP"
                elif phase in ("UP", "DOWN"):
                    last_phase = phase

            last_form = form
            last_score = score
            last_cue = cue
            last_flaws = flaws

    except Exception as e:
        print(f"[NIM LOG] {e}")

    return jsonify({
        "exercise": current_exercise,
        "reps": reps,
        "phase": last_phase,
        "form": last_form,
        "score": last_score,
        "cue": last_cue,
        "flaws": last_flaws,
    })


@app.route("/switch_exercise", methods=["POST"])
def switch_exercise():
    global current_exercise, last_phase
    current_exercise = "squat" if current_exercise == "curl" else "curl"
    last_phase = "READY"
    return jsonify({"exercise": current_exercise})


@app.route("/reset", methods=["POST"])
def reset():
    global reps
    reps = 0
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

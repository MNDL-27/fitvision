import base64
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, jsonify, render_template_string, request, send_from_directory
from flask_cors import CORS

from exercise_tracker import ExerciseTracker
from movement_analyzer import analyze_full_body_movement
from nvidia_nim import NvidiaNimCoach
from pose_detector import PoseDetector
from smooth_motion_engine import HysteresisExerciseEngine, LandmarkSmoother, MovementDebouncer

STATIC_DIR = Path(__file__).parent / "static"
app = Flask(__name__, static_folder=str(STATIC_DIR))
CORS(app)

# BlazePose Full (complexity=1) for complete 33-point body tracking
pose_detector = PoseDetector(complexity=1, detection_con=0.35, track_con=0.35)

# MediaPipe Hands (Exp 5: Hand Tracking) for all 5 fingers & 21 landmarks
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.35,
    min_tracking_confidence=0.35,
)
crop_hands_detector = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.25,
)

# Smooth Motion & Hysteresis Engines
landmark_smoother = LandmarkSmoother(alpha=0.75)
hysteresis_engine = HysteresisExerciseEngine(exercise="curl")
movement_debouncer = MovementDebouncer(window_size=5)
tracker = ExerciseTracker(exercise="curl")
nim_coach = NvidiaNimCoach()

latest_nim = {
    "score": 100,
    "form": "GOOD",
    "cue": "Stand in frame or select a demo video to begin.",
    "details": "NVIDIA NIM Cloud Biomechanics ready.",
}

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>FitVision — 60 FPS Fluid Motion & Pose Analysis</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; -webkit-tap-highlight-color: transparent; }
        body { background: #050811; color: #f8fafc; min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 10px; }
        header { text-align: center; margin-bottom: 6px; width: 100%; max-width: 640px; }
        h1 { font-size: 1.25rem; font-weight: 800; color: #38bdf8; display: flex; align-items: center; justify-content: center; gap: 8px; }
        .badge { font-size: 0.65rem; padding: 2px 8px; border-radius: 9999px; text-transform: uppercase; color: #fff; font-weight: 800; }
        .badge.nim { background: #16a34a; box-shadow: 0 0 10px rgba(22,163,74,0.5); }
        .badge.cv { background: #0284c7; }

        #toast { position: fixed; top: 14px; background: #0284c7; color: #fff; padding: 8px 16px; border-radius: 9999px; font-weight: 700; font-size: 0.82rem; box-shadow: 0 4px 15px rgba(0,0,0,0.5); z-index: 100; opacity: 0; transition: opacity 0.3s; pointer-events: none; }

        /* Mode Selector Tabs */
        .source-bar { display: flex; gap: 6px; width: 100%; max-width: 640px; margin-bottom: 8px; }
        .tab-btn { flex: 1; padding: 9px 4px; background: #1e293b; border: 1px solid #334155; border-radius: 10px; color: #94a3b8; font-size: 0.78rem; font-weight: 700; cursor: pointer; text-align: center; transition: 0.15s; }
        .tab-btn:active { transform: scale(0.96); }
        .tab-btn.active { background: #0284c7; color: #fff; border-color: #38bdf8; box-shadow: 0 0 10px rgba(2,132,199,0.5); }

        .viewport { position: relative; width: 100%; max-width: 640px; margin: 0 auto; background: #000; border-radius: 16px; overflow: hidden; box-shadow: 0 12px 35px -5px rgba(0,0,0,0.8); border: 2px solid #1e293b; }
        #video-player { width: 100%; height: auto; max-height: 65vh; display: block; }
        #overlay { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }

        .status-pill { position: absolute; top: 10px; left: 10px; background: rgba(15, 23, 42, 0.85); backdrop-filter: blur(8px); border: 1px solid #334155; padding: 5px 12px; border-radius: 9999px; font-size: 0.72rem; font-weight: 700; color: #38bdf8; display: flex; align-items: center; gap: 6px; z-index: 5; }
        .pulse { width: 8px; height: 8px; border-radius: 50%; background: #22c55e; animation: pulse 1.5s infinite; }
        @keyframes pulse { 0% { opacity: 1; transform: scale(1); } 50% { opacity: 0.4; transform: scale(1.3); } 100% { opacity: 1; transform: scale(1); } }

        /* Full Movement Live Card */
        .movement-card { width: 100%; max-width: 640px; background: #0f172a; border: 1px solid #38bdf8; border-radius: 12px; padding: 10px 14px; margin-top: 8px; }
        .movement-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
        .movement-title { font-size: 0.75rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; }
        .movement-name { font-size: 1.05rem; font-weight: 800; color: #38bdf8; }
        .movement-cue { font-size: 0.82rem; color: #e2e8f0; font-weight: 600; margin-top: 2px; }

        /* Angles Grid */
        .angles-bar { display: flex; gap: 6px; width: 100%; max-width: 640px; margin-top: 6px; }
        .angle-chip { flex: 1; background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 6px; text-align: center; font-size: 0.72rem; }
        .angle-chip b { display: block; font-size: 0.95rem; color: #4ade80; }

        /* Live NVIDIA NIM Coach Card */
        .nim-card { width: 100%; max-width: 640px; background: #0f172a; border: 2px solid #22c55e; border-radius: 12px; padding: 12px; margin-top: 8px; box-shadow: 0 4px 20px rgba(34,197,94,0.15); }
        .nim-header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1e293b; padding-bottom: 6px; margin-bottom: 6px; }
        .nim-title { font-size: 0.85rem; font-weight: 800; color: #4ade80; display: flex; align-items: center; gap: 6px; }
        .nim-score { font-size: 0.9rem; font-weight: 800; color: #facc15; }
        .nim-cue { font-size: 0.95rem; font-weight: 800; color: #f8fafc; margin-bottom: 4px; line-height: 1.3; }
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
        <h1>FitVision AI <span class="badge cv">60 FPS Fluid Motion</span> <span class="badge nim">NVIDIA NIM</span></h1>
    </header>

    <!-- Source Selector -->
    <div class="source-bar">
        <button class="tab-btn active" id="tab-curl" onclick="loadVideo('/static/curl_clean.mp4', 'curl', this)">🏋️ Curls Demo</button>
        <button class="tab-btn" id="tab-squat" onclick="loadVideo('/static/squat.mp4', 'squat', this)">🏋️ Squats Demo</button>
        <button class="tab-btn" id="tab-cam" onclick="startCamera(this)">📹 Live Camera</button>
        <button class="tab-btn" id="tab-upload" onclick="document.getElementById('file-input').click()">📁 Upload</button>
    </div>
    <input type="file" id="file-input" accept="video/*,image/*" onchange="handleFileUpload(event)">

    <div class="viewport">
        <video id="video-player" src="/static/curl_clean.mp4" playsinline autoplay loop muted preload="auto"></video>
        <canvas id="overlay"></canvas>
        <div class="status-pill" id="status-pill">
            <div class="pulse" id="status-pulse"></div>
            <span id="cv-status-text">CV: Fluid 60 FPS Tracking</span>
        </div>
    </div>

    <!-- Real-time Body Movement Identification -->
    <div class="movement-card">
        <div class="movement-header">
            <div class="movement-title">Identified Body Movement:</div>
            <span style="font-size: 0.75rem; color: #4ade80;" id="movement-hand">Hand: 5-Finger Articulation</span>
        </div>
        <div class="movement-name" id="movement-name">Standing / Ready</div>
        <div class="movement-cue" id="movement-cue">Maintain good posture</div>
    </div>

    <!-- Live Multi-Joint Angles Bar -->
    <div class="angles-bar">
        <div class="angle-chip">L Elbow<b id="ang-larm">0°</b></div>
        <div class="angle-chip">R Elbow<b id="ang-rarm">0°</b></div>
        <div class="angle-chip">L Knee<b id="ang-lknee">0°</b></div>
        <div class="angle-chip">R Knee<b id="ang-rknee">0°</b></div>
        <div class="angle-chip">Torso<b id="ang-torso">0°</b></div>
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
            <div class="stat-label">Active Angle</div>
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
        Decoupled 60 FPS LERP Client Rendering + 33-point Pose & 21-point Hand Tracking + NVIDIA NIM Biomechanics.
    </div>

    <script>
        const video = document.getElementById('video-player');
        const overlay = document.getElementById('overlay');
        const octx = overlay.getContext('2d');

        const sendCanvas = document.createElement('canvas');
        const sctx = sendCanvas.getContext('2d');

        let inFlight = false;
        let isCamera = false;
        let currentExercise = 'curl';
        let prevReps = 0;
        let latestMovement = null;

        // Smooth Interpolation State (Decouples Rendering from Network Latency!)
        let currentLandmarks = {};
        let targetLandmarks = {};
        let currentHands = [];
        let targetHands = [];
        let displayAngle = 0.0;
        let targetAngle = 0.0;
        let displayProgress = 0.0;
        let targetProgress = 0.0;

        let lastFpsTime = performance.now();
        let frameCount = 0;
        let fps = 60;

        // Full clean natural skeleton: robust face connections + body bones
        const FULL_SKELETON_CONNECTIONS = [
            // Head & Neck (Clean anatomical alignment, no spiderweb!)
            [7, 2], [8, 5],     // Ears to Eyes
            [2, 0], [5, 0],     // Eyes to Nose
            [9, 10],            // Mouth
            [0, 11], [0, 12],   // Neck to Shoulders
            // Shoulders & Arms
            [11, 12],           // Collar / Shoulders
            [11, 13], [13, 15], // Left arm: shoulder -> elbow -> wrist
            [12, 14], [14, 16], // Right arm: shoulder -> elbow -> wrist
            // Torso & Spine
            [11, 23], [12, 24], // Torso sides
            [23, 24],           // Hips / Pelvis
            // Legs (ONLY drawn when in frame)
            [23, 25], [25, 27], // Left leg: hip -> knee -> ankle
            [24, 26], [26, 28], // Right leg: hip -> knee -> ankle
            // Feet
            [27, 29], [27, 31], [29, 31],
            [28, 30], [28, 32], [30, 32]
        ];

        // 21-point Hand Skeleton: ALL 5 FINGERS (Exp 5)
        const HAND_CONNECTIONS = [
            [0, 1], [1, 2], [2, 3], [3, 4],        // Thumb
            [0, 5], [5, 6], [6, 7], [7, 8],        // Index
            [0, 9], [9, 10], [10, 11], [11, 12],   // Middle
            [0, 13], [13, 14], [14, 15], [15, 16], // Ring
            [0, 17], [17, 18], [18, 19], [19, 20], // Pinky
            [5, 9], [9, 13], [13, 17]              // Knuckles
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

        function syncDimensions() {
            overlay.width = video.videoWidth || 640;
            overlay.height = video.videoHeight || 480;
            sendCanvas.width = 480;
            sendCanvas.height = Math.round(480 * ((video.videoHeight || 480) / (video.videoWidth || 640)));
        }
        video.addEventListener('loadedmetadata', syncDimensions);
        window.addEventListener('resize', syncDimensions);

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
            syncDimensions();
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
                syncDimensions();
                isCamera = true;
                showToast('Camera Active');
            } catch (err) {
                alert('Camera error: ' + err.message);
                loadVideo('/static/curl_clean.mp4', 'curl', document.getElementById('tab-curl'));
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
            syncDimensions();
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
            const url = nextEx === 'curl' ? '/static/curl_clean.mp4' : '/static/squat.mp4';
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

        // Network pipeline: sends frames and receives latest targets
        async function sendCvFrame() {
            if (inFlight || video.readyState < 2) return;
            inFlight = true;

            sctx.drawImage(video, 0, 0, sendCanvas.width, sendCanvas.height);
            const base64Data = sendCanvas.toDataURL('image/jpeg', 0.65).split(',')[1];

            try {
                const res = await fetch('/process_frame', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: base64Data, width: overlay.width, height: overlay.height })
                });
                if (res.ok) {
                    const data = await res.json();
                    targetLandmarks = data.landmarks || {};
                    targetHands = data.hands || [];
                    targetAngle = data.angle || 0.0;
                    targetProgress = data.progress || 0.0;

                    document.getElementById('stat-reps').innerText = data.reps;
                    document.getElementById('stat-exercise').innerText = data.exercise.toUpperCase();
                    document.getElementById('stat-phase').innerText = data.stage;

                    if (data.movement) {
                        latestMovement = data.movement;
                        document.getElementById('movement-name').innerText = data.movement.movement;
                        document.getElementById('movement-cue').innerText = data.movement.posture_cue;
                        const handDesc = (data.hands && data.hands.length > 0) ? 'All 5 Fingers Articulated (' + data.hands.length + ' Hand' + (data.hands.length > 1 ? 's' : '') + ')' : data.movement.hand_state;
                        document.getElementById('movement-hand').innerText = 'Hand: ' + handDesc;
                        document.getElementById('ang-larm').innerText = (data.movement.left_arm !== null ? data.movement.left_arm + '°' : '--');
                        document.getElementById('ang-rarm').innerText = (data.movement.right_arm !== null ? data.movement.right_arm + '°' : '--');
                        document.getElementById('ang-lknee').innerText = (data.movement.left_knee !== null ? data.movement.left_knee + '°' : '--');
                        document.getElementById('ang-rknee').innerText = (data.movement.right_knee !== null ? data.movement.right_knee + '°' : '--');
                        document.getElementById('ang-torso').innerText = (data.movement.torso_angle !== null ? data.movement.torso_angle + '°' : '--');
                    }

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

        // DECOUPLED 60 FPS RENDER LOOP WITH LERP (Linear Interpolation)
        function render60Fps() {
            // Measure actual client rendering FPS
            frameCount++;
            const now = performance.now();
            if (now - lastFpsTime >= 1000) {
                fps = frameCount;
                frameCount = 0;
                lastFpsTime = now;
            }

            octx.clearRect(0, 0, overlay.width, overlay.height);

            // Interpolate Angle & Progress ONLY if genuinely visible
            if (targetAngle !== null && targetAngle !== undefined) {
                displayAngle += (targetAngle - displayAngle) * 0.35;
                document.getElementById('stat-angle').innerText = Math.round(displayAngle) + '°';
            } else {
                displayAngle = null;
                document.getElementById('stat-angle').innerText = '--';
            }
            displayProgress += (targetProgress - displayProgress) * 0.35;

            // Interpolate Landmark Coordinates (LERP) - strictly purge vanished/occluded joints
            for (const id in currentLandmarks) {
                if (!targetLandmarks[id] || targetLandmarks[id][2] < 0.55) {
                    delete currentLandmarks[id];
                }
            }

            const lerpFactor = 0.42;
            for (const id in targetLandmarks) {
                const target = targetLandmarks[id];
                if (target[2] >= 0.55) {
                    if (!currentLandmarks[id]) {
                        currentLandmarks[id] = [target[0], target[1], target[2]];
                    } else {
                        currentLandmarks[id][0] += (target[0] - currentLandmarks[id][0]) * lerpFactor;
                        currentLandmarks[id][1] += (target[1] - currentLandmarks[id][1]) * lerpFactor;
                        currentLandmarks[id][2] = target[2];
                    }
                }
            }

            // Interpolate Hand Coordinates
            if (targetHands && targetHands.length > 0) {
                if (currentHands.length !== targetHands.length) {
                    currentHands = JSON.parse(JSON.stringify(targetHands));
                } else {
                    for (let h = 0; h < targetHands.length; h++) {
                        for (let p = 0; p < targetHands[h].length; p++) {
                            currentHands[h][p][0] += (targetHands[h][p][0] - currentHands[h][p][0]) * lerpFactor;
                            currentHands[h][p][1] += (targetHands[h][p][1] - currentHands[h][p][1]) * lerpFactor;
                        }
                    }
                }
            } else {
                currentHands = [];
            }

            const w = overlay.width;
            const h = overlay.height;
            function mapX(x) {
                return isCamera ? (w - x) : x;
            }

            const lms = currentLandmarks;

            // 1. Draw Clean Anatomical Bones (ONLY when visibility >= 0.55)
            octx.strokeStyle = '#06b6d4';
            octx.lineWidth = 3.5;
            octx.lineCap = 'round';
            octx.lineJoin = 'round';

            for (const [p1, p2] of FULL_SKELETON_CONNECTIONS) {
                if (lms[p1] && lms[p2]) {
                    const vis1 = lms[p1][2] !== undefined ? lms[p1][2] : 1.0;
                    const vis2 = lms[p2][2] !== undefined ? lms[p2][2] : 1.0;
                    const isHeadConnection = (p1 <= 10 || p2 <= 10);
                    const minThresh = isHeadConnection ? 0.65 : 0.55;

                    if (vis1 >= minThresh && vis2 >= minThresh) {
                        const isSpine = (p1 === 11 && p2 === 23) || (p1 === 12 && p2 === 24);
                        if (isSpine && latestMovement && latestMovement.torso_angle > 35) {
                            octx.strokeStyle = '#ef4444'; // Red alert on bad spine posture!
                            octx.lineWidth = 5.0;
                        } else {
                            octx.strokeStyle = '#06b6d4';
                            octx.lineWidth = 3.5;
                        }
                        octx.beginPath();
                        octx.moveTo(mapX(lms[p1][0]), lms[p1][1]);
                        octx.lineTo(mapX(lms[p2][0]), lms[p2][1]);
                        octx.stroke();
                    }
                }
            }

            // 2. Draw ONLY Genuinely Detected Landmark Dots (vis >= 0.55)
            for (const id_str in lms) {
                const id = parseInt(id_str);
                const pt = lms[id];
                const vis = pt[2] !== undefined ? pt[2] : 1.0;
                const minThresh = id <= 10 ? 0.65 : 0.55;

                if (vis >= minThresh) {
                    const px = mapX(pt[0]);
                    const py = pt[1];

                    octx.fillStyle = '#ffffff';
                    octx.beginPath();
                    octx.arc(px, py, 5.5, 0, 2 * Math.PI);
                    octx.fill();

                    if (id <= 10) octx.fillStyle = '#facc15';      // Face: Yellow
                    else if (id <= 16) octx.fillStyle = '#ec4899'; // Arms & Shoulders: Pink
                    else if (id <= 22) octx.fillStyle = '#38bdf8'; // Hands: Cyan
                    else if (id <= 28) octx.fillStyle = '#22c55e'; // Hips & Knees: Green
                    else octx.fillStyle = '#f97316';               // Feet: Orange

                    octx.beginPath();
                    octx.arc(px, py, 3.5, 0, 2 * Math.PI);
                    octx.fill();
                }
            }

            // 3. Draw All 5 Fingers on Hands (Exp 5)
            if (currentHands && currentHands.length > 0) {
                for (const hand of currentHands) {
                    octx.strokeStyle = '#38bdf8';
                    octx.lineWidth = 2.5;
                    for (const [p1, p2] of HAND_CONNECTIONS) {
                        if (hand[p1] && hand[p2]) {
                            octx.beginPath();
                            octx.moveTo(mapX(hand[p1][0]), hand[p1][1]);
                            octx.lineTo(mapX(hand[p2][0]), hand[p2][1]);
                            octx.stroke();
                        }
                    }

                    hand.forEach((pt, idx) => {
                        const hx = mapX(pt[0]);
                        const hy = pt[1];
                        const isTip = [4, 8, 12, 16, 20].includes(idx);
                        octx.fillStyle = isTip ? '#22c55e' : '#facc15';
                        octx.beginPath();
                        octx.arc(hx, hy, isTip ? 4.5 : 3, 0, 2 * Math.PI);
                        octx.fill();
                        octx.strokeStyle = '#ffffff';
                        octx.lineWidth = 1;
                        octx.stroke();
                    });
                }
            }

            // 4. Highlight Active Joint with Angle Text (ONLY if joint is visible and angle exists)
            const activeId = currentExercise === 'curl' ? 13 : 25;
            if (displayAngle !== null && lms[activeId] && lms[activeId][2] >= 0.55) {
                const jx = mapX(lms[activeId][0]);
                const jy = lms[activeId][1];

                octx.fillStyle = '#22c55e';
                octx.beginPath();
                octx.arc(jx, jy, 11, 0, 2 * Math.PI);
                octx.fill();
                octx.strokeStyle = '#ffffff';
                octx.lineWidth = 2;
                octx.stroke();

                octx.font = 'bold 16px sans-serif';
                octx.fillStyle = '#ffffff';
                octx.fillText(Math.round(displayAngle) + '°', jx + 14, jy - 8);
            }

            // 5. FPS & Locking Status
            octx.fillStyle = '#22c55e';
            octx.font = 'bold 14px monospace';
            const count = Object.keys(lms).length;
            octx.fillText('DISPLAY: ' + fps + ' FPS | 33 POINTS ACTIVE (' + count + ' TRACKED)', 12, 24);

            // 6. Smooth Progress Bar
            const prog = Math.max(0, Math.min(1, displayProgress / 100));
            octx.fillStyle = '#1e293b';
            octx.fillRect(0, h - 8, w, 8);
            octx.fillStyle = '#22c55e';
            octx.fillRect(0, h - 8, w * prog, 8);

            requestAnimationFrame(render60Fps);
        }

        // Background network frame sender (runs around 20-30 times/sec)
        async function networkLoop() {
            if (!inFlight && !video.paused) {
                await sendCvFrame();
            }
            setTimeout(networkLoop, 35);
        }

        window.addEventListener('DOMContentLoaded', () => {
            syncDimensions();
            video.play().catch(() => {});
            requestAnimationFrame(render60Fps);
            setTimeout(networkLoop, 500);
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

    h, w = frame.shape[:2]

    # 1. Full 33-point Pose Landmark Extraction (Exp 6)
    raw_landmarks = {}
    active_joint = None

    pose_detector.find_pose(frame, draw=False)
    if hasattr(pose_detector, "results") and pose_detector.results and pose_detector.results.pose_landmarks:
        for idx, lm in enumerate(pose_detector.results.pose_landmarks.landmark):
            sx = int(lm.x * target_w)
            sy = int(lm.y * target_h)
            raw_landmarks[idx] = (sx, sy, float(lm.visibility))

    # Apply Landmark Smoother
    landmarks_dict = landmark_smoother.smooth(raw_landmarks)
    print(f"[DEBUG] frame={frame.shape} raw={len(raw_landmarks)} smoothed={len(landmarks_dict)}")

    # 2. Complete 21-point Hand & 5-Finger Tracking (Exp 5)
    all_hands = []
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h_res = hands_detector.process(rgb)

    if h_res and h_res.multi_hand_landmarks:
        for hand in h_res.multi_hand_landmarks:
            all_hands.append([[int(lm.x * target_w), int(lm.y * target_h)] for lm in hand.landmark])
    elif hasattr(pose_detector, "results") and pose_detector.results and pose_detector.results.pose_landmarks:
        for w_idx in [15, 16]:
            rw = pose_detector.results.pose_landmarks.landmark[w_idx]
            if rw.visibility > 0.3:
                cx, cy = int(rw.x * w), int(rw.y * h)
                pad = int(min(h, w) * 0.22)
                y1, y2 = max(0, cy - pad), min(h, cy + pad)
                x1, x2 = max(0, cx - pad), min(w, cx + pad)
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0 and (y2 - y1 > 30) and (x2 - x1 > 30):
                    crop_up = cv2.resize(crop, (224, 224))
                    c_res = crop_hands_detector.process(cv2.cvtColor(crop_up, cv2.COLOR_BGR2RGB))
                    if c_res and c_res.multi_hand_landmarks:
                        for hand in c_res.multi_hand_landmarks:
                            pts = []
                            for lm in hand.landmark:
                                fx = int((x1 + lm.x * (x2 - x1)) / w * target_w)
                                fy = int((y1 + lm.y * (y2 - y1)) / h * target_h)
                                pts.append([fx, fy])
                            all_hands.append(pts)

    # 3. Comprehensive Movement Analysis & Debouncing
    raw_movement = analyze_full_body_movement(landmarks_dict, tracker.exercise)
    stable_movement_name = movement_debouncer.update(raw_movement.get("movement", "Ready"))
    raw_movement["movement"] = stable_movement_name

    # 4. Hysteresis Exercise Rep Counting
    active_angle = raw_movement.get("active_angle")
    is_gesture = raw_movement.get("is_gesture", False)
    stage, reps, feedback = hysteresis_engine.update(active_angle, is_gesture)

    # Calculate smooth progress percentage ONLY if angle exists
    if active_angle is not None:
        if tracker.exercise == "curl":
            progress = float(np.clip(np.interp(active_angle, (45, 140), (100, 0)), 0, 100))
        else:
            progress = float(np.clip(np.interp(active_angle, (95, 155), (100, 0)), 0, 100))
        angle_out = round(active_angle, 1)
    else:
        progress = 0.0
        angle_out = None

    return jsonify({
        "landmarks": landmarks_dict,
        "hands": all_hands,
        "active_side": tracker.active_side,
        "movement": raw_movement,
        "reps": reps,
        "cadence": hysteresis_engine.last_rep_duration,
        "rep_history": list(hysteresis_engine.rep_history),
        "exercise": tracker.exercise,
        "stage": stage,
        "feedback": feedback,
        "angle": angle_out,
        "progress": round(progress, 1),
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
    hysteresis_engine.set_exercise(ex)
    return jsonify({"exercise": tracker.exercise})


@app.route("/switch_exercise", methods=["POST"])
def switch_exercise():
    next_ex = "squat" if tracker.exercise == "curl" else "curl"
    tracker.set_exercise(next_ex)
    hysteresis_engine.set_exercise(next_ex)
    return jsonify({"exercise": tracker.exercise})


@app.route("/reset", methods=["POST"])
def reset():
    tracker.reps = 0
    hysteresis_engine.reps = 0
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

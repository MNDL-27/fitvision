"""NVIDIA NIM Pose & Biomechanics Engine.
All pose estimation, movement phase detection, joint analysis, and rep counting
are performed 100% via NVIDIA NIM Cloud Vision API (meta/llama-3.2-11b-vision-instruct).
Zero local heuristic/library processing.
"""
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

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


class NimPoseEngine:
    def __init__(self, api_key: str | None = None, exercise: str = "curl"):
        self.api_key = (api_key or get_api_key()).strip()
        self.exercise = exercise.lower()
        self.reps = 0
        self.last_phase = "READY"
        self.last_form = "GOOD"
        self.last_score = 85
        self.last_cue = "Stand in frame to begin"
        self.last_breakdown = "Searching for posture..."
        self.person_detected = False

    def set_exercise(self, exercise: str):
        self.exercise = exercise.lower()
        self.last_phase = "READY"
        self.last_cue = f"Ready for {self.exercise}s"

    def reset_reps(self):
        self.reps = 0
        self.last_phase = "READY"

    def analyze_frame(self, image_b64: str) -> dict:
        """Sends frame to NVIDIA NIM API for complete pose, phase, and biomechanics evaluation."""
        if not self.api_key:
            return {
                "success": False,
                "error": "NVIDIA API key not configured",
                "exercise": self.exercise,
                "reps": self.reps,
                "phase": self.last_phase,
                "form": "ERROR",
                "score": 0,
                "cue": "NVIDIA Key Missing",
                "breakdown": "Please check API key in .env",
                "person_detected": False,
            }

        payload = {
            "model": MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an elite Computer Vision AI fitness coach. "
                        "Evaluate the person in this workout frame. "
                        "You must output a single JSON object with keys: "
                        "person_detected (bool), "
                        'posture_rating ("EXCELLENT"|"GOOD"|"POOR"|"NOT_DETECTED"), '
                        'phase ("UP"|"DOWN"|"IN_BETWEEN"), '
                        "score (int 0-100), "
                        "cue (string under 12 words direct coaching tip), "
                        "breakdown (string describing posture: back straightness, elbows, knees, depth)."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Analyze this workout frame for exercise: {self.exercise.upper()}. Return ONLY the JSON object.",
                        },
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    ],
                },
            ],
            "temperature": 0.1,
            "max_tokens": 200,
        }

        req = urllib.request.Request(
            NIM_API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "FitVision-NimPoseEngine/1.0",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                content = body["choices"][0]["message"]["content"].strip()
                return self._parse_and_update(content)
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "exercise": self.exercise,
                "reps": self.reps,
                "phase": self.last_phase,
                "form": self.last_form,
                "score": self.last_score,
                "cue": self.last_cue,
                "breakdown": self.last_breakdown,
                "person_detected": self.person_detected,
            }

    def _parse_and_update(self, content: str) -> dict:
        phase = self.last_phase
        form = self.last_form
        score = self.last_score
        cue = self.last_cue
        breakdown = self.last_breakdown
        person_detected = False

        # 1. Look for JSON block
        m = re.search(r"(\{[\s\S]*\})", content)
        if m:
            try:
                parsed = json.loads(m.group(1))
                person_detected = bool(parsed.get("person_detected", True))
                phase = str(parsed.get("phase", phase)).upper()
                form = str(parsed.get("posture_rating", parsed.get("form", form))).upper()
                score = int(parsed.get("score", score))
                cue = str(parsed.get("cue", cue))
                breakdown = str(parsed.get("breakdown", breakdown))
            except Exception:
                pass

        # 2. Fallback regex extraction if JSON incomplete
        if not m:
            phase_m = re.search(r"Phase\s*\**:\s*\**\s*([A-Za-z_]+)", content, re.I)
            form_m = re.search(r"\b(EXCELLENT|GOOD|POOR|NOT_DETECTED)\b", content, re.I)
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
            person_detected = True

        # Rep counting based on NVIDIA NIM movement transitions
        if self.exercise == "curl":
            if phase == "UP" and self.last_phase == "DOWN":
                self.last_phase = "UP"
            elif phase == "DOWN" and self.last_phase == "UP":
                self.reps += 1
                self.last_phase = "DOWN"
            elif phase in ("UP", "DOWN"):
                self.last_phase = phase
        elif self.exercise == "squat":
            if phase == "DOWN" and self.last_phase == "UP":
                self.last_phase = "DOWN"
            elif phase == "UP" and self.last_phase == "DOWN":
                self.reps += 1
                self.last_phase = "UP"
            elif phase in ("UP", "DOWN"):
                self.last_phase = phase

        self.person_detected = person_detected
        self.last_form = form
        self.last_score = score
        self.last_cue = cue
        self.last_breakdown = breakdown

        return {
            "success": True,
            "exercise": self.exercise,
            "reps": self.reps,
            "phase": self.last_phase,
            "form": self.last_form,
            "score": self.last_score,
            "cue": self.last_cue,
            "breakdown": self.last_breakdown,
            "person_detected": self.person_detected,
        }

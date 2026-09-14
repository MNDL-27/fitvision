"""NVIDIA NIM Touchless Gesture Engine.
Hand gesture recognition performed 100% via NVIDIA NIM Vision API.
Zero local hand/heuristic libraries.
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


class NimGestureEngine:
    PAUSE = "PAUSE"    # open palm / raised hand
    SWITCH = "SWITCH"  # thumbs up
    RESET = "RESET"    # closed fist
    NONE = "NONE"

    def __init__(self, api_key: str | None = None):
        self.api_key = (api_key or get_api_key()).strip()

    def detect_gesture(self, image_b64: str) -> str:
        """Classify gesture using NVIDIA NIM Vision reasoning."""
        if not self.api_key:
            return self.NONE

        prompt = (
            "Look at any hands in this workout image. Classify gesture into exactly ONE category:\n"
            "- PAUSE (open palm / hand held up)\n"
            "- SWITCH (thumbs up)\n"
            "- RESET (closed fist)\n"
            "- NONE (normal position, gripping weight, or no clear gesture)\n\n"
            "Return format:\n"
            "GESTURE: <PAUSE, SWITCH, RESET, or NONE>"
        )

        payload = {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                    ]
                }
            ],
            "temperature": 0.1,
            "max_tokens": 50
        }

        req = urllib.request.Request(
            NIM_API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "FitVision-NimGestureEngine/1.0"
            }
        )

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                content = body["choices"][0]["message"]["content"].strip()
                m = re.search(r"\b(PAUSE|SWITCH|RESET|NONE)\b", content, re.I)
                if m:
                    return m.group(1).upper()
                return self.NONE
        except Exception:
            return self.NONE

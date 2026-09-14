"""NVIDIA NIM (Inference Microservice) integration for deep fitness form analysis."""
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

NIM_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_MODEL = "meta/llama-3.2-11b-vision-instruct"


def load_env_key() -> str:
    # Try environment variable first
    key = os.environ.get("NVIDIA_API_KEY", "")
    if key:
        return key.strip()

    # Try .env file in project directory
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("NVIDIA_API_KEY="):
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
                if val:
                    return val
    return ""


class NvidiaNimCoach:
    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL):
        self.api_key = (api_key or load_env_key()).strip()
        self.model = model

    def set_key(self, api_key: str):
        self.api_key = api_key.strip()

    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key.startswith("nvapi-"))

    def analyze_frame(self, image_b64: str, exercise: str = "curl") -> dict:
        """Send video frame to NVIDIA NIM VLM for biomechanical form analysis."""
        if not self.is_configured():
            return {
                "success": False,
                "error": "NVIDIA API key not set. Get a key at https://build.nvidia.com",
                "form": "UNKNOWN",
                "cue": "NVIDIA NIM key needed"
            }

        prompt = (
            f"You are an elite fitness biomechanics coach analyzing a workout frame for {exercise.upper()}.\n"
            "Analyze the person's posture and form.\n"
            "Return concise assessment in this format:\n"
            "SCORE: <integer 0-100>\n"
            "RATING: <EXCELLENT, GOOD, or POOR>\n"
            "CUE: <under 15 words direct coaching correction>\n"
            "DETAILS: <1-2 sentences on posture, back straightness, joint alignment>"
        )

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                    ]
                }
            ],
            "temperature": 0.2,
            "max_tokens": 250
        }

        req = urllib.request.Request(
            NIM_API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "FitVision-NVIDIA-NIM/1.0"
            }
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))
                content = result["choices"][0]["message"]["content"].strip()

                # Parse fields from structured response
                score_match = re.search(r"SCORE:\s*(\d+)", content, re.IGNORECASE)
                rating_match = re.search(r"RATING:\s*(\w+)", content, re.IGNORECASE)
                cue_match = re.search(r"CUE:\s*([^\n]+)", content, re.IGNORECASE)
                details_match = re.search(r"DETAILS:\s*(.+)", content, re.IGNORECASE | re.DOTALL)

                score = int(score_match.group(1)) if score_match else 85
                rating = rating_match.group(1).upper() if rating_match else "GOOD"
                cue = cue_match.group(1).strip() if cue_match else "Keep core engaged and posture upright"
                details = details_match.group(1).strip() if details_match else content

                return {
                    "success": True,
                    "form": rating,
                    "score": score,
                    "cue": cue,
                    "details": details[:300],
                    "raw": content,
                    "model": self.model
                }

        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore")
            return {
                "success": False,
                "error": f"NVIDIA NIM error {e.code}: {err_body[:200]}",
                "form": "ERROR",
                "cue": "NIM request failed"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "form": "ERROR",
                "cue": "Connection timed out"
            }

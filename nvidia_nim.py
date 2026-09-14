"""NVIDIA NIM (Inference Microservice) integration for deep fitness form analysis."""
import json
import os
import urllib.error
import urllib.request

NIM_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_MODEL = "meta/llama-3.2-11b-vision-instruct"


class NvidiaNimCoach:
    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL):
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY", "")
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
                "error": "NVIDIA API key not set. Get a free key at https://build.nvidia.com",
                "form": "UNKNOWN",
                "cue": "NVIDIA NIM Key needed"
            }

        prompt = (
            f"You are an elite biomechanics and fitness coach. Analyze this workout frame for exercise: {exercise.upper()}.\n"
            "Assess:\n"
            "1. Posture & joint alignment\n"
            "2. Form defects (e.g., lower back arching, elbow drift/flaring, knees caving inward, incomplete depth/ROM)\n"
            "3. Actionable coaching cue under 15 words\n\n"
            "Return STRICT JSON only:\n"
            "{\n"
            '  "form_rating": "EXCELLENT" | "GOOD" | "POOR",\n'
            '  "issues": ["list of defects or empty"],\n'
            '  "cue": "Short punchy audio/visual cue for the athlete",\n'
            '  "score": 0-100\n'
            "}"
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
            "max_tokens": 300
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
            with urllib.request.urlopen(req, timeout=12) as response:
                result = json.loads(response.read().decode("utf-8"))
                content = result["choices"][0]["message"]["content"].strip()

                # Extract JSON if enclosed in markdown ```json ... ```
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()

                try:
                    parsed = json.loads(content)
                    return {
                        "success": True,
                        "form": parsed.get("form_rating", "GOOD"),
                        "issues": parsed.get("issues", []),
                        "cue": parsed.get("cue", "Maintain steady cadence"),
                        "score": parsed.get("score", 90),
                        "model": self.model
                    }
                except json.JSONDecodeError:
                    return {
                        "success": True,
                        "form": "GOOD",
                        "cue": content[:120],
                        "model": self.model
                    }

        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore")
            return {
                "success": False,
                "error": f"NVIDIA NIM API error {e.code}: {err_body[:200]}",
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

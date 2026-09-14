import unittest
from unittest.mock import MagicMock, patch
from nim_pose_engine import NimPoseEngine
from nim_gesture_engine import NimGestureEngine


class TestNimEngines(unittest.TestCase):
    def test_nim_pose_engine_parsing(self):
        engine = NimPoseEngine(api_key="nvapi-mock")

        mock_content = (
            "Based on the camera feed:\n"
            "Phase: UP\n"
            "Form: EXCELLENT\n"
            "Score: 92\n"
            "Cue: Pin elbows to your torso\n"
            "Flaws: None"
        )
        res = engine._parse_and_update(mock_content)
        self.assertEqual(res["phase"], "UP")
        self.assertEqual(res["form"], "EXCELLENT")
        self.assertEqual(res["score"], 92)
        self.assertEqual(res["cue"], "Pin elbows to your torso")
        self.assertEqual(res["flaws"], [])

    def test_nim_rep_counting_state_machine(self):
        engine = NimPoseEngine(api_key="nvapi-mock", exercise="curl")
        self.assertEqual(engine.reps, 0)

        # 1. Down position
        engine._parse_and_update("Phase: DOWN\nForm: GOOD\nScore: 85")
        self.assertEqual(engine.last_phase, "DOWN")
        self.assertEqual(engine.reps, 0)

        # 2. Curl up
        engine._parse_and_update("Phase: UP\nForm: GOOD\nScore: 85")
        self.assertEqual(engine.last_phase, "UP")
        self.assertEqual(engine.reps, 0)

        # 3. Return down -> rep complete!
        engine._parse_and_update("Phase: DOWN\nForm: GOOD\nScore: 85")
        self.assertEqual(engine.last_phase, "DOWN")
        self.assertEqual(engine.reps, 1)

    def test_nim_gesture_classification(self):
        engine = NimGestureEngine(api_key="nvapi-mock")
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = b'{"choices":[{"message":{"content":"GESTURE: PAUSE"}}]}'
            mock_urlopen.return_value.__enter__.return_value = mock_resp

            gesture = engine.detect_gesture("dummy_base64")
            self.assertEqual(gesture, "PAUSE")


if __name__ == "__main__":
    unittest.main()

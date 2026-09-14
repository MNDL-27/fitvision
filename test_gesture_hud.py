"""Unit tests for GestureController classification and HUDRenderer drawing."""
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from gesture_controller import GestureController
from hud_renderer import HUDRenderer


def make_results(finger_states):
    """finger_states: list of 5 bools (thumb..pinky extended?)."""
    lms = []
    for _ in range(21):
        lms.append(SimpleNamespace(x=0.5, y=0.5, z=0.0))
    tips = GestureController.TIPS
    pips = GestureController.PIPS
    for i in range(5):
        if finger_states[i]:
            lms[tips[i]].y = 0.3
            lms[pips[i]].y = 0.6
        else:
            lms[tips[i]].y = 0.8
            lms[pips[i]].y = 0.6
    # Thumb folds sideways: extended -> tip.x further right than pip
    lms[4].x = 0.8 if finger_states[0] else 0.2
    lms[3].x = 0.5
    return SimpleNamespace(multi_hand_landmarks=[SimpleNamespace(landmark=lms)])


class TestGestureClassification(unittest.TestCase):
    def setUp(self):
        self.gc = GestureController.__new__(GestureController)

    def _gesture_for(self, fingers):
        self.gc.results = make_results(fingers)
        return self.gc.get_gesture()

    def test_open_palm_is_pause(self):
        self.assertEqual(self._gesture_for([1, 1, 1, 1, 1]), 'PAUSE')

    def test_thumbs_up_is_switch(self):
        self.assertEqual(self._gesture_for([1, 0, 0, 0, 0]), 'SWITCH')

    def test_fist_is_reset(self):
        self.assertEqual(self._gesture_for([0, 0, 0, 0, 0]), 'RESET')

    def test_no_results_is_none(self):
        self.gc.results = SimpleNamespace(multi_hand_landmarks=[])
        self.assertEqual(self.gc.get_gesture(), 'NONE')

    def test_fresh_instance_is_none(self):
        gc = GestureController.__new__(GestureController)
        self.assertEqual(gc.get_gesture(), 'NONE')

    def test_other_combo_is_none(self):
        self.assertEqual(self._gesture_for([0, 1, 0, 0, 0]), 'NONE')

    def test_find_hands_no_hand(self):
        gc = GestureController.__new__(GestureController)
        gc.hands = mock.Mock()
        gc.hands.process.return_value = SimpleNamespace(
            multi_hand_landmarks=None)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        res = gc.find_hands(img)
        self.assertIsNone(res.multi_hand_landmarks)
        self.assertEqual(gc.get_gesture(), 'NONE')


class TestHUDRenderer(unittest.TestCase):
    def setUp(self):
        self.renderer = HUDRenderer()
        self.img = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_draw_fps_no_error(self):
        out = self.renderer.draw_fps(self.img.copy())
        self.assertEqual(out.shape, self.img.shape)

    def test_draw_dashboard_no_error(self):
        out = self.renderer.draw_dashboard(
            self.img.copy(), 'Bicep Curl', 10, 'up', 'Good form', 45.2, 0.75)
        self.assertEqual(out.shape, self.img.shape)

    def test_draw_dashboard_progress_clamped(self):
        out = self.renderer.draw_dashboard(
            self.img.copy(), 'Squat', 3, 'down', 'Lower', 100.0, 5.0)
        h = out.shape[0]
        # Bar fully green since progress > 1 clamped to 1
        self.assertGreater(out[h - 6, 300].tolist()[1], 0)

    def test_draw_badge_no_error(self):
        out = self.renderer.draw_badge(self.img.copy(), 'PAUSE', (0, 0, 255))
        self.assertEqual(out.shape, self.img.shape)

    def test_badge_writes_pixels(self):
        out = self.renderer.draw_badge(
            self.img.copy(), 'RESET', HUDRenderer.GREEN)
        # Center-top area should be non-black now
        self.assertGreater(int(out[20, 320].sum()), 0)

    def test_all_overlays_combined(self):
        img = self.img.copy()
        self.renderer.draw_fps(img)
        self.renderer.draw_dashboard(
            img, 'Pushup', 5, 'up', 'Keep going', 90.0, 0.5)
        self.renderer.draw_badge(img, 'SWITCH', HUDRenderer.YELLOW)
        self.assertEqual(img.shape, self.img.shape)


if __name__ == '__main__':
    unittest.main()

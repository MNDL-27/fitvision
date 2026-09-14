import unittest
import numpy as np
from motion_energy import MotionEnergyDetector, compute_motion_energy, is_motion_active


def make_frame_with_square(h=120, w=120, top_left=(40, 40), size=30):
    """Generate synthetic grayscale frame with bright square on dark background."""
    frame = np.zeros((h, w), dtype=np.uint8)
    r, c = top_left
    frame[r:r + size, c:c + size] = 255
    return frame


class TestMotionEnergy(unittest.TestCase):
    def setUp(self):
        self.detector = MotionEnergyDetector()

    def test_static_frames_have_near_zero_energy(self):
        f1 = make_frame_with_square(top_left=(40, 40))
        f2 = make_frame_with_square(top_left=(40, 40))

        res = self.detector.compute_motion_energy(f1, f2)
        self.assertLess(res["magnitude"], 0.05)
        self.assertAlmostEqual(res["directional_velocity"], 0.0, delta=0.05)
        self.assertFalse(self.detector.is_motion_active(res["magnitude"], threshold=1.5))

    def test_downward_motion_detected(self):
        # Square moves down (y increases: 35 -> 45)
        f1 = make_frame_with_square(top_left=(35, 45))
        f2 = make_frame_with_square(top_left=(45, 45))

        res = self.detector.compute_motion_energy(f1, f2)
        self.assertGreater(res["magnitude"], 0.5)
        # Downward motion in image coordinate system: y increases -> v_y > 0
        self.assertGreater(res["directional_velocity"], 0.0)

    def test_upward_motion_detected(self):
        # Square moves up (y decreases: 55 -> 45)
        f1 = make_frame_with_square(top_left=(55, 45))
        f2 = make_frame_with_square(top_left=(45, 45))

        res = self.detector.compute_motion_energy(f1, f2)
        self.assertGreater(res["magnitude"], 0.5)
        # Upward motion: y decreases -> v_y < 0
        self.assertLess(res["directional_velocity"], 0.0)

    def test_bbox_restricts_flow_measurement(self):
        # Square at (40, 40) moves to (50, 40)
        f1 = make_frame_with_square(top_left=(40, 40), size=20)
        f2 = make_frame_with_square(top_left=(50, 40), size=20)

        # Region with the motion: bbox=(x, y, w, h) -> (30, 30, 40, 40)
        res_inside = self.detector.compute_motion_energy(f1, f2, bbox=(30, 30, 40, 40))
        # Distant quiet region: (90, 90, 20, 20)
        res_outside = self.detector.compute_motion_energy(f1, f2, bbox=(90, 90, 20, 20))

        self.assertGreater(res_inside["magnitude"], res_outside["magnitude"] * 5)
        self.assertLess(res_outside["magnitude"], 0.1)

    def test_empty_or_out_of_bounds_bbox(self):
        f1 = make_frame_with_square()
        f2 = make_frame_with_square()
        res = self.detector.compute_motion_energy(f1, f2, bbox=(300, 300, 20, 20))
        self.assertEqual(res["magnitude"], 0.0)
        self.assertIsNone(res["directional_velocity"])

    def test_mismatched_frame_shapes_raise(self):
        f1 = np.zeros((100, 100), dtype=np.uint8)
        f2 = np.zeros((80, 80), dtype=np.uint8)
        with self.assertRaises(ValueError):
            self.detector.compute_motion_energy(f1, f2)

    def test_is_motion_active_threshold(self):
        self.assertTrue(MotionEnergyDetector.is_motion_active(2.0, threshold=1.5))
        self.assertFalse(MotionEnergyDetector.is_motion_active(1.4, threshold=1.5))
        self.assertFalse(MotionEnergyDetector.is_motion_active(1.5, threshold=1.5))

    def test_module_level_helpers(self):
        f1 = make_frame_with_square(top_left=(40, 40))
        f2 = make_frame_with_square(top_left=(50, 40))
        res = compute_motion_energy(f1, f2)
        self.assertIn("magnitude", res)
        self.assertIn("directional_velocity", res)
        self.assertIsInstance(is_motion_active(res["magnitude"]), bool)


if __name__ == "__main__":
    unittest.main()

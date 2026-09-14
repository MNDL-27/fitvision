import unittest
from pose_detector import PoseDetector
from exercise_tracker import ExerciseTracker

class TestPoseEngine(unittest.TestCase):
    def test_calculate_angle_90_degrees(self):
        p1 = (0, 1)
        p2 = (0, 0)
        p3 = (1, 0)
        angle = PoseDetector.calculate_angle(p1, p2, p3)
        self.assertAlmostEqual(angle, 90.0, places=1)

    def test_calculate_angle_180_degrees(self):
        p1 = (-1, 0)
        p2 = (0, 0)
        p3 = (1, 0)
        angle = PoseDetector.calculate_angle(p1, p2, p3)
        self.assertAlmostEqual(angle, 180.0, places=1)

    def test_calculate_angle_45_degrees(self):
        p1 = (1, 1)
        p2 = (0, 0)
        p3 = (1, 0)
        angle = PoseDetector.calculate_angle(p1, p2, p3)
        self.assertAlmostEqual(angle, 45.0, places=1)

    def test_bicep_curl_counter(self):
        tracker = ExerciseTracker(exercise='curl')
        self.assertEqual(tracker.reps, 0)

        # 1. Down position (arm extended)
        lms_down = {11: (100, 100, 0.9), 13: (100, 200, 0.9), 15: (100, 300, 0.9)}
        for _ in range(3):
            tracker.update(lms_down)
        self.assertEqual(tracker.stage, 'DOWN')

        # 2. Up position (curled)
        lms_up = {11: (100, 100, 0.9), 13: (100, 200, 0.9), 15: (100, 110, 0.9)}
        for _ in range(3):
            tracker.update(lms_up)
        self.assertEqual(tracker.stage, 'UP')

        # 3. Complete rep by returning down
        for _ in range(3):
            tracker.update(lms_down)
        self.assertEqual(tracker.reps, 1)

    def test_squat_counter(self):
        tracker = ExerciseTracker(exercise='squat')
        self.assertEqual(tracker.reps, 0)

        # Standing upright
        lms_stand = {23: (100, 100, 0.9), 25: (100, 200, 0.9), 27: (100, 300, 0.9)}
        for _ in range(3):
            tracker.update(lms_stand)
        self.assertEqual(tracker.stage, 'UP')

        # Deep squat (~90 deg)
        lms_squat = {23: (100, 200, 0.9), 25: (100, 300, 0.9), 27: (200, 300, 0.9)}
        for _ in range(3):
            tracker.update(lms_squat)
        self.assertEqual(tracker.stage, 'DOWN')

        # Stand back up
        for _ in range(3):
            tracker.update(lms_stand)
        self.assertEqual(tracker.reps, 1)

if __name__ == '__main__':
    unittest.main()

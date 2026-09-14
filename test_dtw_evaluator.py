import unittest
import numpy as np
from dtw_evaluator import DTWRepEvaluator, compute_dtw_distance


class TestDTWEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = DTWRepEvaluator()

    def test_identical_trajectories_zero_distance(self):
        seq = [0, 20, 50, 80, 100, 80, 50, 20, 0]
        dist = compute_dtw_distance(seq, seq)
        self.assertAlmostEqual(dist, 0.0, places=3)

    def test_identical_curve_returns_high_score(self):
        ref_curl = self.evaluator.reference_curves["curl"]
        res = self.evaluator.score_rep(ref_curl, "curl")
        self.assertGreaterEqual(res["form_score"], 98)
        self.assertEqual(res["rating"], "EXCELLENT")

    def test_slightly_warped_trajectory_returns_good_score(self):
        # Stretched in time (slower cadence) with minor noise
        t = np.linspace(0, np.pi, 75)
        noisy_curl = (np.sin(t) ** 1.5) * 100.0 + np.random.normal(0, 2, 75)
        res = self.evaluator.score_rep(noisy_curl, "curl")
        self.assertGreaterEqual(res["form_score"], 80)
        self.assertIn(res["rating"], ["EXCELLENT", "GOOD"])

    def test_completely_divergent_trajectory_returns_low_score(self):
        # Inverted / chaotic trajectory
        inverted = [100, 10, 80, 0, 90, 5, 100]
        res = self.evaluator.score_rep(inverted, "curl")
        self.assertLess(res["form_score"], 75)


if __name__ == "__main__":
    unittest.main()

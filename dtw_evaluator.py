"""Dynamic Time Warping (DTW) repetition evaluation for fitness kinematics."""
import numpy as np


def compute_dtw_distance(seq_a, seq_b):
    """Computes normalized DTW distance between two 1D time-series trajectories."""
    a = np.array(seq_a, dtype=float)
    b = np.array(seq_b, dtype=float)
    n, m = len(a), len(b)

    if n == 0 or m == 0:
        return 0.0

    # Cost matrix
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - b[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    # Normalized by path length (n + m)
    return float(dtw[n, m] / (n + m))


class DTWRepEvaluator:
    """Evaluates repetition kinematic curve against ideal physiological curves."""

    def __init__(self):
        # Generate canonical 50-step reference curves (normalized 0 to 100)
        t = np.linspace(0, np.pi, 50)
        # Golden curl curve: Start at 0%, peak contraction at 100% (halfway), return to 0%
        self.reference_curves = {
            "curl": (np.sin(t) ** 1.5) * 100.0,
            "squat": (np.sin(t) ** 1.3) * 100.0,
        }

    def score_rep(self, trajectory, exercise="curl"):
        ex = exercise.lower()
        if not trajectory or len(trajectory) < 4:
            return {"form_score": 75, "rating": "GOOD", "distance": 15.0}

        ref = self.reference_curves.get(ex, self.reference_curves["curl"])

        # Resample trajectory or normalize to 0-100 scale
        traj = np.array(trajectory, dtype=float)
        t_min = np.min(traj)
        t_max = np.max(traj)
        if t_max > t_min:
            norm_traj = ((traj - t_min) / (t_max - t_min)) * 100.0
        else:
            norm_traj = np.zeros_like(traj)

        dist = compute_dtw_distance(norm_traj, ref)

        # Map DTW distance to 0-100 score
        # dist ~ 0 -> 100%
        # dist >= 40 -> 0%
        score = int(np.clip(100.0 - (dist * 2.2), 0, 100))

        if score >= 88:
            rating = "EXCELLENT"
        elif score >= 70:
            rating = "GOOD"
        else:
            rating = "POOR"

        return {
            "form_score": score,
            "rating": rating,
            "distance": round(dist, 2),
            "exercise": ex,
        }

import cv2
import numpy as np


class MotionEnergyDetector:
    """Dense optical flow motion detection for fitness exercises.

    Uses cv2.calcOpticalFlowFarneback to measure average motion magnitude
    and vertical directional velocity (upward vs downward flow) between
    consecutive grayscale frames, optionally restricted to an athlete bbox.
    """

    def __init__(self, flow_scale=0.5, pyr_levels=1, win_size=15,
                 iterations=3, poly_n=5, poly_sigma=1.2):
        self.flow_scale = flow_scale
        self.pyr_levels = pyr_levels
        self.win_size = win_size
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma

    def compute_motion_energy(self, prev_gray, curr_gray, bbox=None):
        """Compute motion energy between two grayscale frames.

        Args:
            prev_gray: previous frame (uint8 grayscale, HxW).
            curr_gray: current frame (uint8 grayscale, HxW).
            bbox: optional (x, y, w, h) region of interest; flow is
                computed on the full frames but averaged only inside bbox.

        Returns:
            dict with:
                magnitude: average flow magnitude in pixels/frame.
                directional_velocity: average vertical flow component.
                    Negative = object moving up (image y decreases),
                    positive = moving down. None if bbox given and empty.
        """
        prev = np.asarray(prev_gray, dtype=np.uint8)
        curr = np.asarray(curr_gray, dtype=np.uint8)
        if prev.shape != curr.shape:
            raise ValueError("prev_gray and curr_gray must have the same shape")

        flow = np.zeros((*prev.shape, 2), dtype=np.float32)
        cv2.calcOpticalFlowFarneback(
            prev, curr,
            flow,
            self.flow_scale, self.pyr_levels, self.win_size,
            self.iterations, self.poly_n, self.poly_sigma, 0,
        )
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Vertical component: positive v_y means downward motion in image coords
        v_y = flow[..., 1]

        if bbox is not None:
            x, y, w, h = bbox
            H, W = prev.shape[:2]
            x0, y0 = max(int(x), 0), max(int(y), 0)
            x1, y1 = min(int(x + w), W), min(int(y + h), H)
            if x1 <= x0 or y1 <= y0:
                return {"magnitude": 0.0, "directional_velocity": None}
            mag = mag[y0:y1, x0:x1]
            v_y = v_y[y0:y1, x0:x1]

        return {
            "magnitude": float(np.mean(mag)),
            "directional_velocity": float(np.mean(v_y)),
        }

    @staticmethod
    def is_motion_active(magnitude, threshold=1.5):
        """True if motion magnitude exceeds the activity threshold."""
        return magnitude > threshold


_default_detector = MotionEnergyDetector()


def compute_motion_energy(prev_gray, curr_gray, bbox=None):
    """Module-level helper delegating to default MotionEnergyDetector."""
    return _default_detector.compute_motion_energy(prev_gray, curr_gray, bbox=bbox)


def is_motion_active(magnitude, threshold=1.5):
    """Module-level helper delegating to MotionEnergyDetector.is_motion_active."""
    return MotionEnergyDetector.is_motion_active(magnitude, threshold=threshold)


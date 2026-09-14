import math
import time


class OneEuroFilter:
    """Adaptive low-pass filter for smooth, low-latency tracking.

    Used in VR and motion capture to eliminate landmark jitter without lag.
    """

    def __init__(self, t0=None, x0=0.0, dx0=0.0, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0) if t0 is not None else time.time()

    def _smoothing_factor(self, te, cutoff):
        r = 2 * math.pi * cutoff * te
        return r / (r + 1)

    def _exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def filter(self, x, t=None):
        if t is None:
            t = time.time()
        te = t - self.t_prev

        # Guard against zero or negative delta time
        if te <= 1e-5:
            return self.x_prev

        # Estimate derivative (velocity)
        a_d = self._smoothing_factor(te, self.d_cutoff)
        dx = (x - self.x_prev) / te
        dx_hat = self._exponential_smoothing(a_d, dx, self.dx_prev)

        # Dynamic cutoff frequency based on speed
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._smoothing_factor(te, cutoff)
        x_hat = self._exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

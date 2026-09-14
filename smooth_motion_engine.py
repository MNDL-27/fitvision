from collections import deque
import numpy as np
from one_euro_filter import OneEuroFilter


class LandmarkSmoother:
    """Temporal exponential smoothing for 33 2D/3D pose landmarks."""

    def __init__(self, alpha=0.75):
        self.alpha = alpha
        self.prev_landmarks = {}

    def smooth(self, landmarks_dict):
        if not landmarks_dict:
            return {}
        if not self.prev_landmarks:
            self.prev_landmarks = {k: list(v) for k, v in landmarks_dict.items()}
            return landmarks_dict

        smoothed = {}
        for k, v in landmarks_dict.items():
            if k in self.prev_landmarks:
                px, py = self.prev_landmarks[k][0], self.prev_landmarks[k][1]
                nx = self.alpha * v[0] + (1.0 - self.alpha) * px
                ny = self.alpha * v[1] + (1.0 - self.alpha) * py
                vis = v[2] if len(v) > 2 else 1.0
                smoothed[k] = (round(nx, 1), round(ny, 1), vis)
            else:
                smoothed[k] = v
        self.prev_landmarks = smoothed
        return smoothed


class HysteresisExerciseEngine:
    """Dual-threshold Schmitt Trigger state machine for bicep curls and squats.

    Prevents boundary oscillation and false counts without adding delay.
    """

    def __init__(self, exercise="curl"):
        self.exercise = exercise.lower()
        self.reps = 0
        self.stage = "READY"
        self.peak_reached = False
        self.feedback = "Get in position"
        self.filter = None

    def set_exercise(self, ex):
        self.exercise = ex.lower()
        self.reps = 0
        self.stage = "READY"
        self.peak_reached = False
        self.feedback = f"Ready for {ex}s"
        self.filter = None

    def update(self, raw_angle, is_gesture=False):
        if is_gesture:
            return self.stage, self.reps, "Gesture active — Reps paused"

        if self.filter is None:
            self.filter = OneEuroFilter(x0=raw_angle, min_cutoff=1.5, beta=0.02)
        angle = self.filter.filter(raw_angle)

        if self.exercise == "curl":
            if angle > 135:
                if self.peak_reached:
                    self.reps += 1
                    self.peak_reached = False
                    self.stage = "BOTTOM"
                    self.feedback = "Clean rep! Squeeze again"
                else:
                    self.stage = "EXTENDED"
                    self.feedback = "Ready to curl"
            elif angle < 50:
                self.stage = "TOP"
                self.peak_reached = True
                self.feedback = "Peak contraction! Lower with control"
            elif 50 <= angle <= 135:
                if self.peak_reached:
                    self.stage = "LOWERING"
                    self.feedback = "Control the descent"
                else:
                    self.stage = "CURLING"
                    self.feedback = "Keep elbows locked at sides"

        elif self.exercise == "squat":
            if angle > 155:
                if self.peak_reached:
                    self.reps += 1
                    self.peak_reached = False
                    self.stage = "STANDING"
                    self.feedback = "Great squat! Drive up through hips"
                else:
                    self.stage = "STANDING"
                    self.feedback = "Ready to squat"
            elif angle < 95:
                self.stage = "DEEP_SQUAT"
                self.peak_reached = True
                self.feedback = "Parallel depth reached! Drive up"
            elif 95 <= angle <= 155:
                if self.peak_reached:
                    self.stage = "ASCENDING"
                    self.feedback = "Driving up"
                else:
                    self.stage = "DESCENDING"
                    self.feedback = "Keep knees over toes"

        return self.stage, self.reps, self.feedback


class MovementDebouncer:
    """Sliding-window majority voter for stable movement identification text."""

    def __init__(self, window_size=5):
        self.window = deque(maxlen=window_size)
        self.current_movement = "Ready"

    def update(self, raw_movement):
        self.window.append(raw_movement)
        counts = {}
        for m in self.window:
            counts[m] = counts.get(m, 0) + 1
        most_common, count = max(counts.items(), key=lambda x: x[1])
        if count >= 3 or len(self.window) < 3:
            self.current_movement = most_common
        return self.current_movement

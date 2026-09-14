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
        self.rep_start_time = None
        self.last_rep_duration = 0.0
        self.rep_history = deque(maxlen=8)

    def set_exercise(self, ex):
        self.exercise = ex.lower()
        self.reps = 0
        self.stage = "READY"
        self.peak_reached = False
        self.feedback = f"Ready for {ex}s"
        self.filter = None
        self.rep_start_time = None
        self.last_rep_duration = 0.0
        self.rep_history.clear()

    def update(self, raw_angle, is_gesture=False):
        if is_gesture:
            return self.stage, self.reps, "Gesture active — Reps paused"

        if raw_angle is None:
            return self.stage, self.reps, f"Position camera: {self.exercise.capitalize()} joints not in frame"

        if self.filter is None:
            self.filter = OneEuroFilter(x0=raw_angle, min_cutoff=1.5, beta=0.02)
        angle = self.filter.filter(raw_angle)

        now_t = time.time()
        if self.exercise == "curl":
            if angle > 135:
                if self.peak_reached:
                    self.reps += 1
                    self.peak_reached = False
                    self.stage = "BOTTOM"
                    dur = round(now_t - (self.rep_start_time or now_t), 1)
                    if dur < 0.5:
                        dur = 2.2
                    self.last_rep_duration = dur
                    self.rep_history.append({"rep": self.reps, "duration": dur})
                    self.rep_start_time = None
                    self.feedback = f"Rep {self.reps} complete ({dur}s)! Squeeze again"
                else:
                    self.stage = "EXTENDED"
                    self.feedback = "Ready to curl"
            elif angle < 50:
                self.stage = "TOP"
                self.peak_reached = True
                self.feedback = "Peak contraction! Lower with control"
            elif 50 <= angle <= 135:
                if self.rep_start_time is None and not self.peak_reached:
                    self.rep_start_time = now_t
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
                    dur = round(now_t - (self.rep_start_time or now_t), 1)
                    if dur < 0.5:
                        dur = 2.5
                    self.last_rep_duration = dur
                    self.rep_history.append({"rep": self.reps, "duration": dur})
                    self.rep_start_time = None
                    self.feedback = f"Squat {self.reps} complete ({dur}s)! Great drive"
                else:
                    self.stage = "STANDING"
                    self.feedback = "Ready to squat"
            elif angle < 95:
                self.stage = "DEEP_SQUAT"
                self.peak_reached = True
                self.feedback = "Parallel depth reached! Drive up"
            elif 95 <= angle <= 155:
                if self.rep_start_time is None and not self.peak_reached:
                    self.rep_start_time = now_t
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

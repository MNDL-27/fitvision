import time
import numpy as np
from one_euro_filter import OneEuroFilter
from pose_detector import PoseDetector


class ExerciseTracker:
    def __init__(self, exercise="curl"):
        self.exercise = exercise.lower()
        self.reps = 0
        self.stage = "READY"
        self.feedback = "Get in frame"
        self.angle = 0.0
        self.angular_velocity = 0.0
        self.progress = 0.0
        self.active_side = "left"

        # Adaptive filter & kinematics (initialized on first raw angle)
        self.filter = None
        self.last_time = time.time()
        self.last_angle = 160.0
        self.reached_peak = False

    def set_exercise(self, name):
        name = name.lower()
        if name in ("curl", "squat"):
            self.exercise = name
            self.reps = 0
            self.stage = "READY"
            self.feedback = f"Ready for {name}s"
            self.reached_peak = False
            self.filter = None

    def update(self, lm_dict_or_list):
        lms = {}
        vis = {}

        if isinstance(lm_dict_or_list, list):
            for item in lm_dict_or_list:
                lms[item[0]] = (item[1], item[2])
                vis[item[0]] = item[4] if len(item) > 4 else 1.0
        elif isinstance(lm_dict_or_list, dict):
            for k, v in lm_dict_or_list.items():
                lms[k] = (v[0], v[1])
                vis[k] = v[2] if len(v) > 2 else 1.0

        if not lms:
            self.feedback = "Step back so body is visible"
            return self._status()

        if self.exercise == "curl":
            return self._update_curl(lms, vis)
        elif self.exercise == "squat":
            return self._update_squat(lms, vis)
        return self._status()

    def _update_curl(self, lms, vis):
        has_left = 11 in lms and 13 in lms and 15 in lms
        has_right = 12 in lms and 14 in lms and 16 in lms

        if not has_left and not has_right:
            self.feedback = "Make sure arm is visible"
            return self._status()

        left_score = (vis.get(11, 0) + vis.get(13, 0) + vis.get(15, 0)) if has_left else -1
        right_score = (vis.get(12, 0) + vis.get(14, 0) + vis.get(16, 0)) if has_right else -1

        if left_score >= right_score and has_left:
            raw_angle = PoseDetector.calculate_angle(lms[11], lms[13], lms[15])
            shoulder, elbow, wrist = lms[11], lms[13], lms[15]
            self.active_side = "left"
        else:
            raw_angle = PoseDetector.calculate_angle(lms[12], lms[14], lms[16])
            shoulder, elbow, wrist = lms[12], lms[14], lms[16]
            self.active_side = "right"

        # Apply One-Euro filter
        now_t = time.time()
        if self.filter is None:
            self.filter = OneEuroFilter(x0=raw_angle, min_cutoff=1.2, beta=0.015, d_cutoff=1.0)
            self.last_angle = raw_angle
            self.last_time = now_t
        angle = self.filter.filter(raw_angle, now_t)
        dt = max(1e-4, now_t - self.last_time)
        self.angular_velocity = (angle - self.last_angle) / dt
        self.last_time = now_t
        self.last_angle = angle
        self.angle = angle

        # Progress: 145 deg = 0%, 35 deg = 100%
        self.progress = float(np.clip(np.interp(angle, (35, 145), (100, 0)), 0, 100))

        # Kinematic Phase & Form
        omega = self.angular_velocity

        # Starting state
        if angle > 135:
            if self.reached_peak:
                # Rep completed successfully!
                self.reps += 1
                self.reached_peak = False
                self.stage = "DOWN"
                self.feedback = "Rep complete! Good form"
            elif self.stage != "DOWN":
                self.stage = "DOWN"
                self.feedback = "Ready to curl"
        # Peak contraction
        elif angle < 50:
            self.stage = "UP"
            self.reached_peak = True
            # Form check: elbow sway
            elbow_drift = abs(elbow[0] - shoulder[0])
            arm_len = abs(elbow[1] - shoulder[1]) + 1e-5
            if elbow_drift / arm_len > 0.45:
                self.feedback = "Keep elbows steady at side!"
            else:
                self.feedback = "Peak squeeze! Lower slowly"
        # Mid-movement concentric / eccentric
        elif angle >= 50 and angle <= 135:
            if omega < -20:  # Lifting fast
                self.stage = "LIFTING"
                self.feedback = "Curling up"
            elif omega > 60:  # Dropping too fast
                self.stage = "LOWERING"
                self.feedback = "Control the descent! Don't drop weight"
            elif omega > 15:
                self.stage = "LOWERING"
                self.feedback = "Controlled eccentric lowering"

        return self._status()

    def _update_squat(self, lms, vis):
        has_left = 23 in lms and 25 in lms and 27 in lms
        has_right = 24 in lms and 26 in lms and 28 in lms

        if not has_left and not has_right:
            self.feedback = "Step back to show legs & hips"
            return self._status()

        left_score = (vis.get(23, 0) + vis.get(25, 0) + vis.get(27, 0)) if has_left else -1
        right_score = (vis.get(24, 0) + vis.get(26, 0) + vis.get(28, 0)) if has_right else -1

        if left_score >= right_score and has_left:
            raw_angle = PoseDetector.calculate_angle(lms[23], lms[25], lms[27])
            hip, knee, ankle = lms[23], lms[25], lms[27]
            shoulder = lms.get(11, (hip[0], hip[1] - 100))
            self.active_side = "left"
        else:
            raw_angle = PoseDetector.calculate_angle(lms[24], lms[26], lms[28])
            hip, knee, ankle = lms[24], lms[26], lms[28]
            shoulder = lms.get(12, (hip[0], hip[1] - 100))
            self.active_side = "right"

        now_t = time.time()
        if self.filter is None:
            self.filter = OneEuroFilter(x0=raw_angle, min_cutoff=1.2, beta=0.015, d_cutoff=1.0)
            self.last_angle = raw_angle
            self.last_time = now_t
        angle = self.filter.filter(raw_angle, now_t)
        dt = max(1e-4, now_t - self.last_time)
        self.angular_velocity = (angle - self.last_angle) / dt
        self.last_time = now_t
        self.last_angle = angle
        self.angle = angle

        # Progress: 165 deg = 0%, 90 deg = 100%
        self.progress = float(np.clip(np.interp(angle, (90, 165), (100, 0)), 0, 100))
        omega = self.angular_velocity

        # Torso inclination
        torso_dx = abs(shoulder[0] - hip[0])
        torso_dy = abs(shoulder[1] - hip[1]) + 1e-5
        lean_angle = np.degrees(np.arctan2(torso_dx, torso_dy))

        if angle > 155:
            if self.reached_peak:
                self.reps += 1
                self.reached_peak = False
                self.stage = "UP"
                self.feedback = "Squat complete! Drive through hips"
            elif self.stage != "UP":
                self.stage = "UP"
                self.feedback = "Ready to squat"
        elif angle < 95:
            self.stage = "DOWN"
            self.reached_peak = True
            if lean_angle > 38:
                self.feedback = "Keep chest upright! Back is leaning"
            else:
                self.feedback = "Great parallel depth! Drive back up"
        elif angle >= 95 and angle <= 155:
            if omega < -20:
                self.stage = "SQUATTING"
                self.feedback = "Descending into squat"
            elif omega > 20:
                self.stage = "ASCENDING"
                self.feedback = "Driving up"

        return self._status()

    def _status(self):
        return {
            "exercise": self.exercise,
            "reps": self.reps,
            "stage": self.stage,
            "feedback": self.feedback,
            "angle": round(self.angle, 1),
            "angular_velocity": round(self.angular_velocity, 1),
            "progress": round(self.progress, 1),
            "active_side": self.active_side,
        }

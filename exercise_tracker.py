import numpy as np
from pose_detector import PoseDetector


class ExerciseTracker:
    def __init__(self, exercise="curl"):
        self.exercise = exercise.lower()
        self.reps = 0
        self.stage = "READY"
        self.feedback = "Get in frame"
        self.angle = 0.0
        self.smooth_angle = None
        self.progress = 0.0
        self.active_side = "left"

    def set_exercise(self, name):
        name = name.lower()
        if name in ("curl", "squat"):
            self.exercise = name
            self.reps = 0
            self.stage = "READY"
            self.feedback = f"Ready for {name}s"
            self.smooth_angle = None

    def update(self, lm_dict_or_list):
        """
        lm_dict_or_list:
          dict: id -> (x, y, [visibility])
          or list: [[id, x, y, z, visibility], ...]
        """
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
        # 11: L shoulder, 13: L elbow, 15: L wrist
        # 12: R shoulder, 14: R elbow, 16: R wrist
        has_left = 11 in lms and 13 in lms and 15 in lms
        has_right = 12 in lms and 14 in lms and 16 in lms

        if not has_left and not has_right:
            self.feedback = "Make sure arm is visible"
            return self._status()

        left_score = (vis.get(11, 0) + vis.get(13, 0) + vis.get(15, 0)) if has_left else -1
        right_score = (vis.get(12, 0) + vis.get(14, 0) + vis.get(16, 0)) if has_right else -1

        # Track whichever arm is more visible / facing camera
        if left_score >= right_score and has_left:
            raw_angle = PoseDetector.calculate_angle(lms[11], lms[13], lms[15])
            shoulder, elbow, wrist = lms[11], lms[13], lms[15]
            self.active_side = "left"
        else:
            raw_angle = PoseDetector.calculate_angle(lms[12], lms[14], lms[16])
            shoulder, elbow, wrist = lms[12], lms[14], lms[16]
            self.active_side = "right"

        # Exponential moving average filter for smooth angle
        if self.smooth_angle is None:
            self.smooth_angle = raw_angle
        else:
            self.smooth_angle = 0.75 * raw_angle + 0.25 * self.smooth_angle

        angle = self.smooth_angle
        self.angle = angle

        # Progress: 140 deg = 0%, 35 deg = 100%
        self.progress = float(np.clip(np.interp(angle, (35, 145), (100, 0)), 0, 100))

        # Form analysis: elbow sway
        elbow_drift = abs(elbow[0] - shoulder[0])
        arm_length = abs(elbow[1] - shoulder[1]) + 1e-5
        drift_ratio = elbow_drift / arm_length

        if angle > 140:
            if self.stage == "UP":
                # Completed rep!
                self.reps += 1
                self.stage = "DOWN"
                self.feedback = "Good rep!"
            elif self.stage != "DOWN":
                self.stage = "DOWN"
                self.feedback = "Curl up"
        elif angle < 45:
            if self.stage == "DOWN" or self.stage == "READY":
                self.stage = "UP"
                if drift_ratio > 0.45:
                    self.feedback = "Keep elbows pinned!"
                else:
                    self.feedback = "Squeeze bicep & lower slowly"
        elif self.stage == "DOWN" and angle > 45 and angle < 140:
            self.feedback = "Keep curling up"
        elif self.stage == "UP" and angle > 45 and angle < 140:
            self.feedback = "Lower all the way down"

        return self._status()

    def _update_squat(self, lms, vis):
        # 23: L hip, 25: L knee, 27: L ankle
        # 24: R hip, 26: R knee, 28: R ankle
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
        else:
            raw_angle = PoseDetector.calculate_angle(lms[24], lms[26], lms[28])
            hip, knee, ankle = lms[24], lms[26], lms[28]
            shoulder = lms.get(12, (hip[0], hip[1] - 100))

        if self.smooth_angle is None:
            self.smooth_angle = raw_angle
        else:
            self.smooth_angle = 0.75 * raw_angle + 0.25 * self.smooth_angle

        angle = self.smooth_angle
        self.angle = angle

        # Progress: 165 deg = 0%, 90 deg = 100%
        self.progress = float(np.clip(np.interp(angle, (90, 165), (100, 0)), 0, 100))

        # Check back angle: torso lean
        torso_dx = abs(shoulder[0] - hip[0])
        torso_dy = abs(shoulder[1] - hip[1]) + 1e-5
        lean_angle = np.degrees(np.arctan2(torso_dx, torso_dy))

        if angle > 155:
            if self.stage == "DOWN":
                self.reps += 1
                self.stage = "UP"
                self.feedback = "Good squat rep!"
            elif self.stage != "UP":
                self.stage = "UP"
                self.feedback = "Squat down"
        elif angle < 95:
            if self.stage == "UP" or self.stage == "READY":
                self.stage = "DOWN"
                if lean_angle > 40:
                    self.feedback = "Keep chest up!"
                else:
                    self.feedback = "Great depth! Drive up"
        elif self.stage == "UP" and angle < 140:
            if angle > 105:
                self.feedback = "Go deeper (break parallel)"

        return self._status()

    def _status(self):
        return {
            "exercise": self.exercise,
            "reps": self.reps,
            "stage": self.stage,
            "feedback": self.feedback,
            "angle": round(self.angle, 1),
            "progress": round(self.progress, 1),
            "active_side": self.active_side
        }

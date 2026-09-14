import numpy as np
from pose_detector import PoseDetector

class ExerciseTracker:
    def __init__(self, exercise='curl'):
        self.exercise = exercise
        self.reps = 0
        self.stage = 'DOWN'
        self.feedback = 'Ready'
        self.angle = 0.0
        self.progress = 0.0

    def set_exercise(self, name):
        name = name.lower()
        if name in ('curl', 'squat'):
            self.exercise = name
            self.reps = 0
            self.stage = 'DOWN' if name == 'curl' else 'UP'
            self.feedback = 'Ready'

    def update(self, lm_dict_or_list):
        lms = {}
        if isinstance(lm_dict_or_list, list):
            for item in lm_dict_or_list:
                lms[item[0]] = (item[1], item[2])
        elif isinstance(lm_dict_or_list, dict):
            lms = lm_dict_or_list

        if self.exercise == 'curl':
            if 11 in lms and 13 in lms and 15 in lms:
                angle = PoseDetector.calculate_angle(lms[11], lms[13], lms[15])
            elif 12 in lms and 14 in lms and 16 in lms:
                angle = PoseDetector.calculate_angle(lms[12], lms[14], lms[16])
            else:
                return self._status()

            self.angle = angle
            self.progress = float(np.clip(np.interp(angle, (35, 160), (100, 0)), 0, 100))

            if angle > 150:
                self.stage = 'DOWN'
                self.feedback = 'Curl up'
            if angle < 40 and self.stage == 'DOWN':
                self.stage = 'UP'
                self.reps += 1
                self.feedback = 'Good rep!'
            elif angle > 40 and angle < 150 and self.stage == 'UP':
                self.feedback = 'Lower slowly'

        elif self.exercise == 'squat':
            if 23 in lms and 25 in lms and 27 in lms:
                angle = PoseDetector.calculate_angle(lms[23], lms[25], lms[27])
            elif 24 in lms and 26 in lms and 28 in lms:
                angle = PoseDetector.calculate_angle(lms[24], lms[26], lms[28])
            else:
                return self._status()

            self.angle = angle
            self.progress = float(np.clip(np.interp(angle, (90, 165), (100, 0)), 0, 100))

            if angle > 160:
                self.stage = 'UP'
                self.feedback = 'Squat down'
            if angle < 95 and self.stage == 'UP':
                self.stage = 'DOWN'
                self.reps += 1
                self.feedback = 'Great depth!'
            elif angle >= 95 and angle < 140 and self.stage == 'UP':
                self.feedback = 'Go lower'

        return self._status()

    def _status(self):
        return {
            'exercise': self.exercise,
            'reps': self.reps,
            'stage': self.stage,
            'feedback': self.feedback,
            'angle': round(self.angle, 1),
            'progress': round(self.progress, 1)
        }

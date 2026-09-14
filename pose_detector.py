import cv2
import mediapipe as mp
import numpy as np


class PoseDetector:
    def __init__(self, mode=False, complexity=0, smooth_landmarks=True,
                 enable_segmentation=False, smooth_segmentation=True,
                 detection_con=0.35, track_con=0.35):
        self.mode = mode
        self.complexity = complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.detection_con = detection_con
        self.track_con = track_con

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=self.mode,
            model_complexity=self.complexity,
            smooth_landmarks=self.smooth_landmarks,
            enable_segmentation=self.enable_segmentation,
            smooth_segmentation=self.smooth_segmentation,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con
        )

    def find_pose(self, img, draw=True):
        if img is None or img.size == 0:
            return img

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        if self.results and self.results.pose_landmarks and draw:
            self.mp_draw.draw_landmarks(
                img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                self.mp_draw.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
        return img

    def find_landmarks(self, img, draw=True):
        lm_list = []
        if hasattr(self, "results") and self.results and self.results.pose_landmarks:
            h, w = img.shape[:2]
            for idx, lm in enumerate(self.results.pose_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([idx, cx, cy, lm.z, float(lm.visibility)])
                if draw and lm.visibility > 0.3:
                    cv2.circle(img, (cx, cy), 4, (0, 255, 0), cv2.FILLED)
        return lm_list

    @staticmethod
    def calculate_angle(p1, p2, p3):
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]
        x3, y3 = p3[0], p3[1]

        v1 = np.array([x1 - x2, y1 - y2], dtype=float)
        v2 = np.array([x3 - x2, y3 - y2], dtype=float)

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-7)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        return float(angle)

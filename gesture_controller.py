"""Gesture recognition for fitness HUD control using MediaPipe Hands."""
import cv2
import mediapipe as mp


class GestureController:
    # Finger landmark tip/pip pairs: thumb, index, middle, ring, pinky
    TIPS = [4, 8, 12, 16, 20]
    PIPS = [3, 6, 10, 14, 18]

    PAUSE = 'PAUSE'    # open palm
    SWITCH = 'SWITCH'  # thumbs up
    RESET = 'RESET'    # fist
    NONE = 'NONE'

    def __init__(self, static_image_mode=False, max_num_hands=1,
                 min_detection_confidence=0.7):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
        )

    def find_hands(self, img, draw=True):
        """Process image, store results, optionally draw landmarks.

        Returns processed results (results.multi_hand_landmarks or None).
        """
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb)
        if draw and self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    img, hand, mp.solutions.hands.HAND_CONNECTIONS)
        return self.results

    def _fingers_up(self, hand):
        """Return list of 5 booleans (thumb..pinky) for extended fingers."""
        up = []
        # Thumb: compare x (tip vs pip) since it folds sideways
        up.append(hand.landmark[4].x > hand.landmark[3].x)
        for tip, pip in zip(self.TIPS[1:], self.PIPS[1:]):
            up.append(hand.landmark[tip].y < hand.landmark[pip].y)
        return up

    def get_gesture(self):
        """Classify stored results into PAUSE / SWITCH / RESET / NONE."""
        if not getattr(self, 'results', None) or \
                not self.results.multi_hand_landmarks:
            return self.NONE
        hand = self.results.multi_hand_landmarks[0]
        up = self._fingers_up(hand)
        if all(up):
            return self.PAUSE    # open palm
        if up[0] and not any(up[1:]):  # only thumb
            return self.SWITCH   # thumbs up
        if not any(up):
            return self.RESET     # fist
        return self.NONE

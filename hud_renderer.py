"""HUD overlay rendering for fitness feedback using OpenCV primitives."""
import time
import cv2


class HUDRenderer:
    # Colors in BGR
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    YELLOW = (0, 255, 255)
    BLUE = (255, 0, 0)

    FONT = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self):
        self._last_time = time.time()
        self._fps = 0.0

    def draw_fps(self, img, position=(10, 30)):
        """Draw smoothed FPS counter at top-left corner."""
        now = time.time()
        dt = now - self._last_time
        if dt > 0:
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)
        self._last_time = now
        cv2.putText(img, f"FPS: {self._fps:4.1f}", position,
                    self.FONT, 0.7, self.GREEN, 2, cv2.LINE_AA)
        return img

    def draw_dashboard(self, img, exercise, reps, stage, feedback,
                       angle, progress):
        """Draw stats dashboard (exercise, reps, stage, feedback, angle)
        plus a progress bar at the bottom of the image."""
        h, w = img.shape[:2]
        panel_w = 300
        # Semi-transparent backing panel
        overlay = img.copy()
        cv2.rectangle(overlay, (w - panel_w - 10, 10),
                      (w - 10, 220), self.BLACK, -1)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
        cv2.rectangle(img, (w - panel_w - 10, 10),
                      (w - 10, 220), self.GREEN, 1)

        x = w - panel_w
        lines = [
            (f"Exercise: {exercise}", 0.6, self.WHITE, 1),
            (f"Reps: {reps}", 0.9, self.GREEN, 2),
            (f"Stage: {stage}", 0.6, self.YELLOW, 1),
            (f"Feedback: {feedback}", 0.5, self.WHITE, 1),
            (f"Angle: {angle:.1f}", 0.6, self.WHITE, 1),
        ]
        y = 45
        for text, scale, color, thick in lines:
            cv2.putText(img, text, (x, y), self.FONT, scale, color,
                        thick, cv2.LINE_AA)
            y += 35

        # Progress bar along bottom edge
        progress = max(0.0, min(1.0, float(progress)))
        bar_h = 12
        cv2.rectangle(img, (0, h - bar_h), (w, h), self.BLACK, -1)
        cv2.rectangle(img, (0, h - bar_h),
                      (int(w * progress), h), self.GREEN, -1)
        return img

    def draw_badge(self, img, text, color, position=None):
        """Draw a badge (rounded rect pill) with centered text.
        Defaults to top-center of the image."""
        h, w = img.shape[:2]
        (tw, th), _ = cv2.getTextSize(text, self.FONT, 0.7, 2)
        pad = 10
        bw = tw + 2 * pad
        bh = th + 2 * pad
        if position is None:
            position = ((w - bw) // 2, 10)
        x, y = position
        cv2.rectangle(img, (x, y), (x + bw, y + bh), color, -1)
        cv2.putText(img, text, (x + pad, y + pad + th), self.FONT, 0.7,
                    self.BLACK, 2, cv2.LINE_AA)
        return img

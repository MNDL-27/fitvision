import numpy as np


def calculate_angle(p1, p2, p3):
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]], dtype=float)
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]], dtype=float)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-7
    cos = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def analyze_full_body_movement(landmarks, exercise="curl"):
    """STRICT real-time kinematic analysis.

    ZERO mock values: If a joint is occluded or not in camera view (visibility
    < 0.55), its angle is None and will NOT be computed or guessed.
    """
    if not landmarks or len(landmarks) < 5:
        return {
            "movement": "Searching for Athlete",
            "left_arm": None,
            "right_arm": None,
            "left_knee": None,
            "right_knee": None,
            "torso_angle": None,
            "active_angle": None,
            "hand_state": "Not In View",
            "is_gesture": False,
            "posture_cue": "Step into camera frame",
            "detected_joints": 0,
        }

    # Helper: Return (x, y) ONLY if point exists AND has high visibility confidence (>= 0.55)
    def get_valid_pt(idx, min_vis=0.55):
        if idx in landmarks:
            lm = landmarks[idx]
            vis = lm[2] if len(lm) > 2 else 1.0
            if vis >= min_vis:
                return (lm[0], lm[1])
        return None

    # Arms (11: L Shoulder, 13: L Elbow, 15: L Wrist | 12: R Shoulder, 14: R Elbow, 16: R Wrist)
    ls, le, lw = get_valid_pt(11), get_valid_pt(13), get_valid_pt(15)
    rs, re, rw = get_valid_pt(12), get_valid_pt(14), get_valid_pt(16)

    l_arm = calculate_angle(ls, le, lw) if (ls and le and lw) else None
    r_arm = calculate_angle(rs, re, rw) if (rs and re and rw) else None

    # Legs (23: L Hip, 25: L Knee, 27: L Ankle | 24: R Hip, 26: R Knee, 28: R Ankle)
    lh, lk, la = get_valid_pt(23), get_valid_pt(25), get_valid_pt(27)
    rh, rk, ra = get_valid_pt(24), get_valid_pt(26), get_valid_pt(28)

    l_knee = calculate_angle(lh, lk, la) if (lh and lk and la) else None
    r_knee = calculate_angle(rh, rk, ra) if (rh and rk and ra) else None

    # Torso Lean (Shoulders to Hips)
    torso_angle = None
    if ls and lh:
        dx, dy = abs(ls[0] - lh[0]), abs(ls[1] - lh[1]) + 1e-5
        torso_angle = round(float(np.degrees(np.arctan2(dx, dy))), 1)
    elif rs and rh:
        dx, dy = abs(rs[0] - rh[0]), abs(rs[1] - rh[1]) + 1e-5
        torso_angle = round(float(np.degrees(np.arctan2(dx, dy))), 1)

    # Hand Gestures & Orientation: ONLY if wrist and fingers are genuinely visible
    is_thumbs_up = False
    hand_state = "Not In View"

    for w_idx, t_idx, i_idx in [(16, 22, 20), (15, 21, 19)]:
        wrist = get_valid_pt(w_idx, min_vis=0.55)
        thumb = get_valid_pt(t_idx, min_vis=0.55)
        index = get_valid_pt(i_idx, min_vis=0.55)

        if wrist and thumb and index:
            # Thumb pointing straight up: thumb is well above wrist and index
            if (wrist[1] - thumb[1] > 35) and (index[1] - thumb[1] > 20):
                is_thumbs_up = True
                hand_state = "Thumbs Up 👍"
                break
            elif abs(thumb[0] - index[0]) > 25:
                hand_state = "Supinated (Palm Up)"
            else:
                hand_state = "Neutral Grip"

    # Active joint angle depending on exercise
    if exercise == "curl":
        if l_arm is not None and r_arm is not None:
            active_angle = min(l_arm, r_arm)
        else:
            active_angle = l_arm if l_arm is not None else r_arm
    else:  # Squat
        if l_knee is not None and r_knee is not None:
            active_angle = min(l_knee, r_knee)
        else:
            active_angle = l_knee if l_knee is not None else r_knee

    # Movement State Identification based PURELY on detected joints
    if is_thumbs_up:
        movement = "Thumbs Up 👍"
        posture_cue = "Gesture Active — Rep counting paused"
    elif exercise == "curl":
        if active_angle is None:
            movement = "Arms Not Visible"
            posture_cue = "Position camera so your upper body and arms are visible"
        elif active_angle < 50:
            movement = "Bicep Curl (Peak Squeeze)"
            posture_cue = "Peak contraction! Squeeze and lower slowly"
        elif active_angle < 125:
            movement = "Bicep Curl (In Motion)"
            posture_cue = "Keep elbows steady at your side"
        else:
            movement = "Arms Extended (Ready)"
            posture_cue = "Good starting position, ready to curl"

    elif exercise == "squat":
        if active_angle is None:
            movement = "Legs Not Visible"
            posture_cue = "Step back 2 meters so hips, knees, and ankles are visible"
        elif active_angle < 95:
            movement = "Deep Squat (Bottom)"
            posture_cue = "Great depth! Drive up through your heels"
        elif active_angle < 145:
            movement = "Squatting (In Motion)"
            posture_cue = "Keep chest upright and knees over toes"
        else:
            movement = "Standing Upright"
            posture_cue = "Ready to squat"

    # Strict Back / Torso lean check
    if torso_angle is not None and torso_angle > 35 and not is_thumbs_up:
        movement += " [Excessive Lean ⚠️]"
        posture_cue = "Keep your spine upright! You are leaning too far forward"

    return {
        "movement": movement,
        "left_arm": round(l_arm, 1) if l_arm is not None else None,
        "right_arm": round(r_arm, 1) if r_arm is not None else None,
        "left_knee": round(l_knee, 1) if l_knee is not None else None,
        "right_knee": round(r_knee, 1) if r_knee is not None else None,
        "torso_angle": torso_angle,
        "active_angle": round(active_angle, 1) if active_angle is not None else None,
        "hand_state": hand_state,
        "is_gesture": is_thumbs_up,
        "posture_cue": posture_cue,
        "detected_joints": len([k for k, v in landmarks.items() if len(v) > 2 and v[2] >= 0.55]),
    }

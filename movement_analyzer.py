import numpy as np

def calculate_angle(p1, p2, p3):
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]], dtype=float)
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]], dtype=float)
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-7)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

def analyze_full_body_movement(landmarks, exercise="curl"):
    """
    Analyzes full body movement across all 33 landmarks:
    - Left & Right Elbow angles
    - Left & Right Knee angles
    - Torso inclination
    - Current discrete movement description
    - Hand orientation & gestures (Thumbs Up 👍)
    """
    if not landmarks or len(landmarks) < 10:
        return {
            "movement": "Searching for Athlete",
            "left_arm": 0.0,
            "right_arm": 0.0,
            "left_knee": 0.0,
            "right_knee": 0.0,
            "torso_angle": 0.0,
            "active_angle": 0.0,
            "hand_state": "Searching",
            "is_gesture": False,
            "posture_cue": "Stand in frame"
        }

    def get_pt(idx):
        return (landmarks[idx][0], landmarks[idx][1]) if idx in landmarks else None

    # Arms
    l_arm = calculate_angle(get_pt(11), get_pt(13), get_pt(15)) if (11 in landmarks and 13 in landmarks and 15 in landmarks) else 160.0
    r_arm = calculate_angle(get_pt(12), get_pt(14), get_pt(16)) if (12 in landmarks and 14 in landmarks and 16 in landmarks) else 160.0

    # Legs
    l_knee = calculate_angle(get_pt(23), get_pt(25), get_pt(27)) if (23 in landmarks and 25 in landmarks and 27 in landmarks) else 170.0
    r_knee = calculate_angle(get_pt(24), get_pt(26), get_pt(28)) if (24 in landmarks and 26 in landmarks and 28 in landmarks) else 170.0

    # Torso Lean (shoulders to hips relative to vertical)
    if (11 in landmarks and 23 in landmarks):
        s, h = get_pt(11), get_pt(23)
        dx, dy = abs(s[0] - h[0]), abs(s[1] - h[1]) + 1e-5
        torso_angle = float(np.degrees(np.arctan2(dx, dy)))
    elif (12 in landmarks and 24 in landmarks):
        s, h = get_pt(12), get_pt(24)
        dx, dy = abs(s[0] - h[0]), abs(s[1] - h[1]) + 1e-5
        torso_angle = float(np.degrees(np.arctan2(dx, dy)))
    else:
        torso_angle = 0.0

    # Hand orientation & Thumbs Up gesture check
    # Check if thumb tip is prominently higher than wrist & index
    is_thumbs_up = False
    hand_state = "Neutral"

    for (w_idx, t_idx, i_idx, p_idx, side) in [(16, 22, 20, 18, "Right"), (15, 21, 19, 17, "Left")]:
        if w_idx in landmarks and t_idx in landmarks and i_idx in landmarks:
            wrist = get_pt(w_idx)
            thumb = get_pt(t_idx)
            index = get_pt(i_idx)
            
            # Thumb pointing straight up: thumb_y significantly smaller than wrist_y and index_y
            if (wrist[1] - thumb[1] > 40) and (index[1] - thumb[1] > 20):
                is_thumbs_up = True
                hand_state = "Thumbs Up 👍"
                break
            elif abs(thumb[0] - index[0]) > 25:
                hand_state = "Supinated (Palm Up)"
            else:
                hand_state = "Neutral Grip"

    # Identify primary movement
    active_angle = min(l_arm, r_arm) if exercise == "curl" else min(l_knee, r_knee)

    if is_thumbs_up:
        movement = "Thumbs Up 👍 Detected"
        posture_cue = "Gesture Active — Rep counting paused"
    elif exercise == "curl":
        if active_angle < 50:
            movement = "Bicep Curl (Peak Squeeze)"
            posture_cue = "Great peak contraction! Squeeze & lower slowly"
        elif active_angle < 120:
            movement = "Bicep Curl (In Motion)"
            posture_cue = "Keep elbows pinned to your ribs"
        else:
            movement = "Arms Extended (Starting Position)"
            posture_cue = "Good starting position, ready to curl"
    elif exercise == "squat":
        if active_angle < 100:
            movement = "Deep Squat (Bottom of Rep)"
            posture_cue = "Excellent depth! Drive through heels"
        elif active_angle < 145:
            movement = "Squatting (In Motion)"
            posture_cue = "Keep chest up and knees tracking over toes"
        else:
            movement = "Standing Upright"
            posture_cue = "Ready to squat"

    if torso_angle > 35 and not is_thumbs_up:
        movement += " [Excessive Lean ⚠️]"
        posture_cue = "Straighten your back! Don't lean too forward"

    return {
        "movement": movement,
        "left_arm": round(l_arm, 1),
        "right_arm": round(r_arm, 1),
        "left_knee": round(l_knee, 1),
        "right_knee": round(r_knee, 1),
        "torso_angle": round(torso_angle, 1),
        "active_angle": round(active_angle, 1),
        "hand_state": hand_state,
        "is_gesture": is_thumbs_up,
        "posture_cue": posture_cue
    }

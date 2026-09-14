import numpy as np

def detect_hand_orientation(landmarks, active_side="right"):
    """
    Detect hand orientation (Supinated / Neutral / Pronated)
    from MediaPipe Pose hand landmarks:
    Right: 12 (shoulder), 14 (elbow), 16 (wrist), 18 (pinky), 20 (index), 22 (thumb)
    Left:  11 (shoulder), 13 (elbow), 15 (wrist), 17 (pinky), 19 (index), 21 (thumb)
    """
    if not landmarks:
        return {"orientation": "UNKNOWN", "label": "No Hand", "angle": 0.0}

    offset = 0 if active_side == "left" else 1
    w_idx = 15 + offset
    p_idx = 17 + offset
    i_idx = 19 + offset
    t_idx = 21 + offset
    e_idx = 13 + offset

    if w_idx not in landmarks:
        return {"orientation": "UNKNOWN", "label": "Searching", "angle": 0.0}

    wrist = landmarks[w_idx]
    # Check if finger landmarks are available
    has_fingers = (p_idx in landmarks) and (i_idx in landmarks) and (t_idx in landmarks)
    has_elbow = (e_idx in landmarks)

    if not has_fingers:
        return {"orientation": "NEUTRAL", "label": "Neutral Grip", "angle": 0.0}

    pinky = landmarks[p_idx]
    index = landmarks[i_idx]
    thumb = landmarks[t_idx]

    # dx between thumb and pinky
    # In camera view:
    # For Right hand:
    # If thumb is higher (y smaller) than pinky and index is to the left: Neutral/Hammer
    # If pinky is closer to body than thumb and palm facing camera: Supinated
    dy_thumb_pinky = thumb[1] - pinky[1]
    dx_index_pinky = index[0] - pinky[0]

    # Forearm vector (elbow to wrist)
    if has_elbow:
        elbow = landmarks[e_idx]
        forearm = np.array([wrist[0] - elbow[0], wrist[1] - elbow[1]], dtype=float)
        hand_vec = np.array([index[0] - wrist[0], index[1] - wrist[1]], dtype=float)
        
        # Wrist flexion / extension angle
        cos_w = np.dot(forearm, hand_vec) / (np.linalg.norm(forearm) * np.linalg.norm(hand_vec) + 1e-7)
        wrist_angle = float(np.degrees(np.arccos(np.clip(cos_w, -1.0, 1.0))))
    else:
        wrist_angle = 180.0

    # Determine orientation
    # Thumb higher than pinky by significant margin -> Neutral/Hammer
    if active_side == "right":
        if abs(dy_thumb_pinky) < 25:
            if index[0] > pinky[0]:
                orientation = "SUPINATED"
                label = "Supinated (Palm Up)"
            else:
                orientation = "PRONATED"
                label = "Pronated (Overhand)"
        elif thumb[1] < pinky[1] - 15:
            orientation = "NEUTRAL"
            label = "Hammer Grip (Neutral)"
        else:
            orientation = "SUPINATED"
            label = "Supinated (Palm Up)"
    else:
        if abs(dy_thumb_pinky) < 25:
            if index[0] < pinky[0]:
                orientation = "SUPINATED"
                label = "Supinated (Palm Up)"
            else:
                orientation = "PRONATED"
                label = "Pronated (Overhand)"
        elif thumb[1] < pinky[1] - 15:
            orientation = "NEUTRAL"
            label = "Hammer Grip (Neutral)"
        else:
            orientation = "SUPINATED"
            label = "Supinated (Palm Up)"

    return {
        "orientation": orientation,
        "label": label,
        "wrist_angle": round(wrist_angle, 1),
        "wrist_pos": [wrist[0], wrist[1]]
    }

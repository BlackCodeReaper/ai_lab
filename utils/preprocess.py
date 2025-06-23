# utils/preprocess.py

def extract_landmarks(hand_landmarks):
    """
    Estrae e normalizza i 21 landmark della mano da MediaPipe.
    Ritorna una lista piatta: [x0, y0, ..., x20, y20]
    """
    landmarks = hand_landmarks.landmark
    base_x = landmarks[0].x
    base_y = landmarks[0].y

    normalized = []
    for lm in landmarks:
        normalized.append(lm.x - base_x)
        normalized.append(lm.y - base_y)

    return normalized
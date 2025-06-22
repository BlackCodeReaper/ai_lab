# gesture/predict.py

import cv2
import mediapipe as mp
import joblib
import numpy as np
import time
import os
from collections import deque
from utils.preprocess import extract_landmarks

mp_hands = mp.solutions.hands

MODEL_PATH = os.path.join("model", "gesture_model.pkl")

class GestureRecognizer:
    def __init__(self, static_buffer=5, motion_buffer=5):
        self.model = joblib.load(MODEL_PATH)
        self.hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
        self.static_buffer = deque(maxlen=static_buffer)
        self.motion_buffer = deque(maxlen=motion_buffer)
        self.last_position = None
        self.last_gesture = None
        self.cooldown = 1.5  # secondi
        self.last_trigger_time = time.time()

    def predict_static_gesture(self, landmarks):
        features = np.array(landmarks).reshape(1, -1)
        prediction = self.model.predict(features)[0]
        self.static_buffer.append(prediction)
        return max(set(self.static_buffer), key=self.static_buffer.count)

    def detect_motion(self, x, y):
        self.motion_buffer.append((x, y))
        if len(self.motion_buffer) < self.motion_buffer.maxlen:
            return None

        dx = self.motion_buffer[-1][0] - self.motion_buffer[0][0]
        dy = self.motion_buffer[-1][1] - self.motion_buffer[0][1]

        threshold = 0.05  # sensibilità

        if abs(dx) > abs(dy):
            if dx > threshold:
                return "DESTRA"
            elif dx < -threshold:
                return "SINISTRA"
        else:
            if dy > threshold:
                return "GIÙ"
        return None

    def recognize(self, frame):
        gesture = None
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            for hand in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            # statico
            landmarks = extract_landmarks(lm)
            static_gesture = self.predict_static_gesture(landmarks)

            # dinamico
            wrist = lm.landmark[0]
            gesture_movement = self.detect_motion(1 - wrist.x, wrist.y)

            # gestione cooldown
            now = time.time()
            if now - self.last_trigger_time > self.cooldown:
                if gesture_movement:
                    gesture = gesture_movement
                else:
                    gesture = static_gesture
                self.last_gesture = gesture
                self.last_trigger_time = now
        else:
            self.motion_buffer.clear()

        return frame, self.last_gesture
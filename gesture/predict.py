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
        self.last_static = None
        self.last_trigger_time = 0
        self.motion_cooldown = 0.8
        self.static_cooldown = 0.2

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

        threshold = 0.08
        now = time.time()
        if now - self.last_trigger_time < self.motion_cooldown:
            return None

        if abs(dx) > abs(dy):
            if dx > threshold:
                self.motion_buffer.clear()
                self.last_trigger_time = now
                return "DESTRA"
            elif dx < -threshold:
                self.motion_buffer.clear()
                self.last_trigger_time = now
                return "SINISTRA"
        elif dy > threshold:
            self.motion_buffer.clear()
            self.last_trigger_time = now
            return "GIÙ"
        return None

    
    def recognize(self, frame):
        gesture = None
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        now = time.time()

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            for hand in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            landmarks = extract_landmarks(lm)

            static = self.predict_static_gesture(landmarks)

            # Coordinate corrette del polso (invertite per il mirror)
            wrist = lm.landmark[0]
            wrist_x = 1 - wrist.x
            wrist_y = wrist.y

            # Zone
            left_zone = 0.33
            right_zone = 0.66
            down_threshold = 0.75

            gesture_candidate = None

            if wrist_x < left_zone:
                gesture_candidate = "SINISTRA"
            elif wrist_x > right_zone:
                gesture_candidate = "DESTRA"
            elif left_zone <= wrist_x <= right_zone and wrist_y > down_threshold:
                gesture_candidate = "GIÙ"

            # Applica cooldown per gesti dinamici
            if gesture_candidate and now - self.last_trigger_time > self.motion_cooldown:
                gesture = gesture_candidate
                self.last_trigger_time = now
                self.last_static = None

            # Gesti statici (pugno, OK, L)
            elif static != self.last_static and now - self.last_trigger_time > self.static_cooldown:
                gesture = static
                self.last_static = static
                self.last_trigger_time = now

            # Disegna le linee guida
            third_width = w // 3
            cv2.line(frame, (third_width, 0), (third_width, h), (255, 255, 0), 2)  # Linea verticale sinistra
            cv2.line(frame, (2 * third_width, 0), (2 * third_width, h), (255, 255, 0), 2)  # Linea verticale destra
            center_bottom_y = int(h * down_threshold)
            cv2.line(frame, (third_width, center_bottom_y), (2 * third_width, center_bottom_y), (255, 0, 255), 2)  # linea GIÙ

        return frame, gesture
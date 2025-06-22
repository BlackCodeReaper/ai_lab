# gesture/collect_data.py

import cv2
import mediapipe as mp
import csv
import os
from utils.preprocess import extract_landmarks

mp_hands = mp.solutions.hands

# Mapping tasti → etichette
LABELS = {
    ord('p'): "pugno",
    ord('o'): "Ok",
    ord('l'): "L",
    ord('n'): "nessuno"
}

OUTPUT_CSV = os.path.join("dataset", "gestures.csv")

def main():
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=1,
                           min_detection_confidence=0.5)

    # Crea file CSV se non esiste
    if not os.path.exists("dataset"):
        os.makedirs("dataset")

    file_exists = os.path.exists(OUTPUT_CSV)
    with open(OUTPUT_CSV, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            # intestazione: label,x0,y0,z0,...,x20,y20,z20
            header = ["label"]
            for i in range(21):
                header += [f"x{i}", f"y{i}", f"z{i}"]
            writer.writerow(header)

        print("Premi 'p' (pugno), 'o' (Ok), 'l' (L), 'n' (nessuno) per salvare il gesto. Premi 'q' per uscire.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                landmarks = extract_landmarks(results.multi_hand_landmarks[0])
                for lm in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

            cv2.imshow("Collect Data", frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key in LABELS and results.multi_hand_landmarks:
                label = LABELS[key]
                row = [label] + landmarks
                writer.writerow(row)
                print(f"[+] Salvato gesto: {label}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
# gesture/train_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np
import joblib
import os

CSV_PATH = os.path.join("dataset", "gestures.csv")
MODEL_PATH = os.path.join("model", "gesture_model.pkl")

def main():
    if not os.path.exists(CSV_PATH):
        print(f"[!] Il file {CSV_PATH} non esiste. Raccogli prima i dati con collect_data.py.")
        return

    df = pd.read_csv(CSV_PATH)

    X = df.drop("label", axis=1)
    y = df["label"]

    # Modello Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(clf, X, y, cv=cv)

    print("\n== REPORT ==")
    print("Accuracy per ogni fold:", scores)
    print("Media:", np.mean(scores))
    print("Deviazione standard:", np.std(scores))

    # Salvataggio modello
    if not os.path.exists("model"):
        os.makedirs("model")

    joblib.dump(clf, MODEL_PATH)
    print(f"\n[✓] Modello salvato in {MODEL_PATH}")

if __name__ == "__main__":
    main()
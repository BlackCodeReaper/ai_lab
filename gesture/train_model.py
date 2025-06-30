# gesture/train_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
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

    # Divisione train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Modello Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(clf, X_train, y_train, cv=cv)

    print("\n== REPORT ==")
    print("Accuracy per ogni fold:", scores)
    print("Media:", np.mean(scores))
    print("Deviazione standard:", np.std(scores))

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("Accuracy sul test set:", model.score(X_test, y_test))

    # Salvataggio modello
    if not os.path.exists("model"):
        os.makedirs("model")

    joblib.dump(model, MODEL_PATH)
    print(f"\n[✓] Modello salvato in {MODEL_PATH}")

if __name__ == "__main__":
    main()
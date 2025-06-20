# gesture/train_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modello Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Valutazione
    y_pred = clf.predict(X_test)
    print("\n== REPORT ==")
    print(classification_report(y_test, y_pred))

    # Salvataggio modello
    if not os.path.exists("model"):
        os.makedirs("model")

    joblib.dump(clf, MODEL_PATH)
    print(f"[✓] Modello salvato in {MODEL_PATH}")

if __name__ == "__main__":
    main()
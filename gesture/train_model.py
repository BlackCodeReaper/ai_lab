# gesture/train_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, log_loss
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
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

    # Modello Random Forest e cross-validation
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(clf, X_train, y_train, cv=cv)

    print("\n== REPORT ==")
    print("Accuracy per ogni fold:", scores)
    print("Media:", np.mean(scores))
    print("Deviazione standard:", np.std(scores))

    # Grafico accuracy per fold
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(scores)+1), scores, marker='o', linestyle='--')
    plt.title("Andamento della Accuracy per Fold")
    plt.xlabel("Fold #")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.xticks(range(1, len(scores)+1))
    plt.ylim(0.99, 1)
    plt.show()

    # Grafico cumulative accuracy
    cumulative_avg = np.cumsum(scores) / (np.arange(len(scores)) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(scores)+1), cumulative_avg, marker='o', linestyle='--')
    plt.title("Andamento della media dell'accuracy per Fold")
    plt.xlabel("Fold #")
    plt.ylabel("Media accuracy")
    plt.grid(True)
    plt.xticks(range(1, len(scores)+1))
    plt.ylim(0.99, 1)
    plt.show()

    # Grafico cumulative accuracy
    cumulative_std = [np.std(scores[:i+1]) for i in range(len(scores))]
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(scores)+1), cumulative_std, marker='o', linestyle='--')
    plt.title("Andamento della Deviazione standard per Fold")
    plt.xlabel("Fold #")
    plt.ylabel("Deviazione standard")
    plt.grid(True)
    plt.xticks(range(1, len(scores)+1))
    plt.ylim(0, 0.003)
    plt.show()
    
    loss_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='neg_log_loss')
    loss_scores = -loss_scores  # perché scikit-learn restituisce valori negativi

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_scores)+1), loss_scores, marker='o', linestyle='--')
    plt.title("Log Loss per Fold")
    plt.xlabel("Fold #")
    plt.ylabel("Log Loss")
    plt.grid(True)
    plt.show()

    # Fit finale su tutti i dati di training
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    test_accuracy = model.score(X_test, y_test)
    train_accuracy = model.score(X_train, y_train)

    plt.figure(figsize=(6, 4))
    plt.bar(['Training Set', 'Test Set'], [train_accuracy, test_accuracy], color=['skyblue', 'orange'])
    plt.title('Accuracy del Modello Finale')
    plt.ylabel('Accuracy')
    plt.grid(axis='y')
    plt.ylim(0.99, 1.0)
    plt.show()
    print("Accuracy sul test set:", test_accuracy)

    # Confusion Matrix
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    le = LabelEncoder()
    le.fit(y)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot()
    plt.title("Confusion Matrix sul Test Set")
    plt.show()

    # Visualizzazione t-SNE (solo sul test set per semplicità)
    tsne = TSNE(n_components=2, random_state=42)
    X_test_2d = tsne.fit_transform(X_test)

    # Encoding label per il colore
    y_test_encoded = le.transform(y_test)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test_encoded, cmap='tab10', alpha=0.7)
    plt.title("Visualizzazione t-SNE (Test Set)")
    plt.xlabel("Dimensione 1")
    plt.ylabel("Dimensione 2")

    # Legenda
    colors = [scatter.cmap(scatter.norm(i)) for i in range(len(le.classes_))]
    handles = [mpatches.Patch(color=colors[i], label=le.classes_[i])
               for i in range(len(le.classes_))]
    plt.legend(handles=handles, title="Classi")
    plt.show()

    # Salvataggio modello
    if not os.path.exists("model"):
        os.makedirs("model")

    joblib.dump(model, MODEL_PATH)
    print(f"\n[✓] Modello salvato in {MODEL_PATH}")

if __name__ == "__main__":
    main()
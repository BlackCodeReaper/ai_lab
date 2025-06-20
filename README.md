# 🎮 Tetris con Gesti della Mano (Webcam + AI)

Controlla il gioco del Tetris usando i gesti delle mani attraverso la webcam! Il progetto utilizza:

- **MediaPipe + OpenCV** per tracciare la mano
- **Scikit-learn** per riconoscere 3 gesti statici: 🥊 pugno, ✋ mano aperta, 👆 "L"
- **Pygame** per visualizzare il Tetris e gestire input simulati

## 📁 Struttura del Progetto

```
tetris_gesture_control/
├── game/
│   └── tetris.py              # Logica del gioco Tetris
├── gesture/
│   ├── collect_data.py        # Raccoglie i gesti da webcam
│   ├── train_model.py         # Allena il modello ML
│   ├── predict.py             # Predice gesto in tempo reale
├── model/
│   └── gesture_model.pkl      # Modello salvato
├── utils/
│   └── preprocess.py          # Estrazione e normalizzazione landmark
├── dataset/
│   └── gestures.csv           # CSV dei landmark e delle etichette
└── main.py                    # Avvia webcam + Tetris
```

## 📦 Requisiti

Installa tutte le dipendenze:

```
pip install mediapipe opencv-python pygame scikit-learn numpy joblib pandas
```

## 🧪 Uso del Progetto

### 1. Raccogli i dati dei gesti

Avvia la raccolta e premi:
- `p` → pugno
- `o` → mano aperta
- `l` → gesto "L"
- `q` → per uscire

```
python gesture/collect_data.py
```

👉 Questo crea/aggiorna il file `dataset/gestures.csv`

### 2. Allena il modello ML

Assicurati di avere abbastanza esempi per ogni gesto (almeno 50–100).

```
python gesture/train_model.py
```

👉 Questo crea il file `model/gesture_model.pkl`

### 3. Avvia il gioco controllato dai gesti

```
python main.py
```

👉 A sinistra appare la webcam con il gesto riconosciuto  
👉 A destra il gioco Tetris controllabile con i gesti

## 🖐️ Gesti Riconosciuti

| Tipo       | Gesto                  | Azione in gioco          |
|------------|------------------------|---------------------------|
| Dinamico   | Mano → sinistra        | Sposta blocco a sinistra |
| Dinamico   | Mano → destra          | Sposta blocco a destra   |
| Dinamico   | Mano → giù             | Fa scendere il blocco    |
| Statico ML | ✊ Pugno                | Pausa                    |
| Statico ML | ✋ Mano aperta (5 dita) | Ruota antiorario         |
| Statico ML | 👆 "L" (indice+pollice) | Ruota orario             |

## 🎮 Tasti Mappati

| Gesto        | Tasto simulato        |
|--------------|------------------------|
| Sinistra     | `pygame.K_LEFT` / `a` |
| Destra       | `pygame.K_RIGHT` / `d` |
| Giù          | `pygame.K_DOWN` / `s` |
| Pugno        | `p` (pausa)           |
| Mano aperta  | `q` (ruota ⟲)         |
| L            | `e` (ruota ⟳)         |

## 👨‍💻 Compatibilità

✅ Testato su:
- macOS (incluso Mac Mini M4)
- Python 3.10–3.11
- Visual Studio Code
- Webcam USB o integrata

## 🧠 Suggerimenti

- Mantieni la mano ben visibile al centro della webcam
- Raccogli i dati in condizioni di luce costante
- Durante il gioco evita gesti misti: uno alla volta

Buon divertimento! 🎉
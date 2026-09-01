"""
ECG Heart Disease Detection
STEP 2 - Preprocess ECG Signals
Run: python preprocess.py
"""
import os, pickle, warnings
warnings.filterwarnings('ignore')
import numpy as np
import cv2
import wfdb
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import LabelEncoder

print("=" * 50)
print("  Preprocessing ECG Signals...")
print("=" * 50)

DATA_DIR  = "mitbih_data"
SEQ_LEN   = 360
IMG_SIZE  = (64, 64)
MAX_BEATS = 1000

CLASSES = {
    'N': 'Normal', 'L': 'LBBB', 'R': 'RBBB',
    'A': 'APB',    'V': 'PVC',  'F': 'Fusion',
}

RECORD_IDS = [
    100,101,102,103,104,105,106,107,108,109,
    111,112,113,114,115,116,117,118,119,121,
    122,123,124,200,201,202,203,205,207,208,
    209,210,212,213,214,215,217,219,220,221,
    222,223,228,230,231,232,233,234
]

def bandpass_filter(signal):
    nyq = 180.0
    b, a = butter(4, [0.5/nyq, 40.0/nyq], btype='band')
    return filtfilt(b, a, signal)

def beat_to_image(beat):
    h, w = IMG_SIZE
    canvas = np.full((h, w), 255, dtype=np.uint8)
    xs = np.linspace(0, w-1, len(beat)).astype(np.int32)
    ys = np.clip((h-1 - beat*(h-1)).astype(np.int32), 0, h-1)
    pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
    cv2.polylines(canvas, [pts], False, 0, 2, cv2.LINE_AA)
    return canvas

X_img, X_seq, y_raw = [], [], []
class_counts = {}
half = SEQ_LEN // 2

for rid in RECORD_IDS:
    try:
        rec = wfdb.rdrecord(os.path.join(DATA_DIR, str(rid)))
        ann = wfdb.rdann(os.path.join(DATA_DIR, str(rid)), 'atr')
        signal = rec.p_signal[:, 0]
        signal = bandpass_filter(signal)
        for r, label in zip(ann.sample, ann.symbol):
            if label not in CLASSES:
                continue
            cls = CLASSES[label]
            class_counts[cls] = class_counts.get(cls, 0)
            if class_counts[cls] >= MAX_BEATS:
                continue
            start, end = r - half, r + half
            if start < 0 or end > len(signal):
                continue
            beat = signal[start:end]
            beat = (beat - beat.min()) / (beat.max() - beat.min() + 1e-8)
            X_img.append(beat_to_image(beat)[..., np.newaxis])
            X_seq.append(beat)
            y_raw.append(cls)
            class_counts[cls] += 1
        print(f"  Record {rid} done")
    except Exception as e:
        print(f"  Skip {rid}: {e}")

X_img = np.array(X_img, dtype=np.float32) / 255.0
X_seq = np.array(X_seq, dtype=np.float32)
le    = LabelEncoder()
y     = le.fit_transform(np.array(y_raw))

os.makedirs("processed_data", exist_ok=True)
np.save("processed_data/X_img.npy", X_img)
np.save("processed_data/X_seq.npy", X_seq)
np.save("processed_data/y.npy", y)
with open("processed_data/label_encoder.pkl", 'wb') as f:
    pickle.dump(le, f)

print(f"\nTotal beats : {len(y)}")
print(f"Classes     : {list(le.classes_)}")
print("Done! Next: python train_model.py")
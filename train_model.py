"""
ECG Heart Disease Detection
STEP 3 - Train CNN + LSTM Model
Run: python train_model.py
"""
import os, pickle, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

print("=" * 50)
print("  Training CNN + LSTM Model...")
print("=" * 50)

IMG_SIZE   = (64, 64)
SEQ_LEN    = 360
EPOCHS     = 15
BATCH_SIZE = 64

print("Loading data...")
X_img = np.load("processed_data/X_img.npy")
X_seq = np.load("processed_data/X_seq.npy")
y     = np.load("processed_data/y.npy")
with open("processed_data/label_encoder.pkl", 'rb') as f:
    le = pickle.load(f)

num_classes = len(le.classes_)
print(f"Samples : {len(y)}")
print(f"Classes : {list(le.classes_)}")

X_img_tr, X_img_val, X_seq_tr, X_seq_val, y_tr, y_val = \
    train_test_split(X_img, X_seq, y, test_size=0.2,
                     stratify=y, random_state=42)

def build_model(num_classes):
    img_in = layers.Input(shape=(*IMG_SIZE, 1), name="image")
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(img_in)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)

    seq_in = layers.Input(shape=(SEQ_LEN,), name="sequence")
    s = layers.Reshape((SEQ_LEN, 1))(seq_in)
    s = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.2))(s)
    s = layers.Bidirectional(layers.LSTM(32, dropout=0.2))(s)
    s = layers.Dense(128, activation='relu')(s)
    s = layers.Dropout(0.3)(s)

    merged = layers.Concatenate()([x, s])
    out    = layers.Dense(256, activation='relu')(merged)
    out    = layers.Dropout(0.3)(out)
    out    = layers.Dense(num_classes, activation='softmax')(out)

    model = Model(inputs=[img_in, seq_in], outputs=out, name="CardioAI")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = build_model(num_classes)
model.summary()

os.makedirs("models", exist_ok=True)
callbacks = [
    ModelCheckpoint("models/best_model.keras",
                    monitor='val_accuracy',
                    save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=5,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=3, verbose=1),
]

print(f"\nTraining ({EPOCHS} epochs)...")
history = model.fit(
    [X_img_tr, X_seq_tr], y_tr,
    validation_data=([X_img_val, X_seq_val], y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

best = max(history.history['val_accuracy'])
print(f"\nBest Accuracy: {best*100:.2f}%")
print("Done! Next: python predict_gui.py")
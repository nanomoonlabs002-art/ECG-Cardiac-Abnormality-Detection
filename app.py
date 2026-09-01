import os
import pickle
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import cv2
import streamlit as st
import tensorflow as tf


# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = "models/best_model.keras"
ENCODER_PATH = "processed_data/label_encoder.pkl"

IMG_SIZE = (64, 64)
SEQ_LEN = 360
NORMAL_CLASSES = ["Normal"]


# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model_and_encoder():
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(ENCODER_PATH, "rb") as f:
        le = pickle.load(f)

    return model, le


# -----------------------------
# Prediction Function
# -----------------------------
def predict(image_file, model, le):

    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Cannot read ECG image!")

    img = cv2.resize(img, IMG_SIZE)

    # Image input
    X_img = (img.astype(np.float32) / 255.0)[
        np.newaxis, ..., np.newaxis
    ]

    # Sequence extraction
    rows, cols_arr = np.where(img < 128)

    beat = np.zeros(SEQ_LEN, dtype=np.float32)

    for i, col in enumerate(
        np.linspace(0, img.shape[1] - 1, SEQ_LEN).astype(int)
    ):
        col_pixels = rows[cols_arr == col]

        if len(col_pixels) > 0:
            beat[i] = 1.0 - col_pixels.mean() / img.shape[0]

    X_seq = beat[np.newaxis, :]

    # Prediction
    probs = model.predict([X_img, X_seq], verbose=0)[0]

    top_idx = int(np.argmax(probs))
    top_cls = le.inverse_transform([top_idx])[0]
    top_conf = probs[top_idx] * 100

    is_normal = top_cls in NORMAL_CLASSES

    result = (
        "Normal Beat"
        if is_normal
        else "Arrhythmia Detected"
    )

    top3 = [
        (le.inverse_transform([i])[0], probs[i] * 100)
        for i in np.argsort(probs)[::-1][:3]
    ]

    return result, is_normal, top_cls, top_conf, top3


# -----------------------------
# Web Page
# -----------------------------
st.set_page_config(
    page_title="CardioAI - ECG Detection",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ CardioAI")
st.subheader("ECG Heart Disease Detection")

st.write(
    "Upload an ECG image to detect whether the ECG "
    "shows a Normal Beat or Arrhythmia."
)

uploaded_file = st.file_uploader(
    "Upload ECG Image",
    type=["png", "jpg", "jpeg", "bmp"]
)


# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Uploaded ECG Image",
        use_container_width=True
    )

    if st.button("🔍 Predict ECG"):

        try:

            with st.spinner("Analyzing ECG..."):

                model, le = load_model_and_encoder()

                result, is_normal, top_cls, conf, top3 = predict(
                    uploaded_file,
                    model,
                    le
                )

            if is_normal:
                st.success(
                    f"### {result}\n\n"
                    f"Confidence: {conf:.1f}%"
                )
            else:
                st.error(
                    f"### {result}\n\n"
                    f"Confidence: {conf:.1f}%"
                )

            st.write(f"**Predicted Class:** {top_cls}")

            st.write("### Top Predictions")

            for cls, prob in top3:
                st.write(f"**{cls}** — {prob:.1f}%")

        except Exception as e:

            st.error(f"Prediction Error: {e}")


st.caption(
    "For educational purposes only. "
    "Not a substitute for clinical diagnosis."
)
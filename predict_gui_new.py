"""
ECG Heart Disease Detection - Prediction GUI
Run: python predict_gui_new.py
"""
import os, pickle, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf

MODEL_PATH   = "models/best_model.keras"
ENCODER_PATH = "processed_data/label_encoder.pkl"
IMG_SIZE     = (64, 64)
SEQ_LEN      = 360
NORMAL_CLASSES = ['Normal']

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
with open(ENCODER_PATH, 'rb') as f:
    le = pickle.load(f)
print("Ready!")

def predict(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Cannot read image!")
    img = cv2.resize(img, IMG_SIZE)
    X_img = (img.astype(np.float32) / 255.0)[np.newaxis, ..., np.newaxis]
    rows, cols_arr = np.where(img < 128)
    beat = np.zeros(SEQ_LEN, dtype=np.float32)
    for i, col in enumerate(np.linspace(0, img.shape[1]-1, SEQ_LEN).astype(int)):
        col_pixels = rows[cols_arr == col]
        if len(col_pixels) > 0:
            beat[i] = 1.0 - col_pixels.mean() / img.shape[0]
    X_seq = beat[np.newaxis, :]
    probs = model.predict([X_img, X_seq], verbose=0)[0]
    top_idx = int(np.argmax(probs))
    top_cls = le.inverse_transform([top_idx])[0]
    top_conf = probs[top_idx] * 100
    is_normal = top_cls in NORMAL_CLASSES
    result = "Normal Beat" if is_normal else "Arrhythmia Detected"
    top3 = [(le.inverse_transform([i])[0], probs[i]*100)
            for i in np.argsort(probs)[::-1][:3]]
    return result, is_normal, top_cls, top_conf, top3

root = tk.Tk()
root.title("CardioAI - ECG Heart Disease Detection")
root.state("zoomed")
root.configure(bg='#1a1a2e')
root.resizable(True, True)

hdr = tk.Frame(root, bg='#e53e3e', pady=15)
hdr.pack(fill='x')
tk.Label(hdr, text="CardioAI - ECG Heart Disease Detection",
         font=('Segoe UI', 16, 'bold'), bg='#e53e3e', fg='white').pack()
tk.Label(hdr, text="Upload ECG image - Normal Beat or Arrhythmia Detected",
         font=('Segoe UI', 10), bg='#e53e3e', fg='#ffe0e0').pack()

btn_f = tk.Frame(root, bg='#1a1a2e', pady=12)
btn_f.pack(fill='x')

prev_f = tk.Frame(root, bg='#16213e')
prev_f.pack(padx=30, fill='x')
img_lbl = tk.Label(prev_f, text="No image uploaded yet",
                    bg='#16213e', fg='#718096',
                    font=('Segoe UI', 11), height=6)
img_lbl.pack(pady=6)

res_frm = tk.Frame(root, bg='#1a1a2e', pady=8)
res_frm.pack(fill='x', padx=30)
res_lbl = tk.Label(res_frm, text="", font=('Segoe UI', 22, 'bold'),
                    bg='#1a1a2e', fg='white')
res_lbl.pack()
conf_lbl = tk.Label(res_frm, text="", font=('Segoe UI', 12),
                     bg='#1a1a2e', fg='#a0aec0')
conf_lbl.pack()
cls_lbl = tk.Label(res_frm, text="", font=('Segoe UI', 11),
                    bg='#1a1a2e', fg='#a0aec0')
cls_lbl.pack(pady=2)

top3_frm = tk.Frame(root, bg='#1a1a2e')
top3_frm.pack(fill='x', padx=30, pady=4)

def upload_image():
    path = filedialog.askopenfilename(
        title="Select ECG Image",
        filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp"), ("All", "*.*")]
    )
    if not path:
        return
    try:
        pil = Image.open(path)
        pil.thumbnail((680, 130))
        tk_img = ImageTk.PhotoImage(pil)
        img_lbl.configure(image=tk_img, text="", height=1)
        img_lbl.image = tk_img
    except Exception:
        img_lbl.configure(text="[Preview unavailable]", height=5)
    try:
        result, is_normal, top_cls, conf, top3 = predict(path)
    except Exception as e:
        messagebox.showerror("Error", str(e))
        return
    color = '#27ae60' if is_normal else '#e53e3e'
    icon = "Normal Beat" if is_normal else "Arrhythmia Detected"
    res_frm.configure(bg=color)
    res_lbl.configure(text=icon, bg=color, fg='white')
    conf_lbl.configure(text=f"Confidence: {conf:.1f}%", bg=color, fg='white')
    cls_lbl.configure(text=f"Class: {top_cls}", bg=color,
                       fg='#d0ffd0' if is_normal else '#ffd0d0')
    for w in top3_frm.winfo_children():
        w.destroy()
    tk.Label(top3_frm, text="Top Predictions:",
             font=('Segoe UI', 10, 'bold'),
             bg='#1a1a2e', fg='#718096').pack(anchor='w')
    for cls, prob in top3:
        row = tk.Frame(top3_frm, bg='#1a1a2e')
        row.pack(fill='x', pady=1)
        tk.Label(row, text=f"  {cls}", font=('Segoe UI', 10),
                 width=16, anchor='w', bg='#1a1a2e', fg='#eaeaea').pack(side='left')
        bg_bar = tk.Frame(row, bg='#2d3748', width=280, height=14)
        bg_bar.pack(side='left', padx=4)
        bg_bar.pack_propagate(False)
        tk.Frame(bg_bar,
                 bg='#27ae60' if cls in NORMAL_CLASSES else '#e53e3e',
                 width=int(prob*2.8), height=14).pack(side='left')
        tk.Label(row, text=f"{prob:.1f}%", font=('Segoe UI', 10),
                 bg='#1a1a2e', fg='#eaeaea').pack(side='left', padx=4)

tk.Button(btn_f, text="Upload ECG Image",
          font=('Segoe UI', 13, 'bold'),
          bg='#2b6cb0', fg='white',
          activebackground='#3182ce',
          relief='flat', padx=20, pady=10,
          cursor='hand2',
          command=upload_image).pack()

tk.Label(root,
         text="For educational purposes only. Not a substitute for clinical diagnosis.",
         font=('Segoe UI', 9), bg='#1a1a2e', fg='#4a5568').pack(side='bottom', pady=6)

root.mainloop()

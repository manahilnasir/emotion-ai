"""
Emotion AI
Run: python app.py
Then open http://127.0.0.1:5000
"""

import os
import cv2
import joblib
import numpy as np
import threading
import time
import sqlite3
from datetime import datetime

from flask import Flask, send_from_directory, Response, jsonify, request
from keras.models import model_from_json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=os.path.join(BASE_DIR, "website"))

# ════════════════════════════════════════════════════
#  DATABASE SETUP
# ════════════════════════════════════════════════════
DB_PATH = os.path.join(BASE_DIR, "history.db")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS facial_history (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            emotion    TEXT    NOT NULL,
            confidence REAL    NOT NULL,
            timestamp  TEXT    NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS text_history (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            input_text TEXT    NOT NULL,
            emotion    TEXT    NOT NULL,
            confidence REAL    NOT NULL,
            timestamp  TEXT    NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def save_facial(emotion, confidence):
    conn = get_db()
    conn.execute(
        "INSERT INTO facial_history (emotion, confidence, timestamp) VALUES (?, ?, ?)",
        (emotion, confidence, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()

def save_text(input_text, emotion, confidence):
    conn = get_db()
    conn.execute(
        "INSERT INTO text_history (input_text, emotion, confidence, timestamp) VALUES (?, ?, ?, ?)",
        (input_text, emotion, confidence, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()

init_db()

# ════════════════════════════════════════════════════
#  FACIAL EMOTION MODEL
# ════════════════════════════════════════════════════
facial_dir = os.path.join(BASE_DIR, "facial_emotion")

json_file = open(os.path.join(facial_dir, "emotion_model.json"), "r")
facial_model = model_from_json(json_file.read())
json_file.close()
facial_model.load_weights(os.path.join(facial_dir, "emotion_model.h5"))

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

FACE_LABELS = {0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4:"Neutral", 5:"Sad", 6:"Surprise"}
FACE_COLORS = {
    "Angry":(30,30,220), "Disgust":(30,180,60), "Fear":(180,30,200),
    "Happy":(30,210,210), "Neutral":(180,180,180), "Sad":(220,120,30), "Surprise":(30,190,255),
}

face_lock = threading.Lock()
current_emotion = {"label": "—", "confidence": 0.0, "all": {}}


def extract_features(image):
    return np.array(image).reshape(1, 48, 48, 1) / 255.0


def generate_frames():
    webcam = cv2.VideoCapture(0)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while True:
        success, frame = webcam.read()
        if not success:
            break
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        detected_label, detected_conf, all_scores = "—", 0.0, {}
        for (x, y, w, h) in faces:
            face_img = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
            pred     = facial_model.predict(extract_features(face_img), verbose=0)[0]
            idx      = pred.argmax()
            label    = FACE_LABELS[idx]
            conf     = float(pred[idx]) * 100
            all_scores = {FACE_LABELS[i]: round(float(pred[i])*100, 1) for i in range(7)}
            detected_label, detected_conf = label, round(conf, 1)
            color = FACE_COLORS.get(label, (255,255,255))
            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
            text = f"{label}  {conf:.0f}%"
            font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
            (tw, tht), _ = cv2.getTextSize(text, font, fs, th)
            pad = 6
            cv2.rectangle(frame, (x-1, y-tht-pad*2-4), (x+tw+pad*2, y-2), color, -1)
            cv2.putText(frame, text, (x+pad, y-pad-2), font, fs, (255,255,255), th, cv2.LINE_AA)
        with face_lock:
            # save to DB only when a real face is detected and emotion changed
            if detected_label != "—" and detected_label != current_emotion.get("label"):
                save_facial(detected_label, detected_conf)
            current_emotion.update({"label": detected_label, "confidence": detected_conf, "all": all_scores})
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
        time.sleep(0.03)


# ════════════════════════════════════════════════════
#  TEXT EMOTION MODEL
# ════════════════════════════════════════════════════
text_dir = os.path.join(BASE_DIR, "text_emotion")
pipe_lr  = joblib.load(os.path.join(text_dir, "text_emotion.pkl"))

EMOJI_MAP = {"anger":"😠","disgust":"🤮","fear":"😨","joy":"😂","neutral":"😐","sadness":"😔","shame":"😳","surprise":"😮"}
COLOR_MAP = {"anger":"#e84444","disgust":"#44cc6e","fear":"#b44aee","joy":"#f9c74f","neutral":"#a0a0b8","sadness":"#4a9eff","shame":"#ff7f3e","surprise":"#ff9f43"}


def predict_text_emotion(text):
    pred       = pipe_lr.predict([text])[0]
    probas     = pipe_lr.predict_proba([text])[0]
    all_scores = {cls: round(float(p)*100, 1) for cls, p in zip(pipe_lr.classes_, probas)}
    confidence = round(float(np.max(probas))*100, 1)
    return pred, confidence, all_scores


# ════════════════════════════════════════════════════
#  ROUTES — Website pages
# ════════════════════════════════════════════════════
@app.route("/")
def home():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(app.static_folder, filename)


# ════════════════════════════════════════════════════
#  ROUTES — Facial API
# ════════════════════════════════════════════════════
@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/emotion_data")
def emotion_data():
    with face_lock:
        return jsonify(current_emotion)


# ════════════════════════════════════════════════════
#  ROUTES — Text API
# ════════════════════════════════════════════════════
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    label, confidence, all_scores = predict_text_emotion(text)
    save_text(text, label, confidence)   # ← save to DB
    return jsonify({
        "label":      label,
        "emoji":      EMOJI_MAP.get(label, "❓"),
        "color":      COLOR_MAP.get(label, "#c9a84c"),
        "confidence": confidence,
        "all":        all_scores,
    })


# ════════════════════════════════════════════════════
#  ROUTES — History API
# ════════════════════════════════════════════════════
@app.route("/history/facial")
def facial_history():
    conn = get_db()
    rows = conn.execute(
        "SELECT id, emotion, confidence, timestamp FROM facial_history ORDER BY id DESC LIMIT 50"
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route("/history/text")
def text_history():
    conn = get_db()
    rows = conn.execute(
        "SELECT id, input_text, emotion, confidence, timestamp FROM text_history ORDER BY id DESC LIMIT 50"
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route("/history/clear", methods=["POST"])
def clear_history():
    which = request.get_json().get("type", "all")
    conn = get_db()
    if which in ("facial", "all"):
        conn.execute("DELETE FROM facial_history")
    if which in ("text", "all"):
        conn.execute("DELETE FROM text_history")
    conn.commit()
    conn.close()
    return jsonify({"status": "cleared"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\nEmotion AI running on port {port}\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)

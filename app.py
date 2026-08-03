# -*- coding: utf-8 -*-
import os, cv2, joblib, numpy as np, base64, sqlite3
from datetime import datetime
from flask import Flask, send_from_directory, jsonify, request
from keras.models import model_from_json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=os.path.join(BASE_DIR, "website"), static_url_path="")

# ── DATABASE ──────────────────────────────────────────────────────────────
DB_PATH = os.path.join(BASE_DIR, "history.db")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute(
        "CREATE TABLE IF NOT EXISTS facial_history ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "emotion TEXT NOT NULL, confidence REAL NOT NULL, timestamp TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS text_history ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "input_text TEXT NOT NULL, emotion TEXT NOT NULL,"
        "confidence REAL NOT NULL, timestamp TEXT NOT NULL)"
    )
    conn.commit()
    conn.close()

def save_facial(emotion, confidence):
    conn = get_db()
    conn.execute(
        "INSERT INTO facial_history (emotion,confidence,timestamp) VALUES (?,?,?)",
        (emotion, confidence, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

def save_text(input_text, emotion, confidence):
    conn = get_db()
    conn.execute(
        "INSERT INTO text_history (input_text,emotion,confidence,timestamp) VALUES (?,?,?,?)",
        (input_text, emotion, confidence, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

init_db()

# ── FACIAL MODEL ──────────────────────────────────────────────────────────
facial_dir = os.path.join(BASE_DIR, "facial_emotion")
with open(os.path.join(facial_dir, "emotion_model.json"), "r") as jf:
    facial_model = model_from_json(jf.read())
facial_model.load_weights(os.path.join(facial_dir, "emotion_model.h5"))

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

FACE_LABELS = {0:"Angry",1:"Disgust",2:"Fear",3:"Happy",4:"Neutral",5:"Sad",6:"Surprise"}

def extract_features(img):
    return np.array(img).reshape(1, 48, 48, 1) / 255.0

# ── TEXT MODEL ────────────────────────────────────────────────────────────
text_dir = os.path.join(BASE_DIR, "text_emotion")
pipe_lr   = joblib.load(os.path.join(text_dir, "text_emotion.pkl"))

EMOJI_MAP = {
    "anger":   "\U0001F620",
    "disgust": "\U0001F922",
    "fear":    "\U0001F628",
    "joy":     "\U0001F602",
    "neutral": "\U0001F610",
    "sadness": "\U0001F622",
    "shame":   "\U0001F633",
    "surprise":"\U0001F62E",
}
COLOR_MAP = {
    "anger":"#e84444","disgust":"#44cc6e","fear":"#b44aee","joy":"#f9c74f",
    "neutral":"#a0a0b8","sadness":"#4a9eff","shame":"#ff7f3e","surprise":"#ff9f43",
}

def predict_text_emotion(text):
    pred       = pipe_lr.predict([text])[0]
    probas     = pipe_lr.predict_proba([text])[0]
    all_scores = {cls: round(float(p)*100,1) for cls,p in zip(pipe_lr.classes_, probas)}
    confidence = round(float(np.max(probas))*100, 1)
    return pred, confidence, all_scores

# ── STATIC PAGES ──────────────────────────────────────────────────────────
@app.route("/")
def home():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

# ── FACIAL API ────────────────────────────────────────────────────────────
_last_saved = {"label": None}

@app.route("/predict-face", methods=["POST"])
def predict_face():
    data     = request.get_json(force=True)
    img_data = data.get("image", "")

    # Strip data-URL prefix e.g. "data:image/jpeg;base64,..."
    if "," in img_data:
        img_data = img_data.split(",", 1)[1]

    # Decode base64 -> OpenCV image
    try:
        raw   = base64.b64decode(img_data)
        nparr = np.frombuffer(raw, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("imdecode returned None")
    except Exception as exc:
        return jsonify({"label":"No Face","confidence":0,"all":{},"box":None})

    ih, iw = frame.shape[:2]

    # Equalise histogram for better detection in low-light / webcam conditions
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        gray_eq, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

    if len(faces) == 0:
        return jsonify({"label":"No Face","confidence":0,"all":{},"box":None})

    x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
    face_img   = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
    pred       = facial_model.predict(extract_features(face_img), verbose=0)[0]
    idx        = int(pred.argmax())
    label      = FACE_LABELS[idx]
    conf       = round(float(pred[idx])*100, 1)
    all_scores = {FACE_LABELS[i]: round(float(pred[i])*100,1) for i in range(7)}
    box        = {
        "x": round(x/iw, 4), "y": round(y/ih, 4),
        "w": round(w/iw, 4), "h": round(h/ih, 4),
    }

    if data.get("save") and label != _last_saved["label"]:
        save_facial(label, conf)
        _last_saved["label"] = label

    return jsonify({"label":label,"confidence":conf,"all":all_scores,"box":box})

# ── TEXT API ──────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error":"No text provided"}), 400
    label, confidence, all_scores = predict_text_emotion(text)
    save_text(text, label, confidence)
    return jsonify({
        "label":      label,
        "emoji":      EMOJI_MAP.get(label, "?"),
        "color":      COLOR_MAP.get(label, "#c9a84c"),
        "confidence": confidence,
        "all":        all_scores,
    })

# ── HISTORY API ───────────────────────────────────────────────────────────
@app.route("/history/facial")
def facial_history():
    conn = get_db()
    rows = conn.execute(
        "SELECT id,emotion,confidence,timestamp FROM facial_history ORDER BY id DESC LIMIT 50"
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route("/history/text")
def text_history():
    conn = get_db()
    rows = conn.execute(
        "SELECT id,input_text,emotion,confidence,timestamp FROM text_history ORDER BY id DESC LIMIT 50"
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route("/history/clear", methods=["POST"])
def clear_history():
    which = request.get_json().get("type", "all")
    conn  = get_db()
    if which in ("facial","all"): conn.execute("DELETE FROM facial_history")
    if which in ("text","all"):   conn.execute("DELETE FROM text_history")
    conn.commit()
    conn.close()
    return jsonify({"status":"cleared"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("Emotion AI running on port {}".format(port))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)

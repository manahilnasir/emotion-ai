import os
import joblib
import numpy as np
from flask import Flask, request, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

# ── Load model ────────────────────────────────────────────────────────────────
pipe_lr = joblib.load(os.path.join(BASE_DIR, "text_emotion.pkl"))

# ── Emotion metadata ──────────────────────────────────────────────────────────
EMOJI_MAP = {
    "anger": "😠", "disgust": "🤮", "fear": "😨", "happy": "😄",
    "joy": "😂", "neutral": "😐", "sadness": "😔", "shame": "😳", "surprise": "😮",
}

COLOR_MAP = {
    "anger": "#e84444", "disgust": "#44cc6e", "fear": "#b44aee",
    "happy": "#f9c74f", "joy": "#f9c74f", "neutral": "#a0a0b8",
    "sadness": "#4a9eff", "shame": "#ff7f3e", "surprise": "#ff9f43",
}


def predict_emotion(text):
    prediction = pipe_lr.predict([text])[0]
    probas     = pipe_lr.predict_proba([text])[0]
    classes    = pipe_lr.classes_
    all_scores = {cls: round(float(p) * 100, 1) for cls, p in zip(classes, probas)}
    confidence = round(float(np.max(probas)) * 100, 1)
    return prediction, confidence, all_scores


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    label, confidence, all_scores = predict_emotion(text)
    return jsonify({
        "label":      label,
        "emoji":      EMOJI_MAP.get(label, "❓"),
        "color":      COLOR_MAP.get(label, "#7c5cfc"),
        "confidence": confidence,
        "all":        all_scores,
    })


if __name__ == "__main__":
    app.run(debug=False, threaded=True, port=5002)

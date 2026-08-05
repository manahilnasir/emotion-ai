# 🧠 Emotion AI Buddy

Emotion AI Buddy is a full-stack Flask web application that detects emotions from both **facial expressions** and **text input** using machine learning. The application provides a modern, interactive web interface where users can analyze emotions in real time directly from their browser.

---

## ✨ Features

- 😊 Real-time Facial Emotion Detection
- 💬 Text Emotion Detection
- 🤖 CNN-based Facial Emotion Recognition
- 📝 Machine Learning-based Text Emotion Classification
- 📷 Browser Camera Integration (No software installation required)
- 📊 Confidence Scores for Predictions
- 📈 Emotion Probability Distribution
- 🗂️ Facial & Text Emotion History
- 💾 SQLite Database Storage
- 🎨 Modern Responsive UI with Animations
- 🌐 Deployable on Render

---

## 🛠️ Tech Stack

### Frontend
- HTML5
- CSS3
- JavaScript

### Backend
- Flask
- Python

### Machine Learning
- TensorFlow
- Keras
- Scikit-learn
- OpenCV
- NumPy
- Joblib

### Database
- SQLite

---

## 📂 Project Structure

```text
Emotion-AI-Buddy/
│
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
│
├── website/
│   ├── index.html
│   ├── facial.html
│   ├── text.html
│   ├── history.html
│   ├── styles.css
│   └── assets/
│
├── facial_emotion/
│   ├── emotion_model.h5
│   ├── emotion_model.json
│   └── modeltraining.ipynb
│
├── text_emotion/
│   ├── text_emotion.pkl
│   ├── Text Emotion Detection.ipynb
│   └── emotion_dataset_raw.csv
│
└── screenshots/
```

---

# 🚀 Getting Started

## 1. Clone the repository

```bash
git clone https://github.com/manahilnasir/emotion-ai.git
```

---

## 2. Navigate into the project

```bash
cd emotion-ai-buddy
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Run the application

```bash
python app.py
```

---

## 5. Open in your browser

```
http://127.0.0.1:5000
```

The browser will ask for camera permission when you open the Facial Emotion Detection page.

---

# 🧠 Machine Learning Models

## Facial Emotion Detection

- Convolutional Neural Network (CNN)
- TensorFlow / Keras
- Detects:
  - Angry
  - Disgust
  - Fear
  - Happy
  - Neutral
  - Sad
  - Surprise

---

## Text Emotion Detection

Scikit-learn based text classification model capable of predicting emotions from user input text.

---

# 📊 Features Included

- Live Facial Emotion Detection
- Text Emotion Prediction
- Confidence Percentage
- Emotion Probability Bars
- Prediction History
- Responsive User Interface
- Browser Camera Support
- SQLite Database

---

# 🌍 Deployment

The project is designed to run both locally and on cloud platforms such as **Render**.

Facial emotion detection uses the user's browser camera, making the application compatible with cloud deployment without requiring server-side webcam access.

---

# 📸 Screenshots

Pictures of the website are available in the screenshots folder
https://github.com/manahilnasir/emotion-ai/tree/main/screenshots

# 🎥 Demo
In demo folder , a video is available
https://github.com/manahilnasir/emotion-ai/blob/main/demo/emotion-ai-demo.mp4

# 🚀 Future Improvements

- User Authentication
- Export Emotion History
- More Emotion Categories
- Improved CNN Accuracy
- Dashboard & Analytics
- REST API Documentation

---

# 👩‍💻 Author

**Manahil Nasir**

Final Year BS Computer Science Student  
University of Central Punjab (UCP)

GitHub:
https://github.com/manahilnasir

LinkedIn:
https://www.linkedin.com/in/manahilnasir/

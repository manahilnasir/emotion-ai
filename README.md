# Emotion AI - Face & Text Emotion Detection

Emotion AI is a Flask-based web application that detects emotions from both **facial expressions** and **text input** using machine learning models. The application provides a simple web interface where users can analyze emotions in real time.

## Features

- 😊 Real-time Facial Emotion Detection
- 💬 Text Emotion Detection
- 🤖 Machine Learning Models
- 📷 Webcam Integration using OpenCV
- 📊 Prediction Confidence Scores
- 🗂️ Emotion History Storage
- 🌐 User-friendly Web Interface

## Tech Stack

- Python
- Flask
- TensorFlow / Keras
- Scikit-learn
- OpenCV
- NumPy
- Joblib
- SQLite
- HTML
- CSS
- JavaScript

## Project Structure

```text
emotion-ai/
├── app.py
├── requirements.txt
├── facial_emotion/
│   ├── emotion_model.h5
│   ├── emotion_model.json
│   └── modeltraining.ipynb
├── text_emotion/
│   ├── text_emotion.pkl
│   ├── emotion_dataset_raw.csv
│   └── Text Emotion Detection.ipynb
├── website/
├── screenshots/
├── demo/
│   └── emotion-ai-demo.mp4
├── .gitignore
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/manahilnasir/emotion-ai.git
```

### 2. Navigate to the project folder

```bash
cd emotion-ai
```

### 3. Install the required dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

### 5. Open your browser

Visit:

```text
http://127.0.0.1:5000
```

> Running `python app.py` starts the complete application, including both the Flask backend and the web interface.


## Demo

The repository includes a project demonstration video inside the **demo** folder.

## Future Improvements

- User authentication
- Cloud deployment
- More emotion classes
- REST API integration
- Improved model accuracy
- Real-time analytics dashboard

## Author

**Manahil Nasir**

BS Computer Science Student  
University of Central Punjab (UCP)

GitHub: https://github.com/manahilnasir
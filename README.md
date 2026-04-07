# 🫀 Heart Disease Prediction AI Web App

A machine learning-powered web application that predicts the likelihood of heart disease using a trained PyTorch neural network. The system provides a simple, professional medical-style interface for entering patient health metrics and receiving an AI-generated risk prediction.

> ⚠️ Disclaimer: This project is for educational purposes only and is not a medical diagnostic tool.

---

# 🚀 Features

- 🧠 PyTorch neural network model for classification
- 🌐 Flask-based web application
- 📊 Interactive medical-style input form
- 🎯 Binary prediction (Heart Disease: Yes / No)
- 🎨 Clean, professional healthcare UI
- 🧾 Structured patient input fields based on clinical indicators

---

# 🧠 Machine Learning Model

The model is trained on structured cardiovascular health data (`heart.csv`) and learns patterns from features such as:

- Age
- Sex
- Blood Pressure
- Cholesterol
- Chest Pain Type
- ECG Results
- Maximum Heart Rate
- Exercise-induced Angina
- ST Depression

---

# 🏗️ Project Structure
heart-ai-webapp/
│
├── app.py # Flask backend (serves model + UI)
├── pytorch_heart.py # Model training script (PyTorch NN)
├── heart_model.pth # Trained model weights
├── heart.csv # Dataset used for training
├── requirements.txt # Python dependencies
│
├── templates/
│ └── index.html # Main web UI (medical form + results)
│
├── static/
│ └── style.css # Professional medical UI styling
│
└── README.md # Project documentation
---

# 🧩 System Architecture Diagram
            ┌────────────────────────────┐
            │        User Input          │
            │  (Web Medical Form UI)     │
            └─────────────┬──────────────┘
                          │
                          ▼
            ┌────────────────────────────┐
            │     Flask Backend (app.py) │
            │  - Receives form data      │
            │  - Preprocesses input      │
            └─────────────┬──────────────┘
                          │
                          ▼
            ┌────────────────────────────┐
            │  PyTorch Neural Network    │
            │  (heart_model.pth)         │
            └─────────────┬──────────────┘
                          │
                          ▼
            ┌────────────────────────────┐
            │   Prediction Output        │
            │  0 = No Disease            │
            │  1 = Disease Detected      │
            └─────────────┬──────────────┘
                          │
                          ▼
            ┌────────────────────────────┐
            │   Web UI Result Panel      │
            │  Risk Display (Green/Red)  │
            └────────────────────────────┘


---

# ⚙️ Installation & Setup

```bash
# Clone repository
git clone https://github.com/walta-negassie/heart-ai-webapp.git

# Move into project
cd heart-ai-webapp

# Install dependencies
pip install -r requirements.txt

# Run Flask app
python app.py
```

# Open in browser:
http://127.0.0.1:5000


# 🧪 Example Input Features


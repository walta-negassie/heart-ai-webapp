from flask import Flask, render_template, request
import torch
import torch.nn as nn
import numpy as np
import os

app = Flask(__name__)

# --- maps ---
sex_map    = {'M': 0, 'F': 1}
cp_map     = {'ATA': 0, 'NAP': 1, 'ASY': 2, 'TA': 3}
ecg_map    = {'Normal': 0, 'ST': 1, 'LVH': 2}
angina_map = {'N': 0, 'Y': 1}
slope_map  = {'Up': 0, 'Flat': 1, 'Down': 2}

# --- model ---
class HeartNet(nn.Module):
    def __init__(self, h1, h2):
        super().__init__()
        self.fc1 = nn.Linear(11, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# --- load model ---
checkpoint = torch.load("heart_model.pth", map_location=torch.device("cpu"), weights_only=False)

model = HeartNet(32, 16)
model.load_state_dict(checkpoint["model_state"])
model.eval()

X_min = np.array(checkpoint["X_min"])
X_max = np.array(checkpoint["X_max"])

def normalize(x):
    return (x - X_min) / (X_max - X_min + 1e-8)

def predict(features):
    x = np.array(features, dtype=np.float32)
    x = normalize(x)

    # 🔴 FIX: add batch dimension (VERY IMPORTANT)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        out = model(x)
        return torch.argmax(out, dim=1).item()

# --- routes ---
@app.route('/')
def home():
    return render_template("index.html", prediction=None)

@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.form

    features = [
        float(data['age']),
        float(sex_map[data['sex']]),
        float(cp_map[data['cp']]),
        float(data['bp']),
        float(data['chol']),
        float(data['fbs']),
        float(ecg_map[data['ecg']]),
        float(data['maxhr']),
        float(angina_map[data['angina']]),
        float(data['oldpeak']),
        float(slope_map[data['slope']]),
    ]

    result = predict(features)

    return render_template("index.html", prediction=result)

# --- run app ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
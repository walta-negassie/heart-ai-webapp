import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# --- Load data ---
sex_map    = {'M': 0, 'F': 1}
cp_map     = {'ATA': 0, 'NAP': 1, 'ASY': 2, 'TA': 3}
ecg_map    = {'Normal': 0, 'ST': 1, 'LVH': 2}
angina_map = {'N': 0, 'Y': 1}
slope_map  = {'Up': 0, 'Flat': 1, 'Down': 2}

rows = []
with open('data/heart.csv', 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if float(row['Cholesterol']) == 0 or float(row['RestingBP']) == 0:
            continue

        features = [
            float(row['Age']),
            float(sex_map[row['Sex']]),
            float(cp_map[row['ChestPainType']]),
            float(row['RestingBP']),
            float(row['Cholesterol']),
            float(row['FastingBS']),
            float(ecg_map[row['RestingECG']]),
            float(row['MaxHR']),
            float(angina_map[row['ExerciseAngina']]),
            float(row['Oldpeak']),
            float(slope_map[row['ST_Slope']]),
        ]
        label = int(row['HeartDisease'])
        rows.append(features + [label])

random.shuffle(rows)

data = np.array(rows)
X_all = data[:, :-1]
y_all = data[:, -1]

# Normalize
X_min = X_all.min(axis=0)
X_max = X_all.max(axis=0)
X_norm = (X_all - X_min) / (X_max - X_min + 1e-8)

# Split
split = int(len(rows) * 0.8)
X_train = torch.tensor(X_norm[:split], dtype=torch.float32)
y_train = torch.tensor(y_all[:split], dtype=torch.long)

# --- Model factory ---
class HeartNet(nn.Module):
    def __init__(self, hidden1, hidden2):
        super().__init__()
        self.fc1 = nn.Linear(11, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# --- Experiment function ---
def run_experiment(lr, epochs, h1, h2):
    model = HeartNet(h1, h2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

    return model

# --- Hyperparameter sets ---
learning_rates = [0.001, 0.01, 0.1]
epochs_list = [200, 500]
hidden_configs = [(16, 8), (32, 16)]

# --- Run experiments ---
models = []

for lr in learning_rates:
    for epochs in epochs_list:
        for (h1, h2) in hidden_configs:
            model = run_experiment(lr, epochs, h1, h2)

            print(f"\nLR={lr}, Epochs={epochs}, Hidden=({h1},{h2})")

            models.append((lr, epochs, h1, h2, model))

# --- NOTE ---
# You now select the best model manually in your driver based on test accuracy
torch.save({
    "model_state": model.state_dict(),
    "X_min": X_min,
    "X_max": X_max
}, "heart_model.pth")

print("Model saved as heart_model.pth")
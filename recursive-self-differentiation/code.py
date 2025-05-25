import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from plot_utils import plot_comparison_panels
import matplotlib.pyplot as plt

# === 1. Data Generation ===
np.random.seed(42)

# First 100 vectors: cyclic pattern with increasing noise
data1 = np.zeros((100, 10), dtype=int)
for i in range(100):
    data1[i, i % 10] = 1
    noise_prob = 0.05 + 0.15 * (i / 100)
    for j in range(10):
        if np.random.rand() < noise_prob:
            data1[i, j] = 1 - data1[i, j]

# Next 100 vectors: mostly ones with high probability
data2 = np.random.rand(100, 10) < 0.9
data2 = data2.astype(int)

# Full dataset: shift vs random transition
data = np.vstack([data1, data2])
X = torch.tensor(data[:-1], dtype=torch.float32)
y = torch.tensor(data[1:], dtype=torch.float32)

# === 2. Alternative Pattern Generator ===
def generate_new_data(pattern_type, epoch):
    new_data = np.zeros((200, 10), dtype=int)
    for i in range(200):
        if pattern_type == 0:  # cyclic shift
            new_data[i, i % 10] = 1
        elif pattern_type == 1:  # random mostly ones
            new_data[i] = (np.random.rand(10) < 0.99).astype(int)
        elif pattern_type == 2:  # inverted cyclic shift
            base = np.zeros(10)
            base[i % 10] = 1
            new_data[i] = 1 - base
        elif pattern_type == 3:  # random mostly zeros
            new_data[i] = (np.random.rand(10) < 0.01).astype(int)
        elif pattern_type == 4:  # cluster pattern
            new_data[i, :5] = (np.random.rand(5) < 0.95).astype(int)
            new_data[i, 5:] = (np.random.rand(5) < 0.05).astype(int)
        else:  # sinusoidal trend
            prob = 0.5 + 0.49 * np.sin(i / 50)
            new_data[i] = (np.random.rand(10) < prob).astype(int)

    noise_prob = 0.3 if pattern_type == 1 else 0.25
    noise = (np.random.rand(*new_data.shape) < noise_prob)
    new_data = np.abs(new_data - noise.astype(int))
    return new_data

# === 3. Simple RNN Model ===
class RNNNet(nn.Module):
    def __init__(self, input_size=10, hidden_size=100, output_size=10):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)
        hidden, _ = self.rnn(x)
        self.hidden = hidden.squeeze(1)
        out = self.fc(self.dropout(self.hidden))
        return self.sigmoid(out)

# === 4. τ Tracker ===
class TauTracker:
    def __init__(self):
        self.prev_hidden = None

    def compute_tau(self, model, inputs):
        with torch.no_grad():
            _ = model(inputs)
            hidden = model.hidden
        if self.prev_hidden is None:
            self.prev_hidden = hidden
            return 0.0
        tau = torch.std(hidden).item() + torch.max(torch.abs(hidden)).item() + torch.norm(hidden).item()
        self.prev_hidden = hidden
        return tau

# === 5. UTD Reflexivity Measure ===
def reflexive_diff(model, inputs):
    with torch.no_grad():
        _ = model(inputs)
        hidden = model.hidden
    deltas = [
        1 if torch.norm(hidden[i] - hidden[j]) < 0.05 else 0
        for i in range(len(hidden)) for j in range(i+1, len(hidden))
    ]
    stability = np.mean(deltas) if deltas else 0.0
    variability = torch.std(hidden).item()
    return stability * variability

# === 6. Training Loop ===
def train_model(model, X, y, use_utd=True):
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    tau_tracker = TauTracker()

    epochs = 200
    tau_values, stability_values, accuracy_values, loss_values = [], [], [], []

    X_input = X.clone()
    y_input = y.clone()

    for epoch in range(epochs):
        if epoch > 0 and epoch % 50 == 0:
            pattern_type = np.random.randint(0, 6)
            new_data = generate_new_data(pattern_type, epoch)
            X_input = torch.tensor(new_data[:-1], dtype=torch.float32)
            y_input = torch.tensor(new_data[1:], dtype=torch.float32)
            print(f"Epoch {epoch}: Data changed → Pattern type {pattern_type}")

        model.train()
        optimizer.zero_grad()
        outputs = model(X_input)

        loss = criterion(outputs, y_input)
        l2_norm = sum(p.pow(2).sum() for p in model.parameters())
        loss += 0.03 * l2_norm

        if epoch > 99:
            base = torch.zeros_like(outputs)
            for i in range(len(base)):
                base[i, i % 10] = 1
            loss += criterion(outputs[:99], base[:99])

        loss.backward()

        if use_utd:
            stability = reflexive_diff(model, X_input)
            for param in model.parameters():
                if param.grad is not None:
                    param.grad *= (1.0 + np.exp(stability) * (1 + 2 * epoch / epochs))

        optimizer.step()

        tau = tau_tracker.compute_tau(model, X_input)
        acc = ((outputs > 0.5).float() == y_input).float().mean().item()
        tau_values.append(tau)
        accuracy_values.append(acc)
        loss_values.append(loss.item())
        stability_values.append(
            reflexive_diff(model, X_input) if use_utd and epoch % 2 == 0 else (stability_values[-1] if stability_values else 0.0)
        )

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss={loss.item():.4f}, Tau={tau:.4f}, Stability={stability_values[-1]:.2f}, Acc={acc:.2f}")

    return tau_values, stability_values, accuracy_values, loss_values

# === 7. Run Training (UTD and No-UTD) ===
model_utd = RNNNet()
print("Training with UTD:")
tau_utd, stability_utd, acc_utd, loss_utd = train_model(model_utd, X, y, use_utd=True)

model_noutd = RNNNet()
print("\nTraining without UTD:")
tau_noutd, stability_noutd, acc_noutd, loss_noutd = train_model(model_noutd, X, y, use_utd=False)

# === 8. Evaluation ===
def final_accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        pred = (model(X) > 0.5).float()
        return (pred == y).float().mean().item()

print(f"\nFinal Accuracy (UTD): {final_accuracy(model_utd, X, y):.2f}")
print(f"Final Accuracy (No UTD): {final_accuracy(model_noutd, X, y):.2f}")
print(f"τ Variance (UTD): {np.var(tau_utd):.4f}")
print(f"τ Variance (No UTD): {np.var(tau_noutd):.4f}")
print(f"Avg. Stability (UTD): {np.mean(stability_utd):.2f}")


plot_comparison_panels(
    acc_utd, acc_noutd,
    tau_utd, tau_noutd,
    stability_utd,
    title_prefix="",
    save_path="results/exp01_rnn_comparison.png"
)
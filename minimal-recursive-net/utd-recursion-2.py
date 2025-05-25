import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def generate_three_class_data(n=100, noise=0.1):
    x0 = np.linspace(-1, 1, n).reshape(-1, 1)
    y0 = x0 + np.random.normal(0, noise, size=(n, 1))
    data0 = np.hstack([x0, y0])

    angles = np.linspace(0, 2 * np.pi, n)
    x1 = np.cos(angles) + np.random.normal(0, noise, size=n)
    y1 = np.sin(angles) + np.random.normal(0, noise, size=n)
    data1 = np.stack([x1, y1], axis=1)

    t = np.linspace(0, 4 * np.pi, n)
    x2 = (0.1 * t) * np.cos(t) + np.random.normal(0, noise, size=n)
    y2 = (0.1 * t) * np.sin(t) + np.random.normal(0, noise, size=n)
    data2 = np.stack([x2, y2], axis=1)

    X = np.vstack([data0, data1, data2])
    y = np.array([0]*n + [1]*n + [2]*n)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

class UTDRecursiveNetOrder2(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, out_dim=3):
        super().__init__()
        self.proj1 = nn.Linear(input_dim * 2, hidden_dim)
        self.gru1 = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.proj2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gru2 = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        N = x.size(0)
        xi = x.unsqueeze(1).repeat(1, N, 1)
        xj = x.unsqueeze(0).repeat(N, 1, 1)
        diffs = torch.cat([xi, xj - xi], dim=-1)
        out1 = F.relu(self.proj1(diffs))
        _, h1 = self.gru1(out1)
        h1 = h1.squeeze(0)

        h1i = h1.unsqueeze(1).repeat(1, N, 1)
        h1j = h1.unsqueeze(0).repeat(N, 1, 1)
        delta_h = torch.cat([h1i, h1j - h1i], dim=-1)
        out2 = F.relu(self.proj2(delta_h))
        _, h2 = self.gru2(out2)
        h2 = h2.squeeze(0)

        return self.classifier(h2)

def train_model(model, x, y, epochs=100, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    logits = model(x)
    preds = torch.argmax(logits, dim=1)
    acc = accuracy_score(y.detach().cpu().numpy(), preds.detach().cpu().numpy())
    return acc, preds

# Generate data and train model
x, y = generate_three_class_data(n=100, noise=0.05)
model = UTDRecursiveNetOrder2(input_dim=2)
acc, preds = train_model(model, x, y)

# Convert to numpy
x_np = x.detach().cpu().numpy()
y_np = y.detach().cpu().numpy()
preds_np = preds.detach().cpu().numpy()
correct_mask = (preds_np == y_np)

# Plot true classes
plt.figure(figsize=(5, 5))
plt.scatter(x_np[:, 0], x_np[:, 1], c=y_np, cmap='tab10', s=10)
plt.title("True classes")
plt.axis('equal')
plt.tight_layout()
plt.show()

# Plot predictions
plt.figure(figsize=(5, 5))
plt.scatter(x_np[:, 0], x_np[:, 1], c=preds_np, cmap='tab10', s=10)
plt.title(f"Predicted (Rec=2), acc={acc:.2f}")
plt.axis('equal')
plt.tight_layout()
plt.show()

# Plot correct vs error
plt.figure(figsize=(5, 5))
plt.scatter(x_np[:, 0], x_np[:, 1], c=~correct_mask, cmap='bwr', s=10)
plt.title("Correct (blue) vs Error (red)")
plt.axis('equal')
plt.tight_layout()
plt.show()

# Confusion matrix
cm = confusion_matrix(y_np, preds_np)
plt.figure(figsize=(4, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1,2], yticklabels=[0,1,2])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

print(f"Test accuracy: {acc:.4f}")
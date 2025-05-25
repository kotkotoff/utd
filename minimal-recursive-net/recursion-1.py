import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def generate_structured_difference_data(n=100, noise=0.1):
    # Class 0: diagonal line
    x0 = np.linspace(-1, 1, n).reshape(-1, 1)
    y0 = x0 + np.random.normal(0, noise, size=(n, 1))
    data0 = np.hstack([x0, y0])

    # Class 1: circle
    angles = np.linspace(0, 2 * np.pi, n)
    x1 = np.cos(angles) + np.random.normal(0, noise, size=n)
    y1 = np.sin(angles) + np.random.normal(0, noise, size=n)
    data1 = np.stack([x1, y1], axis=1)

    X = np.vstack([data0, data1])
    y = np.array([0]*n + [1]*n)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

class UTDNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, use_recursion=False):
        super().__init__()
        self.use_recursion = use_recursion
        self.proj = nn.Linear(input_dim * 2, hidden_dim)
        if use_recursion:
            self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        N = x.size(0)
        xi = x.unsqueeze(1).repeat(1, N, 1)      # [N, N, dim]
        xj = x.unsqueeze(0).repeat(N, 1, 1)      # [N, N, dim]
        diffs = torch.cat([xi, xj - xi], dim=-1) # Pairwise differences
        out = self.proj(diffs)                   # [N, N, hidden]
        if self.use_recursion:
            _, h = self.gru(out)                 # [1, N, hidden]
            h = h.squeeze(0)
        else:
            h = out.mean(dim=1)                  # Simple mean over differences
        return self.classifier(h)

def train_utd(model, x, y, epochs=100, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    preds = model(x).argmax(dim=1)
    acc = accuracy_score(y.tolist(), preds.tolist())
    return acc, preds

if __name__ == "__main__":
    x, y = generate_structured_difference_data(n=100, noise=0.1)

    model_plain = UTDNet(input_dim=2, use_recursion=False)
    acc_plain, preds_plain = train_utd(model_plain, x, y)

    model_rec = UTDNet(input_dim=2, use_recursion=True)
    acc_rec, preds_rec = train_utd(model_rec, x, y)

# Boolean mask for correctness
correct_plain = preds_plain.detach().cpu().numpy() == y.detach().cpu().numpy()
correct_rec = preds_rec.detach().cpu().numpy() == y.detach().cpu().numpy()

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].scatter(x[:, 0], x[:, 1], c=~correct_plain, cmap='bwr', s=10)
axs[0].set_title(f"Without recursion (acc={acc_plain:.2f})")
axs[1].scatter(x[:, 0], x[:, 1], c=~correct_rec, cmap='bwr', s=10)
axs[1].set_title(f"With recursion (acc={acc_rec:.2f})")
plt.tight_layout()
plt.show()

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# === Load and prepare Iris dataset ===
data = load_iris()
X = StandardScaler().fit_transform(data.data).astype(np.float32)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train = torch.tensor(X_train)
X_test = torch.tensor(X_test)
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

# === UTDResNetAdaptiveK ===
class ResD1LayerAdaptiveK(nn.Module):
    def __init__(self, input_dim, hidden_dim, k_max=16, alpha=0.5):
        super().__init__()
        self.diff_proj = nn.Linear(input_dim, hidden_dim)
        self.tau_proj = nn.Sequential(nn.Linear(input_dim, 1), nn.Softplus())
        self.agg = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.res_proj = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.k_max = k_max
        self.alpha = alpha

    def forward(self, x):
        B, D = x.shape
        x_i = x.unsqueeze(1).repeat(1, B, 1)
        x_j = x.unsqueeze(0).repeat(B, 1, 1)
        diffs = torch.abs(x_i - x_j)
        d_proj = self.diff_proj(diffs)

        tau = self.tau_proj(x).clamp(min=1e-2) + 1.0
        k_vals = (self.k_max * torch.exp(-self.alpha * tau)).long().clamp(min=1, max=B)

        with torch.no_grad():
            dists = torch.norm(x_i - x_j, dim=-1)
            topk_indices = []
            for i in range(B):
                k_i = k_vals[i].item()
                idx = dists[i].topk(k_i, largest=False).indices
                topk_indices.append(idx)
            max_k = max(len(idx) for idx in topk_indices)
            padded = torch.full((B, max_k), fill_value=0, dtype=torch.long, device=x.device)
            for i, idx in enumerate(topk_indices):
                padded[i, :len(idx)] = idx
            topk_indices = padded

        gathered = torch.gather(d_proj, dim=1, index=topk_indices.unsqueeze(-1).expand(-1, -1, d_proj.size(-1)))
        h_k = gathered.mean(dim=1) / tau
        h = self.agg(h_k)
        x_proj = self.res_proj(x)
        return self.norm(h + x_proj)
class UTDResNetAdaptiveKDepth2(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32, num_classes=3):
        super().__init__()
        self.layer1 = ResD1LayerAdaptiveK(input_dim, hidden_dim)
        self.layer2 = ResD1LayerAdaptiveK(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h1 = self.layer1(x)
        h2 = self.layer2(h1)
        return self.classifier(h2), None
class UTDResNetAdaptiveK(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32, num_classes=3):
        super().__init__()
        self.layer = ResD1LayerAdaptiveK(input_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h = self.layer(x)
        return self.classifier(h), None

# === MLP ===
class MLPNet(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x):
        return self.net(x), None

# === CNN ===
class CNNNet(nn.Module):
    def __init__(self, input_dim=4, num_classes=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * input_dim, num_classes)
        )
    def forward(self, x):
        return self.conv(x.unsqueeze(1)), None

# === GRU ===
class GRUNet(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32, num_classes=3):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        x = x.unsqueeze(1)
        _, h = self.rnn(x)
        return self.fc(h.squeeze(0)), None

# === Train & Eval ===
def train(model, loader, opt, loss_fn):
    model.train()
    for xb, yb in loader:
        opt.zero_grad()
        out, _ = model(xb)
        loss = loss_fn(out, yb)
        loss.backward()
        opt.step()

def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in loader:
            out, _ = model(xb)
            correct += (out.argmax(1) == yb).sum().item()
    return correct / len(loader.dataset)

# === Run models ===
models = {
    "UTDResNetAdaptiveK": UTDResNetAdaptiveK(),
    "UTDResNetAdaptiveK-d2": UTDResNetAdaptiveKDepth2(),
    "MLP": MLPNet(),
    "CNN": CNNNet(),
    "GRU": GRUNet()
}

results = {}
final_metrics = []

for name, model in models.items():
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    accs = []
    for epoch in range(50):
        train(model, train_loader, opt, loss_fn)
        acc = evaluate(model, test_loader)
        accs.append(acc)
    results[name] = accs
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    final_metrics.append((name, round(accs[-1], 3), total_params))

# === Plot accuracy ===
plt.figure(figsize=(8, 5))
for name, accs in results.items():
    plt.plot(range(1, 51), accs, label=name, marker="o", markersize=3)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Model Comparison on Iris Dataset")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Print table ===
from tabulate import tabulate
print(tabulate(final_metrics, headers=["Model", "Final Accuracy", "Parameters"]))

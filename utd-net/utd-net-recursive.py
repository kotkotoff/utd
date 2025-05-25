import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

# === Synthetic data generation ===
def generate_synthetic_data(n_samples=300, input_dim=10, n_classes=4):
    X = np.random.randn(n_samples, input_dim).astype(np.float32)
    y = np.random.randint(0, n_classes, size=n_samples)
    return torch.tensor(X), torch.tensor(y)

X_train, y_train = generate_synthetic_data(300)
X_test, y_test = generate_synthetic_data(100)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)
class ResD1LayerAdaptiveK(nn.Module):
    def __init__(self, input_dim, hidden_dim, k_max=16, alpha=0.5):
        super().__init__()
        self.diff_proj = nn.Linear(input_dim, hidden_dim)
        self.tau_proj = nn.Sequential(nn.Linear(input_dim, 1), nn.Softplus())
        self.agg = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.res_proj = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.k_max = k_max
        self.alpha = alpha

    def forward(self, x):
        B, D = x.shape
        x_i = x.unsqueeze(1).repeat(1, B, 1)  # [B, B, D]
        x_j = x.unsqueeze(0).repeat(B, 1, 1)  # [B, B, D]

        diffs = torch.abs(x_i - x_j)
        d_proj = self.diff_proj(diffs)

        # τ(x) ≥ 1.0
        tau = self.tau_proj(x).clamp(min=1e-2) + 1.0  # [B, 1]
        k_vals = (self.k_max * torch.exp(-self.alpha * tau)).long().clamp(min=1, max=B)  # [B, 1]

        # pairwise L2 distances
        with torch.no_grad():
            dists = torch.norm(x_i - x_j, dim=-1)  # [B, B]
            topk_indices = []
            for i in range(B):
                k_i = k_vals[i].item()
                idx = dists[i].topk(k_i, largest=False).indices
                topk_indices.append(idx)
            # [B, max_k] padded
            max_k = max(len(idx) for idx in topk_indices)
            padded = torch.full((B, max_k), fill_value=0, dtype=torch.long, device=x.device)
            for i, idx in enumerate(topk_indices):
                padded[i, :len(idx)] = idx
            topk_indices = padded

        gathered = torch.gather(
            d_proj, dim=1,
            index=topk_indices.unsqueeze(-1).expand(-1, -1, d_proj.size(-1))
        )  # [B, k_i, H] (with padding)

        h_k = gathered.mean(dim=1) / tau
        h = self.agg(h_k)

        x_proj = self.res_proj(x)
        return self.norm(h + x_proj)
class UTDResNetAdaptiveK(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=32, num_classes=4):
        super().__init__()
        self.layer = ResD1LayerAdaptiveK(input_dim, hidden_dim, k_max=16, alpha=0.3)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h = self.layer(x)
        return self.classifier(h), None
class ResD1EdgeMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # входное различие: [x_i, x_j, x_i - x_j, |x_i - x_j|] → 4D
        self.edge_fn = nn.Sequential(
            nn.Linear(4 * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.tau_proj = nn.Sequential(nn.Linear(input_dim, 1), nn.Softplus())
        self.agg = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.res_proj = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        B, D = x.shape
        x_i = x.unsqueeze(1).repeat(1, B, 1)  # [B, B, D]
        x_j = x.unsqueeze(0).repeat(B, 1, 1)  # [B, B, D]
        delta = x_i - x_j
        abs_delta = torch.abs(delta)
        diff_input = torch.cat([x_i, x_j, delta, abs_delta], dim=-1)  # [B, B, 4D]

        d_proj = self.edge_fn(diff_input)  # [B, B, H]

        tau = self.tau_proj(x).clamp(min=1e-2) + 1.0  # [B, 1]
        h_mean = d_proj.mean(dim=1) / tau
        h = self.agg(h_mean)

        x_proj = self.res_proj(x)
        return self.norm(h + x_proj)
class UTDResNetEdgeMLP(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=32, num_classes=4):
        super().__init__()
        self.layer = ResD1EdgeMLP(input_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h = self.layer(x)
        return self.classifier(h), None

class ResD1LayerV2(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.diff_proj = nn.Linear(input_dim, hidden_dim)
        self.tau_proj = nn.Sequential(nn.Linear(input_dim, 1), nn.Softplus())
        self.agg = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.res_proj = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        B, D = x.shape
        # Дифференциации
        x_i = x.unsqueeze(1).repeat(1, B, 1)
        x_j = x.unsqueeze(0).repeat(B, 1, 1)
        diffs = torch.abs(x_i - x_j)
        d_proj = self.diff_proj(diffs)  # [B, B, H]

        # τ(x) >= 1.0
        tau = self.tau_proj(x).clamp(min=1e-2) + 1.0  # [B, 1]

        # Mean-агрегация и масштабирование
        h_mean = d_proj.mean(dim=1) / tau
        h = self.agg(h_mean)

        x_proj = self.res_proj(x)  # на случай, если размерность отличается
        return self.norm(h + x_proj)
class ResD1LayerTopK(nn.Module):
    def __init__(self, input_dim, hidden_dim, k=8):
        super().__init__()
        self.diff_proj = nn.Linear(input_dim, hidden_dim)
        self.tau_proj = nn.Sequential(nn.Linear(input_dim, 1), nn.Softplus())
        self.agg = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.res_proj = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.k = k

    def forward(self, x):
        B, D = x.shape
        x_i = x.unsqueeze(1).repeat(1, B, 1)  # [B, B, D]
        x_j = x.unsqueeze(0).repeat(B, 1, 1)  # [B, B, D]

        diffs = torch.abs(x_i - x_j)  # [B, B, D]
        d_proj = self.diff_proj(diffs)  # [B, B, H]

        # --- Top-k на основе L2-расстояний
        with torch.no_grad():
            k = min(self.k, B)
            dists = torch.norm(x_i - x_j, dim=-1)  # [B, B]
            topk_indices = dists.topk(k, dim=1, largest=False).indices  # [B, k]

        # Собираем только top-k различия
        gathered = torch.gather(
            d_proj, dim=1,
            index=topk_indices.unsqueeze(-1).expand(-1, -1, d_proj.size(-1))
        )  # [B, k, H]

        tau = self.tau_proj(x).clamp(min=1e-2) + 1.0  # [B, 1]
        h_k = gathered.mean(dim=1) / tau  # [B, H]
        h = self.agg(h_k)

        x_proj = self.res_proj(x)
        return self.norm(h + x_proj)
class UTDResNetTopK(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=32, num_classes=4, k=8):
        super().__init__()
        self.layer = ResD1LayerTopK(input_dim, hidden_dim, k=k)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h = self.layer(x)
        return self.classifier(h), None
class UTDResNetV2Depth2(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=32, num_classes=4):
        super().__init__()
        self.layer1 = ResD1LayerV2(input_dim, hidden_dim)
        self.layer2 = ResD1LayerV2(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h1 = self.layer1(x)
        h2 = self.layer2(h1)
        return self.classifier(h2), None

# === ResD₁ Layer with τ(x) ===
class ResD1Layer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.diff_proj = nn.Linear(input_dim, hidden_dim)
        self.tau_proj = nn.Sequential(nn.Linear(input_dim, 1), nn.Softplus())
        self.agg_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        B, D = x.shape
        x_i = x.unsqueeze(1).repeat(1, B, 1)
        x_j = x.unsqueeze(0).repeat(B, 1, 1)
        diffs = torch.abs(x_i - x_j)
        d_proj = self.diff_proj(diffs)
        tau = self.tau_proj(x).clamp(min=1e-2)
        h_sum = d_proj.sum(dim=1) / tau
        h = self.agg_proj(h_sum)
        return self.norm(h + x)

class UTDResNetV2(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=32, num_classes=4):
        super().__init__()
        self.layer = ResD1LayerV2(input_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h = self.layer(x)
        return self.classifier(h), None

# === UTDResNet ===
class UTDResNet(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=32, num_classes=4):
        super().__init__()
        self.proj_in = nn.Linear(input_dim, hidden_dim)
        self.resd1 = ResD1Layer(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.proj_in(x)
        h = self.resd1(x)
        return self.classifier(h), None

# === MLP ===
class MLPNet(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=32, num_classes=4):
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
    def __init__(self, input_dim=10, num_classes=4):
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
    def __init__(self, input_dim=10, hidden_dim=32, num_classes=4):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        x = x.unsqueeze(1)
        _, h = self.rnn(x)
        return self.fc(h.squeeze(0)), None

# === Transformer ===
class TransformerNet(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=32, num_classes=4):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4),
            num_layers=1
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        x = self.proj(x).unsqueeze(1)
        x = self.encoder(x).squeeze(1)
        return self.classifier(x), None

# === Train and eval ===
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

# === Run all models for 100 epochs ===
models = {
    "UTDResNetEdgeMLP": UTDResNetEdgeMLP(),
    "UTDResNetAdaptiveK": UTDResNetAdaptiveK(),
    "UTDResNetTopK": UTDResNetTopK(),
    "UTDResNetV2-d1": UTDResNetV2(),
    "MLP": MLPNet(),
    "CNN": CNNNet(),
    "GRU": GRUNet(),
    "Transformer": TransformerNet()
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

# === Plot ===
plt.figure(figsize=(10, 5))
for name, accs in results.items():
    plt.plot(range(1, 51), accs, label=name, marker="o", markersize=3)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("UTDResNet vs Classic Models on Synthetic Data (100 Epochs)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Table ===
from tabulate import tabulate
print(tabulate(final_metrics, headers=["Model", "Final Accuracy", "Parameters"]))

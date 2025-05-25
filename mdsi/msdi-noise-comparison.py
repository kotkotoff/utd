class ModularSDIModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        proj_dim: int = 16,
        num_classes: int = 2,
        max_pairs: int = 128,
        memory_momentum: float = 0.95,
        use_sums: bool = True,
        use_mlp: bool = True,
        sdi_only_logits: bool = False,
        normalize_memory: bool = True,
        normalize_diffs: bool = True,
        agg_method: Literal["mean", "sum", "max"] = "mean",
        deterministic_pairs: bool = False,
        diff_dropout_rate: float = 0.0,
        memory_freeze_epoch: Optional[int] = None,
        memory_update_strategy: Literal["moving_average", "batchwise_reset"] = "moving_average",
        memory_attention: bool = False
    ):
        super().__init__()
        assert agg_method in {"mean", "sum", "max"}
        assert memory_update_strategy in {"moving_average", "batchwise_reset"}

        self.num_classes = num_classes
        self.proj_dim = proj_dim
        self.max_pairs = max_pairs
        self.memory_momentum = memory_momentum
        self.use_sums = use_sums
        self.use_mlp = use_mlp
        self.sdi_only_logits = sdi_only_logits
        self.normalize_memory = normalize_memory
        self.normalize_diffs = normalize_diffs
        self.agg_method = agg_method
        self.deterministic_pairs = deterministic_pairs
        self.diff_dropout_rate = diff_dropout_rate
        self.memory_freeze_epoch = memory_freeze_epoch
        self.memory_update_strategy = memory_update_strategy
        self.memory_attention = memory_attention
        self.epoch = 0

        self.proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
        )

        self.dropout = nn.Dropout(diff_dropout_rate) if diff_dropout_rate > 0 else nn.Identity()
        self.register_buffer("sdi_memory", F.normalize(torch.randn(num_classes, proj_dim), dim=1))

        if self.use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(proj_dim, proj_dim),
                nn.ReLU(),
                nn.Linear(proj_dim, num_classes)
            )

        if self.memory_attention:
            self.memory_attn = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=1, batch_first=True)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def compute_sdi_vector(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)

        if z.shape[0] < 2:
            return torch.zeros(1, self.proj_dim, device=z.device)

        if self.deterministic_pairs:
            idx = torch.arange(len(z) - 1)
            pairs = torch.stack([idx, idx + 1], dim=1)
        else:
            pairs = torch.combinations(torch.arange(z.shape[0], device=z.device), r=2)
            if len(pairs) > self.max_pairs:
                idx = torch.randperm(len(pairs))[:self.max_pairs]
                pairs = pairs[idx]

        diffs = z[pairs[:, 0]] - z[pairs[:, 1]]
        parts = [diffs]
        if self.use_sums:
            parts.append(z[pairs[:, 0]] + z[pairs[:, 1]])

        z = torch.cat(parts, dim=0)

        if self.normalize_diffs:
            z = F.normalize(z, dim=1)

        z = self.dropout(z)

        if self.agg_method == "mean":
            return z.mean(dim=0, keepdim=True)
        elif self.agg_method == "sum":
            return z.sum(dim=0, keepdim=True)
        elif self.agg_method == "max":
            z, _ = z.max(dim=0, keepdim=True)
            return z

    def update_memory(self, sdi_vecs: torch.Tensor, y: torch.Tensor):
        if self.memory_freeze_epoch is not None and self.epoch >= self.memory_freeze_epoch:
            return
        with torch.no_grad():
            for c in range(self.num_classes):
                mask = (y == c)
                if mask.any():
                    new_mean = sdi_vecs[mask].mean(dim=0)
                    updated = (
                        self.memory_momentum * self.sdi_memory[c] +
                        (1 - self.memory_momentum) * new_mean
                        if self.memory_update_strategy == "moving_average"
                        else new_mean
                    )
                    self.sdi_memory[c] = F.normalize(updated, dim=0) if self.normalize_memory else updated

    def compute_logits(self, sdi_vecs: torch.Tensor, B: int) -> torch.Tensor:
        mem = F.normalize(self.sdi_memory, dim=1) if self.normalize_memory else self.sdi_memory
        logits = torch.matmul(sdi_vecs, mem.T)

        if self.memory_attention:
            mem_exp = mem.unsqueeze(0).expand(B, -1, -1)
            query = sdi_vecs.unsqueeze(1)
            attended, _ = self.memory_attn(query=query, key=mem_exp, value=mem_exp)
            logits = logits + attended.squeeze(1) @ mem.T

        if self.use_mlp and not self.sdi_only_logits:
            mlp_logits = self.mlp(sdi_vecs)
            logits = logits + mlp_logits

        return logits

    def forward(self, X: torch.Tensor, y: Optional[torch.Tensor] = None, update_memory: bool = False):
        if X.dim() == 2:
            X = X.unsqueeze(0)
        B = X.shape[0]
        sdi_vecs = torch.cat([self.compute_sdi_vector(X[b]) for b in range(B)], dim=0)

        if update_memory and y is not None:
            self.update_memory(sdi_vecs, y)

        logits = self.compute_logits(sdi_vecs, B)
        return logits, sdi_vecs
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


# === Dataset generation ===
def generate_structured_dataset(n_scenes=100, noise=0.0):
    X, y = [], []
    square = np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]])
    for _ in range(n_scenes // 2):
        pts = square + noise * np.random.randn(4, 2)
        pts -= pts.mean(axis=0)
        X.append(torch.tensor(pts, dtype=torch.float32))
        y.append(0)

        rand = np.random.uniform(-1, 1, (4, 2)) + noise * np.random.randn(4, 2)
        rand -= rand.mean(axis=0)
        X.append(torch.tensor(rand, dtype=torch.float32))
        y.append(1)

    return torch.stack(X), torch.tensor(y)


# === Simplified SDI Model (no recursion) ===
class ModularSDIModel(nn.Module):
    def __init__(self, proj_dim=16, num_classes=2, max_pairs=64, use_mlp=True, use_sums=True):
        super().__init__()
        self.proj_dim = proj_dim
        self.max_pairs = max_pairs
        self.use_mlp = use_mlp
        self.use_sums = use_sums

        self.proj = nn.Sequential(
            nn.Linear(2, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU()
        )

        self.sdi_memory = nn.Parameter(F.normalize(torch.randn(num_classes, proj_dim), dim=1), requires_grad=False)

        if use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(proj_dim, proj_dim),
                nn.ReLU(),
                nn.Linear(proj_dim, num_classes)
            )

    def compute_sdi_vector(self, x):
        z = self.proj(x)
        if z.shape[0] < 2:
            return torch.zeros(1, self.proj_dim, device=z.device)

        pairs = torch.combinations(torch.arange(z.shape[0], device=z.device), r=2)
        if len(pairs) > self.max_pairs:
            pairs = pairs[torch.randperm(len(pairs))[:self.max_pairs]]

        diffs = z[pairs[:, 0]] - z[pairs[:, 1]]
        parts = [diffs]
        if self.use_sums:
            parts.append(z[pairs[:, 0]] + z[pairs[:, 1]])
        z = torch.cat(parts, dim=0)
        z = F.normalize(z, dim=1)
        return z.mean(dim=0, keepdim=True)

    def update_memory(self, sdi_vecs, y):
        with torch.no_grad():
            for c in range(self.sdi_memory.size(0)):
                mask = (y == c)
                if mask.any():
                    mean_vec = sdi_vecs[mask].mean(dim=0)
                    self.sdi_memory[c] = F.normalize(mean_vec, dim=0)

    def forward(self, X, y=None, update_memory=False):
        if X.dim() == 2:
            X = X.unsqueeze(0)
        B = X.shape[0]
        sdi_vecs = torch.cat([self.compute_sdi_vector(X[b]) for b in range(B)], dim=0)

        if update_memory and y is not None:
            self.update_memory(sdi_vecs, y)

        logits = sdi_vecs @ self.sdi_memory.T
        if self.use_mlp:
            logits = logits + self.mlp(sdi_vecs)
        return logits, sdi_vecs


# === Training and evaluation ===
def train_on_noise(noise, epochs=20):
    X, y = generate_structured_dataset(n_scenes=100, noise=noise)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=4, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=4)

    model = ModularSDIModel()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    accs = []

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            logits, _ = model(xb, yb, update_memory=True)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                logits, _ = model(xb)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        accs.append(correct / total)

    return accs


# === Plotting accuracy vs epochs ===
def plot_accuracy_curves():
    noise_levels = [0.01, 0.5, 1.0]
    plt.figure(figsize=(8, 5))
    for noise in noise_levels:
        accs = train_on_noise(noise, epochs=20)
        plt.plot(range(1, 21), accs, label=f"Noise {noise}", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("MSDI Accuracy vs Epochs at Different Noise Levels")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# === Run the experiment ===
plot_accuracy_curves()

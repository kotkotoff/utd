import GEOparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from tqdm import tqdm

# === 1. Load and normalize GEO data ===
gse = GEOparse.get_GEO("GSE2034", destdir="./")
data = gse.pivot_samples("VALUE")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data.values)
sample_ids = list(gse.gsms.keys())[:100]
X_scaled = X_scaled[:100]

# === 2. Extract relapse labels ===
relapse_flags = []
for sid in sample_ids:
    meta = gse.gsms[sid].metadata.get("characteristics_ch1", [])
    match = [x for x in meta if "bone relapses" in x.lower()]
    relapse_flags.append(int(match[0].split(":")[-1].strip()) if match else 0)

# === 3. Select top-k variable genes and convert to MSDI scenes ===
top_k = 256
gene_var = np.var(X_scaled, axis=0)
top_genes = np.argsort(gene_var)[-top_k:]
X_topk = X_scaled[:, top_genes]

scenes = []
labels = []

for i, row in enumerate(X_topk):
    if np.isnan(row).any():
        continue
    scene = np.stack([np.arange(top_k), row], axis=1)
    scenes.append(scene)
    labels.append(relapse_flags[i])

X_tensor = torch.tensor(scenes, dtype=torch.float32)
y_tensor = torch.tensor(labels, dtype=torch.long)
print(f"Loaded {len(scenes)} valid samples.")

# === 4. Define MSDI model ===
class MSDIModel(nn.Module):
    def __init__(self, input_dim=2, proj_dim=32):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):  # x: (B, N, 2)
        z = self.proj(x)  # (B, N, D)
        d1 = z[:, None, :, :] - z[:, :, None, :]  # (B, N, N, D)
        triu_idx = torch.triu_indices(z.shape[1], z.shape[1], offset=1)
        d1 = d1[:, triu_idx[0], triu_idx[1], :]  # (B, M, D)
        agg = d1.mean(dim=1)  # (B, D)
        return self.classifier(agg)

# === 5. Prepare DataLoaders ===
dataset = TensorDataset(X_tensor, y_tensor)
train_set, test_set = random_split(dataset, [80, len(dataset) - 80], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
test_loader = DataLoader(test_set, batch_size=8)

# === 6. Train the model ===
model = MSDIModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(15):
    model.train()
    for xb, yb in train_loader:
        loss = loss_fn(model(xb), yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# === 7. Evaluate ===
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        pred = torch.argmax(model(xb), dim=1)
        y_true.extend(yb.tolist())
        y_pred.extend(pred.tolist())

print(classification_report(y_true, y_pred))

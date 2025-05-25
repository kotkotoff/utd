import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

# === LOAD DATA: EMOTION ===
dataset = load_dataset("emotion")
texts = dataset["train"]["text"] + dataset["test"]["text"]
labels = dataset["train"]["label"] + dataset["test"]["label"]

train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, stratify=labels, random_state=42
)

model = SentenceTransformer("all-MiniLM-L6-v2")
X_train = model.encode(train_texts, convert_to_tensor=True)
X_test = model.encode(test_texts, convert_to_tensor=True)
y_train = torch.tensor(train_labels)
y_test = torch.tensor(test_labels)
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)
input_dim = X_train.shape[1]

# === MODELS ===
class UTDResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=6):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.res = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        h = F.relu(self.proj(x))
        r = F.relu(self.res(h))
        return self.fc(self.norm(h + r)), None

class UTDResNetAdaptiveK(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=6, k_min=4, k_max=16):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.tau = nn.Sequential(nn.Linear(input_dim, 1), nn.Sigmoid())
        self.k_min = k_min
        self.k_max = k_max
        self.res = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        tau = self.tau(x).squeeze(-1)
        k_vals = (self.k_max - (self.k_max - self.k_min) * tau).round().clamp(min=1).int()
        with torch.no_grad():
            B = x.shape[0]
            dists = torch.cdist(x, x)
            topk_indices = [dists[i].topk(k=k_vals[i].item(), largest=False).indices for i in range(B)]
        h = F.relu(self.proj(x))
        agg = torch.stack([h[idx].mean(dim=0) for idx in topk_indices])
        r = F.relu(self.res(agg))
        return self.fc(self.norm(h + r)), None

class UTDAttnHeadNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_heads=4, num_classes=6):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_heads)])
        self.attn_weights = nn.Parameter(torch.ones(num_heads))
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        h = F.relu(self.proj(x))
        head_outputs = []
        for head in self.heads:
            h_proj = head(h)
            scores = torch.cdist(h_proj, h_proj)
            scores = torch.softmax(-scores, dim=-1)
            h_head = torch.matmul(scores, h_proj)
            head_outputs.append(h_head)
        stacked = torch.stack(head_outputs)
        attn = torch.softmax(self.attn_weights, dim=0).view(-1, 1, 1)
        combined = (stacked * attn).sum(dim=0)
        return self.fc(self.norm(combined)), None

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=6):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_classes))
    def forward(self, x): return self.net(x), None

class CNN(nn.Module):
    def __init__(self, input_dim, num_classes=6):
        super().__init__()
        self.conv = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.fc = nn.Linear(16 * input_dim, num_classes)
    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv(x))
        return self.fc(x.view(x.size(0), -1)), None

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=6):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        x = x.unsqueeze(1)
        _, h = self.rnn(x)
        return self.fc(h.squeeze(0)), None

class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes=6):
        super().__init__()
        self.qkv = nn.Linear(input_dim, input_dim * 3)
        self.fc = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [z.unsqueeze(1) for z in qkv]
        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / x.size(1)**0.5, dim=-1)
        return self.fc(torch.bmm(attn, v).squeeze(1)), None

# === TRAINING FUNCTIONS ===
def train_epoch(model, loader, optimizer, loss_fn):
    model.train()
    for xb, yb in loader:
        optimizer.zero_grad()
        out, _ = model(xb)
        loss = loss_fn(out, yb)
        loss.backward()
        optimizer.step()

def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in loader:
            out, _ = model(xb)
            correct += (out.argmax(1) == yb).sum().item()
    return correct / len(loader.dataset)

# === MAIN LOOP ===
models = {
    "UTDResNet": UTDResNet(input_dim),
    "UTDResNetAdaptiveK": UTDResNetAdaptiveK(input_dim),
    "UTDAttnHeadNet": UTDAttnHeadNet(input_dim),
    "MLP": MLP(input_dim),
    "CNN": CNN(input_dim),
    "GRU": GRUModel(input_dim),
    "Transformer": TransformerModel(input_dim)
}

results = {}
for name, model in models.items():
    opt = Adam(model.parameters(), lr=1e-3)
    accs = []
    for epoch in range(10):
        train_epoch(model, train_loader, opt, nn.CrossEntropyLoss())
        acc = evaluate(model, test_loader)
        accs.append(acc)
    results[name] = accs

# === PLOTTING RESULTS ===
for name, accs in results.items():
    plt.plot(range(1, 11), accs, label=name)
plt.title("Emotion Dataset — Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

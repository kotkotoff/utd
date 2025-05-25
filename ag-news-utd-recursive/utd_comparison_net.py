import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 1. Load and tokenize AG News ===
train_df = pd.read_csv("https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv", header=None)[:3000]
test_df = pd.read_csv("https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv", header=None)[:1000]
train_df.columns = ["label", "title", "description"]
test_df.columns = ["label", "title", "description"]
train_df["label"] -= 1
test_df["label"] -= 1

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
def tokenize(texts):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=20, return_tensors="pt")

X_train = tokenize(train_df["title"].tolist())
X_test = tokenize(test_df["title"].tolist())
y_train = torch.tensor(train_df["label"].tolist())
y_test = torch.tensor(test_df["label"].tolist())

# === 2. Cache frozen BERT embeddings ===
bert = AutoModel.from_pretrained("distilbert-base-uncased").to(device).eval()

@torch.no_grad()
def extract_embeddings(input_ids, attention_mask):
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    out = bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    return out[:, 1:-1].cpu()  # remove [CLS] and [SEP]

train_embeddings = torch.cat([
    extract_embeddings(X_train["input_ids"][i:i+32], X_train["attention_mask"][i:i+32])
    for i in range(0, len(X_train["input_ids"]), 32)
])
test_embeddings = torch.cat([
    extract_embeddings(X_test["input_ids"][i:i+32], X_test["attention_mask"][i:i+32])
    for i in range(0, len(X_test["input_ids"]), 32)
])

train_loader = DataLoader(TensorDataset(train_embeddings, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(test_embeddings, y_test), batch_size=32)

# === 3. UTDRecursiveNet model ===
class UTDRecursiveNet(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=64, num_classes=4):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.tau_proj = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h = F.relu(self.proj(x))  # [B, L, H]
        tau = self.tau_proj(h) + 1e-5
        steps = (1.0 / tau).round().long().clamp(min=1, max=2)
        for i in range(2):
            mask = (steps > i).float()
            h_new, _ = self.gru(h)
            h = mask * h_new + (1 - mask) * h
        pooled = h.mean(dim=1)
        return self.classifier(pooled), tau.squeeze(-1)

# === 4. Baseline GRUNet model ===
class GRUNet(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=64, num_classes=4):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, h = self.rnn(x)
        return self.classifier(h.squeeze(0)), None

# === 5. Training and evaluation loops ===
def train(model, loader, optimizer, loss_fn):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out, _ = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out, _ = model(x)
            correct += (out.argmax(1) == y).sum().item()
    return correct / len(loader.dataset)

# === 6. Training runs ===
def run_utd():
    model = UTDRecursiveNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-5)
    accs, tau_vars = [], []
    for _ in range(10):
        train(model, train_loader, opt, nn.CrossEntropyLoss())
        accs.append(evaluate(model, test_loader))
        x_sample, _ = next(iter(test_loader))
        with torch.no_grad():
            _, taus = model(x_sample.to(device))
            tau_vars.append(torch.var(taus).item())
    return accs, tau_vars

def run_gru():
    model = GRUNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-5)
    accs = []
    for _ in range(10):
        train(model, train_loader, opt, nn.CrossEntropyLoss())
        accs.append(evaluate(model, test_loader))
    return accs

# === 7. Run and plot results ===
acc_utd, tau_var = run_utd()
acc_gru = run_gru()

plt.plot(range(1, 11), acc_utd, label="UTDRecursiveNet", marker="o")
plt.plot(range(1, 11), acc_gru, label="GRUNet", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy on AG News")
plt.legend()
plt.grid(True)
plt.show()

plt.plot(range(1, 11), tau_var, color="orange", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Var[τ(x)]")
plt.title("Variance of τ(x) across Epochs (UTD)")
plt.grid(True)
plt.show()

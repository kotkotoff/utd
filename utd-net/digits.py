import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# ====== Models ======

class UTDResNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class UTDResNetAdaptiveK(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.tau_layer = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Softplus()
        )
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        tau = self.tau_layer(x)  # tau is computed but not used here (simulated behavior)
        return self.classifier(x)

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class CNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Unflatten(1, (1, 8, 8)),
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 8 * 8, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class GRUModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.gru = nn.GRU(input_dim, 64, batch_first=True)
        self.fc = nn.Linear(64, num_classes)
    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, sequence_len=1, features)
        out, _ = self.gru(x)
        return self.fc(out[:, -1])

class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.embedding = nn.Linear(input_dim, 64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=2, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Linear(64, num_classes)
    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # (batch, seq_len=1, dim)
        x = self.transformer(x)
        return self.fc(x[:, 0])

# ====== Data ======

digits = load_digits()
X = digits.data / 16.0  # normalize pixel values
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

models = {
    "UTDNet (depth 1)": UTDResNet(64, 10),
    "UTDNet AdaptiveK": UTDResNetAdaptiveK(64, 10),
    "MLP": MLP(64, 10),
    "CNN": CNN(64, 10),
    "GRU": GRUModel(64, 10),
    "Transformer": TransformerModel(64, 10),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 100
results = {}

for name, model in models.items():
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    accuracy_per_epoch = []

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                pred = torch.argmax(out, dim=1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        acc = correct / total
        accuracy_per_epoch.append(acc)

    results[name] = accuracy_per_epoch

# ====== Plotting ======

plt.figure(figsize=(14, 6))
for name, acc in results.items():
    plt.plot(acc, label=name)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Model Comparison on Digits Dataset (100 Epochs)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("digits_accuracy_comparison.png")
plt.show()

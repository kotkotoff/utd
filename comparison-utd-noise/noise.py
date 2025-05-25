import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch_geometric.data import Data, DataLoader
from sklearn.metrics import accuracy_score

# Dataset generation (modified noise fraction)
def generate_tau_sensitive_data(n_samples=600, n_features=8, batch_size=100, noise_fraction=0.5):
    torch.manual_seed(42)
    np.random.seed(42)
    X, y = [], []
    for i in range(n_samples):
        base = np.random.randn(n_features)
        if i % 2 == 0:
            diff = np.ones(n_features) * 0.8
            label = 0
        else:
            diff = -np.ones(n_features) * 0.8
            label = 1
        point = base + diff
        if np.random.rand() < noise_fraction:
            point = np.random.randn(n_features) * 5
        X.append(point)
        y.append(label)
    X, y = np.array(X), np.array(y)
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    num_batches = len(X) // batch_size
    batch = torch.tensor([i for i in range(num_batches) for _ in range(batch_size)], dtype=torch.long)
    return Data(
        x=torch.tensor(X, dtype=torch.float32),
        y=torch.tensor(y, dtype=torch.long),
        batch=batch
    )

# UTDGraphNetNoise
import torch
import torch.nn as nn
import torch.nn.functional as F

class UTDGraphNetNoise(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=16, output_dim=2, num_layers=2, max_recursion=10, tau_threshold=0.005):
        super(UTDGraphNetNoise, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_recursion = max_recursion
        self.tau_threshold = tau_threshold

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.diff_layers = nn.ModuleList([
            nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(num_layers)
        ])
        self.tau_layer = nn.Linear(hidden_dim, 1)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def compute_d1(self, h, edge_index):
        row, col = edge_index
        h_i, h_j = h[row], h[col]
        diff = torch.abs(h_i - h_j)
        return row, diff

    def compute_tau(self, h):
        return F.softplus(self.tau_layer(h)).squeeze()

    def forward(self, x, edge_index):
        h = F.relu(self.input_proj(x))
        for layer in self.diff_layers:
            row, diff = self.compute_d1(h, edge_index)
            diff_agg = torch.zeros_like(h).scatter_add_(0, row.unsqueeze(-1).expand(-1, self.hidden_dim), diff)
            h = F.relu(layer(torch.cat([h, diff_agg], dim=-1)))

            tau = self.compute_tau(h)
            n_updates = torch.clamp((1.0 / tau).floor().long(), max=self.max_recursion)

            h_new = h.clone()
            for i in range(h.shape[0]):
                if n_updates[i] > 0 and tau[i] < self.tau_threshold:
                    for _ in range(n_updates[i]):
                        h_new[i] = self.gru(diff_agg[i], h[i])
            h = h_new

        return self.output(h), tau


# MLP with dropout
class MLP(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=16, output_dim=2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

# Generate edge_index
def create_edge_index(x, epsilon=0.95):
    cos_sim = F.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=2)
    row, col = torch.where(cos_sim > epsilon)
    edge_index = torch.stack([row, col], dim=0)
    return edge_index

# Training function
def train(model, data, epochs=50, lr=1e-3, tau_threshold=0.005, lambda_stability=0.1, is_utd=True):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    losses, accuracies = [], []

    for epoch in range(epochs):
        optimizer.zero_grad()
        x, y, edge_index = data.x, data.y, create_edge_index(data.x) if is_utd else None
        if is_utd:
            output, tau = model(x, edge_index)
            tau = model.compute_tau(model.input_proj(x))
            loss_stability = torch.mean(torch.relu(tau - tau_threshold))
            loss = criterion(output, y) + lambda_stability * loss_stability
        else:
            output = model(x)
            loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1).detach().numpy()
        acc = accuracy_score(y.numpy(), pred)
        losses.append(loss.item())
        accuracies.append(acc)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Accuracy: {acc:.4f}")

    return losses, accuracies

# Generate clean test data
def generate_clean_test_data(n_samples=200, n_features=8):
    torch.manual_seed(43)
    np.random.seed(43)
    X, y = [], []
    for i in range(n_samples):
        base = np.random.randn(n_features)
        if i % 2 == 0:
            diff = np.ones(n_features) * 0.8
            label = 0
        else:
            diff = -np.ones(n_features) * 0.8
            label = 1
        X.append(base + diff)
        y.append(label)
    X, y = np.array(X), np.array(y)
    return Data(
        x=torch.tensor(X, dtype=torch.float32),
        y=torch.tensor(y, dtype=torch.long)
    )

# Comparison
def compare_utd_mlp():
    train_data = generate_tau_sensitive_data(n_samples=600, n_features=8, batch_size=100, noise_fraction=0.5)
    test_data = generate_clean_test_data(n_samples=200, n_features=8)

    utd_model = UTDGraphNetNoise(input_dim=8, hidden_dim=16, output_dim=2, tau_threshold=0.005, max_recursion=10)
    mlp_model = MLP(input_dim=8, hidden_dim=16, output_dim=2)

    print("Training UTDGraphNetNoise...")
    utd_losses, utd_accuracies = train(utd_model, train_data, epochs=50, is_utd=True)
    print("\nTraining MLP...")
    mlp_losses, mlp_accuracies = train(mlp_model, train_data, epochs=50, is_utd=False)

    utd_model.eval()
    mlp_model.eval()
    with torch.no_grad():
        test_edge_index = create_edge_index(test_data.x)
        utd_logits, _ = utd_model(test_data.x, test_edge_index)
        utd_pred = utd_logits.argmax(dim=1).numpy()
        mlp_pred = mlp_model(test_data.x).argmax(dim=1).numpy()
        utd_test_acc = accuracy_score(test_data.y.numpy(), utd_pred)
        mlp_test_acc = accuracy_score(test_data.y.numpy(), mlp_pred)

    print("\nTest Accuracy Comparison:")
    print(f"UTDGraphNetNoise: {utd_test_acc:.4f}")
    print(f"MLP: {mlp_test_acc:.4f}")

    return utd_accuracies, mlp_accuracies, utd_test_acc, mlp_test_acc

if __name__ == "__main__":
    utd_acc, mlp_acc, utd_test_acc, mlp_test_acc = compare_utd_mlp()
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from sklearn.metrics import accuracy_score

# Store results here
results = {}

# --- MLP (no graph) ---
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, edge_index=None, batch=None):
        return self.net(x)

# --- GCN ---
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch=None):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

# --- GAT ---
class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=1)
        self.conv2 = GATConv(hidden_dim, output_dim, heads=1)

    def forward(self, x, edge_index, batch=None):
        x = F.elu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

# --- TauAttentionGNN (scalar τ) ---
class TauAttentionGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, tau_dim=8, max_recursion=5):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.tau_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, tau_dim),
            nn.Sigmoid()
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.max_recursion = max_recursion

    def compute_d1(self, h, edge_index):
        row, col = edge_index
        d1 = torch.abs(h[row] - h[col])
        return row, d1

    def forward(self, x, edge_index, batch=None):
        h = F.relu(self.input_proj(x))
        tau = self.tau_encoder(x).mean(dim=-1, keepdim=True)
        for _ in range(self.max_recursion):
            row, d1 = self.compute_d1(h, edge_index)
            weighted = d1 * tau[row]
            agg = torch.zeros_like(h).index_add(0, row, weighted)
            h = self.gru(agg, h)
        return self.output(h)

# --- TauAttentionDirectionalGNN (softmax τ_ij) ---
class TauAttentionDirectionalGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, max_recursion=5):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.attn = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.max_recursion = max_recursion

    def forward(self, x, edge_index, batch=None):
        h = F.relu(self.input_proj(x))
        row, col = edge_index
        for _ in range(self.max_recursion):
            h_i, h_j = h[row], h[col]
            d_ij = torch.abs(h_i - h_j)
            attn_input = torch.cat([h_i, h_j], dim=1)
            attn_score = self.attn(attn_input).squeeze(-1)
            attn_exp = torch.exp(attn_score - attn_score.max())
            attn_sum = torch.zeros(h.size(0), device=x.device).index_add(0, row, attn_exp)
            alpha = attn_exp / (attn_sum[row] + 1e-8)
            agg = torch.zeros_like(h).index_add(0, row, d_ij * alpha.unsqueeze(-1))
            h = self.gru(agg, h)
        return self.output(h)

# --- TauRecursiveGNN ---
class TauRecursiveGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, max_recursion=5):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.tau_proj = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softplus()
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.max_recursion = max_recursion

    def forward(self, x, edge_index, batch=None):
        h = F.relu(self.input_proj(x))
        tau_val = self.tau_proj(x).squeeze()
        steps = torch.clamp((1.0 / tau_val).long(), max=self.max_recursion)

        for _ in range(self.max_recursion):
            row, col = edge_index
            d1 = torch.abs(h[row] - h[col])
            agg = torch.zeros_like(h).index_add(0, row, d1)
            mask = steps > 0
            new_h = self.gru(agg, h)
            h = torch.where(mask.unsqueeze(1), new_h, h)
            steps = steps - 1

        return self.output(h)


# --- Training and evaluation ---
from sklearn.metrics import accuracy_score

def train_and_eval(model_class, name, **kwargs):
    model = model_class(data.num_node_features, 32, len(data.y.unique()), **kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=1)
    acc = accuracy_score(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu())
    results[name] = acc

# --- Run all models ---
results = {}
train_and_eval(MLP, "MLP")
train_and_eval(GCN, "GCN")
train_and_eval(GAT, "GAT")
train_and_eval(TauAttentionGNN, "TauAttentionGNN")
train_and_eval(TauAttentionDirectionalGNN, "TauDirectionalGNN")
train_and_eval(TauRecursiveGNN, "TauRecursiveGNN")

# --- Print results ---
import pandas as pd
df_results = pd.DataFrame.from_dict(results, orient='index', columns=['Accuracy'])
df_results = df_results.sort_values(by='Accuracy', ascending=False)
print(df_results)

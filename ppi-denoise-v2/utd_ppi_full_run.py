
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import PPI
from sklearn.metrics import f1_score
import gc

# Load the full PPI dataset
ppi_train = PPI(root='ppi_data', split='train')
ppi_test = PPI(root='ppi_data', split='test')

# Define the UTDGraphNetDenoiseV2 model
class UTDGraphNetDenoiseV2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, k=5, steps=3, tau_thresh=0.05):
        super().__init__()
        self.k = k
        self.steps = steps
        self.tau_thresh = tau_thresh

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.diff_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        self.tau_head = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        h = F.relu(self.input_proj(x))
        for _ in range(self.steps):
            row, col = edge_index
            h_i, h_j = h[row], h[col]
            d1 = torch.abs(h_i - h_j)
            diff_agg = torch.zeros_like(h).index_add(0, row, d1)

            var_accum = torch.zeros(h.shape[0], device=h.device)
            count = torch.zeros(h.shape[0], device=h.device)
            for i in range(d1.size(0)):
                vi = d1[i].var().item()
                var_accum[row[i]] += vi
                count[row[i]] += 1
            var_mean = (var_accum / (count + 1e-6)).unsqueeze(1)

            tau = self.tau_head(var_mean)
            alpha = torch.sigmoid(self.tau_thresh - tau)
            new_h = self.gru(diff_agg, h)
            h = alpha * new_h + (1 - alpha) * h

        return self.output(h)

# Training function over individual graphs
def train_per_graph(model, graphs, optimizer, criterion, epochs=5):
    model.train()
    for epoch in range(epochs):
        for graph in graphs:
            optimizer.zero_grad()
            out = model(graph.x, graph.edge_index)
            loss = criterion(out, graph.y.float())
            loss.backward()
            optimizer.step()
        gc.collect()
        torch.cuda.empty_cache()

# Evaluation function
@torch.no_grad()
def eval_per_graph(model, graphs):
    model.eval()
    y_true, y_pred = [], []
    for graph in graphs:
        out = model(graph.x, graph.edge_index)
        pred = (out > 0).int()
        y_true.append(graph.y.cpu())
        y_pred.append(pred.cpu())
    return f1_score(torch.cat(y_true), torch.cat(y_pred), average='micro')

# Instantiate and train the model
model = UTDGraphNetDenoiseV2(input_dim=50, hidden_dim=64, output_dim=121)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCEWithLogitsLoss()

train_per_graph(model, ppi_train, optimizer, criterion, epochs=5)
f1 = eval_per_graph(model, ppi_test)
print(f"UTDGraphNetDenoiseV2 full PPI micro-F1: {f1:.4f}")

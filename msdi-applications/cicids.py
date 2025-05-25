import torch.nn as nn
import torch.nn.functional as F

class MSDITabular(nn.Module):
    def __init__(self, input_dim=80, proj_dim=64, num_classes=2, use_mlp=True):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU()
        )
        self.use_mlp = use_mlp
        self.sdi_memory = nn.Parameter(F.normalize(torch.randn(num_classes, proj_dim), dim=1), requires_grad=False)
        if use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(proj_dim, proj_dim),
                nn.ReLU(),
                nn.Linear(proj_dim, num_classes)
            )

    def forward(self, x, y=None, update_memory=False):
        z = self.proj(x)
        if update_memory and y is not None:
            with torch.no_grad():
                for c in range(self.sdi_memory.shape[0]):
                    mask = (y == c)
                    if mask.any():
                        mean = z[mask].mean(dim=0)
                        self.sdi_memory[c] = F.normalize(mean, dim=0)

        logits = z @ self.sdi_memory.T
        if self.use_mlp:
            logits = logits + self.mlp(z)
        return logits

def train_msdi(model, train_loader, test_loader, epochs=10):
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    accs = []

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            logits = model(xb, yb, update_memory=True)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += len(yb)
        acc = correct / total
        accs.append(acc)
        print(f"Epoch {epoch+1}: accuracy = {acc:.4f}")
    
    return accs

model = MSDITabular(input_dim=80)
accs = train_msdi(model, train_loader, test_loader, epochs=10)

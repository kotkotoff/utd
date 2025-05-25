import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt

# Generate 2D synthetic dataset with 3 classes
X, y = make_classification(
    n_samples=1200,
    n_features=2,
    n_redundant=0,
    n_clusters_per_class=1,
    n_classes=3,
    class_sep=1.2,
    random_state=42
)

# Normalize features
X = StandardScaler().fit_transform(X)

# Create train/test masks
idx_train, idx_test = train_test_split(np.arange(len(X)), stratify=y, test_size=0.3, random_state=42)
train_mask = torch.zeros(len(X), dtype=torch.bool)
test_mask = torch.zeros(len(X), dtype=torch.bool)
train_mask[idx_train] = True
test_mask[idx_test] = True

# Build k-NN graph
k = 10
nn = NearestNeighbors(n_neighbors=k)
nn.fit(X)
edges = []
for i, neighbors in enumerate(nn.kneighbors(X, return_distance=False)):
    for j in neighbors:
        edges.append([i, j])
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# Create torch_geometric Data object
data = Data(
    x=torch.tensor(X, dtype=torch.float),
    y=torch.tensor(y, dtype=torch.long),
    edge_index=edge_index,
    train_mask=train_mask,
    test_mask=test_mask
)
data.num_nodes = data.x.size(0)

# Optional: visualize the dataset
plt.figure(figsize=(5, 5))
for cls in np.unique(y):
    plt.scatter(X[y == cls, 0], X[y == cls, 1], label=f"Class {cls}", alpha=0.6)
plt.title("Synthetic 2D Dataset with 3 Classes")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()

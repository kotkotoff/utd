import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def compute_D2(D1_nodes):
    m = D1_nodes.size(0)
    u = D1_nodes.unsqueeze(1).repeat(1, m, 1)
    v = D1_nodes.unsqueeze(0).repeat(m, 1, 1)
    return (u - v).norm(p=1, dim=2)

def compute_D3(D2_matrix):
    m = D2_matrix.size(0)
    u = D2_matrix.unsqueeze(1).repeat(1, m, 1)
    v = D2_matrix.unsqueeze(0).repeat(m, 1, 1)
    return (u - v).norm(p=1, dim=2)

n_nodes = 20
feature_dim = 10
proj_dim = 16
features = torch.randn(n_nodes, feature_dim)
linear = nn.Linear(feature_dim, proj_dim)

with torch.no_grad():
    Z = linear(features)

pairs = [(i, j) for i in range(n_nodes) for j in range(n_nodes) if i != j]
np.random.shuffle(pairs)
pairs = pairs[:32]
D1_nodes = torch.stack([Z[i] - Z[j] for i, j in pairs])

D2 = compute_D2(D1_nodes)
D3 = compute_D3(D2)

def graph_stats(D):
    theta_values = np.linspace(0.05, 0.95, 19)
    components = []
    avg_degree = []
    var_measure = []

    for theta in theta_values:
        A = (D < torch.quantile(D, theta)).float()
        deg = A.sum(dim=1).numpy()
        avg_degree.append(deg.mean())
        var_measure.append(D.var().item())

        visited = set()
        count = 0
        for i in range(len(A)):
            if i in visited:
                continue
            stack = [i]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                neighbors = torch.where(A[node] > 0)[0].tolist()
                stack.extend(neighbors)
            count += 1
        components.append(count)

    return theta_values, components, avg_degree, var_measure

theta, comp2, deg2, var2 = graph_stats(D2)
_, comp3, deg3, var3 = graph_stats(D3)


fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

axs[0].plot(theta, comp2, label="Depth 2", marker='o')
axs[0].plot(theta, comp3, label="Depth 3", marker='x')
axs[0].set_ylabel("Connected Components")
axs[0].legend()
axs[0].grid(True)

axs[1].plot(theta, deg2, label="Depth 2", marker='s')
axs[1].plot(theta, deg3, label="Depth 3", marker='^')
axs[1].set_ylabel("Average Degree")
axs[1].legend()
axs[1].grid(True)

axs[2].plot(theta, var2, label="Depth 2", marker='D')
axs[2].plot(theta, var3, label="Depth 3", marker='*')
axs[2].set_ylabel("Var($\\mathcal{D}_n$)")
axs[2].set_xlabel("Threshold $\\theta$")
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()

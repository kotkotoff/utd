import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# --- Learnable Differentiation
class LearnableDifferentiationLayer(nn.Module):
    def __init__(self, in_dim, proj_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, proj_dim)

    def forward(self, X):
        return self.linear(X)

# --- Recursive Differentiation Layer
class RecursiveDifferentiationLayer(nn.Module):
    def __init__(self, depth=2, max_pairs=64):
        super().__init__()
        self.depth = depth
        self.max_pairs = max_pairs

    def forward(self, Z):
        D_current = Z
        D_layers = []
        for _ in range(self.depth):
            n = D_current.size(0)
            pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
            sampled = random.sample(pairs, min(len(pairs), self.max_pairs))
            D1_nodes = torch.stack([D_current[i] - D_current[j] for (i, j) in sampled])
            m = D1_nodes.size(0)
            u = D1_nodes.unsqueeze(1).repeat(1, m, 1)
            v = D1_nodes.unsqueeze(0).repeat(m, 1, 1)
            D2 = (u - v).norm(p=1, dim=2)
            D_current = D1_nodes
            D_layers.append(D2)
        return D2, D_layers

def analyze_thresholds(D2, thresholds):
    results = []
    for q in thresholds:
        A = (D2 < torch.quantile(D2, q)).float()
        G = nx.from_numpy_array(A.numpy())
        n_components = nx.number_connected_components(G)
        degrees = np.array([d for _, d in G.degree()])
        avg_deg = degrees.mean() if len(degrees) > 0 else 0
        tau = 1 / (degrees + 1e-6)
        tau_var = tau.var() if len(tau) > 0 else 0
        results.append((float(q), n_components, avg_deg, tau_var))
    return results

def compare_depths(depths, n_nodes=20, feature_dim=10, proj_dim=16, max_pairs=64):
    all_results = {}

    for d in depths:
        features = torch.randn(n_nodes, feature_dim)
        diff1 = LearnableDifferentiationLayer(in_dim=feature_dim, proj_dim=proj_dim)
        recursive_diff = RecursiveDifferentiationLayer(depth=d, max_pairs=max_pairs)

        with torch.no_grad():
            Z = diff1(features)
            D_final, _ = recursive_diff(Z)
            thresholds = torch.linspace(0.05, 0.95, steps=61)
            results = analyze_thresholds(D_final, thresholds)
            df = pd.DataFrame(results, columns=["θ", "Components", "AvgDegree", "Var(τ)"])
            all_results[d] = df

    return all_results

def plot_comparisons(results_by_depth):
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    for d, df in results_by_depth.items():
        axs[0].plot(df["θ"], df["Components"], label=f'depth={d}')
        axs[1].plot(df["θ"], df["AvgDegree"], label=f'depth={d}')
        axs[2].plot(df["θ"], df["Var(τ)"], label=f'depth={d}')

    axs[0].set_ylabel("Connected Components")
    axs[0].set_title("Recursive Differentiation by Depth")

    axs[1].set_ylabel("Average Degree")
    axs[2].set_ylabel("Var[τ(x)]")
    axs[2].set_xlabel("Quantile Threshold θ")

    for ax in axs:
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()

depth_values = [1, 2, 3]
results_by_depth = compare_depths(depth_values)
plot_comparisons(results_by_depth)

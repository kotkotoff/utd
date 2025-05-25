import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import os

# --- Parameters ---
N = 1000
outlier_frac = 0.1
eps = 0.2
delta = 1e-3
levels = 15
np.random.seed(0)

# --- Generate 3 clusters + uniform outliers ---
n_per_cluster = int(N * (1 - outlier_frac) / 3)
clusters = [np.random.normal(loc=center, scale=0.6, size=(n_per_cluster, 2))
            for center in [(-1, -1), (1, -1), (0, 1)]]
outliers = np.random.uniform(low=-3, high=3, size=(N - 3 * n_per_cluster, 2))
X_init = np.vstack(clusters + [outliers])

# --- Initialize data structures ---
X_levels = [X_init.copy()]  # list of point positions per level
tau_n_list = np.zeros((N, levels))
depth = np.full(N, levels)

# --- Recursive updates ---
for n in range(levels):
    X_prev = X_levels[-1]
    X_next = X_prev.copy()
    D = distance_matrix(X_prev, X_prev)
    for i in range(N):
        neighbors = np.where(D[i] < eps)[0]
        if len(neighbors) > 0:
            X_next[i] = np.mean(X_prev[neighbors], axis=0)
        # else: keep same
        tau_n_list[i, n] = np.linalg.norm(X_next[i] - X_prev[i])
        if tau_n_list[i, n] < delta and depth[i] == levels:
            depth[i] = n + 1
            
    X_levels.append(X_next)

# --- τ(x) statistics per node ---
tau_mean = np.mean(tau_n_list, axis=1)
tau_var = np.var(tau_n_list, axis=1)

# --- τ dynamics per level ---
tau_mean_per_level = np.mean(tau_n_list, axis=0)
tau_var_per_level = np.var(tau_n_list, axis=0)
depth_levels = np.arange(1, levels + 1)

# --- Ensure output dir ---
os.makedirs("depth-rhythm-invariant", exist_ok=True)

# --- Plot all 4 subplots ---
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# (1) Depth vs Mean τ(x)
axes[0, 0].scatter(tau_mean, depth, s=6, alpha=0.6, color='blue')
axes[0, 0].set_xlabel('Mean τ(x)')
axes[0, 0].set_ylabel('Recursive Depth d(x)')
axes[0, 0].set_title('Recursive Depth vs. Mean τ(x)')

# (2) Mean/Var of τ(x) across levels
axes[0, 1].plot(depth_levels, tau_mean_per_level, label='Mean τ(x)', marker='o', color='orange')
axes[0, 1].plot(depth_levels, tau_var_per_level, label='Var[τ(x)]', marker='o', color='red')
axes[0, 1].set_xlabel('Recursion Depth')
axes[0, 1].set_ylabel('Value')
axes[0, 1].legend()
axes[0, 1].set_title('τ(x) Dynamics Across Recursion Levels')

# (3) Var[τ(x)] vs Mean τ(x)
axes[1, 0].scatter(tau_mean, tau_var, s=6, alpha=0.6, color='darkgreen')
axes[1, 0].set_xlabel('Mean τ(x)')
axes[1, 0].set_ylabel('Var[τ(x)]')
axes[1, 0].set_title('Rhythmic Variance vs. Mean τ(x)')

# (4) Histogram of d(x)
axes[1, 1].hist(depth, bins=np.arange(1, levels + 2) - 0.5, color='purple', alpha=0.7)
axes[1, 1].set_xlabel('Recursive Depth d(x)')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title('Histogram of Recursive Depths')
axes[1, 1].set_xticks(np.arange(1, levels + 1))

plt.tight_layout()
plt.savefig("depth-rhythm-invariant/recursion_depth.png", dpi=300)
plt.show()

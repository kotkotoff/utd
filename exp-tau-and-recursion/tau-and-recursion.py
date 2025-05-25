import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

# Parameters
N = 1000
outlier_frac = 0.1
eps = 0.15
delta = 0.01
levels = 15
beta = 5.0
np.random.seed(0)

# --- Generate data: 3 Gaussians + uniform noise ---
n_per_cluster = int(N * (1 - outlier_frac) / 3)
clusters = [np.random.normal(loc=center, scale=0.5, size=(n_per_cluster, 2))
            for center in [(-1, -1), (1, -1), (0, 1)]]
n_outliers = N - 3 * n_per_cluster
outliers = np.random.uniform(low=-3, high=3, size=(n_outliers, 2))
X = np.vstack(clusters + [outliers])

# --- Compute distance matrix and epsilon-graph ---
D = distance_matrix(X, X)
G_eps = (D < eps).astype(int)
np.fill_diagonal(G_eps, 0)

# --- Initialization ---
tau_n_list = np.zeros((N, levels))
depth = np.full(N, levels)
D_prev = np.array([
    np.mean(D[i][np.where(G_eps[i])[0]]) if G_eps[i].sum() > 0 else 1.0
    for i in range(N)
])

# --- Recursive differentiation ---
for n in range(levels):
    D_curr = np.zeros(N)
    for i in range(N):
        neighbors = np.where(G_eps[i])[0]
        if len(neighbors) > 0:
            D_curr[i] = np.mean(D_prev[neighbors])
        else:
            D_curr[i] = D_prev[i]
        tau_n_list[i, n] = abs(D_curr[i] - D_prev[i])
        if tau_n_list[i, n] < delta and depth[i] == levels:
            depth[i] = n + 1
    D_prev = D_curr

# --- Compute metrics ---
tau_mean = np.mean(tau_n_list, axis=1)
tau_var = np.var(tau_n_list, axis=1)
r_refl = 1 / tau_mean * np.exp(-beta * tau_var)

tau_mean_per_level = np.mean(tau_n_list, axis=0)
tau_var_per_level = np.var(tau_n_list, axis=0)

# --- Plot 1: depth vs mean τ; τ dynamics across levels ---
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(tau_mean, depth, s=5, alpha=0.5)
plt.xlabel("Mean τ(x)")
plt.ylabel("Depth d(x)")
plt.title("Depth vs. Mean τ(x)")

plt.subplot(1, 2, 2)
plt.plot(range(1, levels + 1), tau_mean_per_level, label="Mean τ(x)")
plt.plot(range(1, levels + 1), tau_var_per_level, label="Var[τ(x)]")
plt.xlabel("Recursion Level")
plt.ylabel("Value")
plt.title("τ(x) Dynamics Across Levels")
plt.legend()

plt.tight_layout()
plt.show()

# --- Plot 2: Reflexivity vs τ and histogram of depth ---
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(tau_mean, r_refl, s=5, alpha=0.5)
plt.xlabel("Mean τ(x)")
plt.ylabel("Reflexivity $R_{\\mathrm{refl}}(x)$")
plt.title("Reflexivity vs. Mean τ(x)")

plt.subplot(1, 2, 2)
plt.hist(depth, bins=15, color='gray', edgecolor='black')
plt.xlabel("Recursive Depth d(x)")
plt.ylabel("Count")
plt.title("Histogram of Recursive Depth")

plt.tight_layout()
plt.show()

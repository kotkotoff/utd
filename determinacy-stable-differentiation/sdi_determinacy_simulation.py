
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import euclidean
from sklearn.cluster import AgglomerativeClustering

# Parameters
np.random.seed(42)
sigma = 0.12
delta = 0.15
epsilons = [0.15, 0.4]

# Generate clustered dataset
cluster_centers = [(1, 1), (3, 1), (2, 3)]
clustered_data = np.vstack([
    np.random.normal(loc=center, scale=sigma, size=(7 if i < 2 else 6, 2))
    for i, center in enumerate(cluster_centers)
])

# Generate uniform dataset
uniform_data = np.random.uniform(low=0, high=4, size=(20, 2))

# Helper function to compute instability
def compute_tau(X):
    dists = pairwise_distances(X)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists, axis=1)
    max_dist = np.max(dists)
    return min_dist / max_dist

# Differentiation + Graph formation + Connected components
def differentiate_and_cluster(X, epsilon, delta):
    D = pairwise_distances(X)
    tau = compute_tau(X)
    mask = (D < epsilon)
    np.fill_diagonal(mask, 0)
    instable_mask = (tau >= delta)
    mask[instable_mask, :] = 0
    mask[:, instable_mask] = 0
    model = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='single', distance_threshold=0.5)
    labels = model.fit_predict(~mask)
    return labels

# Perform differentiation for both epsilons
labels_eps1 = differentiate_and_cluster(clustered_data, epsilons[0], delta)
labels_eps2 = differentiate_and_cluster(clustered_data, epsilons[1], delta)
labels_uniform = differentiate_and_cluster(uniform_data, epsilons[1], delta)

# Plotting
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 120,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.3
})

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# (a) SDI at epsilon = 0.15
axs[0, 0].scatter(clustered_data[:, 0], clustered_data[:, 1], c=labels_eps1, cmap='tab10', edgecolor='k')
axs[0, 0].set_title("SDI at ε = 0.15")

# (b) SDI at epsilon = 0.4
axs[0, 1].scatter(clustered_data[:, 0], clustered_data[:, 1], c=labels_eps2, cmap='tab10', edgecolor='k')
axs[0, 1].set_title("SDI at ε = 0.4")

# (c) Convergence curve
eps_curve = []
for e in np.linspace(0.1, 0.5, 20):
    labels = differentiate_and_cluster(clustered_data, e, delta)
    eps_curve.append(len(set(labels)))
axs[1, 0].plot(np.linspace(0.1, 0.5, 20), eps_curve, marker='o', color='steelblue')
axs[1, 0].set_title("Convergence curve")
axs[1, 0].set_xlabel("ε")
axs[1, 0].set_ylabel("Number of SDI")

# (d) Trivial SDI for uniform distribution
axs[1, 1].scatter(uniform_data[:, 0], uniform_data[:, 1], c=labels_uniform, cmap='tab10', edgecolor='k')
axs[1, 1].set_title("Trivial SDI (uniform data)")

for ax in axs.flatten():
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.suptitle("Convergence and nesting of SDI structures", fontsize=16, y=1.03)
plt.show()

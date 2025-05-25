
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
import networkx as nx

# ----------------------------
# Generate clustered data
# ----------------------------
np.random.seed(42)
cluster_centers = [(1, 1), (4, 1), (2.5, 4)]
points_per_subcluster = 12
sigma = 0.08

X = []
for cx, cy in cluster_centers:
    for dx, dy in [(-0.2, -0.2), (0.2, -0.2), (0, 0.2)]:
        subcenter = (cx + dx, cy + dy)
        subcluster = np.random.normal(loc=subcenter, scale=sigma, size=(points_per_subcluster, 2))
        X.append(subcluster)
X = np.vstack(X)

# ----------------------------
# Define differentiation function and SDI condition
# ----------------------------
def differentiation(x_i, x_j, eps):
    dist = np.linalg.norm(x_i - x_j)
    if dist < eps:
        return dist
    else:
        return None

def instability(x_idx, D, D_max):
    dists = np.delete(D[x_idx], x_idx)
    return np.min(dists) / D_max

# ----------------------------
# Build differentiation graph G_eps
# ----------------------------
eps_range = np.linspace(0.1, 0.5, 50)
theta = 0.2
num_components = []

D = euclidean_distances(X)
D_max = np.max(D)

for eps in eps_range:
    G = nx.Graph()
    for i in range(len(X)):
        sigma_i = instability(i, D, D_max)
        if sigma_i >= theta:
            continue
        for j in range(i + 1, len(X)):
            sigma_j = instability(j, D, D_max)
            if sigma_j >= theta:
                continue
            d = differentiation(X[i], X[j], eps)
            if d is not None:
                G.add_edge(i, j)
    num_components.append(nx.number_connected_components(G))

# ----------------------------
# Plot phase transition
# ----------------------------
plt.figure(figsize=(8, 5))
plt.plot(eps_range, num_components, color='steelblue', linewidth=2)
plt.xlabel(r"Differentiation threshold $\varepsilon$")
plt.ylabel("Number of SDI components")
plt.title("Phase Transition in Differentiational Graphs")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("phase_transition_simulation.png", dpi=300)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import pairwise_distances

np.random.seed(42)
sigma = 0.15
epsilon_clustered = 0.7
epsilon_uniform = 0.5
max_iter = 20
convergence_threshold = 1e-4

# Generate data
cluster1 = np.random.normal(loc=(1, 1), scale=sigma, size=(25, 2))
cluster2 = np.random.normal(loc=(3, 3), scale=sigma, size=(25, 2))
clustered_data = np.vstack([cluster1, cluster2])
uniform_data = np.random.uniform(0, 4, size=(50, 2))

def run_recursive_differentiation(X, epsilon, max_iter=20, tol=1e-4):
    centroids_history = []
    current_data = X.copy()
    for t in range(max_iter):
        D = pairwise_distances(current_data)
        adjacency = (D < epsilon).astype(int)
        np.fill_diagonal(adjacency, 0)
        G = nx.Graph(adjacency)
        components = list(nx.connected_components(G))
        centroids = np.array([np.mean(current_data[list(c)], axis=0) for c in components])
        centroids_history.append(centroids)
        if t > 0:
            prev = centroids_history[-2]
            if len(prev) == len(centroids) and np.linalg.norm(prev - centroids) < tol:
                break
        current_data = centroids
    return G, components, centroids_history

G_clustered, comp_clustered, centroids_hist_clustered = run_recursive_differentiation(clustered_data, epsilon_clustered)
G_uniform, comp_uniform, centroids_hist_uniform = run_recursive_differentiation(uniform_data, epsilon_uniform)

depth_clustered = len(centroids_hist_clustered)
depth_uniform = len(centroids_hist_uniform)

def compute_entropy(components, total_points):
    sizes = np.array([len(c) for c in components])
    p = sizes / total_points
    return -np.sum(p * np.log(p + 1e-10))

entropy_clustered = compute_entropy(comp_clustered, 50)
entropy_uniform = compute_entropy(comp_uniform, 50)

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

axs[0, 0].scatter(clustered_data[:, 0], clustered_data[:, 1], c='blue', label='Points')
for (i, j) in G_clustered.edges():
    axs[0, 0].plot(*zip(clustered_data[i], clustered_data[j]), color='gray', alpha=0.5)
for c in comp_clustered:
    center = np.mean(clustered_data[list(c)], axis=0)
    axs[0, 0].scatter(*center, c='green', marker='*', s=150)
axs[0, 0].set_title("Clustered: Components + SDI")

axs[0, 1].scatter(uniform_data[:, 0], uniform_data[:, 1], c='orange')
for (i, j) in G_uniform.edges():
    axs[0, 1].plot(*zip(uniform_data[i], uniform_data[j]), color='gray', alpha=0.5)
axs[0, 1].set_title("Uniform: No structure")

axs[1, 0].bar(["Clustered", "Uniform"], [depth_clustered, depth_uniform], color=['steelblue', 'orange'])
axs[1, 0].set_title("Depth of Recursion")
axs[1, 0].set_ylabel("Iterations")

axs[1, 1].bar(["Clustered", "Uniform"], [entropy_clustered, entropy_uniform], color=['steelblue', 'orange'])
axs[1, 1].set_title("Semantic Entropy")
axs[1, 1].set_ylabel("H(D*)")

plt.tight_layout()
plt.suptitle("Recursive differentiation dynamics", fontsize=16, y=1.03)
plt.show()

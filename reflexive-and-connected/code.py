import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist, squareform

# --- PARAMETERS ---
np.random.seed(42)
n = 20
eps = 0.5
tau_thresh = 0.2

# --- DATASET 1: clustered ---
cluster1 = np.random.normal(loc=[1, 1], scale=0.1, size=(10, 2))
cluster2 = np.random.normal(loc=[3, 3], scale=0.1, size=(10, 2))
X_clustered = np.vstack([cluster1, cluster2])

# --- DATASET 2: uniform ---
X_uniform = np.random.uniform(low=0, high=4, size=(20, 2))

# --- DIFFERENTIATION MEASURES ---
def compute_tau(X):
    dist = pairwise_distances(X)
    np.fill_diagonal(dist, np.inf)
    min_dists = np.min(dist, axis=1)
    max_dist = np.max(dist)
    return min_dists / max_dist

def build_graph(X, tau, eps, tau_thresh):
    n = len(X)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(X[i] - X[j])
            if dist < eps and tau[i] < tau_thresh and tau[j] < tau_thresh:
                G.add_edge(i, j)
    return G

def centroid(X, indices):
    return np.mean(X[indices], axis=0)

# --- VISUALIZATION FUNCTION ---
def plot_graphs(X1, G1, centroids1, X2, G2, centroids2):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # Clustered
    axs[0].set_title("Clustered: Recursive Differentiation Enabled")
    pos = {i: X1[i] for i in range(len(X1))}
    nx.draw(G1, pos, ax=axs[0], node_color='lightblue', with_labels=False, node_size=50, edge_color='gray')
    axs[0].scatter(*zip(*centroids1), c='green', s=100, marker='*')

    # Uniform
    axs[1].set_title("Uniform: No Recursive Differentiation")
    pos = {i: X2[i] for i in range(len(X2))}
    nx.draw(G2, pos, ax=axs[1], node_color='salmon', with_labels=False, node_size=50, edge_color='gray')
    axs[1].scatter(*zip(*centroids2), c='green', s=100, marker='*')

    for ax in axs:
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 4)
        ax.set_aspect('equal')
        ax.grid(True)

    plt.tight_layout()
    plt.show()

# --- CLUSTERED DATA ---
tau_clustered = compute_tau(X_clustered)
G_clustered = build_graph(X_clustered, tau_clustered, eps, tau_thresh)
components_clustered = list(nx.connected_components(G_clustered))
centroids_clustered = [centroid(X_clustered, list(c)) for c in components_clustered]

# --- UNIFORM DATA ---
tau_uniform = compute_tau(X_uniform)
G_uniform = build_graph(X_uniform, tau_uniform, eps, tau_thresh)
components_uniform = list(nx.connected_components(G_uniform))
centroids_uniform = [centroid(X_uniform, list(c)) for c in components_uniform]

# --- PLOT ---
plot_graphs(X_clustered, G_clustered, centroids_clustered,
            X_uniform, G_uniform, centroids_uniform)

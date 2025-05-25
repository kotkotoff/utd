
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics.pairwise import euclidean_distances

np.random.seed(42)
n_points = 60
timesteps = 40
shift_time = 20

# Generate 2D positions + discrete color aspect
positions = np.random.randn(n_points, 2)
colors = np.random.randint(0, 3, size=(n_points, 1)) * 1.0
X = np.hstack((positions, colors))  # aspects: x, y, color

# Initial and context vectors
aspect_weights = np.array([1.0, 1.0, 0.0])  # only x, y initially
context_vector = np.tile(aspect_weights, (timesteps, 1))
context_vector[shift_time:, 2] = 1.0  # color becomes relevant after shift

utility = []
aspect_change_count = []
alignment = []
previous_aspects = aspect_weights.copy()

for t in range(timesteps):
    context = context_vector[t]
    current_weights = aspect_weights.copy()

    # Update aspect weights
    delta = context - current_weights
    current_weights += 0.1 * delta
    current_weights = np.clip(current_weights, 0, 1)

    # Weighted differentiation
    WX = X * current_weights
    D = euclidean_distances(WX, WX)
    eps = 1.5

    # Build differentiation graph
    G = nx.Graph()
    for i in range(n_points):
        for j in range(i + 1, n_points):
            if D[i, j] < eps:
                G.add_edge(i, j)

    # SDI and utility
    sdi_components = list(nx.connected_components(G))
    sdi_size = len(sdi_components)
    target_sdi = 3 if t < shift_time else 5
    U = np.exp(-abs(sdi_size - target_sdi))
    utility.append(U)

    # Aspect change and alignment
    change_count = np.sum(np.abs(current_weights - previous_aspects) > 0.01)
    aspect_change_count.append(change_count)
    previous_aspects = current_weights.copy()
    align = np.dot(current_weights, context) / (np.linalg.norm(current_weights) * np.linalg.norm(context) + 1e-6)
    alignment.append(align)

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(utility, color='darkgreen')
axes[0].axvline(shift_time, color='red', linestyle='--')
axes[0].set_title("Utility over time")
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Utility")

axes[1].plot(aspect_change_count, color='darkblue')
axes[1].axvline(shift_time, color='red', linestyle='--')
axes[1].set_title("Aspect changes per step")
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Change count")

axes[2].plot(alignment, color='purple')
axes[2].axvline(shift_time, color='red', linestyle='--')
axes[2].set_title("Aspect-context alignment")
axes[2].set_xlabel("Time")
axes[2].set_ylabel("Cosine alignment")

plt.tight_layout()
plt.savefig("co_evolution_true_simulation.png", dpi=300)
plt.show()

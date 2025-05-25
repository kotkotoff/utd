
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

np.random.seed(42)
cluster1 = np.random.normal(loc=(1, 1), scale=0.1, size=(10, 2))
cluster2 = np.random.normal(loc=(3, 1), scale=0.1, size=(8, 2))
cluster3 = np.random.normal(loc=(2, 3), scale=0.1, size=(7, 2))
cluster4 = np.random.normal(loc=(4, 3), scale=0.1, size=(5, 2))
uncertain = np.random.uniform(0, 5, size=(5, 2))
X_init = np.vstack([cluster1, cluster2, cluster3, cluster4, uncertain])

eps = 0.3
n = len(X_init)
u_values = []
sigma_values = []

for i in range(n):
    count_perp = 0
    valid_dists = []
    for j in range(n):
        if i != j:
            dist = np.linalg.norm(X_init[i] - X_init[j])
            if dist < eps:
                valid_dists.append(dist)
            else:
                count_perp += 1
    u_i = count_perp / (n - 1)
    u_values.append(u_i)
    if valid_dists:
        min_d = np.min(valid_dists)
        max_d = np.max([np.linalg.norm(x - y) for x in X_init for y in X_init])
        sigma_i = min_d / max_d
    else:
        sigma_i = 1.0
    sigma_values.append(sigma_i)

angle = np.pi / 4
scale = 0.5
rotation_matrix = scale * np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle),  np.cos(angle)]])
X_trans = X_init @ rotation_matrix.T + np.array([0.5, 0.5])

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

for group in [cluster1, cluster2, cluster3, cluster4]:
    hull = ConvexHull(group)
    ax[0].fill(group[hull.vertices, 0], group[hull.vertices, 1], alpha=0.3)
ax[0].scatter(X_init[:, 0], X_init[:, 1], c='gray')
ax[0].scatter(uncertain[:, 0], uncertain[:, 1], marker='x', color='black', label='Boundary points')
ax[0].set_title("Initial SDI and Epistemic Boundary")
ax[0].legend()

for group in [cluster1, cluster2, cluster3, cluster4]:
    transformed = group @ rotation_matrix.T + np.array([0.5, 0.5])
    hull = ConvexHull(transformed)
    ax[1].fill(transformed[hull.vertices, 0], transformed[hull.vertices, 1], alpha=0.3)
ax[1].scatter(X_trans[:, 0], X_trans[:, 1], c='gray')
ax[1].scatter(X_trans[-5:, 0], X_trans[-5:, 1], marker='x', color='black', label='Boundary points')
for i in range(n):
    ax[1].arrow(X_init[i, 0], X_init[i, 1],
                X_trans[i, 0] - X_init[i, 0],
                X_trans[i, 1] - X_init[i, 1],
                color='blue', alpha=0.3, head_width=0.05, length_includes_head=True)
ax[1].set_title("Reconfigured SDI and Boundary")

plt.tight_layout()
plt.savefig("reconfiguration_true_simulation.png", dpi=300)
plt.show()

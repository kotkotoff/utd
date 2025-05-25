import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.stats import ttest_ind

# Parameters
N = 50
dim = 2
k_neighbors = 5
epsilon = 0.1
delta_tau = 1e-4
learning_rate = 0.2
np.random.seed(42)

# Generate synthetic data in 2D
X = np.random.rand(N, dim)

# Compute initial tau(x): mean distance to k nearest neighbors
dist_mat = distance_matrix(X, X)
np.fill_diagonal(dist_mat, np.inf)
sorted_dists = np.sort(dist_mat, axis=1)
tau_init = np.mean(sorted_dists[:, :k_neighbors], axis=1)

# Stabilization dynamics
tau_t = tau_init.copy()
tau_prev = tau_t.copy()
depth = np.zeros(N)

for t in range(1000):
    tau_next = tau_t.copy()
    for i in range(N):
        neighbors = np.where(dist_mat[i] < epsilon)[0]
        if len(neighbors) == 0:
            continue
        neighbor_mean = np.mean(tau_t[neighbors])
        tau_next[i] = tau_t[i] - learning_rate * (tau_t[i] - neighbor_mean)
    diff = np.abs(tau_next - tau_t)
    converged = (diff < delta_tau)
    depth[~converged] += 1
    if np.all(converged):
        break
    tau_t = tau_next

# Partition by threshold
delta = np.mean(tau_init)
low_group = tau_init <= delta
high_group = tau_init > delta

# Compute expected depths
depth_low = depth[low_group]
depth_high = depth[high_group]

mean_low = np.mean(depth_low)
mean_high = np.mean(depth_high)
t_stat, p_value = ttest_ind(depth_high, depth_low, equal_var=False)

print(f"Mean depth (low tau):  {mean_low:.2f}")
print(f"Mean depth (high tau): {mean_high:.2f}")
print(f"Welch's t-test: t = {t_stat:.2f}, p = {p_value:.4f}")

# Visualization
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(depth_low, alpha=0.7, label='Low τ(x)', bins=10)
plt.hist(depth_high, alpha=0.7, label='High τ(x)', bins=10)
plt.xlabel('Recursive Depth d(x)')
plt.ylabel('Count')
plt.legend()
plt.title('Depth Distribution by Initial τ(x)')

plt.subplot(1, 2, 2)
plt.scatter(tau_init, depth, c='black')
plt.axvline(delta, color='red', linestyle='--', label='Threshold δ')
plt.xlabel('Initial τ(x)')
plt.ylabel('Recursive Depth d(x)')
plt.title('Depth vs Initial τ(x)')
plt.legend()

plt.tight_layout()
plt.show()

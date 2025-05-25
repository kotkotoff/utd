# Phase-Coordinated Differentiation Simulation (Colab-ready)
import numpy as np
import matplotlib.pyplot as plt

# Parameters
np.random.seed(42)
grid_size = 40
T = 200
K = 5.0  # coupling strength
noise_level = 0.1

# Initialize phases and frequencies
theta = 2 * np.pi * np.random.rand(grid_size, grid_size)
omega = 2 * np.pi * np.random.rand(grid_size, grid_size)
phases = [theta.copy()]

# Metrics over time
chirality = []
coherence = []
tau_inv = []
entropy = []
phase_var = []

# Get Moore neighborhood (wrap-around)
def get_neighbors(i, j):
    neighbors = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if (di == 0 and dj == 0):
                continue
            ni, nj = (i + di) % grid_size, (j + dj) % grid_size
            neighbors.append((ni, nj))
    return neighbors

# Simulation loop
for t in range(T):
    new_theta = theta.copy()
    total_changes = 0
    theta_vec_sum = np.array([0.0, 0.0])
    local_cos = []

    for i in range(grid_size):
        for j in range(grid_size):
            neighbors = get_neighbors(i, j)
            sum_sin = sum(np.sin(theta[ni, nj] - theta[i, j]) for ni, nj in neighbors)
            eta = noise_level * np.random.randn()
            dtheta = omega[i, j] + (K / len(neighbors)) * sum_sin + eta
            new_theta[i, j] = (theta[i, j] + dtheta * 0.05) % (2 * np.pi)

            # Chirality vector
            theta_vec_sum += np.array([np.cos(theta[i, j]), np.sin(theta[i, j])])

            # Local coherence
            cos_vals = [np.cos(theta[i, j] - theta[ni, nj]) for ni, nj in neighbors]
            local_cos.append(np.mean(cos_vals))

            # Track significant changes (SDI analog)
            if abs(dtheta) > 0.1:
                total_changes += 1

    theta = new_theta
    phases.append(theta.copy())

    # Metrics
    tau_inv.append(total_changes / (grid_size * grid_size))
    phase_var.append(np.var(theta))
    coherence.append(np.mean(local_cos))
    chirality.append(np.linalg.norm(theta_vec_sum) / (grid_size * grid_size))

    # Approximate entropy via histogram
    hist, _ = np.histogram(theta, bins=50, range=(0, 2 * np.pi), density=True)
    hist = hist[hist > 0]
    ent = -np.sum(hist * np.log(hist))
    entropy.append(ent)

# Plot metrics
fig, axs = plt.subplots(3, 2, figsize=(12, 10))
axs[0, 0].plot(tau_inv)
axs[0, 0].set_title("Differentiation Rhythm τ⁻¹(t)")
axs[0, 1].plot(entropy)
axs[0, 1].set_title("Entropy H(t)")
axs[1, 0].plot(chirality)
axs[1, 0].set_title("Chirality Magnitude")
axs[1, 1].plot(coherence)
axs[1, 1].set_title("Local Coherence χ_local(t)")
axs[2, 1].axis("off")

plt.tight_layout()
plt.show()

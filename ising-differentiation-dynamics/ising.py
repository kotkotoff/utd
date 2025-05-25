import numpy as np
import matplotlib.pyplot as plt

# ==== Parameters ==== #
N = 30                      # Lattice size (N x N)
steps = 30                  # Number of update steps per trial
trials = 20                 # Number of independent trials per temperature
T_values = np.linspace(0.01, 3.0, 40)  # Range of noise levels ("temperature")

avg_D1 = []  # Average disagreement metric (D1)
avg_M = []   # Average absolute magnetization

# ==== Neighborhood function (4-nearest neighbors with periodic boundary) ==== #
def neighbors(i, j):
    return [
        ((i - 1) % N, j),
        ((i + 1) % N, j),
        (i, (j - 1) % N),
        (i, (j + 1) % N)
    ]

# ==== Main simulation loop ==== #
for T in T_values:
    D1_sum = 0
    M_sum = 0
    for _ in range(trials):
        # Initialize random spin configuration
        grid = np.random.choice([-1, 1], size=(N, N))

        # Evolution over time steps
        for _ in range(steps):
            new_grid = np.copy(grid)
            for i in range(N):
                for j in range(N):
                    s = grid[i, j]
                    neigh = [grid[x, y] for x, y in neighbors(i, j)]
                    d1 = sum(1 if s != n else 0 for n in neigh)

                    # Update rule based on disagreement
                    if d1 >= 3:
                        new_grid[i, j] *= -1
                    elif d1 == 2 and np.random.rand() < T:
                        new_grid[i, j] *= -1
            grid = np.copy(new_grid)

        # Compute average local disagreement D1
        total_d1 = 0
        for i in range(N):
            for j in range(N):
                s = grid[i, j]
                neigh = [grid[x, y] for x, y in neighbors(i, j)]
                total_d1 += sum(1 if s != n else 0 for n in neigh)
        D1_avg = total_d1 / (N * N)
        D1_sum += D1_avg

        # Compute absolute magnetization
        M_sum += np.abs(grid.sum()) / (N * N)

    avg_D1.append(D1_sum / trials)
    avg_M.append(M_sum / trials)

# ==== Plotting results ==== #
plt.figure(figsize=(8, 4))
plt.plot(T_values, avg_D1, label='Average $D_1$', color='royalblue')
plt.plot(T_values, avg_M, label='Magnetization $|M|$', color='darkorange')
plt.xlabel("Noise level $T$")
plt.ylabel("Value")
plt.title("UTD-Ising: $D_1$ and $|M|$ vs Noise")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("utd_ising_comparison.png", dpi=300)
plt.show()

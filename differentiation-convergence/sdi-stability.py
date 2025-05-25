
import numpy as np
import matplotlib.pyplot as plt

def count_stable_deltas(x, threshold_d1=0.01, threshold_d2=0.01):
    n = len(x)
    d1_matrix = np.abs(x[:, None] - x[None, :])
    d1_upper = d1_matrix[np.triu_indices(n, k=1)]
    d1_filtered = d1_upper[d1_upper > threshold_d1]

    stable_count = 0
    for i in range(len(d1_filtered)):
        for j in range(i + 1, len(d1_filtered)):
            if abs(d1_filtered[i] - d1_filtered[j]) < threshold_d2:
                stable_count += 1
    return stable_count

random_counts = []
for _ in range(100):
    x_rand = np.random.rand(20)
    count = count_stable_deltas(x_rand)
    random_counts.append(count)

x_grid = np.linspace(0, 1, 20)
x_sin = np.sin(np.linspace(0, 2 * np.pi, 20))
grid_count = count_stable_deltas(x_grid)
sin_count = count_stable_deltas(x_sin)

plt.figure(figsize=(8, 4))
plt.hist(random_counts, bins=20, color='gray', alpha=0.7, label='Random (100 runs)')
plt.axvline(grid_count, color='blue', linestyle='--', label=f'Grid: {grid_count}')
plt.axvline(sin_count, color='green', linestyle='--', label=f'Sine: {sin_count}')
plt.xlabel("Number of stable δ (|δ_i - δ_j| < 0.01)")
plt.ylabel("Frequency")
plt.title("Stable δ in Random vs Structured Data")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

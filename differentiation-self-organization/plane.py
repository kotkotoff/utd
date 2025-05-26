import numpy as np
import matplotlib.pyplot as plt

# Parameters
steps_list = [0, 25, 200, 1000]
N = 40
eta = 0.2
np.random.seed(42)

# Recursive differentiation: compute first and second spatial differences
def compute_D2(dx, dy):
    ddx = np.roll(dx, -1, axis=1) - dx
    ddy = np.roll(dy, -1, axis=0) - dy
    D2 = ddx + ddy
    D2 = (D2 - np.mean(D2)) / (np.std(D2) + 1e-6)  # normalize to prevent overflow
    return D2

# Run simulation
snapshots = []
for steps in steps_list:
    X_current = np.random.randn(N, N)
    for _ in range(steps):
        dx = np.roll(X_current, -1, axis=1) - X_current
        dy = np.roll(X_current, -1, axis=0) - X_current
        D2 = compute_D2(dx, dy)
        X_current += eta * D2
    snapshots.append(X_current.copy())

# Plot
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
titles = [f"Step {s}" for s in steps_list]
for ax, snap, title in zip(axs.flat, snapshots, titles):
    im = ax.imshow(snap, cmap='viridis', interpolation='nearest')
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, orientation='vertical')

plt.tight_layout()
plt.show()

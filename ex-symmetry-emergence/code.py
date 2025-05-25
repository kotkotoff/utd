import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from collections import Counter
import matplotlib.colors as mcolors

N = 50
steps = 100
tau = np.random.rand(N, N)
tau_history = []
entropy_history = []
dominant_frac_history = []
unique_patterns_history = []

def extract_patterns(grid):
    patterns = []
    for i in range(1, grid.shape[0]-1):
        for j in range(1, grid.shape[1]-1):
            patch = grid[i-1:i+2, j-1:j+2]
            mean = np.mean(patch)
            binary_patch = (patch > mean).astype(int)
            pattern = tuple(binary_patch.flatten())
            patterns.append(pattern)
    return patterns

def pattern_stats(grid):
    patterns = extract_patterns(grid)
    counter = Counter(patterns)
    total = sum(counter.values())
    probs = np.array(list(counter.values())) / total
    ent = entropy(probs, base=2)
    dominant_frac = np.max(probs)
    return ent, dominant_frac, len(counter)

for t in range(steps):
    tau_new = tau.copy()
    for i in range(1, N-1):
        for j in range(1, N-1):
            neighborhood = tau[i-1:i+2, j-1:j+2]
            tau_new[i, j] = 0.6 * tau[i, j] + 0.4 * np.mean(neighborhood) + np.random.normal(0, 0.01)
    tau = np.clip(tau_new, 0, 1)
    tau_history.append(tau.copy())

    ent, dom_frac, uniq = pattern_stats(tau)
    entropy_history.append(ent)
    dominant_frac_history.append(dom_frac)
    unique_patterns_history.append(uniq)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].plot(entropy_history, color='steelblue')
axes[0, 0].set_title("Pattern Entropy Over Time")
axes[0, 0].set_xlabel("Step")
axes[0, 0].set_ylabel("Entropy")

axes[0, 1].plot(dominant_frac_history, color='darkgreen')
axes[0, 1].set_title("Dominant Pattern Fraction")
axes[0, 1].set_xlabel("Step")
axes[0, 1].set_ylabel("Fraction")

axes[1, 0].plot(unique_patterns_history, color='indigo')
axes[1, 0].set_title("Number of Unique Patterns")
axes[1, 0].set_xlabel("Step")
axes[1, 0].set_ylabel("Count")

im = axes[1, 1].imshow(tau_history[-1], cmap='viridis', interpolation='none')
axes[1, 1].set_title("Final $\tau(x)$ Configuration")
plt.colorbar(im, ax=axes[1, 1], shrink=0.8, pad=0.02)

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy as shannon_entropy
from sklearn.preprocessing import normalize

# Parameters
np.random.seed(42)
n = 100          # number of nodes
T = 20           # recursion steps
gamma = 0.6      # memory weight
noise_level = 0.01

# Random binary initialization
X = (np.random.rand(n, 10) > 0.5).astype(float)

# Initial structural similarity matrix
def compute_D0(X):
    return (X @ X.T) / X.shape[1]

# Recursive update using structural composition and soft memory
def recursive_structural(D_prev):
    D_new = (D_prev @ D_prev.T) / D_prev.shape[1]
    D_new += noise_level * np.random.randn(*D_new.shape)
    D_new = np.clip(D_new, 0, 1)
    return gamma * D_prev + (1 - gamma) * D_new

# Tracking metrics
tau_inv = []
entropies = []
coherences = []
unique_rows = []

D = compute_D0(X)

for t in range(T):
    if t > 0:
        delta = np.abs(D - D_prev)
        tau_inv.append(np.sum(delta) / (n * n))
    else:
        tau_inv.append(1.0)

    # Entropy: frequency of rounded row patterns
    rounded = np.round(D, 2)
    row_strings = ["".join(map(str, row)) for row in rounded]
    _, counts = np.unique(row_strings, return_counts=True)
    entropies.append(shannon_entropy(counts))

    # Coherence: cosine-like similarity between rows
    norms = np.linalg.norm(D, axis=1, keepdims=True) + 1e-6
    D_norm = D / norms
    coherence = (D_norm @ D_norm.T).mean()
    coherences.append(coherence)

    # Structural complexity: number of unique rows
    unique_rows.append(len(set(row_strings)))

    D_prev = D.copy()
    D = recursive_structural(D)

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs[0, 0].plot(tau_inv)
axs[0, 0].set_title("Differentiation Rhythm τ⁻¹(t)")
axs[0, 1].plot(entropies)
axs[0, 1].set_title("Entropy H(t)")
axs[1, 0].plot(coherences)
axs[1, 0].set_title("Local Coherence χ_local(t)")
axs[1, 1].plot(unique_rows)
axs[1, 1].set_title("Structural Complexity")

plt.tight_layout()
plt.show()

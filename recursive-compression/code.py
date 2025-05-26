import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

# Set random seed for reproducibility
np.random.seed(42)

# Disable LaTeX rendering to avoid parsing errors
plt.rc('text', usetex=False)

# Simulation parameters
n_points = 20  # Number of words
aspects = ['cosine', 'euclidean']  # Aspects: cosine (alpha_1), euclidean (alpha_2)
recursion_levels = 3  # Number of recursion levels
epsilon_stab = 0.8  # Stability threshold
epsilon_non_triv = 0.005  # Non-triviality threshold

# Generate 20 random words (5 letters each)
words = [''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), 5)) for _ in range(n_points)]

# Compute letter frequencies for each word
X = np.zeros((n_points, 26))  # Matrix: 20 words x 26 letters
for i, word in enumerate(words):
    for char in word:
        X[i, ord(char) - ord('a')] += 1
    # Normalize frequencies
    X[i] = X[i] / np.sum(X[i]) if np.sum(X[i]) > 0 else np.ones(26) / 26

# Define aspect functions
def alpha_cosine(x_i, x_j):
    # Cosine distance between frequency vectors
    dot = np.dot(x_i, x_j)
    norm_i = np.linalg.norm(x_i)
    norm_j = np.linalg.norm(x_j)
    return 1 - dot / (norm_i * norm_j) if norm_i * norm_j > 0 else 0.0

def alpha_euclidean(x_i, x_j):
    # Euclidean distance between frequency vectors
    return np.linalg.norm(x_i - x_j)

# First-order differentiation: D_1(x_i, x_j, alpha)
def D1(x_i, x_j, aspect):
    if aspect == 'cosine':
        diff = alpha_cosine(x_i, x_j)
    else:  # euclidean
        diff = alpha_euclidean(x_i, x_j)
    return diff if diff > epsilon_non_triv else np.nan

# Higher-order differentiation: D_m(delta_i, delta_j, alpha)
def D_m(delta_i, delta_j, aspect):
    if np.isnan(delta_i) or np.isnan(delta_j):
        return np.nan
    diff = np.abs(delta_i - delta_j)
    return diff if diff > epsilon_non_triv else np.nan

# Stability check for distinctions
def is_stable(delta, aspect):
    if np.isnan(delta):
        return False
    delta_prime = D_m(delta, delta, aspect)
    return not np.isnan(delta_prime) and delta_prime < epsilon_stab

# Compute Shannon entropy of distinctions
def compute_entropy(deltas):
    valid_deltas = deltas[~np.isnan(deltas)]
    if len(valid_deltas) <= 1:
        print("Warning: Insufficient valid distinctions for entropy calculation")
        return 0.0
    hist, _ = np.histogram(valid_deltas, bins=50, density=True)
    hist = hist[hist > 0]
    return entropy(hist, base=np.e)  # Entropy in nats

# Run recursive differentiation
entropies = {'cosine': [], 'euclidean': []}
distinctions = {'cosine': [], 'euclidean': []}

# Initial distinctions (m=0) using alpha_1 (cosine)
deltas_0 = []
for i in range(n_points):
    for j in range(i + 1, n_points):
        delta = D1(X[i], X[j], 'cosine')
        if not np.isnan(delta):
            deltas_0.append(delta)
H_0 = compute_entropy(np.array(deltas_0))
entropies['cosine'].append(H_0)
distinctions['cosine'].append(deltas_0)

# Recursive differentiation with alpha_1 (cosine) for m=1,2,3
current_deltas = deltas_0
for m in range(1, recursion_levels + 1):
    new_deltas = []
    for i in range(len(current_deltas)):
        for j in range(i + 1, len(current_deltas)):
            delta = D_m(current_deltas[i], current_deltas[j], 'cosine')
            if is_stable(delta, 'cosine'):
                new_deltas.append(delta)
    current_deltas = new_deltas if new_deltas else [0.1]  # Fallback to avoid empty lists
    H_m = compute_entropy(np.array(current_deltas))
    entropies['cosine'].append(H_m)
    distinctions['cosine'].append(current_deltas)

# Apply D_3 with alpha_2 (euclidean)
deltas_euclidean = []
for i in range(n_points):
    for j in range(i + 1, n_points):
        delta = D1(X[i], X[j], 'euclidean')
        if not np.isnan(delta):
            deltas_euclidean.append(delta)
H_3_euclidean = compute_entropy(np.array(deltas_euclidean))
entropies['euclidean'].append(H_3_euclidean)
distinctions['euclidean'].append(deltas_euclidean)

# Print results
print("\nSimulation Results:")
print(f"Initial Entropy (H_0): {H_0:.2f} nats")
for m in range(recursion_levels + 1):
    print(f"Entropy at m={m} (alpha_1, cosine): {entropies['cosine'][m]:.2f} nats")
print(f"Entropy at m=3 (alpha_2, euclidean): {H_3_euclidean:.2f} nats")

# Save results for LaTeX
with open('recursive_complexity_results.txt', 'w') as f:
    f.write(f"Initial Entropy (H_0): {H_0:.2f} nats\n")
    f.write(f"Entropy at m=3 (alpha_1): {entropies['cosine'][3]:.2f} nats\n")
    f.write(f"Entropy at m=3 (alpha_2): {H_3_euclidean:.2f} nats\n")

# Visualization settings
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'grid.linestyle': '--',
    'grid.alpha': 0.7
})

# Plot 1: Heatmap of letter frequencies
plt.figure(figsize=(10, 6))
plt.imshow(X, cmap='viridis', aspect='auto')
plt.colorbar(label='Letter Frequency')
plt.title('Letter Frequencies in 20 Words')
plt.xlabel('Letter (a-z)')
plt.ylabel('Word')
plt.xticks(range(26), list('abcdefghijklmnopqrstuvwxyz'))
plt.show()

# Plot 2: Combined entropy and distinction histograms
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), tight_layout=True)

# Left subplot: Entropy over recursion levels
ax1.plot(range(recursion_levels + 1), entropies['cosine'], marker='o', linestyle='-', color='blue', label=r'$\alpha_1$ (Cosine)')
ax1.plot([3], [H_3_euclidean], marker='s', color='red', label=r'$\alpha_2$ (Euclidean, m=3)')
ax1.set_title('Entropy Compression and Restoration')
ax1.set_xlabel('Recursion Level $m$')
ax1.set_ylabel('Entropy (nats)')
ax1.legend()
ax1.grid(True)

# Right subplot: Distinction histograms at m=3
valid_cosine = np.array([d for d in distinctions['cosine'][3] if not np.isnan(d)])
valid_euclidean = np.array([d for d in distinctions['euclidean'][0] if not np.isnan(d)])
if len(valid_cosine) > 0:
    ax2.hist(valid_cosine, bins=15, alpha=0.5, color='blue', label=r'$\alpha_1$ (Ordered)', density=True)
else:
    print("Warning: No valid distinctions for alpha_1 at m=3")
if len(valid_euclidean) > 0:
    ax2.hist(valid_euclidean, bins=15, alpha=0.5, color='red', label=r'$\alpha_2$ (Disordered)', density=True)
else:
    print("Warning: No valid distinctions for alpha_2 at m=3")
ax2.set_title('Distinction Distributions at $m=3$')
ax2.set_xlabel('Distinction Value')
ax2.set_ylabel('Density')
ax2.legend()
ax2.grid(True)

plt.show()
import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 100                  # Number of nodes
d = 5                    # Dimensionality
epsilon = 0.15           # Differentiation threshold
delta = 0.6              # Anomaly detection threshold
tau_min = 0.1            # Minimum differentiation rhythm
max_depth = 10           # Max recursion depth

np.random.seed(42)

# Base signal: sinusoidal manifold
t = np.linspace(0, 2 * np.pi, L)
X = np.stack([np.sin(t + i) for i in range(d)], axis=1)

# Inject anomalies (structured spikes)
anomaly_idx = np.arange(20, 30)
X[anomaly_idx] += np.random.normal(2, 0.4, (len(anomaly_idx), d))

# Inject stable cosine pattern
pattern_idx = np.arange(0, 10)
cosine_pattern = np.stack([np.cos(t[:10] + shift) for shift in np.linspace(0, np.pi, d)], axis=1)  # shape (10, d)
X[pattern_idx, :] = cosine_pattern + np.random.normal(0, 0.01, size=(10, d))

# Add noisy cluster
noise_idx = np.arange(70, 80)
X[noise_idx] = np.random.normal(0, 1, (len(noise_idx), d))

# Tau vector
tau = np.random.uniform(tau_min, 0.4, L)
tau[anomaly_idx] += 0.15
tau[pattern_idx] = tau_min + np.random.normal(0, 0.01, len(pattern_idx))

# Build differentiation graph
def build_diff_graph(X, epsilon):
    L = X.shape[0]
    adj = np.zeros((L, L))
    for i in range(L):
        for j in range(i + 1, L):
            dist = np.linalg.norm(X[i] - X[j])
            if dist < epsilon:
                adj[i, j] = adj[j, i] = dist
    return adj

G = build_diff_graph(X, epsilon)

# Recursive differentiation step
def differentiate_level(X, G, tau):
    L = X.shape[0]
    new_X = np.zeros_like(X)
    for i in range(L):
        neighbors = np.where(G[i] > 0)[0]
        if len(neighbors) > 0:
            diffs = np.array([np.linalg.norm(X[i] - X[j]) for j in neighbors])
            weights = np.exp(-tau[i] * diffs)
            weights /= np.sum(weights) + 1e-10
            new_X[i] = np.sum([weights[j] * X[neighbors[j]] for j in range(len(neighbors))], axis=0)
        else:
            new_X[i] = X[i]
    return new_X

# Run recursive updates and track anomaly scores
X_current = X.copy()
anomaly_scores = []

for _ in range(max_depth):
    X_current = differentiate_level(X_current, G, tau)
    mu = np.mean(X_current, axis=0)
    alpha = np.linalg.norm(X_current - mu, axis=1)
    anomaly_scores.append(alpha)

# Final anomaly score
final_scores = anomaly_scores[-1]

# Plot anomaly scores
plt.figure(figsize=(10, 5))
plt.plot(final_scores, label='Anomaly Score', color='midnightblue', linewidth=2)
plt.axhline(delta, color='crimson', linestyle='--', linewidth=1.5, label='Threshold δ')
plt.scatter(anomaly_idx, final_scores[anomaly_idx], color='red', label='True Anomalies', zorder=10)
plt.xlabel('Node Index')
plt.ylabel('Deviation Score α(x)')
plt.title('Recursive Anomaly Detection (Theorem 8.15.1)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

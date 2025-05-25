import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Parameters
num_nodes = 50         # Number of elements
steps = 30             # Recursion steps
alpha = 0.2            # Neighbor influence
tau_min = 0.05         # Min rhythm
omega_crit = 1 / tau_min  # Frequency cutoff

# Initialize signals with high-frequency components
np.random.seed(42)
X = np.random.randn(num_nodes)
X[10:15] += 7  # Strong high-frequency perturbation
X[30:33] -= 7

# Recursive smoothing step
def recursive_step(X):
    new_X = np.copy(X)
    for i in range(len(X)):
        left = X[(i - 1) % len(X)]
        right = X[(i + 1) % len(X)]
        new_X[i] = (1 - alpha) * X[i] + alpha * (left + right) / 2
    return new_X

# Run recursion
X_initial = X.copy()
X_current = X.copy()
for step in range(steps):
    X_current = recursive_step(X_current)

# Compute spectra
freq = fftfreq(num_nodes, d=1.0)
fft_vals_start = fft(X_initial)
freq_power_start = np.abs(fft_vals_start)**2
if np.max(freq_power_start) > 0:
    freq_power_start /= np.max(freq_power_start)

fft_vals_end = fft(X_current)
freq_power_end = np.abs(fft_vals_end)**2
if np.max(freq_power_end) > 0:
    freq_power_end /= np.max(freq_power_end)

# Check theorem
high_freq_energy = np.sum(freq_power_end[freq > omega_crit]) / (np.sum(freq_power_end) + 1e-10)
print(f"Theorem 8.15.3: High-frequency suppression: {1 - high_freq_energy:.3f}")

# Plot spectrum
plt.figure(figsize=(8, 5))
plt.plot(freq, freq_power_start, 'b-', label='Initial', linewidth=2, alpha=0.8)
plt.plot(freq, freq_power_end, 'r--', label='Final', linewidth=2, alpha=0.8)
plt.axvline(omega_crit, color='k', linestyle='--', label=f'ω_crit = {omega_crit:.1f}')
plt.xlabel('Frequency')
plt.ylabel('Normalized Power')
plt.title('Frequency Spectrum (Theorem 8.15.3)')
plt.legend()
plt.xlim(0, 0.5)  # Focus on positive frequencies
plt.ylim(0, 1.5)
plt.grid(True)
plt.tight_layout()
plt.savefig('utd_harmonic_filtering_thinner.png')
plt.show()
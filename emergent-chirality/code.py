import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve


# Grid size and time steps
size = 20
steps = 50000

# Initialize random grid with scalar field R
R = np.random.rand(size, size)

# Define differentiation rhythm τ as local standard deviation
def local_tau(R):
    kernel = np.array([[0.05, 0.1, 0.05],
                       [0.1 , 0.4, 0.1 ],
                       [0.05, 0.1, 0.05]])
    return convolve(R, kernel)

# Compute gradient and chirality
def compute_gradient(R):
    gx, gy = np.gradient(R)
    return gx, gy

# Simulation history
tau_inv_list = []
chi_local_list = []
depth_list = []
chirality_magnitude_list = []

for t in range(steps):
    tau = local_tau(R)
    tau_inv = 1.0 / (np.mean(tau) + 1e-6)
    gx, gy = compute_gradient(R)
    grad_norm = np.sqrt(gx**2 + gy**2)
    chi_local = np.mean(grad_norm)
    R = R + 0.1 * (np.random.rand(size, size) - 0.5) * tau_inv  # inject noise modulated by τ⁻¹
    depth_list.append(np.mean(R))
    tau_inv_list.append(tau_inv)
    chi_local_list.append(chi_local)
    chirality_magnitude_list.append(np.linalg.norm([np.sum(gx), np.sum(gy)]))

# Export to dataframe
import pandas as pd
df = pd.DataFrame({
    'step': list(range(steps)),
    'tau_inv': tau_inv_list,
    'chi_local': chi_local_list,
    'depth': depth_list,
    'chirality': chirality_magnitude_list
})

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(df['step'], df['tau_inv'])
plt.title(r"$\tau^{-1}(t)$ (Inverse Differentiation Rhythm)")
plt.xlabel("Step")
plt.ylabel(r"$\tau^{-1}$")

plt.subplot(2, 2, 2)
plt.plot(df['step'], df['chi_local'])
plt.title(r"$\chi_{\mathrm{local}}(t)$ (Gradient Coherence)")
plt.xlabel("Step")
plt.ylabel("Mean Gradient Norm")

plt.subplot(2, 2, 3)
plt.plot(df['step'], df['depth'])
plt.title(r"Recursive Field Mean Value")
plt.xlabel("Step")
plt.ylabel("Mean Field Value")

plt.subplot(2, 2, 4)
plt.plot(df['step'], df['chirality'])
plt.title("Chirality Magnitude (Norm of Total Gradient)")
plt.xlabel("Step")
plt.ylabel("Chirality")

plt.tight_layout()
plt.show()


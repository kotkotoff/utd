
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, find_objects

# Parameters
N = 64
T = 100
threshold = 0.15

np.random.seed(42)
delta_t = np.random.randint(0, 2, (N, N))
history = [delta_t.copy()]
tau_map = np.zeros((N, N))
stability_counter = np.zeros((N, N))

def differentiate(d1, d2):
    return (d1 != d2).astype(int)

for t in range(1, T+1):
    delta_new = np.zeros_like(delta_t)
    for i in range(1, N-1):
        for j in range(1, N-1):
            neighborhood = delta_t[i-1:i+2, j-1:j+2]
            delta_new[i, j] = 1 if np.sum(neighborhood) > 4 else 0

    D1 = differentiate(delta_t, delta_new)
    stable = D1 < threshold
    stability_counter[stable] += 1
    tau_map[~stable] = np.where(tau_map[~stable] == 0, t, tau_map[~stable])
    delta_t = delta_new.copy()
    history.append(delta_t)

# --- 1. Plot SDI Soliton Candidates ---
def plot_sdi_candidates(tau_map, T):
    threshold = 0.9 * T
    sdi_mask = tau_map > threshold
    labeled, num_features = label(sdi_mask)
    regions = find_objects(labeled)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(tau_map, cmap='magma', interpolation='nearest')
    ax.set_title("SDI Soliton Candidates on $\\tau(x)$")
    plt.colorbar(im, ax=ax)

    for region in regions:
        if region is not None:
            x_slice, y_slice = region
            x0, x1 = x_slice.start, x_slice.stop
            y0, y1 = y_slice.start, y_slice.stop
            ax.add_patch(plt.Rectangle((y0, x0), y1 - y0, x1 - x0, fill=False, edgecolor='cyan', linewidth=1.5))

    plt.tight_layout()
    plt.show()

# --- 2. Plot SDI Region Stability ---
def plot_sdi_stability(history, x0, x1, y0, y1, T):
    bounding_box_areas = []
    total_intensities = []

    for t in range(T + 1):
        delta = history[t]
        region = delta[x0:x1, y0:y1]

        if np.any(region):
            coords = np.argwhere(region)
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)
            area = (x_max - x_min + 1) * (y_max - y_min + 1)
        else:
            area = 0

        bounding_box_areas.append(area)
        total_intensities.append(np.sum(region))

    fig, ax = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    ax[0].plot(range(T + 1), bounding_box_areas, label="Bounding Box Area")
    ax[0].set_ylabel("Area (pixels)")
    ax[0].set_title("Spatial Spread of SDI Region")

    ax[1].plot(range(T + 1), total_intensities, label="Total Intensity", color='darkgreen')
    ax[1].set_ylabel("Active Cells")
    ax[1].set_xlabel("Recursive Step $t$")

    for a in ax:
        a.grid(True)

    plt.tight_layout()
    plt.show()

# --- Run visualizations ---
plot_sdi_candidates(tau_map, T)

# Fallback to max τ region if no features detected
max_tau_index = np.unravel_index(np.argmax(tau_map), tau_map.shape)
i0, j0 = max_tau_index
r = 4
x0, x1 = max(0, i0 - r), min(N, i0 + r)
y0, y1 = max(0, j0 - r), min(N, j0 + r)
plot_sdi_stability(history, x0, x1, y0, y1, T)

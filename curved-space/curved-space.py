import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

# ======================
# Parameters
# ======================
N = 100     # Number of nodes
d = 5       # Dimensionality of internal differentiational state
steps = 10  # Number of evolution steps
np.random.seed(42)

# ======================
# Initialize internal differentiation states
# ======================
R = np.random.rand(N, d)

# Store surface evolution
depth_series = []

# ======================
# Recursive Differentiation Loop
# ======================
for t in range(steps):
    # Simulate change in internal structure
    R += 0.01 * np.random.randn(N, d)

    # Level 1: Pairwise differentiation
    D1 = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=2)

    # Level 2: Differentiation of differentiation
    D2 = np.linalg.norm(D1[:, None, :] - D1[None, :, :], axis=2)

    # Level 3: Differentiation of level-2 differences
    D3 = np.linalg.norm(D2[:, None, :] - D2[None, :, :], axis=2)

    # Embed into 2D space using MDS over D3
    coords = MDS(n_components=2, dissimilarity='precomputed', random_state=42).fit_transform(D3)
    X, Y = coords[:, 0], coords[:, 1]

    # Measure depth of D1 for each node (how differentiated it is from all others)
    Z = D1.std(axis=1)

    # Interpolate onto regular grid
    grid_x, grid_y = np.mgrid[X.min():X.max():200j, Y.min():Y.max():200j]
    grid_z = griddata((X, Y), Z, (grid_x, grid_y), method='cubic')

    # Store interpolated surface for animation
    depth_series.append(grid_z)

# ======================
# Plot Animation
# ======================
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

def update_frame(i):
    ax.clear()
    surf = ax.plot_surface(grid_x, grid_y, depth_series[i], cmap='plasma', edgecolor='none')
    ax.set_title(f"Recursive Surface (step {i})")
    ax.set_xlabel("MDS X")
    ax.set_ylabel("MDS Y")
    ax.set_zlabel("Differentiational Depth")
    return surf,

ani = animation.FuncAnimation(fig, update_frame, frames=len(depth_series), interval=500)

# ======================
# Save Animation
# ======================
video_path = "/mnt/data/differentiation_surface_evolution.mp4"
ani.save(video_path, writer='ffmpeg', fps=2)

# Output video file path
print("Saved to:", video_path)

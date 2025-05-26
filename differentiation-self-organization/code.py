import numpy as np
import matplotlib.pyplot as plt

# --- Step 1: Generate unstructured input ---
def generate_scene(noise=0.2, n_points=32):
    return np.random.randn(n_points, 2) * noise

# --- Step 2: Recursive differentiation without normalization ---
def compute_diffs(points, level=1):
    current = points.copy()
    for _ in range(level):
        if len(current) < 2:
            break
        pairs = np.random.choice(len(current), size=(len(current)*2, 2), replace=True)
        diffs = current[pairs[:, 0]] - current[pairs[:, 1]]
        current = diffs  # no normalization
    return current

# --- Step 3: Run for multiple levels ---
scene = generate_scene()
levels = [compute_diffs(scene, level=i) for i in range(4)]

# --- Step 4: Plot in 2x2 layout ---
titles = [
    "Level 0: No Recursion",
    "Level 1: First Differentiation",
    "Level 2: Second Differentiation",
    "Level 3: Third Differentiation"
]

fig, axs = plt.subplots(2, 2, figsize=(10, 8))
for i, ax in enumerate(axs.flatten()):
    ax.scatter(levels[i][:, 0], levels[i][:, 1], s=10, color='blue')
    ax.set_title(titles[i])
    ax.grid(True)

plt.tight_layout()
plt.show()

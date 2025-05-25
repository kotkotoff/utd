import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
x = np.random.rand(20)
n = len(x)
threshold = 0.01  
counts = []

d1 = np.abs(x[:, None] - x[None, :])
d1 = d1[np.triu_indices(n, k=1)]
d1 = d1[d1 > threshold]
counts.append(len(d1))

if len(d1) > 0:
    d2 = np.abs(d1[:, None] - d1[None, :])
    d2 = d2[np.triu_indices(len(d1), k=1)]
    d2 = d2[d2 > threshold]
    counts.append(len(d2))
else:
    counts.append(0)

if len(d2) > 0:
    d3 = np.abs(d2[:, None] - d2[None, :])
    d3 = d3[np.triu_indices(len(d2), k=1)]
    d3 = d3[d3 > threshold]
    counts.append(len(d3))
else:
    counts.append(0)

plt.figure(figsize=(6, 4))
plt.plot([1, 2, 3], counts, marker='o')
plt.xlabel('Level of Differentiation')
plt.ylabel('Number of Distinctions')
plt.title('Recursive Growth of Distinctions')
plt.grid(True)
plt.tight_layout()
plt.show()

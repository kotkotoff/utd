import numpy as np
from sklearn.metrics import pairwise_distances, f1_score
from sklearn.cluster import KMeans

np.random.seed(42)
n_samples = 100
n_features = 50

cluster1 = np.clip(np.random.normal(0.6, 0.1, size=(50, n_features)), 0, 1)
cluster2 = np.clip(np.random.normal(0.3, 0.1, size=(50, n_features)), 0, 1)
X = np.vstack([cluster1, cluster2])
y_true = np.array([0] * 50 + [1] * 50)

D = pairwise_distances(X, metric='euclidean') / np.sqrt(n_features)
mu = np.median(X, axis=0)
sigma = np.mean((X > mu) != 0, axis=1)

mask = sigma < 0.2
X_filtered = X[mask]
y_filtered = y_true[mask]

kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
labels_pred = kmeans.fit_predict(X_filtered)

# Ensure correct label mapping for F1
if f1_score(y_filtered, labels_pred) < f1_score(y_filtered, 1 - labels_pred):
    labels_pred = 1 - labels_pred

score = f1_score(y_filtered, labels_pred)
print(f"F1-score (gene clustering): {score:.2f}")

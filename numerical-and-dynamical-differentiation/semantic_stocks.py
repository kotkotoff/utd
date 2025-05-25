import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

np.random.seed(42)
n_stocks = 50
n_days = 252
n_clusters = 5

base_patterns = [np.random.normal(0, 0.01, size=n_days) for _ in range(n_clusters)]
stocks = []

for i in range(n_stocks):
    cluster_id = i % n_clusters
    noise = np.random.normal(0, 0.02, size=n_days)
    stock_series = base_patterns[cluster_id] + noise
    stocks.append(stock_series)

X = np.array(stocks)
window = 30
step = 30
time_points = list(range(window, n_days + 1, step))

print("Time window end → #clusters recovered")
for t in time_points:
    X_window = X[:, t-window:t]
    X_flat = X_window.reshape(n_stocks, -1)
    D = pairwise_distances(X_flat, metric="euclidean")
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(D)
    n_unique = len(np.unique(labels))
    print(f"Day {t:3d} → {n_unique} clusters")

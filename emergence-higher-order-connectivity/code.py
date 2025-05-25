import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# --- Step 1: Generate clustered and uniform datasets ---
np.random.seed(42)
cluster1 = np.random.normal(loc=(1, 1), scale=0.15, size=(10, 2))
cluster2 = np.random.normal(loc=(3, 1), scale=0.15, size=(10, 2))
cluster3 = np.random.normal(loc=(2, 3), scale=0.15, size=(10, 2))
data_clustered = np.vstack([cluster1, cluster2, cluster3])
data_uniform = np.random.uniform(0, 4, size=(30, 2))

# --- Step 2: Compute τ(x) ---
def compute_tau(data):
    dist = np.linalg.norm(data[:, None] - data[None, :], axis=-1)
    np.fill_diagonal(dist, np.inf)
    return np.min(dist, axis=1) / np.max(dist)

tau_clustered = compute_tau(data_clustered)
tau_uniform = compute_tau(data_uniform)

# --- Step 3: Build G^τ_ε ---
def build_diff_graph(data, tau_vals, eps=0.5, tau_thresh=0.2):
    G = nx.Graph()
    for i, pt in enumerate(data):
        G.add_node(i, pos=pt, tau=tau_vals[i], label=1)
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            d = np.linalg.norm(data[i] - data[j])
            if d < eps and tau_vals[i] < tau_thresh and tau_vals[j] < tau_thresh:
                G.add_edge(i, j, d1=1.0)
    return G

Gτ_clustered = build_diff_graph(data_clustered, tau_clustered)
Gτ_uniform = build_diff_graph(data_uniform, tau_uniform)

# --- Step 4: Extract components and build Δ_C ---
def build_second_order_graph(components, threshold=0.1):
    G2 = nx.Graph()
    for i, pts in enumerate(components):
        G2.add_node(i, pos=np.mean(pts, axis=0))
    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            d_sum = 0
            for x in components[i]:
                for y in components[j]:
                    d_sum += 1 if np.linalg.norm(x - y) < 0.5 else 0
            D2 = d_sum / (len(components[i]) * len(components[j]))
            if D2 > threshold:
                G2.add_edge(i, j, d2=D2)
    return G2

components_clustered = [list(c) for c in nx.connected_components(Gτ_clustered)]
points_clustered = [[data_clustered[i] for i in comp] for comp in components_clustered]
G2_clustered = build_second_order_graph(points_clustered)

components_uniform = [list(c) for c in nx.connected_components(Gτ_uniform)]
points_uniform = [[data_uniform[i] for i in comp] for comp in components_uniform]
G2_uniform = build_second_order_graph(points_uniform)

# --- Step 5: Plot all 4 graphs ---
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# (a) Clustered Gτ
pos_a = nx.spring_layout(Gτ_clustered, seed=42)
nx.draw_networkx_nodes(Gτ_clustered, pos=pos_a,
                       node_size=200, node_color='steelblue',
                       edgecolors='white', linewidths=0.5, alpha=0.9, ax=axs[0, 0])
nx.draw_networkx_edges(Gτ_clustered, pos=pos_a, width=2.0, alpha=0.3, ax=axs[0, 0])
axs[0, 0].set_title("(a) Clustered $G^\tau_\varepsilon$")
axs[0, 0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

# (b) Clustered Δ_C
pos_b = nx.get_node_attributes(G2_clustered, "pos")
nx.draw_networkx_nodes(G2_clustered, pos=pos_b, node_color="indianred", node_shape='*',
                       node_size=300, edgecolors='black', linewidths=1.0, ax=axs[0, 1])
nx.draw_networkx_edges(G2_clustered, pos=pos_b, edge_color="gray", width=2.0, alpha=0.6, ax=axs[0, 1])
axs[0, 1].set_title("(b) Clustered $\Delta_C$")
axs[0, 1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

# (c) Uniform Gτ
pos_c = nx.spring_layout(Gτ_uniform, seed=42)
nx.draw_networkx_nodes(Gτ_uniform, pos=pos_c,
                       node_size=200, node_color='slategray',
                       edgecolors='white', linewidths=0.5, alpha=0.9, ax=axs[1, 0])
nx.draw_networkx_edges(Gτ_uniform, pos=pos_c, width=2.0, alpha=0.3, ax=axs[1, 0])
axs[1, 0].set_title("(c) Uniform $G^\tau_\varepsilon$")
axs[1, 0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

# (d) Uniform Δ_C
pos_d = nx.get_node_attributes(G2_uniform, "pos")
nx.draw_networkx_nodes(G2_uniform, pos=pos_d, node_color="purple", node_shape='*',
                       node_size=300, edgecolors='black', linewidths=1.0, ax=axs[1, 1])
nx.draw_networkx_edges(G2_uniform, pos=pos_d, edge_color="gray", width=2.0, alpha=0.6, ax=axs[1, 1])
axs[1, 1].set_title("(d) Uniform $\Delta_C$")
axs[1, 1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

plt.tight_layout()
plt.savefig("higher_order_diff_plot.png", dpi=300)
plt.show()

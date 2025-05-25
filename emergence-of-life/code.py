import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# --- Parameters ---
np.random.seed(42)
N = 50              # number of nodes
T = 50              # time steps
C_crit = 12.0       # biological threshold
lambda_decay = 0.85 # decay factor for weakly connected nodes

# --- Create clustered graph ---
G = nx.powerlaw_cluster_graph(N, 3, 0.4)

# --- Initialize differentiation parameters ---
d = {n: np.random.uniform(0.8, 1.2) for n in G.nodes}
tau = {n: np.random.uniform(0.1, 0.3) for n in G.nodes}
sigma = {n: np.random.uniform(0.05, 0.15) for n in G.nodes}

# --- Biological complexity function ---
def C_bio():
    return sum(d[n] + tau[n] + sigma[n] for n in G.nodes)

# --- Local neighborhood within radius ---
def get_local_neighborhood(G, node, radius=2):
    return nx.ego_graph(G, node, radius=radius).nodes

# --- Differentiation dynamics ---
def differentiate():
    for n in G.nodes:
        neighborhood = get_local_neighborhood(G, n, radius=2)
        neighborhood = [m for m in neighborhood if m != n]
        if not neighborhood:
            continue

        local_C = np.mean([d[m] + tau[m] + sigma[m] for m in neighborhood])
        current_C = d[n] + tau[n] + sigma[n]

        d[n] += 0.1 * np.mean([d[m] for m in neighborhood])
        tau[n] += 0.05 * np.mean([abs(tau[n] - tau[m]) < 0.1 for m in neighborhood])
        sigma[n] += 0.01 * np.std([d[m] for m in neighborhood])

        if current_C > local_C + 0.1:
            d[n] *= 1.05
            tau[n] *= 1.05
            sigma[n] *= 1.03

        if G.degree[n] < 3:
            d[n] *= lambda_decay
            tau[n] *= lambda_decay
            sigma[n] *= lambda_decay

        d[n] = min(d[n], 10.0)
        tau[n] = min(tau[n], 1.0)
        sigma[n] = min(sigma[n], 1.0)

# --- Simulation ---
C_vals = []
for t in range(T):
    differentiate()
    C_vals.append(C_bio())

# --- SDI detection ---
def find_sdi_candidates():
    candidates = []
    for comp in nx.connected_components(G):
        if len(comp) >= 4:
            subgraph = G.subgraph(comp)
            total = sum(d[n] + tau[n] + sigma[n] for n in subgraph.nodes)
            avg_d = np.mean([d[n] for n in subgraph.nodes])
            if total > C_crit and all(subgraph.degree[n] > 2 for n in subgraph) and avg_d > 2.5:
                candidates.append(list(comp))
    return candidates

SDIs = find_sdi_candidates()

# --- Output ---
print(f"Final biological complexity C_bio = {C_vals[-1]:.2f}")
print(f"Number of SDI candidates: {len(SDIs)}")
if SDIs:
    print(f"First SDI found (size {len(SDIs[0])}): {SDIs[0]}")

# --- Plot ---
plt.plot(C_vals)
plt.axhline(C_crit, color='r', linestyle='--', label='C_crit')
plt.title("Biological Complexity Over Time")
plt.xlabel("Iteration")
plt.ylabel("C_bio(S)")
plt.legend()
plt.grid()
plt.show()

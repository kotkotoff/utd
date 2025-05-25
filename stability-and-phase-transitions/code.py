import networkx as nx
import matplotlib.pyplot as plt
from itertools import product

# === Step 1: Reduced domain and single aspect ===
X = [1, 2, 3]
aspect = '<'
delta_1 = []

for a, b in product(X, X):
    result = a < b
    delta_1.append((a, b, aspect, result))

# === Step 2: Build Δ₂ and graph ===
G = nx.DiGraph()
G.add_nodes_from(delta_1, layer=1)

delta_2 = []
stable_nodes = []

for d1, d2 in product(delta_1, delta_1):
    a1, b1, op1, r1 = d1
    a2, b2, op2, r2 = d2

    # Only same aspect (trivially true here)
    diff = (r1 == r2)
    is_stable = (d1 == d2) and diff

    d2_node = (d1, d2, diff)
    delta_2.append(d2_node)
    G.add_node(d2_node, layer=2)
    G.add_edge(d1, d2_node)
    G.add_edge(d2, d2_node)

    if is_stable:
        stable_nodes.append(d1)

# === Improved layout ===
pos = {}
x_gap = 2.0
y_gap = 1.5

delta1_nodes = [node for node in G.nodes if node in delta_1]
for i, node in enumerate(delta1_nodes):
    pos[node] = (i * x_gap, 0)

delta2_nodes = [node for node in G.nodes if node not in delta_1]
for node in delta2_nodes:
    src1, src2, _ = node
    x1, _ = pos[src1]
    x2, _ = pos[src2]
    pos[node] = ((x1 + x2) / 2, -y_gap)

# === Coloring and Labels ===
node_colors = []
for node in G.nodes:
    if node in stable_nodes:
        node_colors.append('red')
    elif node in delta_1:
        node_colors.append('lightblue')
    else:
        node_colors.append('lightgreen')

labels = {}
for node in stable_nodes:
    a, b, op, _ = node
    labels[node] = f"{a} {op} {b}"

# === Drawing ===
plt.figure(figsize=(10, 5))
nx.draw(G, pos, with_labels=False, node_color=node_colors,
        node_size=700, edge_color='gray', width=1.0, alpha=0.9)
nx.draw_networkx_labels(G, pos, labels=labels, font_size=9)

plt.title("Δ₁ → Δ₂ with Stable Distinctions (Red)", fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.savefig("delta1_delta2_graph_small.png", dpi=300)
plt.show()

# === Print stable distinctions ===
print("Stable distinctions in Δ₁:")
for node in stable_nodes:
    a, b, op, res = node
    print(f"D₁({a}, {b}, {op}) = {res}")

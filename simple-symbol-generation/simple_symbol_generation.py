"""
UTD experiment for emotion classification using IMDB dataset.
Implements differentiation graph G_{\tau\varepsilon} and reflexive differentiation
as described in Chapter 9, Section 9.1 of 'Universal Theory of Differentiation'.
Includes graph visualization.
"""

import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import pickle
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import logging
import os
from plot_utils import visualize_diff_graph

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Check NetworkX version
logging.info(f"NetworkX version: {nx.__version__}")

# Configuration
MODEL_NAME = 'all-MiniLM-L6-v2'
DATASET_NAME = 'imdb'
TRAIN_SIZE = 100
TEST_SIZE = 50
EPSILON = 0.5  # Threshold for D_1
DELTA = 5.0    # Threshold for tau
REFLEXIVE_THRESHOLD = 0.2  # Threshold for reflexive differentiation
SEED = 42
FIGURES_DIR = 'figures'

# Create figures directory
os.makedirs(FIGURES_DIR, exist_ok=True)

# Load model
model = SentenceTransformer(MODEL_NAME)

# Load data
def load_data(split, size):
    try:
        dataset = load_dataset(DATASET_NAME, split=split)
        data = dataset.shuffle(seed=SEED).select(range(size))
        return data['text'], data['label']
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        raise

train_sentences, train_labels = load_data('train', TRAIN_SIZE)
test_sentences, test_labels = load_data('test', TEST_SIZE)
logging.info(f"Loaded {len(train_sentences)} training and {len(test_sentences)} test samples")

# Get embeddings
def get_embeddings(sentences):
    try:
        return model.encode(sentences, show_progress_bar=True)
    except Exception as e:
        logging.error(f"Failed to generate embeddings: {e}")
        raise

train_embeddings = get_embeddings(train_sentences)
test_embeddings = get_embeddings(test_sentences)

# Compute tau (differentiation rhythm, Section 8.1)
def compute_tau(emb):
    return np.linalg.norm(emb)

# Build differentiation graph (Section 9.1)
def build_graph(sentences, embeddings, labels, epsilon, delta):
    G = nx.Graph()
    # Add nodes
    for i, (sent, emb, label) in enumerate(zip(sentences, embeddings, labels)):
        G.add_node(i, sent=sent, emb=emb.tolist(), tau=compute_tau(emb), label=label)
    
    # Compute D_1 (1 - cosine similarity, Definition 1.2.1)
    cos_sim = 1 - cdist(embeddings, embeddings, metric='cosine')
    d1_values = 1 - cos_sim
    np.fill_diagonal(d1_values, 0)
    
    # Add edges
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            if (d1_values[i, j] < epsilon and 
                G.nodes[i]['tau'] < delta and 
                G.nodes[j]['tau'] < delta):
                G.add_edge(i, j, d1=d1_values[i, j])
    
    return G, d1_values

# Visualize graph
def visualize_graph(G, name="Graph", filename="diff_graph.png"):
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=SEED)
    
    # Node colors based on labels (0: red, 1: blue)
    node_colors = ['red' if G.nodes[n]['label'] == 0 else 'blue' for n in G.nodes]
    
    # Node sizes based on tau (scaled for visibility)
    node_sizes = [1000 * G.nodes[n]['tau'] / max([G.nodes[i]['tau'] for i in G.nodes]) for n in G.nodes]
    
    # Edge widths based on D_1
    edge_widths = [10 * G[u][v]['d1'] for u, v in G.edges]
    
    # Draw graph
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.7)
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title(f"{name} (Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()})")
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Graph visualization saved as {os.path.join(FIGURES_DIR, filename)}")

train_G, train_d1_values = build_graph(train_sentences, train_embeddings, train_labels, EPSILON, DELTA)
test_G, test_d1_values = build_graph(test_sentences, test_embeddings, test_labels, EPSILON, DELTA)

# Diagnostics
def log_graph_stats(G, d1_values, name="Graph"):
    tau_values = [G.nodes[i]['tau'] for i in G.nodes]
    logging.info(f"{name} stats: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    logging.info(f"Tau range: min={min(tau_values):.2f}, max={max(tau_values):.2f}, mean={np.mean(tau_values):.2f}")
    logging.info(f"D_1 range: min={np.min(d1_values):.2f}, max={np.max(d1_values):.2f}, mean={np.mean(d1_values):.2f}")

log_graph_stats(train_G, train_d1_values, "Training Graph")
log_graph_stats(test_G, test_d1_values, "Test Graph")

# Visualize graphs
visualize_diff_graph(train_G, "Training Differentiation Graph", save_path=os.path.join(FIGURES_DIR, "train_diff_graph.png"))
visualize_diff_graph(test_G, "Test Differentiation Graph", save_path=os.path.join(FIGURES_DIR, "test_diff_graph.png"))

# Save graph
def save_graph(G, filename="diff_graph.gpickle"):
    try:
        nx.write_gpickle(G, os.path.join(FIGURES_DIR, filename))
        logging.info(f"Graph saved as {os.path.join(FIGURES_DIR, filename)}")
    except Exception as e:
        logging.error(f"Failed to save graph with nx.write_gpickle: {e}")
        try:
            with open(os.path.join(FIGURES_DIR, filename.replace('.gpickle', '.pkl')), 'wb') as f:
                pickle.dump(G, f)
            logging.info(f"Graph saved as {os.path.join(FIGURES_DIR, filename.replace('.gpickle', '.pkl'))}")
        except Exception as e:
            logging.error(f"Failed to save graph with pickle: {e}")

save_graph(train_G, "train_diff_graph.gpickle")
save_graph(test_G, "test_diff_graph.gpickle")

# Reflexive differentiation (D_{n+1}, Section 9.1)
def reflexive_diff(G, labels, threshold=REFLEXIVE_THRESHOLD):
    delta_n1 = []
    for node in G.nodes:
        neighbors = list(G.neighbors(node))
        d1 = np.mean([G[node][n]['d1'] for n in neighbors]) if neighbors else 0
        delta_prime = 1 if d1 < threshold and labels[node] == 1 else 0
        delta_n1.append((node, delta_prime))
    return delta_n1

# Project to codes
def project_to_codes(delta_n1):
    codes = [(node, dp) for node, dp in delta_n1]
    symbolization = sum(1 for _, dp in codes if dp in [0, 1]) / len(codes)
    return codes, symbolization

# Run experiment
train_delta_n1 = reflexive_diff(train_G, train_labels)
train_codes, train_symbolization = project_to_codes(train_delta_n1)
train_stable_count = sum(1 for _, dp in train_delta_n1 if dp == train_labels[_]) / len(train_delta_n1)

test_delta_n1 = reflexive_diff(test_G, test_labels)
test_codes, test_symbolization = project_to_codes(test_delta_n1)
test_stable_count = sum(1 for _, dp in test_delta_n1 if dp == test_labels[_]) / len(test_delta_n1)

# Evaluate accuracy
pred_codes = [c[1] for c in test_codes]
accuracy = accuracy_score(test_labels, pred_codes)

# Log results
logging.info(f"Training SDI stability: {train_stable_count:.2f}")
logging.info(f"Training SDI symbolization: {train_symbolization:.2f}")
logging.info(f"Test accuracy: {accuracy:.2f}")
logging.info(f"Test SDI stability: {test_stable_count:.2f}")
logging.info(f"Test SDI symbolization: {test_symbolization:.2f}")

# Save results
results = {
    'train_stability': train_stable_count,
    'train_symbolization': train_symbolization,
    'test_accuracy': accuracy,
    'test_stability': test_stable_count,
    'test_symbolization': test_symbolization
}
with open(os.path.join(FIGURES_DIR, 'experiment_results.pkl'), 'wb') as f:
    pickle.dump(results, f)
logging.info(f"Results saved as {os.path.join(FIGURES_DIR, 'experiment_results.pkl')}")
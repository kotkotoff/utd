import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS

texts = {
    "animal": [
        "The dog barked loudly in the night.",
        "Cats like to sleep on warm surfaces.",
        "Birds can fly across long distances.",
        "A lion roared in the savannah.",
        "Wolves hunt in packs during the night.",
        "The horse galloped across the field.",
        "Dolphins are highly intelligent animals.",
        "The snake slithered silently through the grass.",
        "Bears hibernate during the winter.",
        "The frog leaped into the pond."
    ],
    "tech": [
        "Smartphones are evolving every year.",
        "Artificial intelligence is transforming industries.",
        "The new processor increases computing speed.",
        "Technology startups are growing rapidly.",
        "Quantum computing may redefine cryptography.",
        "The robot vacuum cleaned the house.",
        "Blockchain ensures data integrity.",
        "3D printing allows custom manufacturing.",
        "Software updates improve security.",
        "Cloud computing stores data remotely."
    ],
    "emotion": [
        "She felt a wave of joy when she saw him.",
        "He was overcome by sadness.",
        "Anger surged through his veins.",
        "The movie left her in tears.",
        "He smiled with gratitude.",
        "Anxiety crept in before the exam.",
        "Excitement filled the room before the concert.",
        "They embraced with love.",
        "She trembled with fear.",
        "Peace settled over him in the forest."
    ]
}

all_texts = []
labels = []
for label, group in texts.items():
    all_texts.extend(group)
    labels.extend([label] * len(group))

vectorizer = TfidfVectorizer(binary=True)
X = vectorizer.fit_transform(all_texts).toarray()

D = pairwise_distances(X, metric="hamming")
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
X_mds = mds.fit_transform(D)

fig, ax = plt.subplots(figsize=(8, 6))
colors = {'animal': 'steelblue', 'tech': 'green', 'emotion': 'indianred'}

for cat in colors:
    idx = [i for i, l in enumerate(labels) if l == cat]
    ax.scatter(X_mds[idx, 0], X_mds[idx, 1],
               c=colors[cat], label=cat, s=60, marker='x')

ax.set_title(r"Semantic Differentiation via $D_1 + \tau$", fontsize=13)
ax.set_xlabel("MDS Dimension 1")
ax.set_ylabel("MDS Dimension 2")
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(title="Theme", loc="upper left")
fig.tight_layout()

fig.savefig("semantic_differentiation_plot_styled.png", dpi=300)
print("Saved to semantic_differentiation_plot_styled.png")

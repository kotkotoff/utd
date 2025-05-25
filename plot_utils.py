import matplotlib.pyplot as plt
import networkx as nx
import os

def visualize_diff_graph(G, title="Differentiation Graph", save_path=None, seed=42):
    """
    Visualizes a differentiation graph with:
    - Node color: class label (0 / 1 → slategray / steelblue)
    - Node size: scaled τ(x)
    - Edge width: scaled D₁ distance
    """

    pos = nx.spring_layout(G, seed=seed)
    
    # Collect visual properties
    tau_vals = [G.nodes[n]["tau"] for n in G.nodes]
    tau_max = max(tau_vals)
    node_sizes = [300 + 800 * (tau / tau_max) for tau in tau_vals]

    labels = [G.nodes[n]["label"] for n in G.nodes]
    node_colors = ["slategray" if l == 0 else "steelblue" for l in labels]

    edge_widths = [2.0 * G[u][v]["d1"] for u, v in G.edges]

    plt.figure(figsize=(9, 7))
    nx.draw_networkx_nodes(G, pos,
                           node_size=node_sizes,
                           node_color=node_colors,
                           edgecolors='white',
                           linewidths=0.5,
                           alpha=0.9)
    nx.draw_networkx_edges(G, pos,
                           width=edge_widths,
                           alpha=0.3)
    
    nx.draw_networkx_labels(G, pos,
                            font_size=9,
                            font_color="black")
    
    plt.title(f"{title} — Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}", fontsize=13)
    plt.axis("off")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Graph saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_comparison_panels(
    acc_utd, acc_no_utd,
    tau_utd, tau_no_utd,
    stability_utd,
    title_prefix="",
    save_path=None
):
    """
    Draws 3-panel comparison plot: Accuracy, Tau(x), and Stability over epochs
    for UTD and non-UTD models.

    Args:
        acc_utd: list of accuracy values with UTD
        acc_no_utd: list of accuracy values without UTD
        tau_utd: list of tau values with UTD
        tau_no_utd: list of tau values without UTD
        stability_utd: list of stability values with UTD
        title_prefix: optional string to prefix each subplot title
        save_path: if specified, saves the plot to the given file
    """
    epochs = list(range(1, len(acc_utd) + 1))

    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11
    })

    fig, axs = plt.subplots(3, 1, figsize=(9, 6), sharex=True)

    # Accuracy plot
    axs[0].plot(epochs, acc_utd, label='UTD', color='steelblue', linewidth=1.8)
    axs[0].plot(epochs, acc_no_utd, label='No UTD', color='slategray', linewidth=1.8, linestyle='--')
    axs[0].set_ylabel("Accuracy")
    axs[0].set_title(f"{title_prefix}Accuracy over Epochs", pad=8)
    axs[0].legend(loc="lower right")
    axs[0].grid(True, linestyle="--", linewidth=0.5, alpha=0.3)

    # Tau(x) plot
    axs[1].plot(epochs, tau_utd, label='UTD', color='darkorange', linewidth=1.8)
    axs[1].plot(epochs, tau_no_utd, label='No UTD', color='dimgray', linewidth=1.8, linestyle='--')
    axs[1].set_ylabel("τ(x)")
    axs[1].set_title(f"{title_prefix}Tau over Epochs", pad=8)
    axs[1].legend(loc="upper right")
    axs[1].grid(True, linestyle="--", linewidth=0.5, alpha=0.3)

    # Stability plot
    axs[2].plot(epochs, stability_utd, label='UTD', color='indianred', linewidth=1.8)
    axs[2].set_ylabel("Stability")
    axs[2].set_xlabel("Epoch")
    axs[2].set_title(f"{title_prefix}Stability over Epochs", pad=8)
    axs[2].legend(loc="upper right")
    axs[2].grid(True, linestyle="--", linewidth=0.5, alpha=0.3)

    # Optional: mark epoch transitions
    for ax in axs:
        for x in [49, 99, 149]:
            ax.axvline(x=x, color='gray', linestyle=':', linewidth=1, alpha=0.4)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    else:
        plt.show()

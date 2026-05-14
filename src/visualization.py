"""Causal graph plotting and LEWIS score bar charts."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx


def plot_causal_graph(adj_matrix, node_names, title="Causal Graph", save_path=None):
    """
    Plot a causal DAG from an adjacency matrix.
    adj_matrix[i][j] = 1 means i -> j.
    """
    G = nx.DiGraph()
    G.add_nodes_from(node_names)

    for i in range(len(node_names)):
        for j in range(len(node_names)):
            if adj_matrix[i][j] == 1:
                G.add_edge(node_names[i], node_names[j])

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42, k=2)
    nx.draw(G, pos, ax=ax, with_labels=True, node_color="lightblue",
            node_size=2000, font_size=12, font_weight="bold",
            arrows=True, arrowsize=20, edge_color="gray",
            connectionstyle="arc3,rad=0.1")
    ax.set_title(title, fontsize=14)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_nesuf_comparison(scores_causal, scores_no_graph, feature_names,
                          title="Nesuf: Causal Graph vs No Graph", save_path=None):
    """
    Plot Nesuf scores comparing causal graph vs no graph (Figure 6 style).
    """
    x = np.arange(len(feature_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.barh(x + width / 2, scores_causal, width, label="Causal graph", color="forestgreen")
    bars2 = ax.barh(x - width / 2, scores_no_graph, width, label="No graph", color="lightgray")

    ax.set_xlabel("Importance", fontsize=12)
    ax.set_yticks(x)
    ax.set_yticklabels(feature_names, fontsize=11)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1.05)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_reversal_probabilities(nec_scores, suf_scores, feature_names,
                                title="Reversal Probability Scores", save_path=None):
    """
    Plot Nec and Suf scores side by side (Figure 7 style).
    """
    x = np.arange(len(feature_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(x + width / 2, nec_scores, width, label="Nec", color="steelblue")
    ax.barh(x - width / 2, suf_scores, width, label="Suf", color="darkorange")

    ax.set_xlabel("Probability", fontsize=12)
    ax.set_yticks(x)
    ax.set_yticklabels(feature_names, fontsize=11)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1.05)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_lewis_vs_shap(lewis_scores, shap_scores, feature_names,
                       title="LEWIS (Causal) vs SHAP (Correlational)", save_path=None):
    """
    Side-by-side horizontal bar chart comparing LEWIS Nesuf and SHAP importance rankings.
    Both are normalized to [0, 1] for fair comparison.
    """
    x = np.arange(len(feature_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(x + width / 2, lewis_scores, width, label="LEWIS (causal)", color="forestgreen")
    ax.barh(x - width / 2, shap_scores, width, label="SHAP (correlational)", color="steelblue")

    ax.set_xlabel("Normalized Importance", fontsize=12)
    ax.set_yticks(x)
    ax.set_yticklabels(feature_names, fontsize=11)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1.05)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_experiment_results(results_df, metric="MAE_mean", title=None, save_path=None):
    """
    Plot experiment results as grouped bar chart.
    results_df should have columns: method, prior, MAE_mean, SPR_mean.
    """
    methods = results_df["method"].unique()
    priors = results_df["prior"].unique()

    x = np.arange(len(methods))
    width = 0.25
    offsets = np.linspace(-width, width, len(priors))

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, p in enumerate(priors):
        subset = results_df[results_df["prior"] == p]
        values = [subset[subset["method"] == m][metric].values[0]
                  if len(subset[subset["method"] == m]) > 0 else 0
                  for m in methods]
        ax.bar(x + offsets[i], values, width, label=f"Prior ({p})")

    ax.set_ylabel(metric, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10, rotation=45, ha="right")
    ax.set_title(title or f"Experiment Results ({metric})", fontsize=14)
    ax.legend(fontsize=10)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig

"""
Module: Causal Discovery
Purpose: DirectLiNGAM (primary) + PC algorithm (cross-check), bootstrap stability analysis.
Inputs:  data/processed/train.csv
Outputs: models/adj_matrix_directlingam.npy, results/plots/causal_graph_directlingam.png,
         results/tables/causal_method_agreement.csv, results/tables/dag_stability_scores.csv,
         results/plots/dag_stability_heatmap.png
Reference: Takahashi et al. (2024) arXiv:2402.02678, Section III
"""

import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

logger = logging.getLogger(__name__)

FEATURE_COLS = ["volatility", "momentum", "volume_avg", "rsi_14", "max_drawdown", "vix", "eur_usd"]
TARGET_COL = "label"


def run_directlingam_with_prior_b(X_train_with_target, feature_cols, target_col):
    """
    Run DirectLiNGAM with prior (b): target is sink variable (no outgoing edges).

    Args:
        X_train_with_target: DataFrame with features + target column.
        feature_cols: List of feature column names.
        target_col: Name of target column.

    Returns:
        Adjacency matrix (numpy array), feature names list including target.
    """
    import lingam

    all_cols = feature_cols + [target_col]
    data = X_train_with_target[all_cols].values.astype(float)
    n_features = len(feature_cols)
    target_idx = n_features  # target is last column

    # Prior (b): target is sink — forbid edges FROM target TO any feature
    prior_knowledge = -1 * np.ones((len(all_cols), len(all_cols)), dtype=int)
    np.fill_diagonal(prior_knowledge, 0)
    prior_knowledge[target_idx, :] = 1   # forbid all outgoing from target
    prior_knowledge[target_idx, target_idx] = 0  # no self-loop

    model = lingam.DirectLiNGAM(prior_knowledge=prior_knowledge)
    model.fit(data)

    adj_matrix = (np.abs(model.adjacency_matrix_) > 0.01).astype(int).T
    logger.info(f"DirectLiNGAM(b): {adj_matrix.sum()} edges discovered")

    return adj_matrix, all_cols


def run_pc_crosscheck(X_train_with_target, feature_cols, target_col, alpha=0.05):
    """
    Run PC algorithm independently for cross-validation of DirectLiNGAM results.

    Args:
        X_train_with_target: DataFrame with features + target.
        feature_cols: Feature column names.
        target_col: Target column name.
        alpha: Significance level for independence tests.

    Returns:
        PC adjacency matrix (numpy array).
    """
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
    from causallearn.graph.GraphNode import GraphNode

    all_cols = feature_cols + [target_col]
    data = X_train_with_target[all_cols].values.astype(float)
    n_vars = len(all_cols)
    target_idx = len(feature_cols)

    # Prior (b) for PC: target is sink
    nodes = [GraphNode(f"X{i}") for i in range(n_vars)]
    bk = BackgroundKnowledge()
    for i in range(n_vars):
        if i != target_idx:
            bk.add_forbidden_by_node(nodes[target_idx], nodes[i])

    cg = pc(data, alpha=alpha, indep_test="fisherz", background_knowledge=bk)

    adj_matrix = np.zeros((n_vars, n_vars), dtype=int)
    graph = cg.G.graph
    for i in range(n_vars):
        for j in range(n_vars):
            if graph[i, j] == -1 and graph[j, i] == 1:
                adj_matrix[i, j] = 1

    logger.info(f"PC algorithm: {adj_matrix.sum()} edges discovered")
    return adj_matrix


def compare_methods(adj_lingam, adj_pc, col_names, results_dir="results/tables"):
    """
    Compare DirectLiNGAM and PC results: count agreed/disagreed edges.

    Args:
        adj_lingam: DirectLiNGAM adjacency matrix.
        adj_pc: PC adjacency matrix.
        col_names: Column names.
        results_dir: Directory to save agreement analysis.

    Returns:
        dict with agreement statistics.
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    n = len(col_names)
    agreed = 0
    disagreed = 0
    lingam_only = 0
    pc_only = 0
    details = []

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            l_edge = adj_lingam[i, j]
            p_edge = adj_pc[i, j]

            if l_edge and p_edge:
                agreed += 1
                details.append({"from": col_names[i], "to": col_names[j],
                                "directlingam": 1, "pc": 1, "agreement": "both"})
            elif l_edge and not p_edge:
                lingam_only += 1
                details.append({"from": col_names[i], "to": col_names[j],
                                "directlingam": 1, "pc": 0, "agreement": "directlingam_only"})
            elif not l_edge and p_edge:
                pc_only += 1
                details.append({"from": col_names[i], "to": col_names[j],
                                "directlingam": 0, "pc": 1, "agreement": "pc_only"})

    total_edges = agreed + lingam_only + pc_only
    agreement_rate = agreed / total_edges if total_edges > 0 else 0.0

    pd.DataFrame(details).to_csv(
        os.path.join(results_dir, "causal_method_agreement.csv"), index=False
    )

    logger.info(f"Edge agreement: {agreed} agreed, {lingam_only} DirectLiNGAM-only, "
                f"{pc_only} PC-only (rate: {agreement_rate:.2%})")

    return {"agreed": agreed, "lingam_only": lingam_only, "pc_only": pc_only,
            "agreement_rate": agreement_rate}


def bootstrap_stability(X_train_with_target, feature_cols, target_col,
                        n_bootstrap=30, results_dir="results"):
    """
    Bootstrap stability analysis: re-run DirectLiNGAM on 30 bootstrap samples.
    Flag edges with stability < 0.5 as UNSTABLE.

    Args:
        X_train_with_target: Training data with target.
        feature_cols: Feature column names.
        target_col: Target column name.
        n_bootstrap: Number of bootstrap samples (30 for speed, 100 for full).
        results_dir: Base results directory.

    Returns:
        Stability matrix (n_vars x n_vars), stability report DataFrame.
    """
    import lingam

    all_cols = feature_cols + [target_col]
    data = X_train_with_target[all_cols].values.astype(float)
    n_vars = len(all_cols)
    target_idx = len(feature_cols)
    n_samples = len(data)

    # Prior knowledge for all bootstrap runs
    prior_knowledge = -1 * np.ones((n_vars, n_vars), dtype=int)
    np.fill_diagonal(prior_knowledge, 0)
    prior_knowledge[target_idx, :] = 1
    prior_knowledge[target_idx, target_idx] = 0

    edge_counts = np.zeros((n_vars, n_vars))

    for b in range(n_bootstrap):
        idx = np.random.choice(n_samples, size=int(0.8 * n_samples), replace=True)
        sample = data[idx]

        try:
            model = lingam.DirectLiNGAM(prior_knowledge=prior_knowledge)
            model.fit(sample)
            adj = (np.abs(model.adjacency_matrix_) > 0.01).astype(int).T
            edge_counts += adj
        except Exception:
            continue

    stability = edge_counts / n_bootstrap

    # Save stability scores
    Path(os.path.join(results_dir, "tables")).mkdir(parents=True, exist_ok=True)
    stability_df = pd.DataFrame(stability, index=all_cols, columns=all_cols)
    stability_df.to_csv(os.path.join(results_dir, "tables", "dag_stability_scores.csv"))

    # Flag unstable edges
    n_stable = (stability >= 0.7).sum().sum()
    n_unstable = ((stability > 0) & (stability < 0.5)).sum().sum()
    logger.info(f"Bootstrap stability ({n_bootstrap} runs): "
                f"{n_stable} stable edges (>=0.7), {n_unstable} unstable (<0.5)")

    # Stability heatmap
    Path(os.path.join(results_dir, "plots")).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(stability_df, annot=True, fmt=".2f", cmap="YlOrRd",
                vmin=0, vmax=1, linewidths=0.5)
    plt.title(f"DAG Edge Stability ({n_bootstrap} bootstrap runs)")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "plots", "dag_stability_heatmap.png"), dpi=150)
    plt.close()

    return stability, stability_df


def plot_causal_graph(adj_matrix, col_names, results_dir="results/plots", filename="causal_graph_directlingam.png"):
    """
    Visualize causal graph (DAG) from adjacency matrix.
    Color coding: green=stock features, blue=macro features, red=target.

    Args:
        adj_matrix: Adjacency matrix.
        col_names: Column names.
        results_dir: Directory to save plot.
        filename: Output filename.
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    macro_features = {"vix", "eur_usd"}
    target_features = {"label"}

    G = nx.DiGraph()
    for i, name in enumerate(col_names):
        G.add_node(name)

    # Only show edges with weight > threshold
    for i in range(len(col_names)):
        for j in range(len(col_names)):
            if adj_matrix[i, j] and i != j:
                G.add_edge(col_names[i], col_names[j])

    if G.number_of_edges() == 0:
        logger.warning("No edges in causal graph to plot")
        return

    # Node colors
    color_map = []
    for node in G.nodes():
        if node in target_features:
            color_map.append("#ff6b6b")   # red for target
        elif node in macro_features:
            color_map.append("#4ecdc4")   # teal for macro
        else:
            color_map.append("#95e1d3")   # light green for stock

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=2, seed=42)
    nx.draw(G, pos, with_labels=True, node_color=color_map, node_size=2000,
            font_size=10, font_weight="bold", arrows=True, arrowsize=20,
            edge_color="gray", width=2, alpha=0.9)
    plt.title("Discovered Causal Graph (DirectLiNGAM with Prior b)")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, filename), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Causal graph saved: {filename} ({G.number_of_edges()} edges)")


def run(train_df, feature_cols=None, models_dir="models", results_dir="results"):
    """
    Execute Step 6: Full causal discovery pipeline.

    Args:
        train_df: Training DataFrame with features + label.
        feature_cols: Feature column names.
        models_dir: Directory to save adjacency matrix.
        results_dir: Base results directory.

    Returns:
        dict with adj_matrix, col_names, pc_adj, agreement, stability.
    """
    logger.info("=" * 60)
    logger.info("STEP 6: CAUSAL DISCOVERY")
    logger.info("=" * 60)

    Path(models_dir).mkdir(parents=True, exist_ok=True)

    if feature_cols is None:
        feature_cols = [c for c in FEATURE_COLS if c in train_df.columns]

    target_col = TARGET_COL if TARGET_COL in train_df.columns else "label"

    # 6.1 DirectLiNGAM (primary method)
    logger.info("Running DirectLiNGAM with prior (b)...")
    adj_matrix, col_names = run_directlingam_with_prior_b(train_df, feature_cols, target_col)
    np.save(os.path.join(models_dir, "adj_matrix_directlingam.npy"), adj_matrix)

    # 6.2 PC algorithm (cross-check)
    logger.info("Running PC algorithm for cross-check...")
    pc_adj = run_pc_crosscheck(train_df, feature_cols, target_col)

    # 6.3 Compare methods
    agreement = compare_methods(adj_matrix, pc_adj, col_names,
                                os.path.join(results_dir, "tables"))

    # 6.4 Bootstrap stability (30 runs for speed)
    logger.info("Running bootstrap stability analysis (30 runs)...")
    stability, stability_df = bootstrap_stability(
        train_df, feature_cols, target_col, n_bootstrap=30, results_dir=results_dir
    )

    # 6.5 Plot causal graph
    plot_causal_graph(adj_matrix, col_names, os.path.join(results_dir, "plots"))

    return {
        "adj_matrix": adj_matrix,
        "col_names": col_names,
        "pc_adj": pc_adj,
        "agreement": agreement,
        "stability": stability,
    }

"""
Module: 06_causal_discovery
Purpose: Run DirectLiNGAM causal discovery with PC cross-check and stability analysis.
Inputs:  data/processed/train.csv, models/dataset_metadata.json
Outputs: models/adj_matrix_directlingam.npy, results/tables/causal_method_agreement.csv,
         results/tables/dag_stability_scores.csv, results/tables/regime_stability_report.csv,
         results/plots/causal_graph_directlingam.png, results/plots/dag_stability_heatmap.png
Reference: Takahashi et al. (2024) arXiv:2402.02678
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
from lingam import DirectLiNGAM
from sklearn.preprocessing import LabelEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 6: Causal discovery")
    parser.add_argument("--bootstrap-runs", type=int, default=100)
    parser.add_argument("--bootstrap-frac", type=float, default=0.8)
    parser.add_argument("--edge-threshold", type=float, default=0.1)
    return parser.parse_args()


def build_prior_knowledge(n_features: int, target_idx: int) -> np.ndarray:
    prior = np.full((n_features + 1, n_features + 1), -1, dtype=int)
    # Target (investment decision) cannot cause historical features
    prior[target_idx, :] = 0
    np.fill_diagonal(prior, 0)
    return prior


def binary_adjacency(adj: np.ndarray, threshold: float) -> np.ndarray:
    b = (np.abs(adj) > threshold).astype(int)
    np.fill_diagonal(b, 0)
    return b


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten()
    b_flat = b.flatten()
    inter = np.logical_and(a_flat == 1, b_flat == 1).sum()
    union = np.logical_or(a_flat == 1, b_flat == 1).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def plot_graph(adj: np.ndarray, nodes: list[str], macro_features: set[str], out_path: Path, threshold: float) -> None:
    g = nx.DiGraph()
    for node in nodes:
        g.add_node(node)

    for i, src in enumerate(nodes):
        for j, dst in enumerate(nodes):
            w = adj[i, j]
            if i != j and abs(w) > threshold:
                g.add_edge(src, dst, weight=float(w))

    colors = []
    for node in nodes:
        if node == "investment_decision":
            colors.append("#d62728")
        elif node in macro_features:
            colors.append("#1f77b4")
        else:
            colors.append("#2ca02c")

    pos = nx.spring_layout(g, seed=42)
    edge_width = [1.0 + 3.0 * abs(g[u][v]["weight"]) for u, v in g.edges]

    plt.figure(figsize=(14, 10))
    nx.draw_networkx_nodes(g, pos, node_color=colors, node_size=1400, alpha=0.95)
    nx.draw_networkx_labels(g, pos, font_size=9)
    nx.draw_networkx_edges(g, pos, arrows=True, width=edge_width, alpha=0.7, edge_color="#444444")
    plt.title("DirectLiNGAM Causal Graph")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()

    root = Path(__file__).resolve().parents[1]
    processed_dir = root / "data" / "processed"
    models_dir = root / "models"
    tables_dir = root / "results" / "tables"
    plots_dir = root / "results" / "plots"

    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    with open(models_dir / "dataset_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    feature_cols = metadata["feature_cols"]
    target_col = "investment_decision"

    train = pd.read_csv(processed_dir / "train.csv")
    train = train.sort_values("Date").reset_index(drop=True)

    le = LabelEncoder()
    y = le.fit_transform(train["label"].astype(str))
    X = train[feature_cols].to_numpy(dtype=float)

    data_for_discovery = np.column_stack([X, y])
    col_names = feature_cols + [target_col]

    target_idx = len(feature_cols)
    prior = build_prior_knowledge(len(feature_cols), target_idx)

    model = DirectLiNGAM(prior_knowledge=prior)
    model.fit(data_for_discovery)
    adj = np.asarray(model.adjacency_matrix_, dtype=float)

    np.save(models_dir / "adj_matrix_directlingam.npy", adj)
    pd.DataFrame(adj, index=col_names, columns=col_names).to_csv(
        tables_dir / "adj_matrix_directlingam.csv"
    )

    # PC cross-check.
    try:
        cg = pc(data_for_discovery, alpha=0.05, indep_test=fisherz)
        pc_adj_raw = np.asarray(cg.G.graph)
        pc_bin = (np.abs(pc_adj_raw) > 0).astype(int)
    except Exception:
        pc_bin = np.zeros_like(adj, dtype=int)

    lingam_bin = binary_adjacency(adj, args.edge_threshold)

    agreement_rows = []
    for i, src in enumerate(col_names):
        for j, dst in enumerate(col_names):
            if i == j:
                continue
            agreement_rows.append(
                {
                    "source": src,
                    "target": dst,
                    "edge_lingam": int(lingam_bin[i, j]),
                    "edge_pc": int(pc_bin[i, j]),
                    "agree": int(lingam_bin[i, j] == pc_bin[i, j]),
                }
            )

    agreement_df = pd.DataFrame(agreement_rows)
    agreement_df.to_csv(tables_dir / "causal_method_agreement.csv", index=False)

    # Bootstrap stability.
    rng = np.random.default_rng(42)
    n = len(train)
    edge_counts = np.zeros_like(adj, dtype=float)

    for _ in range(args.bootstrap_runs):
        idx = rng.choice(n, size=max(20, int(n * args.bootstrap_frac)), replace=True)
        sample = data_for_discovery[idx]
        bs_model = DirectLiNGAM(prior_knowledge=prior)
        bs_model.fit(sample)
        bs_adj = np.asarray(bs_model.adjacency_matrix_, dtype=float)
        edge_counts += binary_adjacency(bs_adj, args.edge_threshold)

    stability = edge_counts / float(args.bootstrap_runs)

    stability_rows = []
    for i, src in enumerate(col_names):
        for j, dst in enumerate(col_names):
            if i == j:
                continue
            score = float(stability[i, j])
            stability_rows.append(
                {
                    "source": src,
                    "target": dst,
                    "stability_score": score,
                    "stable_ge_0_5": int(score >= 0.5),
                    "reliable_ge_0_7": int(score >= 0.7),
                }
            )

    pd.DataFrame(stability_rows).to_csv(tables_dir / "dag_stability_scores.csv", index=False)

    plt.figure(figsize=(12, 10))
    sns.heatmap(stability, xticklabels=col_names, yticklabels=col_names, cmap="viridis", vmin=0, vmax=1)
    plt.title("DAG Stability Heatmap (Bootstrap)")
    plt.tight_layout()
    plt.savefig(plots_dir / "dag_stability_heatmap.png", dpi=160)
    plt.close()

    # Regime stability test.
    regime_adj: dict[str, np.ndarray] = {}
    for regime in ["Bull", "Crisis", "Neutral"]:
        reg_df = train[train["regime"] == regime]
        if len(reg_df) < 200:
            continue
        y_reg = le.transform(reg_df["label"].astype(str))
        X_reg = reg_df[feature_cols].to_numpy(dtype=float)
        reg_data = np.column_stack([X_reg, y_reg])

        reg_model = DirectLiNGAM(prior_knowledge=prior)
        reg_model.fit(reg_data)
        regime_adj[regime] = binary_adjacency(np.asarray(reg_model.adjacency_matrix_), args.edge_threshold)

    regime_rows = []
    for a, b in combinations(sorted(regime_adj.keys()), 2):
        regime_rows.append({"regime_a": a, "regime_b": b, "jaccard_similarity": jaccard(regime_adj[a], regime_adj[b])})

    pd.DataFrame(regime_rows).to_csv(tables_dir / "regime_stability_report.csv", index=False)

    macro_features = {"ecb_rate", "us_10y_yield", "vix", "eur_usd", "de_inflation", "us_inflation"}
    plot_graph(adj, col_names, macro_features, plots_dir / "causal_graph_directlingam.png", args.edge_threshold)

    print(f"Saved adjacency matrix: {models_dir / 'adj_matrix_directlingam.npy'}")
    print(f"Saved PC agreement: {tables_dir / 'causal_method_agreement.csv'}")
    print(f"Saved DAG stability: {tables_dir / 'dag_stability_scores.csv'}")


if __name__ == "__main__":
    main()

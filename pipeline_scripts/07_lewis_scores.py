"""
Module: 07_lewis_scores
Purpose: Compute multi-class LEWIS-style Nec/Suf/Nesuf scores (pairwise one-vs-rest).
Inputs:  data/processed/train.csv, data/processed/test_predictions.csv,
         models/adj_matrix_directlingam.npy, models/dataset_metadata.json
Outputs: results/tables/lewis_scores_causal.csv, results/tables/lewis_scores_no_graph.csv,
         results/tables/reversal_scores.csv, results/plots/reversal_probabilities.png
Reference: Takahashi et al. (2024) arXiv:2402.02678
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer


def _safe_prob(num: float, den: float, eps: float = 1e-9) -> float:
    return float(num / max(den, eps))


def _p_y1_given_x(df: pd.DataFrame, feature: str, x_val: int, y_col: str) -> float:
    sub = df[df[feature] == x_val]
    if sub.empty:
        return float(df[y_col].mean())
    return float(sub[y_col].mean())


def _p_y1_do_x(
    df: pd.DataFrame,
    feature: str,
    x_val: int,
    y_col: str,
    parent_cols: list[str],
) -> float:
    if not parent_cols:
        return _p_y1_given_x(df, feature, x_val, y_col)

    # Ultra-fast Vectorized parent marginals and conditional means
    parent_dist = df.groupby(parent_cols, observed=True).size()
    parent_dist = parent_dist / parent_dist.sum()
    
    sub_df = df[df[feature] == x_val]
    if sub_df.empty:
        return _p_y1_given_x(df, feature, x_val, y_col)

    # Cond means of y_col grouped by parent_cols
    cond_means = sub_df.groupby(parent_cols, observed=True)[y_col].mean()

    # Join probabilities
    merged = parent_dist.to_frame("weight").join(cond_means.to_frame("mean"), how="left")
    
    # Fill any missing parent states with the generic conditional mean
    marginal_mean = _p_y1_given_x(df, feature, x_val, y_col)
    merged["mean"] = merged["mean"].fillna(marginal_mean)

    weighted = (merged["weight"] * merged["mean"]).sum()
    return float(weighted)


def compute_feature_scores(
    df_disc: pd.DataFrame,
    feature: str,
    y_col: str,
    parent_cols: list[str],
    use_graph: bool,
) -> dict[str, float]:
    vals = sorted(df_disc[feature].dropna().unique().astype(int).tolist())
    if len(vals) < 2:
        return {"Nec": 0.0, "Suf": 0.0, "Nesuf": 0.0, "maxNesuf": 0.0}

    # PRE-CALCULATE ALL MARGINALS AND DO-INTERVENTIONS ONCE OUTSIDE THE LOOP!
    # This prevents the exponential slowdown inside the nested for-loop.
    p1_given_dict = {v: _p_y1_given_x(df_disc, feature, v, y_col) for v in vals}
    if use_graph:
        p1_do_dict = {v: _p_y1_do_x(df_disc, feature, v, y_col, parent_cols) for v in vals}
    else:
        p1_do_dict = p1_given_dict

    nec_max = 0.0
    suf_max = 0.0
    nesuf_max = 0.0

    for x in vals:
        for xp in vals:
            if x == xp:
                continue

            p1_x = p1_given_dict[x]
            p1_xp = p1_given_dict[xp]
            p1_do_x = p1_do_dict[x]
            p1_do_xp = p1_do_dict[xp]

            # Binary LEWIS-style extension for positive class o=1 and contrasting class o'=0.
            nec = _safe_prob(max(0.0, p1_x - p1_do_xp), p1_x)
            suf = _safe_prob(max(0.0, p1_do_x - p1_xp), (1.0 - p1_xp))
            nesuf = max(0.0, p1_do_x - p1_do_xp)

            nec_max = max(nec_max, nec)
            suf_max = max(suf_max, suf)
            nesuf_max = max(nesuf_max, nesuf)

    return {"Nec": nec_max, "Suf": suf_max, "Nesuf": nesuf_max, "maxNesuf": nesuf_max}


def aggregate_pairwise_scores(
    df_disc: pd.DataFrame,
    feature_cols: list[str],
    adj: np.ndarray,
    use_graph: bool,
    positive_classes: list[str],
) -> pd.DataFrame:
    rows = []

    for feature_idx, feature in enumerate(feature_cols):
        print(f"   => Computing pairwise {'causal' if use_graph else 'no-graph'} scores for {feature} ({feature_idx+1}/{len(feature_cols)})...", flush=True)
        class_scores = []

        for positive in positive_classes:
            y_col = f"y_bin_{positive}"
            if use_graph:
                parents_idx = [i for i in range(len(feature_cols)) if abs(adj[i, feature_idx]) > 0]
                parent_cols = [feature_cols[i] for i in parents_idx]
            else:
                parent_cols = []

            class_scores.append(
                compute_feature_scores(
                    df_disc=df_disc,
                    feature=feature,
                    y_col=y_col,
                    parent_cols=parent_cols,
                    use_graph=use_graph,
                )
            )

        row = {
            "feature": feature,
            "Nec": float(np.mean([c["Nec"] for c in class_scores])),
            "Suf": float(np.mean([c["Suf"] for c in class_scores])),
            "Nesuf": float(np.mean([c["Nesuf"] for c in class_scores])),
            "maxNesuf": float(np.max([c["maxNesuf"] for c in class_scores])),
        }
        rows.append(row)

    return pd.DataFrame(rows).sort_values("maxNesuf", ascending=False)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    processed_dir = root / "data" / "processed"
    models_dir = root / "models"
    tables_dir = root / "results" / "tables"
    plots_dir = root / "results" / "plots"

    with open(models_dir / "dataset_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    feature_cols = metadata["feature_cols"]

    train = pd.read_csv(processed_dir / "train.csv")
    test_pred = pd.read_csv(processed_dir / "test_predictions.csv")

    adj = np.load(models_dir / "adj_matrix_directlingam.npy")

    discretizer = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="uniform")
    train_disc = discretizer.fit_transform(train[feature_cols])
    test_disc = discretizer.transform(test_pred[feature_cols])

    df_disc = pd.DataFrame(test_disc, columns=feature_cols).astype(int)
    df_disc["y_pred"] = test_pred["y_pred"].astype(str)

    positive_classes = ["Buy", "Sell"]
    for c in positive_classes:
        df_disc[f"y_bin_{c}"] = (df_disc["y_pred"] == c).astype(int)

    scores_causal = aggregate_pairwise_scores(
        df_disc=df_disc,
        feature_cols=feature_cols,
        adj=adj,
        use_graph=True,
        positive_classes=positive_classes,
    )

    scores_no_graph = aggregate_pairwise_scores(
        df_disc=df_disc,
        feature_cols=feature_cols,
        adj=adj,
        use_graph=False,
        positive_classes=positive_classes,
    )

    reversal = scores_causal[["feature", "Nec", "Suf"]].copy()

    scores_causal.to_csv(tables_dir / "lewis_scores_causal.csv", index=False)
    scores_no_graph.to_csv(tables_dir / "lewis_scores_no_graph.csv", index=False)
    reversal.to_csv(tables_dir / "reversal_scores.csv", index=False)

    # Reversal probability chart.
    plot_df = reversal.sort_values("Nec", ascending=True)
    y_pos = np.arange(len(plot_df))

    plt.figure(figsize=(10, 7))
    plt.barh(y_pos - 0.2, plot_df["Nec"], height=0.35, label="Nec")
    plt.barh(y_pos + 0.2, plot_df["Suf"], height=0.35, label="Suf")
    plt.yticks(y_pos, plot_df["feature"])
    plt.xlabel("Score")
    plt.title("Reversal Probabilities (LEWIS)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "reversal_probabilities.png", dpi=150)
    plt.close()

    print(f"Saved LEWIS causal scores: {tables_dir / 'lewis_scores_causal.csv'}")
    print(f"Saved LEWIS no-graph scores: {tables_dir / 'lewis_scores_no_graph.csv'}")


if __name__ == "__main__":
    main()

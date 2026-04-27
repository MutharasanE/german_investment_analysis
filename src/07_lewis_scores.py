"""
Module: LEWIS Scores
Purpose: Compute multi-class LEWIS counterfactual scores (Nec, Suf, Nesuf) using
         pairwise decomposition. This is the novel contribution: extending binary
         LEWIS to 3-class (Buy/Hold/Sell).
Inputs:  Trained model predictions, adjacency matrix from Step 6
Outputs: results/tables/lewis_scores_causal.csv, results/tables/lewis_scores_no_graph.csv,
         results/tables/reversal_scores.csv, results/plots/reversal_probabilities.png
Reference: Takahashi et al. (2024) arXiv:2402.02678, Equations (3)-(8)
           Galhotra et al. (2021) LEWIS: SIGMOD 2021
"""

import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import KBinsDiscretizer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .backdoor import compute_do_probability, compute_conditional_probability

logger = logging.getLogger(__name__)

N_BINS = 10  # Discretization bins, same as Takahashi et al. (2024)


def discretize_features(X_train, X_test, n_bins=N_BINS):
    """
    Discretize features using equal-width binning (same as paper).
    Fit on training data, transform both train and test.

    Args:
        X_train: Training features DataFrame.
        X_test: Test features DataFrame.
        n_bins: Number of bins (default 10, per paper).

    Returns:
        Discretized X_train, X_test DataFrames, fitted discretizer.
    """
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="uniform")
    X_train_disc = pd.DataFrame(
        discretizer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    ).astype(int)
    X_test_disc = pd.DataFrame(
        discretizer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    ).astype(int)
    return X_train_disc, X_test_disc, discretizer


def compute_pairwise_lewis(df_disc, feature_cols, target_col, adj_matrix, col_names,
                           positive_label, negative_label, use_graph=True):
    """
    Compute LEWIS scores for a binary decomposition (positive vs negative class).

    For each feature, computes maxNesuf — the maximum necessity-sufficiency score
    over all value pairs (x, x') where x > x'.

    Args:
        df_disc: Discretized DataFrame with features + target.
        feature_cols: Feature column names.
        target_col: Target column name.
        adj_matrix: Causal adjacency matrix (or identity for no-graph baseline).
        col_names: Column names matching adj_matrix.
        positive_label: Label treated as positive class.
        negative_label: Label treated as negative class.
        use_graph: If True, use causal graph; if False, assume P(Y|do(X))=P(Y|X).

    Returns:
        DataFrame with feature, maxNesuf, max_nec, max_suf columns.
    """
    if not use_graph:
        # No-graph baseline: identity matrix (no causal edges)
        adj_matrix = np.zeros_like(adj_matrix)

    results = []
    for feat in feature_cols:
        unique_vals = sorted(df_disc[feat].unique())
        max_nesuf = 0.0
        max_nec = 0.0
        max_suf = 0.0

        for i in range(len(unique_vals)):
            for j in range(i + 1, len(unique_vals)):
                x_low, x_high = unique_vals[i], unique_vals[j]

                # Nesuf(x, x') = P(o'|do(X=x')) - P(o'|do(X=x))
                p_neg_do_low = compute_do_probability(
                    df_disc, feat, x_low, target_col, negative_label,
                    adj_matrix, col_names
                )
                p_neg_do_high = compute_do_probability(
                    df_disc, feat, x_high, target_col, negative_label,
                    adj_matrix, col_names
                )
                nesuf = p_neg_do_low - p_neg_do_high

                # Nec(x, x') = (P(o'|do(X=x')) - P(o'|x)) / P(o|x)
                p_neg_given_high = compute_conditional_probability(
                    df_disc, target_col, negative_label, feat, x_high
                )
                p_pos_given_high = compute_conditional_probability(
                    df_disc, target_col, positive_label, feat, x_high
                )
                nec = (p_neg_do_low - p_neg_given_high) / p_pos_given_high if p_pos_given_high > 0 else 0.0

                # Suf(x, x') = (P(o|do(X=x)) - P(o|x')) / P(o'|x')
                p_pos_do_high = compute_do_probability(
                    df_disc, feat, x_high, target_col, positive_label,
                    adj_matrix, col_names
                )
                p_pos_given_low = compute_conditional_probability(
                    df_disc, target_col, positive_label, feat, x_low
                )
                p_neg_given_low = compute_conditional_probability(
                    df_disc, target_col, negative_label, feat, x_low
                )
                suf = (p_pos_do_high - p_pos_given_low) / p_neg_given_low if p_neg_given_low > 0 else 0.0

                max_nesuf = max(max_nesuf, nesuf)
                max_nec = max(max_nec, nec)
                max_suf = max(max_suf, suf)

        results.append({
            "feature": feat,
            "maxNesuf": max_nesuf,
            "max_Nec": max_nec,
            "max_Suf": max_suf,
        })

    return pd.DataFrame(results)


def compute_multiclass_lewis(df_disc, feature_cols, target_col, adj_matrix, col_names, use_graph=True):
    """
    Multi-class LEWIS extension: compute scores via pairwise decomposition.

    For 3-class (Buy=2, Hold=1, Sell=0):
      - Buy vs Rest: positive=2, negative={0,1}
      - Sell vs Rest: positive=0, negative={1,2}

    The global score per feature is the average maxNesuf across pairwise comparisons.

    Args:
        df_disc: Discretized DataFrame.
        feature_cols: Feature column names.
        target_col: Target column name.
        adj_matrix: Causal adjacency matrix.
        col_names: Column names.
        use_graph: Whether to use causal graph.

    Returns:
        DataFrame with aggregated multi-class LEWIS scores per feature.
    """
    # Create binary targets for pairwise decomposition
    df_buy = df_disc.copy()
    df_buy[target_col] = (df_disc[target_col] == 2).astype(int)  # Buy=1, Rest=0

    df_sell = df_disc.copy()
    df_sell[target_col] = (df_disc[target_col] == 0).astype(int)  # Sell=1, Rest=0

    # Compute LEWIS for each pairwise comparison
    buy_scores = compute_pairwise_lewis(
        df_buy, feature_cols, target_col, adj_matrix, col_names,
        positive_label=1, negative_label=0, use_graph=use_graph
    )
    buy_scores = buy_scores.rename(columns={
        "maxNesuf": "nesuf_buy", "max_Nec": "nec_buy", "max_Suf": "suf_buy"
    })

    sell_scores = compute_pairwise_lewis(
        df_sell, feature_cols, target_col, adj_matrix, col_names,
        positive_label=1, negative_label=0, use_graph=use_graph
    )
    sell_scores = sell_scores.rename(columns={
        "maxNesuf": "nesuf_sell", "max_Nec": "nec_sell", "max_Suf": "suf_sell"
    })

    # Merge and average
    merged = buy_scores.merge(sell_scores, on="feature")
    merged["maxNesuf_avg"] = (merged["nesuf_buy"] + merged["nesuf_sell"]) / 2
    merged["max_Nec_avg"] = (merged["nec_buy"] + merged["nec_sell"]) / 2
    merged["max_Suf_avg"] = (merged["suf_buy"] + merged["suf_sell"]) / 2

    return merged.sort_values("maxNesuf_avg", ascending=False)


def plot_reversal_probabilities(scores_df, results_dir="results/plots"):
    """
    Plot reversal probabilities (Nec and Suf) as bar chart.

    Args:
        scores_df: DataFrame with feature, max_Nec_avg, max_Suf_avg.
        results_dir: Directory to save plot.
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    features = scores_df["feature"].values
    x = np.arange(len(features))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, scores_df["max_Nec_avg"], width, label="Necessity (Nec)", color="#ff6b6b")
    ax.bar(x + width/2, scores_df["max_Suf_avg"], width, label="Sufficiency (Suf)", color="#4ecdc4")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Score")
    ax.set_title("Reversal Probabilities: Necessity vs Sufficiency (Multi-class Average)")
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "reversal_probabilities.png"), dpi=150)
    plt.close()
    logger.info("Reversal probability plot saved")


def run(train_df, test_df, model, adj_matrix, col_names, feature_cols=None,
        results_dir="results"):
    """
    Execute Step 7: Compute multi-class LEWIS scores with and without causal graph.

    Args:
        train_df: Training DataFrame.
        test_df: Test DataFrame.
        model: Trained CatBoost model.
        adj_matrix: Causal adjacency matrix from Step 6.
        col_names: Column names matching adj_matrix.
        feature_cols: Feature column names.
        results_dir: Base results directory.

    Returns:
        dict with causal_scores, no_graph_scores, reversal_scores.
    """
    logger.info("=" * 60)
    logger.info("STEP 7: LEWIS SCORES (MULTI-CLASS EXTENSION)")
    logger.info("=" * 60)

    Path(os.path.join(results_dir, "tables")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(results_dir, "plots")).mkdir(parents=True, exist_ok=True)

    if feature_cols is None:
        feature_cols = [c for c in col_names if c != "label"]

    target_col = "label"

    # Prepare data: combine train features with model predictions for LEWIS
    X_train = train_df[feature_cols]
    y_pred_train = model.predict(X_train).flatten().astype(int)

    # Discretize features
    logger.info(f"Discretizing features into {N_BINS} bins...")
    X_train_disc, _, _ = discretize_features(X_train, X_train)

    # Add predicted labels as target for LEWIS computation
    df_disc = X_train_disc.copy()
    df_disc[target_col] = y_pred_train

    # 7.1 LEWIS with causal graph
    logger.info("Computing LEWIS scores WITH causal graph...")
    causal_scores = compute_multiclass_lewis(
        df_disc, feature_cols, target_col, adj_matrix, col_names, use_graph=True
    )
    causal_scores.to_csv(os.path.join(results_dir, "tables", "lewis_scores_causal.csv"), index=False)
    logger.info("Causal LEWIS scores:")
    for _, row in causal_scores.iterrows():
        logger.info(f"  {row['feature']}: maxNesuf={row['maxNesuf_avg']:.4f}")

    # 7.2 LEWIS without causal graph (baseline)
    logger.info("Computing LEWIS scores WITHOUT causal graph (baseline)...")
    no_graph_scores = compute_multiclass_lewis(
        df_disc, feature_cols, target_col, adj_matrix, col_names, use_graph=False
    )
    no_graph_scores.to_csv(os.path.join(results_dir, "tables", "lewis_scores_no_graph.csv"), index=False)

    # 7.3 Plot reversal probabilities
    plot_reversal_probabilities(causal_scores, os.path.join(results_dir, "plots"))

    # 7.4 Save reversal scores
    reversal_df = causal_scores[["feature", "max_Nec_avg", "max_Suf_avg"]].copy()
    reversal_df.to_csv(os.path.join(results_dir, "tables", "reversal_scores.csv"), index=False)

    return {
        "causal_scores": causal_scores,
        "no_graph_scores": no_graph_scores,
        "reversal_scores": reversal_df,
    }

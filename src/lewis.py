"""
LEWIS explanatory scores: Necessity (Nec), Sufficiency (Suf),
and Necessity-Sufficiency (Nesuf).

Based on Galhotra et al. (2021) and Takahashi et al. (2024).
Scores are computed using counterfactual probabilities derived
from backdoor adjustment on the causal graph.
"""

import numpy as np
import pandas as pd
from .backdoor import compute_do_probability, compute_conditional_probability


def _graph_target(target_col, col_names):
    """Resolve the graph target column. If target_col (e.g. Y_pred) isn't in
    col_names, fall back to 'Y' or the last column in col_names."""
    if target_col in col_names:
        return target_col
    if "Y" in col_names:
        return "Y"
    return col_names[-1]


def compute_nesuf(df, feature_col, target_col, x_high, x_low,
                  adj_matrix, col_names, positive_label=1, negative_label=0):
    """
    Compute Necessity-Sufficiency score (Equation 8):
    Nesuf(x, x') = P(o'|do(X=x')) - P(o'|do(X=x))
    """
    gt = _graph_target(target_col, col_names)
    p_neg_do_low = compute_do_probability(
        df, feature_col, x_low, target_col, negative_label,
        adj_matrix, col_names, graph_outcome_col=gt
    )
    p_neg_do_high = compute_do_probability(
        df, feature_col, x_high, target_col, negative_label,
        adj_matrix, col_names, graph_outcome_col=gt
    )
    return p_neg_do_low - p_neg_do_high


def compute_nec(df, feature_col, target_col, x_high, x_low,
                adj_matrix, col_names, positive_label=1, negative_label=0):
    """
    Compute Necessity score (Equation 6):
    Nec(x, x') = (P(o'|do(X=x')) - P(o'|x)) / P(o|x)
    """
    gt = _graph_target(target_col, col_names)
    p_neg_do_low = compute_do_probability(
        df, feature_col, x_low, target_col, negative_label,
        adj_matrix, col_names, graph_outcome_col=gt
    )
    p_neg_given_high = compute_conditional_probability(
        df, target_col, negative_label, feature_col, x_high
    )
    p_pos_given_high = compute_conditional_probability(
        df, target_col, positive_label, feature_col, x_high
    )

    if p_pos_given_high == 0:
        return 0.0
    return (p_neg_do_low - p_neg_given_high) / p_pos_given_high


def compute_suf(df, feature_col, target_col, x_high, x_low,
                adj_matrix, col_names, positive_label=1, negative_label=0):
    """
    Compute Sufficiency score (Equation 7):
    Suf(x, x') = (P(o|do(X=x)) - P(o|x')) / P(o'|x')
    """
    gt = _graph_target(target_col, col_names)
    p_pos_do_high = compute_do_probability(
        df, feature_col, x_high, target_col, positive_label,
        adj_matrix, col_names, graph_outcome_col=gt
    )
    p_pos_given_low = compute_conditional_probability(
        df, target_col, positive_label, feature_col, x_low
    )
    p_neg_given_low = compute_conditional_probability(
        df, target_col, negative_label, feature_col, x_low
    )

    if p_neg_given_low == 0:
        return 0.0
    return (p_pos_do_high - p_pos_given_low) / p_neg_given_low


def compute_max_nesuf(df, feature_col, target_col, adj_matrix, col_names,
                      positive_label=1, negative_label=0):
    """
    Compute maxNesuf(X) - the global explanation score.
    Maximum Nesuf over all pairs (x, x') where x > x'.
    """
    unique_vals = sorted(df[feature_col].unique())
    max_score = 0.0
    best_pair = (None, None)

    for i in range(len(unique_vals)):
        for j in range(i + 1, len(unique_vals)):
            x_low, x_high = unique_vals[i], unique_vals[j]
            score = compute_nesuf(
                df, feature_col, target_col, x_high, x_low,
                adj_matrix, col_names, positive_label, negative_label
            )
            if score > max_score:
                max_score = score
                best_pair = (x_low, x_high)

    return max_score, best_pair


def compute_all_scores(df, feature_cols, target_col, adj_matrix, col_names,
                       positive_label=1, negative_label=0):
    """
    Compute maxNesuf for all features. Returns a DataFrame with scores.
    """
    results = []
    for feat in feature_cols:
        max_nesuf, pair = compute_max_nesuf(
            df, feat, target_col, adj_matrix, col_names,
            positive_label, negative_label
        )
        results.append({
            "feature": feat,
            "maxNesuf": max_nesuf,
            "best_pair_low": pair[0],
            "best_pair_high": pair[1],
        })
    return pd.DataFrame(results).sort_values("maxNesuf", ascending=False)


def compute_reversal_scores(df, feature_col, target_col, adj_matrix, col_names,
                            positive_label=1, negative_label=0):
    """
    Compute max Nec and max Suf scores for a feature (reversal probabilities).
    """
    unique_vals = sorted(df[feature_col].unique())
    max_nec = 0.0
    max_suf = 0.0

    for i in range(len(unique_vals)):
        for j in range(i + 1, len(unique_vals)):
            x_low, x_high = unique_vals[i], unique_vals[j]

            nec = compute_nec(
                df, feature_col, target_col, x_high, x_low,
                adj_matrix, col_names, positive_label, negative_label
            )
            suf = compute_suf(
                df, feature_col, target_col, x_high, x_low,
                adj_matrix, col_names, positive_label, negative_label
            )

            max_nec = max(max_nec, nec)
            max_suf = max(max_suf, suf)

    return max_nec, max_suf


def compute_all_reversal_scores(df, feature_cols, target_col, adj_matrix, col_names,
                                positive_label=1, negative_label=0):
    """Compute Nec and Suf for all features."""
    results = []
    for feat in feature_cols:
        nec, suf = compute_reversal_scores(
            df, feat, target_col, adj_matrix, col_names,
            positive_label, negative_label
        )
        results.append({"feature": feat, "Nec": nec, "Suf": suf})
    return pd.DataFrame(results)

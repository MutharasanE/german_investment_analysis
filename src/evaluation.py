"""
Evaluation metrics: MAE and Spearman rank correlation for Nesuf scores.
Implements Equations 9 and 10 from Takahashi et al. (2024).
"""

import numpy as np
from scipy import stats


def compute_mae(true_scores, estimated_scores):
    """
    Compute Mean Absolute Error between true and estimated maxNesuf scores.
    Equation 9: MAE = (1/N) * sum |maxNesuf_true - maxNesuf_est|
    """
    true_arr = np.array(true_scores)
    est_arr = np.array(estimated_scores)
    return np.mean(np.abs(true_arr - est_arr))


def compute_spearman(true_scores, estimated_scores):
    """
    Compute Spearman's rank correlation between true and estimated Nesuf rankings.
    Equation 10: SPR = 1 - (6 * sum(d_i^2)) / (n * (n^2 - 1))
    Returns value in [-1, 1]. Closer to 1 = better rank agreement.
    """
    correlation, p_value = stats.spearmanr(true_scores, estimated_scores)
    return correlation


def evaluate_trial(true_scores_dict, estimated_scores_dict, feature_names):
    """
    Evaluate a single trial. Returns MAE and Spearman for the trial.

    Parameters:
        true_scores_dict: {feature_name: maxNesuf_true}
        estimated_scores_dict: {feature_name: maxNesuf_est}
        feature_names: list of feature names to evaluate
    """
    true_vals = [true_scores_dict[f] for f in feature_names]
    est_vals = [estimated_scores_dict[f] for f in feature_names]

    mae = compute_mae(true_vals, est_vals)
    spr = compute_spearman(true_vals, est_vals)
    return mae, spr


def aggregate_trials(trial_results):
    """
    Aggregate results from multiple trials.

    Parameters:
        trial_results: list of (mae, spr) tuples

    Returns:
        dict with mean MAE, std error of MAE, mean SPR
    """
    maes = [r[0] for r in trial_results]
    sprs = [r[1] for r in trial_results]

    return {
        "MAE_mean": np.mean(maes),
        "MAE_std_error": np.std(maes, ddof=1) / np.sqrt(len(maes)),
        "SPR_mean": np.mean(sprs),
        "SPR_std_error": np.std(sprs, ddof=1) / np.sqrt(len(sprs)),
        "n_trials": len(trial_results),
    }


def format_result(result):
    """Format aggregated result as a string matching paper's table format."""
    return (
        f"MAE: {result['MAE_mean']:.4f} ± {result['MAE_std_error']:.4f} | "
        f"SPR: {result['SPR_mean']:.4f}"
    )

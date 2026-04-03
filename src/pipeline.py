"""
Main experiment pipeline.
Reproduces the experiments from Takahashi et al. (2024) and extends to investment data.
"""

import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

# Suppress numpy warnings from empty slices in probability computation
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")
# Suppress LiNGAM HSIC kernel overflow warnings (harmless — large values in RBF kernel)
warnings.filterwarnings("ignore", message="overflow encountered in exp")
warnings.filterwarnings("ignore", message="invalid value encountered in subtract")
warnings.filterwarnings("ignore", message="invalid value encountered in add")
warnings.filterwarnings("ignore", message="invalid value encountered in reduce")
warnings.filterwarnings("ignore", message="invalid value encountered in scalar subtract")

from .data_generation import (
    THREE_VAR_GENERATORS, THREE_VAR_TRUE_GRAPHS,
    generate_8var_data, EIGHT_VAR_TRUE_GRAPH, EIGHT_VAR_FEATURE_NAMES,
)
from .discretization import discretize_dataframe
from .causal_discovery import METHODS, METHOD_PRIORS
from .lewis import compute_all_scores
from .evaluation import evaluate_trial, aggregate_trials, format_result


def run_3var_experiment(n_samples=5000, n_bins=10, func_type="linear",
                        distribution="uniform", n_trials=100):
    """
    Reproduce Section III: 3-variable analysis.
    Compute Nesuf for structures A-E using the TRUE causal graph.
    """
    results = {}

    for struct_name, generator in THREE_VAR_GENERATORS.items():
        nesuf_x_list = []
        nesuf_z_list = []

        for trial in tqdm(range(n_trials), desc=f"Structure {struct_name}"):
            df = generator(n_samples, func_type=func_type, distribution=distribution)
            df_disc = discretize_dataframe(df, target_col="Y", n_bins=n_bins)

            adj = THREE_VAR_TRUE_GRAPHS[struct_name]
            col_names = ["X", "Z", "Y"]

            # Train classifier to get predictions
            from catboost import CatBoostClassifier
            features = df_disc[["X", "Z"]]
            target = df_disc["Y"]
            model = CatBoostClassifier(iterations=100, depth=6, verbose=0)
            model.fit(features, target)
            df_disc["Y_pred"] = model.predict(features).flatten()

            # Compute maxNesuf using predicted values
            scores = compute_all_scores(
                df_disc, ["X", "Z"], "Y_pred", adj, col_names
            )
            x_score = scores[scores["feature"] == "X"]["maxNesuf"].values[0]
            z_score = scores[scores["feature"] == "Z"]["maxNesuf"].values[0]
            nesuf_x_list.append(x_score)
            nesuf_z_list.append(z_score)

        results[struct_name] = {
            "X": {"mean": np.mean(nesuf_x_list), "std": np.std(nesuf_x_list)},
            "Z": {"mean": np.mean(nesuf_z_list), "std": np.std(nesuf_z_list)},
        }

    return results


def run_8var_experiment(n_samples=5000, n_bins=10, func_type="linear",
                        distribution="uniform", mixed=False, n_trials=100,
                        methods=None, priors=None):
    """
    Reproduce Section IV: 8-variable experiment with causal discovery.

    Compares estimated Nesuf (from discovered graph) vs true Nesuf (from known graph).
    """
    feature_names = ["X1", "X2", "X3", "X4", "X5", "X6", "X7"]
    col_names = EIGHT_VAR_FEATURE_NAMES  # includes Y
    target_idx = 7  # Y is the last column

    if methods is None:
        methods = ["DirectLiNGAM", "PC"]
    if priors is None:
        priors = ["0", "a", "b"]

    all_results = []

    for method_name in methods:
        supported_priors = METHOD_PRIORS.get(method_name, ["0"])
        method_fn = METHODS[method_name]

        for prior in priors:
            if prior not in supported_priors:
                continue

            trial_results = []
            print(f"\n--- {method_name} (prior={prior}) ---")

            for trial in tqdm(range(n_trials), desc=f"{method_name}({prior})"):
                # Generate data
                df = generate_8var_data(n_samples, func_type=func_type,
                                        distribution=distribution, mixed=mixed)
                df_disc = discretize_dataframe(df, target_col="Y", n_bins=n_bins)

                # Train classifier
                from catboost import CatBoostClassifier
                features = df_disc[feature_names]
                target = df_disc["Y"]
                model = CatBoostClassifier(iterations=200, depth=6, verbose=0)
                model.fit(features, target)
                df_disc["Y_pred"] = model.predict(features).flatten()

                # Compute TRUE Nesuf using known graph
                true_scores_df = compute_all_scores(
                    df_disc, feature_names, "Y_pred",
                    EIGHT_VAR_TRUE_GRAPH, col_names
                )
                true_scores = dict(zip(
                    true_scores_df["feature"], true_scores_df["maxNesuf"]
                ))

                # Run causal discovery
                data_array = df_disc[col_names].values.astype(float)
                try:
                    est_adj = method_fn(data_array, prior=prior, target_idx=target_idx)
                except Exception as e:
                    print(f"  Trial {trial}: {method_name} failed ({e}), skipping")
                    continue

                # Compute ESTIMATED Nesuf using discovered graph
                est_scores_df = compute_all_scores(
                    df_disc, feature_names, "Y_pred",
                    est_adj, col_names
                )
                est_scores = dict(zip(
                    est_scores_df["feature"], est_scores_df["maxNesuf"]
                ))

                # Evaluate
                mae, spr = evaluate_trial(true_scores, est_scores, feature_names)
                trial_results.append((mae, spr))

            if trial_results:
                agg = aggregate_trials(trial_results)
                agg["method"] = method_name
                agg["prior"] = prior
                all_results.append(agg)
                print(f"  {format_result(agg)}")

    # Also compute "No graph" baseline (P(Y|do(X)) = P(Y|X))
    print("\n--- No graph (baseline) ---")
    no_graph_trials = []
    no_graph_adj = np.zeros_like(EIGHT_VAR_TRUE_GRAPH)

    for trial in tqdm(range(n_trials), desc="No graph"):
        df = generate_8var_data(n_samples, func_type=func_type,
                                distribution=distribution, mixed=mixed)
        df_disc = discretize_dataframe(df, target_col="Y", n_bins=n_bins)

        from catboost import CatBoostClassifier
        features = df_disc[feature_names]
        target = df_disc["Y"]
        model = CatBoostClassifier(iterations=200, depth=6, verbose=0)
        model.fit(features, target)
        df_disc["Y_pred"] = model.predict(features).flatten()

        true_scores_df = compute_all_scores(
            df_disc, feature_names, "Y_pred",
            EIGHT_VAR_TRUE_GRAPH, col_names
        )
        true_scores = dict(zip(true_scores_df["feature"], true_scores_df["maxNesuf"]))

        est_scores_df = compute_all_scores(
            df_disc, feature_names, "Y_pred",
            no_graph_adj, col_names
        )
        est_scores = dict(zip(est_scores_df["feature"], est_scores_df["maxNesuf"]))

        mae, spr = evaluate_trial(true_scores, est_scores, feature_names)
        no_graph_trials.append((mae, spr))

    agg = aggregate_trials(no_graph_trials)
    agg["method"] = "No graph"
    agg["prior"] = "-"
    all_results.append(agg)
    print(f"  {format_result(agg)}")

    return pd.DataFrame(all_results)


def run_investment_pipeline(df, feature_cols, target_col, n_bins=10,
                            methods=None, priors=None):
    """
    Run the causal XAI pipeline on real investment data.
    Single run (no trials needed since data is fixed).

    Returns:
        results: dict with causal graph, LEWIS scores, reversal probabilities
    """
    from .discretization import discretize_dataframe
    from .lewis import compute_all_scores, compute_all_reversal_scores
    from .visualization import plot_causal_graph, plot_nesuf_comparison, plot_reversal_probabilities

    if methods is None:
        methods = ["DirectLiNGAM"]
    if priors is None:
        priors = ["b"]

    col_names = feature_cols + [target_col]

    # Discretize
    df_disc = discretize_dataframe(
        df[col_names], target_col=target_col, n_bins=n_bins, method="equal_freq"
    )

    # Train classifier
    from catboost import CatBoostClassifier
    features = df_disc[feature_cols]
    target = df_disc[target_col]
    model = CatBoostClassifier(iterations=300, depth=6, verbose=0)
    model.fit(features, target)
    df_disc["Y_pred"] = model.predict(features).flatten()

    accuracy = (df_disc["Y_pred"] == df_disc[target_col]).mean()
    print(f"Classifier accuracy: {accuracy:.4f}")

    results = {}

    for method_name in methods:
        method_fn = METHODS[method_name]
        supported_priors = METHOD_PRIORS.get(method_name, ["0"])

        for prior in priors:
            if prior not in supported_priors:
                continue

            key = f"{method_name}({prior})"
            print(f"\nRunning {key}...")

            # Causal discovery
            target_idx = col_names.index(target_col)
            data_array = df_disc[col_names].values.astype(float)
            est_adj = method_fn(data_array, prior=prior, target_idx=target_idx)

            # LEWIS scores with causal graph
            scores_causal = compute_all_scores(
                df_disc, feature_cols, "Y_pred", est_adj, col_names
            )

            # LEWIS scores without causal graph (baseline)
            no_graph_adj = np.zeros_like(est_adj)
            scores_no_graph = compute_all_scores(
                df_disc, feature_cols, "Y_pred", no_graph_adj, col_names
            )

            # Reversal probabilities
            reversal = compute_all_reversal_scores(
                df_disc, feature_cols, "Y_pred", est_adj, col_names
            )

            results[key] = {
                "adj_matrix": est_adj,
                "scores_causal": scores_causal,
                "scores_no_graph": scores_no_graph,
                "reversal": reversal,
                "model": model,
                "accuracy": accuracy,
            }

            print(f"\nNesuf scores ({key}):")
            print(scores_causal[["feature", "maxNesuf"]].to_string(index=False))

    return results

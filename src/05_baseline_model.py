"""
Module: Baseline Model + Main Model with SHAP
Purpose: Train Logistic Regression baseline AND CatBoost 3-class classifier,
         tune CatBoost hyperparameters, compute SHAP, produce confusion matrices
         and classification reports for both models.
Inputs:  data/processed/train.csv, data/processed/test.csv
Outputs: models/catboost_best.cbm, models/logistic_baseline.pkl,
         results/plots/confusion_matrix.png, results/plots/confusion_matrix_baseline.png,
         results/tables/classification_report.csv, results/tables/classification_report_baseline.csv,
         results/tables/shap_scores.csv, results/tables/model_comparison.csv,
         results/plots/shap_summary_plot.png, results/plots/shap_bar_plot.png
Reference: Takahashi et al. (2024) arXiv:2402.02678
"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    cohen_kappa_score, matthews_corrcoef
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

LABEL_NAMES = ["Sell", "Hold", "Buy"]
TARGET_COL = "label"
FEATURE_COLS = ["volatility", "momentum", "volume_avg", "rsi_14", "max_drawdown", "vix", "eur_usd"]


def tune_hyperparameters(X_train, y_train):
    """
    Grid search with TimeSeriesSplit CV. Small grid to keep training fast.

    Args:
        X_train: Training features.
        y_train: Training labels.

    Returns:
        dict: Best parameters, all results.
    """
    param_grid = {
        "iterations": [500, 1000],
        "learning_rate": [0.01, 0.05, 0.1],
        "depth": [4, 6, 8],
        "l2_leaf_reg": [1, 3, 5],
        "bagging_temperature": [0.5, 1.0],
    }

    tscv = TimeSeriesSplit(n_splits=5)
    best_score = -1
    best_params = {}
    all_results = []

    # Generate all combinations
    from itertools import product
    keys = list(param_grid.keys())
    combos = list(product(*param_grid.values()))

    logger.info(f"Hyperparameter tuning: {len(combos)} combinations x 3-fold TimeSeriesSplit")

    for combo in combos:
        params = dict(zip(keys, combo))
        scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            model = CatBoostClassifier(
                loss_function="MultiClass",
                eval_metric="Accuracy",
                random_seed=42,
                verbose=0,
                early_stopping_rounds=30,
                **params,
            )
            model.fit(
                X_train.iloc[train_idx], y_train.iloc[train_idx],
                eval_set=(X_train.iloc[val_idx], y_train.iloc[val_idx]),
                verbose=0,
            )
            preds = model.predict(X_train.iloc[val_idx]).flatten()
            acc = accuracy_score(y_train.iloc[val_idx], preds)
            scores.append(acc)

        mean_score = np.mean(scores)
        all_results.append({**params, "mean_cv_accuracy": round(mean_score, 4)})

        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    logger.info(f"Best params: {best_params} (CV accuracy: {best_score:.4f})")
    return {"best_params": best_params, "all_results": all_results, "best_cv_score": best_score}


def train_final_model(X_train, y_train, X_test, y_test, best_params):
    """
    Train final CatBoost model with best hyperparameters.

    Args:
        X_train, y_train: Training data.
        X_test, y_test: Test data (for early stopping eval).
        best_params: Best hyperparameters from grid search.

    Returns:
        Trained CatBoostClassifier model.
    """
    model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="Accuracy",
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50,
        **best_params,
    )
    model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=100)
    return model


def compute_shap_values(model, X_test, feature_cols, results_dir="results"):
    """
    Compute SHAP values using TreeExplainer.

    Args:
        model: Trained CatBoost model.
        X_test: Test features.
        feature_cols: Feature column names.
        results_dir: Base results directory.

    Returns:
        DataFrame with SHAP importance per feature.
    """
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    n_features = len(feature_cols)
    logger.info(f"SHAP values type={type(shap_values).__name__}, "
                f"shape={shap_values.shape if hasattr(shap_values, 'shape') else [np.array(s).shape for s in shap_values]}")

    # For multi-class: shap_values may be a list of 2D arrays OR a 3D ndarray.
    # Newer SHAP versions return 3D ndarray — axis order varies:
    #   (n_samples, n_features, n_classes) or (n_classes, n_samples, n_features)
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # Detect axis order: if last dim == n_classes (3), transpose to (n_classes, n_samples, n_features)
        if shap_values.shape[2] == 3 and shap_values.shape[1] == n_features:
            # Shape is (n_samples, n_features, n_classes)
            shap_values = shap_values.transpose(2, 0, 1)
        # Now guaranteed: (n_classes, n_samples, n_features)
        shap_importance = pd.DataFrame({
            "feature": feature_cols,
            "shap_sell": np.abs(shap_values[0]).mean(axis=0),
            "shap_hold": np.abs(shap_values[1]).mean(axis=0),
            "shap_buy": np.abs(shap_values[2]).mean(axis=0),
            "shap_mean": np.abs(shap_values).mean(axis=(0, 1)),
        })
        # Convert to list of 2D arrays for SHAP plotting functions
        shap_values_list = [shap_values[i] for i in range(shap_values.shape[0])]
    elif isinstance(shap_values, list):
        shap_importance = pd.DataFrame({
            "feature": feature_cols,
            "shap_sell": np.abs(shap_values[0]).mean(axis=0),
            "shap_hold": np.abs(shap_values[1]).mean(axis=0),
            "shap_buy": np.abs(shap_values[2]).mean(axis=0),
            "shap_mean": np.abs(np.array(shap_values)).mean(axis=(0, 1)),
        })
        shap_values_list = shap_values
    else:
        # Binary or single-output: 2D array (n_samples, n_features)
        shap_importance = pd.DataFrame({
            "feature": feature_cols,
            "shap_mean": np.abs(shap_values).mean(axis=0),
        })
        shap_values_list = shap_values

    shap_importance = shap_importance.sort_values("shap_mean", ascending=False)
    shap_importance.to_csv(os.path.join(results_dir, "tables", "shap_scores.csv"), index=False)

    # SHAP summary plot
    plt.figure(figsize=(10, 6))
    if isinstance(shap_values_list, list):
        shap.summary_plot(shap_values_list, X_test, feature_names=feature_cols,
                          class_names=LABEL_NAMES, show=False)
    else:
        shap.summary_plot(shap_values_list, X_test, feature_names=feature_cols, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "plots", "shap_summary_plot.png"), dpi=150)
    plt.close()

    # SHAP bar plot (mean importance)
    plt.figure(figsize=(8, 5))
    plt.barh(shap_importance["feature"], shap_importance["shap_mean"])
    plt.xlabel("Mean |SHAP value|")
    plt.title("SHAP Feature Importance (Correlational Baseline)")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "plots", "shap_bar_plot.png"), dpi=150)
    plt.close()

    logger.info("SHAP values computed and plots saved")
    return shap_importance, shap_values


def plot_confusion_matrix(y_true, y_pred, results_dir="results/plots",
                          filename="confusion_matrix.png",
                          title="Confusion Matrix (Buy / Hold / Sell)"):
    """
    Plot and save 3-class confusion matrix heatmap.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        results_dir: Directory to save plot.
        filename: Output filename.
        title: Plot title.
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, filename), dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved to {results_dir}/{filename}")


def generate_classification_report(y_true, y_pred, results_dir="results/tables"):
    """
    Generate and save full classification report with all metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        results_dir: Directory to save report CSV.

    Returns:
        dict with all metrics.
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    report = classification_report(y_true, y_pred, target_names=LABEL_NAMES,
                                   output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).T
    report_df.to_csv(os.path.join(results_dir, "classification_report.csv"))

    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    metrics = {
        "accuracy": round(acc, 4),
        "cohen_kappa": round(kappa, 4),
        "matthews_corrcoef": round(mcc, 4),
        "per_class": report,
    }

    logger.info(f"Test accuracy: {acc:.4f} | Cohen's Kappa: {kappa:.4f} | MCC: {mcc:.4f}")
    return metrics


def train_baseline_logistic(X_train, y_train, X_test, y_test, models_dir="models",
                            results_dir="results"):
    """
    Train Logistic Regression as a simple baseline model.

    Args:
        X_train, y_train: Training data.
        X_test, y_test: Test data.
        models_dir: Directory to save model.
        results_dir: Base results directory.

    Returns:
        dict with model, predictions, and metrics.
    """
    tscv = TimeSeriesSplit(n_splits=5)
    best_C, best_score = 1.0, -1

    for C in [0.01, 0.1, 1.0, 10.0]:
        scores = []
        for train_idx, val_idx in tscv.split(X_train):
            lr = LogisticRegression(C=C, max_iter=1000, solver="lbfgs", random_state=42)
            lr.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            acc = accuracy_score(y_train.iloc[val_idx], lr.predict(X_train.iloc[val_idx]))
            scores.append(acc)
        mean_acc = np.mean(scores)
        if mean_acc > best_score:
            best_score = mean_acc
            best_C = C

    logger.info(f"Baseline LR best C={best_C} (CV accuracy: {best_score:.4f})")

    lr_model = LogisticRegression(C=best_C, max_iter=1000, solver="lbfgs", random_state=42)
    lr_model.fit(X_train, y_train)
    joblib.dump(lr_model, os.path.join(models_dir, "logistic_baseline.pkl"))

    y_pred_lr = lr_model.predict(X_test)

    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred_lr,
                          os.path.join(results_dir, "plots"),
                          filename="confusion_matrix_baseline.png",
                          title="Confusion Matrix — Logistic Regression (Baseline)")

    # Classification report
    report = classification_report(y_test, y_pred_lr, target_names=LABEL_NAMES,
                                   output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).T
    report_df.to_csv(os.path.join(results_dir, "tables", "classification_report_baseline.csv"))

    acc = accuracy_score(y_test, y_pred_lr)
    kappa = cohen_kappa_score(y_test, y_pred_lr)
    mcc = matthews_corrcoef(y_test, y_pred_lr)

    metrics = {
        "accuracy": round(acc, 4),
        "cohen_kappa": round(kappa, 4),
        "matthews_corrcoef": round(mcc, 4),
        "best_C": best_C,
        "cv_accuracy": round(best_score, 4),
    }
    logger.info(f"Baseline LR — Test accuracy: {acc:.4f} | Kappa: {kappa:.4f} | MCC: {mcc:.4f}")
    return {"model": lr_model, "y_pred": y_pred_lr, "metrics": metrics}


def save_model_comparison(baseline_metrics, catboost_metrics, results_dir="results"):
    """Save side-by-side model comparison table."""
    comp = pd.DataFrame({
        "Metric": ["Accuracy", "Cohen's Kappa", "MCC"],
        "Logistic Regression": [
            baseline_metrics["accuracy"],
            baseline_metrics["cohen_kappa"],
            baseline_metrics["matthews_corrcoef"],
        ],
        "CatBoost": [
            catboost_metrics["accuracy"],
            catboost_metrics["cohen_kappa"],
            catboost_metrics["matthews_corrcoef"],
        ],
    })
    comp["Improvement"] = comp["CatBoost"] - comp["Logistic Regression"]
    comp.to_csv(os.path.join(results_dir, "tables", "model_comparison.csv"), index=False)
    logger.info("Model comparison saved to results/tables/model_comparison.csv")
    return comp


def run(train_df, test_df, feature_cols=None, models_dir="models", results_dir="results"):
    """
    Execute Step 5: Train CatBoost, compute SHAP, produce confusion matrix.

    Args:
        train_df: Training DataFrame.
        test_df: Test DataFrame.
        feature_cols: Feature column names.
        models_dir: Directory to save model.
        results_dir: Base results directory.

    Returns:
        dict with model, predictions, metrics, shap_importance.
    """
    logger.info("=" * 60)
    logger.info("STEP 5: BASELINE + MAIN MODEL + SHAP")
    logger.info("=" * 60)

    Path(models_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(results_dir, "plots")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(results_dir, "tables")).mkdir(parents=True, exist_ok=True)

    if feature_cols is None:
        feature_cols = [c for c in FEATURE_COLS if c in train_df.columns]

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL] if TARGET_COL in train_df.columns else train_df["label"]
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL] if TARGET_COL in test_df.columns else test_df["label"]

    # 5.1 Baseline model (Logistic Regression)
    logger.info("Training baseline model (Logistic Regression)...")
    baseline = train_baseline_logistic(X_train, y_train, X_test, y_test,
                                       models_dir, results_dir)

    # 5.2 CatBoost hyperparameter tuning
    logger.info("CatBoost hyperparameter tuning...")
    tuning = tune_hyperparameters(X_train, y_train)
    pd.DataFrame(tuning["all_results"]).to_csv(
        os.path.join(results_dir, "tables", "hyperparameter_tuning_results.csv"), index=False
    )

    # 5.3 Train final CatBoost model
    logger.info("Training final CatBoost model...")
    model = train_final_model(X_train, y_train, X_test, y_test, tuning["best_params"])
    model.save_model(os.path.join(models_dir, "catboost_best.cbm"))
    logger.info(f"Model saved to {models_dir}/catboost_best.cbm")

    # 5.4 CatBoost predictions
    y_pred = model.predict(X_test).flatten().astype(int)

    # 5.5 CatBoost confusion matrix
    plot_confusion_matrix(y_test, y_pred, os.path.join(results_dir, "plots"),
                          filename="confusion_matrix.png",
                          title="Confusion Matrix — CatBoost (Main Model)")

    # 5.6 CatBoost classification report
    metrics = generate_classification_report(y_test, y_pred, os.path.join(results_dir, "tables"))

    # 5.7 Model comparison
    comparison = save_model_comparison(baseline["metrics"], metrics, results_dir)
    logger.info(f"\n{comparison.to_string(index=False)}")

    # Save metadata early (before SHAP) so downstream steps can load it
    metadata = {
        "feature_cols": feature_cols,
        "best_params": tuning["best_params"],
        "cv_accuracy": tuning["best_cv_score"],
        "test_accuracy": metrics["accuracy"],
        "cohen_kappa": metrics["cohen_kappa"],
        "matthews_corrcoef": metrics["matthews_corrcoef"],
        "n_train": len(train_df),
        "n_test": len(test_df),
        "baseline": {
            "model": "LogisticRegression",
            "best_C": baseline["metrics"]["best_C"],
            "cv_accuracy": baseline["metrics"]["cv_accuracy"],
            "test_accuracy": baseline["metrics"]["accuracy"],
            "cohen_kappa": baseline["metrics"]["cohen_kappa"],
            "matthews_corrcoef": baseline["metrics"]["matthews_corrcoef"],
        },
    }
    with open(os.path.join(models_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # 5.8 SHAP explanations (on CatBoost — the main model)
    logger.info("Computing SHAP values...")
    shap_importance, shap_values = compute_shap_values(
        model, X_test, feature_cols, results_dir
    )

    return {
        "model": model,
        "y_pred": y_pred,
        "y_test": y_test.values if hasattr(y_test, "values") else y_test,
        "metrics": metrics,
        "baseline_metrics": baseline["metrics"],
        "shap_importance": shap_importance,
        "shap_values": shap_values,
        "feature_cols": feature_cols,
        "X_train": X_train,
        "X_test": X_test,
    }

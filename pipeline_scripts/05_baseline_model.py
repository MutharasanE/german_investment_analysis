"""
Module: 05_baseline_model
Purpose: Train CatBoost multiclass baseline and compute SHAP explanations.
Inputs:  data/processed/train.csv, validation.csv, test.csv, models/dataset_metadata.json
Outputs: models/catboost_multiclass.cbm, results/tables/hyperparameter_tuning_results.csv,
         results/tables/classification_report.csv, results/tables/shap_scores.csv,
         results/plots/confusion_matrix.png, results/plots/shap_summary_plot.png,
         results/plots/shap_bar_plot.png, data/processed/test_predictions.csv
Reference: Takahashi et al. (2024) arXiv:2402.02678
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from catboost import CatBoostClassifier
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder


RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 5: Baseline model and SHAP")
    parser.add_argument("--max-combos", type=int, default=30, help="Max parameter combinations to evaluate")
    parser.add_argument("--cv-splits", type=int, default=5, help="TimeSeriesSplit folds for CV")
    parser.add_argument("--max-train-rows", type=int, default=0, help="Cap rows used for train/CV (0 = all)")
    parser.add_argument("--max-val-rows", type=int, default=0, help="Cap rows used for validation (0 = all)")
    parser.add_argument("--max-test-rows", type=int, default=0, help="Cap rows used for test inference (0 = all)")
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP computation/plots")
    parser.add_argument(
        "--task-type",
        choices=["GPU", "CPU"],
        default="GPU",
        help="CatBoost training device type",
    )
    parser.add_argument("--devices", default="0", help="CatBoost GPU device ids when task-type=GPU")
    return parser.parse_args()


def build_catboost_model(
    task_type: str,
    devices: str,
    verbose: int,
    **params: object,
) -> CatBoostClassifier:
    model_params: dict[str, object] = {
        "loss_function": "MultiClass",
        "eval_metric": "TotalF1", # Better metric for imbalanced multiclass
        "auto_class_weights": "Balanced", # Automatically balance class weights
        "random_seed": RANDOM_STATE,
        "verbose": verbose,
        **params,
    }
    if task_type == "GPU":
        model_params["task_type"] = "GPU"
        model_params["devices"] = devices
    return CatBoostClassifier(**model_params)


def build_param_grid() -> list[dict[str, object]]:
    grid = {
        "iterations": [500, 1000, 2000],
        "learning_rate": [0.01, 0.05, 0.1],
        "depth": [4, 6, 8],
        "l2_leaf_reg": [1, 3, 5],
        "bagging_temperature": [0.5, 1.0],
    }
    keys = list(grid.keys())
    values = list(grid.values())
    combos = []
    for val_tuple in itertools.product(*values):
        combos.append(dict(zip(keys, val_tuple)))
    return combos


def evaluate_time_series_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    params: dict[str, object],
    task_type: str,
    devices: str,
    n_splits: int = 5,
) -> float:
    if len(X) <= n_splits:
        raise ValueError(f"Not enough rows ({len(X)}) for cv_splits={n_splits}")
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = build_catboost_model(task_type=task_type, devices=devices, verbose=0, **params)
        try:
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
        except Exception as exc:
            if task_type != "GPU":
                raise
            print(f"GPU CV fold failed, retrying on CPU: {exc}")
            model = build_catboost_model(task_type="CPU", devices=devices, verbose=0, **params)
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
        preds = model.predict(X_val).reshape(-1)
        from sklearn.metrics import f1_score
        scores.append(f1_score(y_val, preds, average="macro"))

    return float(np.mean(scores))


def plot_confusion(cm: np.ndarray, labels: list[str], out_path: Path) -> None:
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def normalize_shap_values(shap_values: object, n_classes: int, n_features: int) -> np.ndarray:
    if isinstance(shap_values, list):
        arr = np.asarray([np.asarray(sv) for sv in shap_values])
        # shape expected: (classes, n_samples, n_features)
        if arr.ndim == 3:
            return arr

    arr = np.asarray(shap_values)
    # CatBoost can return shape (n_samples, classes, features)
    if arr.ndim == 3 and arr.shape[1] == n_classes:
        return np.transpose(arr, (1, 0, 2))

    # Fallback if a single matrix appears.
    if arr.ndim == 2:
        expanded = np.zeros((n_classes, arr.shape[0], arr.shape[1]))
        for i in range(n_classes):
            expanded[i] = arr
        return expanded

    raise ValueError(f"Unsupported SHAP shape: {arr.shape}, expected classes={n_classes}, features={n_features}")


def _cap_rows(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if max_rows <= 0 or len(df) <= max_rows:
        return df
    return df.iloc[:max_rows].copy()


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

    train = pd.read_csv(processed_dir / "train.csv")
    validation = pd.read_csv(processed_dir / "validation.csv")
    test = pd.read_csv(processed_dir / "test.csv")

    train = _cap_rows(train, args.max_train_rows)
    validation = _cap_rows(validation, args.max_val_rows)
    test = _cap_rows(test, args.max_test_rows)

    full_train = (
        pd.concat([train, validation], ignore_index=True)
        .sort_values("Date")
        .reset_index(drop=True)
    )

    X_train = full_train[feature_cols]
    y_train_raw = full_train["label"].astype(str).to_numpy()

    X_val = validation[feature_cols]
    y_val_raw = validation["label"].astype(str).to_numpy()

    X_test = test[feature_cols]
    y_test_raw = test["label"].astype(str).to_numpy()

    le = LabelEncoder()
    le.fit(np.unique(np.concatenate([y_train_raw, y_val_raw, y_test_raw])))
    y_train = le.transform(y_train_raw)
    y_val = le.transform(y_val_raw)
    y_test = le.transform(y_test_raw)

    import optuna

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 500, 2000, step=500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 1, 10),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        }
        cv_score = evaluate_time_series_cv(
            X_train,
            y_train,
            params,
            task_type=args.task_type,
            devices=args.devices,
            n_splits=args.cv_splits,
        )
        return cv_score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    n_trials = args.max_combos if args.max_combos > 0 else 20
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_score = study.best_value

    tuning_df = study.trials_dataframe()
    tuning_df.to_csv(tables_dir / "hyperparameter_tuning_results.csv", index=False)

    assert best_params is not None

    model = build_catboost_model(
        task_type=args.task_type,
        devices=args.devices,
        verbose=100,
        **best_params,
    )
    try:
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
    except Exception as exc:
        if args.task_type != "GPU":
            raise
        print(f"Final GPU fit failed, retrying on CPU: {exc}")
        model = build_catboost_model(task_type="CPU", devices=args.devices, verbose=100, **best_params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)

    model.save_model(str(models_dir / "catboost_multiclass.cbm"))

    y_pred = model.predict(X_test).reshape(-1).astype(int)
    y_pred_labels = le.inverse_transform(y_pred)

    # Prefer model probabilities. Fallback to softmax on raw scores.
    try:
        y_proba = model.predict_proba(X_test)
    except Exception:
        raw = model.predict(X_test, prediction_type="RawFormulaVal")
        y_proba = softmax(np.asarray(raw), axis=1)

    cm = confusion_matrix(y_test_raw, y_pred_labels, labels=list(le.classes_))
    plot_confusion(cm, list(le.classes_), plots_dir / "confusion_matrix.png")

    report = classification_report(
        y_test_raw,
        y_pred_labels,
        labels=list(le.classes_),
        target_names=list(le.classes_),
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).T

    overall_metrics = pd.DataFrame(
        [
            {
                "metric": "accuracy",
                "value": accuracy_score(y_test_raw, y_pred_labels),
            },
            {
                "metric": "cohen_kappa",
                "value": cohen_kappa_score(y_test_raw, y_pred_labels),
            },
            {
                "metric": "mcc",
                "value": matthews_corrcoef(y_test_raw, y_pred_labels),
            },
        ]
    )

    report_df.to_csv(tables_dir / "classification_report.csv")
    overall_metrics.to_csv(tables_dir / "classification_overall_metrics.csv", index=False)

    if args.skip_shap:
        print("Skipping SHAP as requested (--skip-shap).")
    else:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            shap_array = normalize_shap_values(shap_values, n_classes=len(le.classes_), n_features=len(feature_cols))

            shap_rows = {"feature": feature_cols}
            for class_idx, class_label in enumerate(le.classes_):
                shap_rows[f"shap_{class_label.lower()}"] = np.abs(shap_array[class_idx]).mean(axis=0)

            shap_rows["shap_mean"] = np.abs(shap_array).mean(axis=(0, 1))
            shap_df = pd.DataFrame(shap_rows).sort_values("shap_mean", ascending=False)
            shap_df.to_csv(tables_dir / "shap_scores.csv", index=False)

            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_array.mean(axis=0), X_test, show=False)
            plt.tight_layout()
            plt.savefig(plots_dir / "shap_summary_plot.png", dpi=150)
            plt.close()

            plt.figure(figsize=(10, 6))
            shap_df_plot = shap_df.sort_values("shap_mean", ascending=True)
            plt.barh(shap_df_plot["feature"], shap_df_plot["shap_mean"], color="#1f77b4")
            plt.title("SHAP Mean Absolute Importance")
            plt.xlabel("Mean |SHAP value|")
            plt.tight_layout()
            plt.savefig(plots_dir / "shap_bar_plot.png", dpi=150)
            plt.close()
        except Exception as exc:
            print(f"SHAP generation failed, continuing without SHAP outputs: {exc}")

    pred_out = test.copy()
    pred_out["y_true"] = y_test_raw
    pred_out["y_pred"] = y_pred_labels
    for i, class_label in enumerate(le.classes_):
        pred_out[f"proba_{class_label}"] = y_proba[:, i]
    pred_out.to_csv(processed_dir / "test_predictions.csv", index=False)

    with open(models_dir / "label_encoder_classes.json", "w", encoding="utf-8") as f:
        json.dump({"classes": list(le.classes_)}, f, indent=2)

    print(f"\nBest CV Macro F1: {best_score:.4f}")
    print(f"Best params: {best_params}")
    print(f"Saved model: {models_dir / 'catboost_multiclass.cbm'}")
    print(f"Rows used - train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")


if __name__ == "__main__":
    main()

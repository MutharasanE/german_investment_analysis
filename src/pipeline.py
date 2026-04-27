"""
Master pipeline runner. Executes all steps in order.
Usage: python src/pipeline.py --steps all
       python src/pipeline.py --steps 1,2,3
       python src/pipeline.py --steps 5,6,7  (skip download if data exists)
"""

import argparse
import importlib.util
import logging
import warnings
import time
import sys
import os
from pathlib import Path

# Suppress known harmless warnings
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="invalid value encountered")
warnings.filterwarnings("ignore", category=FutureWarning)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure output directories exist before setting up logging
for d in ["data/raw", "data/processed", "results/plots", "results/tables",
          "results/reports", "models", "notebooks"]:
    Path(d).mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("results/pipeline.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

STEPS = {
    1: ("Data Download",         "01_data_download"),
    2: ("Feature Engineering",   "02_feature_engineering"),
    3: ("Labeling",              "03_labeling"),
    4: ("Data Preparation",      "04_data_preparation"),
    5: ("Baseline + SHAP",       "05_baseline_model"),
    6: ("Causal Discovery",      "06_causal_discovery"),
    7: ("LEWIS Scores",          "07_lewis_scores"),
    8: ("Evaluation",            "08_evaluation"),
    9: ("Comparison",            "09_comparison"),
    10: ("Governance Report",    "10_governance"),
}


def load_step(module_name):
    """
    Import a step module from src/ by filename.
    Uses importlib.util to handle numeric prefixes in module names.
    Sets __package__ = "src" so relative imports (e.g., from .backdoor) work.
    """
    file_path = PROJECT_ROOT / "src" / f"{module_name}.py"
    full_name = f"src.{module_name}"
    spec = importlib.util.spec_from_file_location(full_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "src"
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _preload_state(state, steps_to_run):
    """
    Load previously saved artifacts from disk so that later steps can run
    without re-running earlier steps. Only loads what's needed and available.
    """
    import json
    import numpy as np
    import pandas as pd
    from catboost import CatBoostClassifier

    feature_cols = ["volatility", "momentum", "volume_avg", "rsi_14",
                    "max_drawdown", "vix", "eur_usd"]
    min_step = min(steps_to_run)

    # If skipping steps 1-4, load train/test from disk
    if min_step >= 5:
        train_path = PROJECT_ROOT / "data" / "processed" / "train.csv"
        test_path = PROJECT_ROOT / "data" / "processed" / "test.csv"
        if train_path.exists() and test_path.exists():
            state["train"] = pd.read_csv(train_path, index_col=0, parse_dates=True)
            state["test"] = pd.read_csv(test_path, index_col=0, parse_dates=True)
            state["feature_cols"] = [c for c in feature_cols if c in state["train"].columns]
            logger.info(f"Pre-loaded train ({len(state['train'])}) and test ({len(state['test'])}) from disk")

    # If skipping step 5, load model + SHAP from disk
    if min_step >= 6:
        model_path = PROJECT_ROOT / "models" / "catboost_best.cbm"
        if model_path.exists():
            model = CatBoostClassifier()
            model.load_model(str(model_path))
            state["model"] = model
            logger.info("Pre-loaded CatBoost model from disk")

            if "test" in state and "feature_cols" in state:
                X_test = state["test"][state["feature_cols"]]
                state["X_test"] = X_test
                state["X_train"] = state["train"][state["feature_cols"]]
                y_pred = model.predict(X_test).flatten().astype(int)
                state["y_pred"] = y_pred
                state["y_test"] = state["test"]["label"].values if "label" in state["test"].columns else None

        meta_path = PROJECT_ROOT / "models" / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            state["metrics"] = {
                "accuracy": meta.get("test_accuracy"),
                "cohen_kappa": meta.get("cohen_kappa"),
            }

        shap_path = PROJECT_ROOT / "results" / "tables" / "shap_scores.csv"
        if shap_path.exists():
            state["shap_importance"] = pd.read_csv(shap_path)

    # If skipping step 6, load adjacency matrix from disk
    if min_step >= 7:
        adj_path = PROJECT_ROOT / "models" / "adj_matrix_directlingam.npy"
        if adj_path.exists():
            state["adj_matrix"] = np.load(adj_path)
            state["col_names"] = feature_cols + ["label"]
            logger.info("Pre-loaded adjacency matrix from disk")

        agree_path = PROJECT_ROOT / "results" / "tables" / "causal_method_agreement.csv"
        if agree_path.exists():
            agree_df = pd.read_csv(agree_path)
            agreed = (agree_df["agreement"] == "both").sum() if "agreement" in agree_df.columns else 0
            total = len(agree_df)
            state["agreement"] = {"agreed": agreed, "agreement_rate": agreed / total if total else 0}

    # If skipping step 7, load LEWIS scores from disk
    if min_step >= 8:
        lc_path = PROJECT_ROOT / "results" / "tables" / "lewis_scores_causal.csv"
        lng_path = PROJECT_ROOT / "results" / "tables" / "lewis_scores_no_graph.csv"
        if lc_path.exists():
            state["lewis_causal"] = pd.read_csv(lc_path)
        if lng_path.exists():
            state["lewis_no_graph"] = pd.read_csv(lng_path)


def run_pipeline(steps_to_run):
    """
    Execute pipeline steps in order, passing results between steps.

    Args:
        steps_to_run: List of step numbers to execute.
    """
    start_time = time.time()
    logger.info("=" * 70)
    logger.info("MASTER THESIS PIPELINE - Causal XAI for Investment Decisions")
    logger.info("DAX 40 | Daily | 3-month train / 1-month test | Buy/Hold/Sell")
    logger.info("=" * 70)

    # Shared state between steps — pre-load from disk if skipping early steps
    state = {}
    _preload_state(state, steps_to_run)

    for step_num in sorted(steps_to_run):
        if step_num not in STEPS:
            logger.warning(f"Unknown step {step_num}, skipping")
            continue

        step_name, module_name = STEPS[step_num]
        logger.info(f"\n{'='*60}")
        logger.info(f">>> STEP {step_num}: {step_name}")
        logger.info(f"{'='*60}")

        step_start = time.time()
        try:
            mod = load_step(module_name)

            if step_num == 1:
                result = mod.run()
                state["stock_data"] = result["stock_data"]
                state["macro_data"] = result["macro_data"]

            elif step_num == 2:
                result = mod.run(state["stock_data"], state["macro_data"])
                state["all_features"] = result

            elif step_num == 3:
                result = mod.run(state["all_features"], state["stock_data"])
                state["dataset"] = result

            elif step_num == 4:
                result = mod.run(state["dataset"])
                state["train"] = result["train"]
                state["test"] = result["test"]
                state["feature_cols"] = result["feature_cols"]

            elif step_num == 5:
                result = mod.run(state["train"], state["test"],
                                 state.get("feature_cols"))
                state["model"] = result["model"]
                state["y_pred"] = result["y_pred"]
                state["y_test"] = result["y_test"]
                state["metrics"] = result["metrics"]
                state["baseline_metrics"] = result.get("baseline_metrics")
                state["shap_importance"] = result["shap_importance"]
                state["shap_values"] = result["shap_values"]
                state["X_test"] = result["X_test"]
                state["X_train"] = result["X_train"]
                state["feature_cols"] = result["feature_cols"]

            elif step_num == 6:
                result = mod.run(state["train"], state.get("feature_cols"))
                state["adj_matrix"] = result["adj_matrix"]
                state["col_names"] = result["col_names"]
                state["agreement"] = result["agreement"]

            elif step_num == 7:
                result = mod.run(
                    state["train"], state["test"], state["model"],
                    state["adj_matrix"], state["col_names"],
                    state.get("feature_cols"),
                )
                state["lewis_causal"] = result["causal_scores"]
                state["lewis_no_graph"] = result["no_graph_scores"]

            elif step_num == 8:
                y_prob = None
                if state.get("model") is not None and state.get("X_test") is not None:
                    y_prob = state["model"].predict_proba(state["X_test"])
                result = mod.run(
                    state["y_test"], state["y_pred"], y_prob,
                    state["lewis_causal"], state["lewis_no_graph"],
                    state.get("feature_cols", []),
                )

            elif step_num == 9:
                result = mod.run(
                    state["shap_importance"], state["lewis_causal"],
                    state["lewis_no_graph"],
                )

            elif step_num == 10:
                result = mod.run({
                    "metrics": state.get("metrics", {}),
                    "agreement": state.get("agreement", {}),
                })

            elapsed = time.time() - step_start
            logger.info(f"<<< Step {step_num} completed in {elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - step_start
            logger.error(f"!!! Step {step_num} FAILED after {elapsed:.1f}s: {e}")
            import traceback
            logger.error(traceback.format_exc())
            logger.info("Continuing to next step...")

    total_time = time.time() - start_time
    logger.info(f"\n{'='*70}")
    logger.info(f"PIPELINE COMPLETE - Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Master Thesis Pipeline Runner")
    parser.add_argument("--steps", type=str, default="all",
                        help="Steps to run: 'all' or comma-separated (e.g., '1,2,3')")
    args = parser.parse_args()

    if args.steps == "all":
        steps = list(range(1, 11))
    else:
        steps = [int(s.strip()) for s in args.steps.split(",")]

    run_pipeline(steps)


if __name__ == "__main__":
    main()

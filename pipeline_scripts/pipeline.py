"""
Module: pipeline
Purpose: Master pipeline runner that executes thesis steps in order.
Inputs:  CLI step selection and existing data/artifacts from previous steps.
Outputs: results/pipeline.log and step-wise generated artifacts.
Reference: Takahashi et al. (2024) arXiv:2402.02678
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path


STEPS = {
    1: ("Data Download", "01_data_download.py"),
    2: ("Feature Engineering", "02_feature_engineering.py"),
    3: ("Labeling", "03_labeling.py"),
    4: ("Data Preparation", "04_data_preparation.py"),
    5: ("Baseline + SHAP", "05_baseline_model.py"),
    6: ("Causal Discovery", "06_causal_discovery.py"),
    7: ("LEWIS Scores", "07_lewis_scores.py"),
    8: ("Evaluation", "08_evaluation.py"),
    9: ("Comparison", "09_comparison.py"),
    10: ("Governance Report", "10_governance.py"),
}


def parse_steps(arg: str) -> list[int]:
    if arg.strip().lower() == "all":
        return sorted(STEPS)

    out = []
    for token in arg.split(","):
        token = token.strip()
        if not token:
            continue
        step = int(token)
        if step not in STEPS:
            raise ValueError(f"Unknown step: {step}")
        out.append(step)

    return out


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )


def run_step(src_dir: Path, step_num: int) -> bool:
    step_name, script_name = STEPS[step_num]
    script_path = src_dir / script_name

    logging.info("Starting Step %s: %s", step_num, step_name)

    try:
        subprocess.run([sys.executable, str(script_path)], check=True)
        logging.info("Completed Step %s: %s", step_num, step_name)
        return True
    except subprocess.CalledProcessError as exc:
        logging.exception("Step %s failed with exit code %s", step_num, exc.returncode)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Master pipeline runner")
    parser.add_argument("--steps", default="all", help="all or comma-separated step numbers, e.g. 1,2,3")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    src_dir = root / "src"
    setup_logging(root / "results" / "pipeline.log")

    steps = parse_steps(args.steps)
    logging.info("Requested steps: %s", steps)

    failed_steps = []
    for step_num in steps:
        ok = run_step(src_dir, step_num)
        if not ok:
            failed_steps.append(step_num)

    if failed_steps:
        logging.error("Pipeline completed with failures in steps: %s", failed_steps)
    else:
        logging.info("Pipeline completed successfully for all requested steps.")


if __name__ == "__main__":
    main()

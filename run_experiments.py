"""
Entry point: Run all experiments from Takahashi et al. (2024).
Usage: python run_experiments.py [--experiment 3var|8var|all] [--trials 100] [--parallel]
"""

import argparse
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.pipeline import run_3var_experiment, run_8var_experiment


def _run_single_8var_config(cfg, n_samples, n_bins, n_trials):
    """Run a single 8-var config (used for parallel execution)."""
    label = f"{cfg['func_type']}_{cfg['distribution']}"

    if cfg["func_type"] == "linear":
        methods = ["DirectLiNGAM", "PC", "NOTEARS"]
    else:
        methods = ["RESIT", "PC", "NOTEARS-MLP"]

    results_df = run_8var_experiment(
        n_samples=n_samples, n_bins=n_bins,
        func_type=cfg["func_type"], distribution=cfg["distribution"],
        mixed=cfg["mixed"], n_trials=n_trials,
        methods=methods, priors=["0", "a", "b"],
    )

    results_dir = Path("results/experiments")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_dir / f"8var_{label}.csv", index=False)
    return label, results_df


def main():
    parser = argparse.ArgumentParser(description="Run causal XAI experiments")
    parser.add_argument("--experiment", choices=["3var", "8var", "all"], default="all")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--parallel", action="store_true",
                        help="Run 8-var configs in parallel (uses 4 CPU cores)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers (default: 4)")
    args = parser.parse_args()

    results_dir = Path("results/experiments")
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.experiment in ("3var", "all"):
        print("=" * 60)
        print("EXPERIMENT 1: 3-variable structures (Section III)")
        print("=" * 60)

        for func_type in ["linear", "nonlinear"]:
            print(f"\n--- Function: {func_type} ---")
            results = run_3var_experiment(
                n_samples=args.samples, n_bins=args.bins,
                func_type=func_type, distribution="uniform",
                n_trials=args.trials,
            )
            print(f"\nResults ({func_type}):")
            for struct, scores in results.items():
                print(f"  {struct}: X={scores['X']['mean']:.3f}, Z={scores['Z']['mean']:.3f}")

            with open(results_dir / f"3var_{func_type}.json", "w") as f:
                json.dump(results, f, indent=2, default=str)

    if args.experiment in ("8var", "all"):
        print("\n" + "=" * 60)
        print("EXPERIMENT 2: 8-variable causal discovery (Section IV)")
        print("=" * 60)

        configs = [
            {"func_type": "linear", "distribution": "uniform", "mixed": False},
            {"func_type": "linear", "distribution": "gaussian", "mixed": False},
            {"func_type": "nonlinear", "distribution": "uniform", "mixed": False},
            {"func_type": "nonlinear", "distribution": "gaussian", "mixed": False},
        ]

        if args.parallel:
            print(f"\nRunning {len(configs)} configs in parallel ({args.workers} workers)...")
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(
                        _run_single_8var_config, cfg,
                        args.samples, args.bins, args.trials,
                    ): cfg
                    for cfg in configs
                }
                for future in as_completed(futures):
                    label, results_df = future.result()
                    print(f"\nCompleted: {label}")
                    print(results_df.to_string(index=False))
                    print(f"Saved to results/8var_{label}.csv")
        else:
            for cfg in configs:
                label = f"{cfg['func_type']}_{cfg['distribution']}"
                print(f"\n--- Config: {label} ---")

                if cfg["func_type"] == "linear":
                    methods = ["DirectLiNGAM", "PC", "NOTEARS"]
                else:
                    methods = ["RESIT", "PC", "NOTEARS-MLP"]

                results_df = run_8var_experiment(
                    n_samples=args.samples, n_bins=args.bins,
                    func_type=cfg["func_type"], distribution=cfg["distribution"],
                    mixed=cfg["mixed"], n_trials=args.trials,
                    methods=methods, priors=["0", "a", "b"],
                )
                results_df.to_csv(results_dir / f"8var_{label}.csv", index=False)
                print(f"\nSaved to results/8var_{label}.csv")

    print("\nAll experiments completed!")


if __name__ == "__main__":
    main()

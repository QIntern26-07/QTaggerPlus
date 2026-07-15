"""CLI entrypoint: run classical baselines over CIC-MalMem-2022."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger

from common import data
from common.evaluate import aggregate_metrics
from classical import run
from classical.models import MODEL_NAMES


def aggregate_records(records) -> dict:
    """Mean/std per metric across a model x task's fold records."""
    agg = {"model": records[0]["model"], "task": records[0]["task"]}
    agg.update(aggregate_metrics([r["metrics"] for r in records]))
    return agg


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CIC-MalMem classical baselines")
    p.add_argument("--csv", default="data/cic_malmem/Obfuscated-MalMem2022.csv")
    p.add_argument("--models", nargs="+", default=list(MODEL_NAMES))
    p.add_argument("--tasks", nargs="+", default=["binary", "multiclass"])
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--trials", type=int, default=25)
    p.add_argument("--inner-splits", type=int, default=3)
    p.add_argument(
        "--n-jobs", type=int, default=-1,
        help="parallelism for inner-CV fold evaluation and the final model refit "
             "(default -1 = all cores). Lower this if training competes with other "
             "programs for RAM/CPU on this machine.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mlflow", action="store_true")
    p.add_argument("--tracking-uri", default=None)
    p.add_argument(
        "--n-components", type=int, default=None,
        help="PCA target dimensionality (aligns with quantum qubit count). "
             "Omit to disable PCA.",
    )
    p.add_argument("--out", default="results/cic/metrics.csv")
    p.add_argument("--predictions-dir", default="results/cic")
    p.add_argument(
        "--load-quantum-splits", action="store_true",
        help="run on the exact row subsample and folds a prior quantum run "
             "persisted, instead of the full dataset with freshly computed "
             "folds — required for a real classical-vs-quantum comparison, "
             "since QSVM's O(n^2) kernel cost makes full-dataset quantum runs "
             "infeasible.",
    )
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logger.add("run.log", rotation="10 MB")
    df = data.load_cic_malmem(args.csv)

    rows = []
    for task in args.tasks:
        # Per-task row set: binary = all rows, multiclass = malware-only 15-class
        # (see data.task_xy). Must match what the quantum CLI persisted, so a
        # multiclass sample_idx indexes into the malware-only frame here too.
        X, y = data.task_xy(df, task)
        Path("data/splits").mkdir(parents=True, exist_ok=True)
        if args.load_quantum_splits:
            sample_idx = data.load_sample_idx(
                f"data/splits/quantum_sample_idx_{task}.json"
            )
            X = X.iloc[sample_idx].reset_index(drop=True)
            y = y.iloc[sample_idx].reset_index(drop=True)
            # Reuse the exact folds the quantum run computed over this same
            # subsample, instead of freshly computing (and silently
            # overwriting them with) different folds.
            folds = data.load_folds(f"data/splits/cic_{task}_quantum_folds.json")
        else:
            folds = data.make_outer_folds(y, n_splits=args.folds, seed=args.seed)
            data.save_folds(folds, f"data/splits/cic_{task}_folds.json")
        for name in args.models:
            records = run.run_nested_cv(
                X, y, task=task, name=name, folds=folds,
                n_trials=args.trials, seed=args.seed, use_mlflow=args.mlflow,
                inner_splits=args.inner_splits, dataset_name="cic-malmem",
                n_jobs=args.n_jobs, tracking_uri=args.tracking_uri,
                extra_params={"framework": "classical",
                              "n_components": args.n_components},
                n_components=args.n_components,
            )
            rows.append(aggregate_records(records))
            Path(args.predictions_dir).mkdir(parents=True, exist_ok=True)
            pred_path = f"{args.predictions_dir}/{name}_{task}_predictions.npz"
            data.save_predictions(records, pred_path)
            logger.info(f"wrote per-fold predictions to {pred_path}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    logger.info(f"wrote {len(rows)} model x task rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

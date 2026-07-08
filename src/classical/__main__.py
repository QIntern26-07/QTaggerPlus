"""CLI entrypoint: run classical baselines over CIC-MalMem-2022."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from common import data
from classical import run
from classical.models import MODEL_NAMES


def aggregate_records(records) -> dict:
    """Mean/std per metric across a model x task's fold records."""
    metric_keys = records[0]["metrics"].keys()
    agg = {"model": records[0]["model"], "task": records[0]["task"]}
    for key in metric_keys:
        vals = np.array([r["metrics"][key] for r in records], dtype=float)
        agg[f"{key}_mean"] = float(vals.mean())
        agg[f"{key}_std"] = float(vals.std(ddof=0))
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
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--out", default="results/cic/metrics.csv")
    p.add_argument("--predictions-dir", default="results/cic")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logger.add("run.log", rotation="10 MB")
    df = data.load_cic_malmem(args.csv)
    X, y_bin, y_multi = data.build_xy(df)
    targets = {"binary": y_bin, "multiclass": y_multi}

    rows = []
    for task in args.tasks:
        y = targets[task]
        folds = data.make_outer_folds(y, n_splits=args.folds, seed=args.seed)
        Path("data/splits").mkdir(parents=True, exist_ok=True)
        data.save_folds(folds, f"data/splits/cic_{task}_folds.json")
        for name in args.models:
            records = run.run_nested_cv(
                X, y, task=task, name=name, folds=folds,
                n_trials=args.trials, seed=args.seed, use_wandb=args.wandb,
                inner_splits=args.inner_splits, dataset_name="cic-malmem",
                n_jobs=args.n_jobs,
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

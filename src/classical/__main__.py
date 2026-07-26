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
    p = argparse.ArgumentParser(description="Classical baselines (CIC-MalMem / EMBER)")
    p.add_argument("--dataset", choices=["cic", "ember", "sorel"], default="cic")
    p.add_argument("--csv", default=None,
                   help="dataset file; defaults per --dataset (CIC csv / EMBER parquet).")
    p.add_argument("--models", nargs="+", default=list(MODEL_NAMES))
    p.add_argument("--tasks", nargs="+", default=None,
                   help="tasks to run; defaults to binary + multiclass for both datasets.")
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
    p.add_argument("--out", default=None,
                   help="metrics CSV; defaults to results/<dataset>/metrics.csv")
    p.add_argument("--predictions-dir", default=None,
                   help="per-fold predictions dir; defaults to results/<dataset>")
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
    if args.dataset == "ember":
        df = data.load_ember(args.csv or "data/ember/ember2018_test.parquet")
    elif args.dataset == "sorel":
        sorel_path = args.csv or "data/sorel/sorel_quantum_subset.parquet"
        if not Path(sorel_path).exists():
            raise SystemExit(
                f"--dataset sorel: {sorel_path} does not exist. SOREL-20M features "
                "require downloading its 71.6 GiB LMDB feature store "
                "(s3://sorel-20m/09-DEC-2020/processed-data/ember_features/data.mdb), "
                "which has not been fetched — this is a deliberate, documented decision, "
                "not a bug. See docs/reports/w4_sorel_labelling_decision.md for the "
                "labelling design and what remains (feature-subset acquisition + sweep)."
            )
        df = data.load_sorel(sorel_path)
    else:
        df = data.load_cic_malmem(args.csv or "data/cic_malmem/Obfuscated-MalMem2022.csv")
    tasks = args.tasks or ["binary", "multiclass"]
    dataset_name = {"cic": "cic-malmem", "ember": "ember-2018",
                    "sorel": "sorel-20m"}[args.dataset]
    out = args.out or f"results/{args.dataset}/metrics.csv"
    predictions_dir = args.predictions_dir or f"results/{args.dataset}"

    rows = []
    for task in tasks:
        # Per-task row set (binary vs. malware-family multiclass) is defined by
        # data.task_xy per dataset. Must match what the quantum CLI persisted, so a
        # multiclass sample_idx indexes into the same per-task frame here too.
        X, y = data.task_xy(df, task, dataset=args.dataset)
        Path("data/splits").mkdir(parents=True, exist_ok=True)
        sample_path, quantum_folds_path = data.split_paths(args.dataset, task)
        if args.load_quantum_splits:
            sample_idx = data.load_sample_idx(sample_path)
            X = X.iloc[sample_idx].reset_index(drop=True)
            y = y.iloc[sample_idx].reset_index(drop=True)
            # Reuse the exact folds the quantum run computed over this same
            # subsample, instead of freshly computing (and silently
            # overwriting them with) different folds.
            folds = data.load_folds(quantum_folds_path)
        else:
            folds = data.make_outer_folds(y, n_splits=args.folds, seed=args.seed)
            data.save_folds(folds, f"data/splits/{args.dataset}_{task}_folds.json")
        for name in args.models:
            records = run.run_nested_cv(
                X, y, task=task, name=name, folds=folds,
                n_trials=args.trials, seed=args.seed, use_mlflow=args.mlflow,
                inner_splits=args.inner_splits, dataset_name=dataset_name,
                n_jobs=args.n_jobs, tracking_uri=args.tracking_uri,
                extra_params={"framework": "classical",
                              "n_components": args.n_components},
                n_components=args.n_components,
            )
            rows.append(aggregate_records(records))
            Path(predictions_dir).mkdir(parents=True, exist_ok=True)
            pred_path = f"{predictions_dir}/{name}_{task}_predictions.npz"
            data.save_predictions(records, pred_path)
            logger.info(f"wrote per-fold predictions to {pred_path}")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    logger.info(f"wrote {len(rows)} model x task rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

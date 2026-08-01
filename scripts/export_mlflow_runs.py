# scripts/export_mlflow_runs.py
"""Export MLflow runs to a committable CSV so results can be shared via git.

`mlflow.db` itself is deliberately not committed: it is an 8MB+ binary SQLite
file that every run rewrites (unresolvable merge conflicts), and its artifact
paths point into the gitignored `mlruns/` directory, so a teammate checking it
out would get broken artifact links anyway. A flat CSV of params + metrics is
small, diffable, reviewable in a PR, and is what the report tables are built
from.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import mlflow
from loguru import logger

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Export MLflow runs to CSV")
    p.add_argument("--experiment", default="qtaggerplus")
    p.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    p.add_argument("--out", default="results/mlflow_runs.csv")
    p.add_argument(
        "--filter", default=None,
        help="MLflow filter string, e.g. \"params.dataset = 'ember-2018'\" — "
             "use to export one sweep instead of the whole experiment.",
    )
    args = p.parse_args(argv)

    mlflow.set_tracking_uri(args.tracking_uri)
    df = mlflow.search_runs(
        experiment_names=[args.experiment], filter_string=args.filter or "",
    )
    if df.empty:
        logger.warning(f"no runs found in experiment '{args.experiment}'")
        return 1

    # `run_id` and the parent link are kept deliberately. The same (dataset,
    # task, model, n_components, encoding) cell is legitimately swept more than
    # once across weeks, so those params alone do NOT identify one CV run — a
    # groupby over them returns 10, 15 or 20 fold rows for a 5-fold sweep, and
    # mixing sweeps silently averages different hyperparameter searches together.
    # Consumers must group by `parent_run_id` and keep only sweeps whose parent
    # is FINISHED, which also excludes folds orphaned by an interrupted sweep
    # (see w4_consolidated_report.md section 6).
    if "tags.mlflow.parentRunId" in df.columns:
        df = df.rename(columns={"tags.mlflow.parentRunId": "parent_run_id"})
    # Drop columns that are machine-local or churn every run — they would make
    # the CSV diff noisy without carrying any result information.
    drop = [c for c in df.columns if c.startswith("tags.mlflow.")]
    drop += [c for c in ("artifact_uri", "experiment_id") if c in df.columns]
    df = df.drop(columns=drop).sort_values("start_time")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    logger.info(f"wrote {len(df)} runs x {len(df.columns)} columns to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

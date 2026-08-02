"""Day 34: paired significance tests, quantum vs classical.

Closes Week 1 Day 6, never run until now. Reports paired t-test and Wilcoxon
signed-rank over the 5 outer-fold macro-F1 values for every QSVM-vs-classical
pair that has a complete fold group in the MLflow export.

McNemar is NOT run for quantum-vs-classical pairs: QSVM per-fold predictions
were never persisted before this week (see quantum/__main__.py, fixed in this
same branch), so no sample-level comparison is possible for existing sweeps.

Run: uv run python scripts/run_significance_tests.py
"""
from __future__ import annotations

import json

import pandas as pd

from classical.compare import paired_ttest, wilcoxon
from common.significance import fold_scores

CSV = "results/mlflow_runs.csv"
CLASSICAL = ("random_forest", "xgboost", "lightgbm", "svm")
CASES = [
    (dataset, task, nc)
    for dataset in ("cic-malmem", "ember-2018")
    for task in ("binary", "multiclass")
    for nc in (1, 3, 6)
]
# "joint" covers CIC sweeps that tuned over both encodings in one run; the
# single names cover EMBER, whose Week 4 sweeps ran one encoding per invocation.
QUANTUM_ENCODINGS = ("angle", "iqp", "joint")


def main() -> int:
    df = pd.read_csv(CSV)
    results, skipped = [], []
    for dataset, task, nc in CASES:
        for encoding in QUANTUM_ENCODINGS:
            try:
                q = fold_scores(df, dataset, task, nc, "qsvm", encoding=encoding)
            except ValueError as exc:
                skipped.append(str(exc))
                continue
            for model in CLASSICAL:
                try:
                    c = fold_scores(df, dataset, task, nc, model)
                except ValueError as exc:
                    skipped.append(str(exc))
                    continue
                results.append({
                    "dataset": dataset, "task": task, "n_components": nc,
                    "quantum": f"qsvm-{encoding}", "classical": model,
                    "qsvm_mean": float(q.mean()),
                    "classical_mean": float(c.mean()),
                    "delta": float(c.mean() - q.mean()),
                    "ttest": paired_ttest(q, c),
                    "wilcoxon": wilcoxon(q, c),
                })
    print(json.dumps({"comparisons": results, "skipped": sorted(set(skipped))},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

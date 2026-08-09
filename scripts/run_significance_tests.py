"""Day 34: paired significance tests, quantum vs classical.

Closes Week 1 Day 6, never run until now. Reports paired t-test and Wilcoxon
signed-rank over the outer-fold macro-F1 values for every QSVM-vs-classical
pair that has a complete fold group in the MLflow export. Week 6: prefers a
six-fold sweep where one exists, since Wilcoxon cannot reach p < 0.05 at n=5.

McNemar lives in scripts/run_mcnemar_cross_framework.py, not here: it needs
per-sample predictions, which QSVM only began persisting in Week 6, so it
applies to the new sweeps and not to the historical export this script reads.

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


# Week 6 added a six-fold sweep on the CIC 15-class cell, because Wilcoxon's
# minimum attainable p at n=5 is 0.0625 — above 0.05 by construction, so every
# non-parametric result before this week was capped out of significance. Prefer
# a six-fold sweep where one exists and fall back to five, but never pair a
# six-fold arm against a five-fold one: paired tests need the same folds on
# both sides, and fold i of one sweep is not fold i of another.
FOLD_COUNTS = (6, 5)


def _scores_any_fold_count(df, dataset, task, nc, model, encoding=None):
    """(scores, n_folds) for the newest clean sweep, preferring six folds."""
    errors = []
    for expected in FOLD_COUNTS:
        try:
            s = fold_scores(df, dataset, task, nc, model, encoding=encoding,
                            expected_folds=expected)
            return s, expected
        except ValueError as exc:
            errors.append(str(exc))
    raise ValueError("; ".join(errors))


def main() -> int:
    df = pd.read_csv(CSV)
    results, skipped = [], []
    for dataset, task, nc in CASES:
        for encoding in QUANTUM_ENCODINGS:
            try:
                q, q_folds = _scores_any_fold_count(
                    df, dataset, task, nc, "qsvm", encoding=encoding)
            except ValueError as exc:
                skipped.append(str(exc))
                continue
            for model in CLASSICAL:
                try:
                    c, c_folds = _scores_any_fold_count(
                        df, dataset, task, nc, model)
                except ValueError as exc:
                    skipped.append(str(exc))
                    continue
                if q_folds != c_folds:
                    skipped.append(
                        f"{dataset}/{task}/nc={nc}/qsvm-{encoding} vs {model}: "
                        f"fold counts differ ({q_folds} vs {c_folds}); the "
                        "classical arm needs a matching re-run before these "
                        "can be paired"
                    )
                    continue
                results.append({
                    "dataset": dataset, "task": task, "n_components": nc,
                    "quantum": f"qsvm-{encoding}", "classical": model,
                    "n_folds": q_folds,
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

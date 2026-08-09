"""Week 6: sample-level McNemar, QSVM vs each classical baseline.

The project report lists this as a limitation: McNemar needs per-sample
predictions from both arms on a shared test set, and QSVM predictions were not
persisted during any earlier sweep, so nothing was recoverable retroactively.
The Week 6 six-fold sweep is the first that writes them.

Both arms scored the identical held-out rows - quantum wrote the subsample and
the folds, classical reloaded them with --load-quantum-splits - so predictions
align by test_idx. The alignment is asserted rather than assumed: a silent
mismatch would produce a p-value for a comparison that never happened.

Run: uv run python scripts/run_mcnemar_cross_framework.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from classical.compare import mcnemar
from common import data

QUANTUM = "results/cic/qsvm_multiclass_nc6_iqp_predictions.npz"
CLASSICAL_DIR = "results/cic/w6_nc6_predictions"
MODELS = ("random_forest", "xgboost", "lightgbm", "svm")
TASK = "multiclass"
OUT = "results/cic/w6_mcnemar.json"


def _aligned(path):
    """Predictions sorted by held-out row index, so two arms line up."""
    p = data.load_predictions(path)
    order = np.argsort(p["test_idx"])
    return p["test_idx"][order], p["y_true"][order], p["y_pred"][order]


def main() -> int:
    q_idx, q_true, q_pred = _aligned(QUANTUM)
    results, skipped = [], []
    for model in MODELS:
        path = Path(CLASSICAL_DIR) / f"{model}_{TASK}_predictions.npz"
        if not path.exists():
            skipped.append(f"{model}: {path} missing")
            continue
        c_idx, c_true, c_pred = _aligned(str(path))
        if not np.array_equal(q_idx, c_idx):
            skipped.append(f"{model}: held-out rows differ from the quantum run")
            continue
        if not np.array_equal(q_true, c_true):
            skipped.append(f"{model}: labels differ on identical rows")
            continue
        res = mcnemar(q_true, q_pred, c_pred)
        results.append({
            "model": model,
            "n_samples": int(len(q_true)),
            "qsvm_correct": int((q_pred == q_true).sum()),
            "classical_correct": int((c_pred == c_true).sum()),
            **res,
        })
        print(json.dumps(results[-1]))
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(
        {"quantum_predictions": QUANTUM, "classical_dir": CLASSICAL_DIR,
         "comparisons": results, "skipped": skipped}, indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

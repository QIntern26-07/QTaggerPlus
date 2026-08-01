"""Day 33: why the fidelity kernel fails on CIC 15-class but not EMBER.

For each dataset, on the FIRST outer training fold at nc=6 with iqp:
build the QSVM Gram, measure concentration and kernel-target alignment, and
measure the same two statistics for a classical RBF kernel on the identical
feature matrix. The RBF control is the decisive comparison — it separates
"these features are hard" from "this kernel is bad".

Run: uv run python scripts/kernel_diagnostics.py
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

from common import data, kernel_diag
from quantum.encoding import n_qubits_for
from quantum.qsvm import QSVM
from quantum.run import _prep

DATASETS = ("cic", "ember")
TASK = "multiclass"
NC = 6
ENCODING = "iqp"
SEED = 42
# Cap workers. The default -1 takes all 20 cores; the Day 29 grid died that way.
# The Gram build is CPU-bound with tiny per-worker state, so 8 is a deliberate
# headroom choice rather than a memory necessity.
N_JOBS = 8

# EMBER's reference frame is the quantum subset, NOT the 200,000-row test
# parquet: ember_family_xy re-downsamples families at load time, so a different
# input frame yields a different pool and the persisted sample indices would
# select the wrong rows. See w5_day31_cic_vs_ember_separability.md A.1.
EMBER_PARQUET = "data/ember/ember2018_quantum_subset.parquet"
CIC_CSV = "data/cic_malmem/Obfuscated-MalMem2022.csv"


def _train_fold(dataset):
    """First outer training fold, exactly as the QSVM sweeps saw it."""
    if dataset == "ember":
        df = data.load_ember(EMBER_PARQUET)
    else:
        df = data.load_cic_malmem(CIC_CSV)
    X, y = data.task_xy(df, TASK, dataset=dataset)
    sample_path, folds_path = data.split_paths(dataset, TASK)
    sample_idx = data.load_sample_idx(sample_path)
    folds = data.load_folds(folds_path)
    train_idx, _ = folds[0]
    Xs = X.to_numpy()[sample_idx]
    ys = y.to_numpy()[sample_idx]
    return Xs[train_idx], ys[train_idx]


def diagnose(dataset):
    X_tr, y_tr = _train_fold(dataset)
    n_qubits = n_qubits_for(ENCODING, NC)
    # _prep needs a second matrix to transform; pass the train fold twice and
    # keep only the first return value. Same PCA + encoding scaling the sweeps
    # used, fit on this train fold only.
    Ztr, _ = _prep(X_tr, X_tr, NC, ENCODING, n_qubits, SEED)

    model = QSVM(encoding=ENCODING, n_components=NC, task=TASK, seed=SEED,
                 n_jobs=N_JOBS)
    # _gram_sym is the exact method QSVM.fit uses to build its training Gram
    # (src/quantum/qsvm.py:165). Called directly here because fit() consumes
    # the matrix internally and never exposes it.
    K_q = model._gram_sym(Ztr)

    # RBF on the SAME matrix. gamma="scale" equivalent: 1/(n_features * var).
    gamma = 1.0 / (Ztr.shape[1] * Ztr.var()) if Ztr.var() > 0 else 1.0
    K_c = rbf_kernel(Ztr, gamma=gamma)

    return {
        "n_train_rows": int(len(Ztr)),
        "n_qubits": int(n_qubits),
        "quantum": {
            "offdiag_std": kernel_diag.offdiag_std(K_q),
            "alignment": kernel_diag.kernel_target_alignment(K_q, y_tr),
        },
        "rbf_control": {
            "gamma": float(gamma),
            "offdiag_std": kernel_diag.offdiag_std(K_c),
            "alignment": kernel_diag.kernel_target_alignment(K_c, y_tr),
        },
    }


def main() -> int:
    out = {d: diagnose(d) for d in DATASETS}
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

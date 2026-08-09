"""Week 6: does bandwidth relieve fidelity-kernel concentration on CIC 15-class?

The project report identifies concentration as the mechanism behind the CIC
15-class collapse and bandwidth as the parameter acting on it, but never swept
bandwidth. This measures the Gram directly - no SVM, no CV - so the mechanism
question is answered before an hour of tuned CV is spent on it.

Run: uv run python scripts/bandwidth_diagnostics.py
"""
from __future__ import annotations

import json
from pathlib import Path

from sklearn.metrics.pairwise import rbf_kernel

from common import data, kernel_diag
from quantum.encoding import n_qubits_for
from quantum.qsvm import QSVM
from quantum.run import _prep

TASK = "multiclass"
NC = 6
ENCODING = "iqp"
SEED = 42
# Cap workers. The default -1 takes all 20 cores; the Day 29 grid died that way.
N_JOBS = 8
CIC_CSV = "data/cic_malmem/Obfuscated-MalMem2022.csv"
OUT = "results/cic/bandwidth_diagnostics.json"

# default_bandwidth(6) = 6**-0.5 = 0.408, the only value ever used. Bracket it
# on both sides: smaller bandwidth means less rotation, so states stay closer
# and the kernel should concentrate LESS, not more.
BANDWIDTHS = [0.05, 0.1, 0.2, 6 ** -0.5, 1.0]


def _train_fold():
    """First outer training fold, exactly as the QSVM sweeps saw it."""
    df = data.load_cic_malmem(CIC_CSV)
    X, y = data.task_xy(df, TASK, dataset="cic")
    sample_path, folds_path = data.split_paths("cic", TASK)
    sample_idx = data.load_sample_idx(sample_path)
    folds = data.load_folds(folds_path)
    train_idx, _ = folds[0]
    Xs = X.to_numpy()[sample_idx]
    ys = y.to_numpy()[sample_idx]
    return Xs[train_idx], ys[train_idx]


def main() -> int:
    X_tr, y_tr = _train_fold()
    n_qubits = n_qubits_for(ENCODING, NC)
    # _prep needs a second matrix to transform; pass the train fold twice and
    # keep only the first return value. Same PCA + encoding scaling the sweeps
    # used, fit on this train fold only.
    Ztr, _ = _prep(X_tr, X_tr, NC, ENCODING, n_qubits, SEED)

    # Same RBF control as scripts/kernel_diagnostics.py:73 — gamma="scale"
    # equivalent, 1/(n_features * var). sklearn's default gamma=1/n_features
    # gives a different spread and would not be comparable with the 0.40x
    # figure the Day 33 report and the project paper both quote.
    gamma = 1.0 / (Ztr.shape[1] * Ztr.var()) if Ztr.var() > 0 else 1.0
    K_rbf = rbf_kernel(Ztr, gamma=gamma)
    rbf_std = kernel_diag.offdiag_std(K_rbf)
    rbf_align = kernel_diag.kernel_target_alignment(K_rbf, y_tr)
    print(f"rbf: gamma={gamma:.4f} offdiag_std={rbf_std:.4f} "
          f"alignment={rbf_align:.4f} n_train={len(y_tr)}")

    records = []
    for bw in BANDWIDTHS:
        model = QSVM(encoding=ENCODING, n_components=NC, bandwidth=bw,
                     task=TASK, seed=SEED, n_jobs=N_JOBS)
        # _gram_sym is the exact method QSVM.fit uses to build its training
        # Gram. Called directly because fit() consumes the matrix internally.
        K_q = model._gram_sym(Ztr)
        rec = {
            "bandwidth": float(bw),
            "offdiag_std": float(kernel_diag.offdiag_std(K_q)),
            "alignment": float(kernel_diag.kernel_target_alignment(K_q, y_tr)),
        }
        rec["ratio_to_rbf"] = rec["offdiag_std"] / rbf_std
        records.append(rec)
        print(json.dumps(rec))

    out = {
        "dataset": "cic-malmem", "task": TASK, "n_components": NC,
        "encoding": ENCODING, "n_train": int(len(y_tr)),
        "rbf_gamma": float(gamma),
        "rbf_offdiag_std": float(rbf_std), "rbf_alignment": float(rbf_align),
        "records": records,
    }
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

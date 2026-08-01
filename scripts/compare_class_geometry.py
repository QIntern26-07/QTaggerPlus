"""Day 31: CIC vs EMBER 15-class feature geometry, and subsample fidelity.

Two questions on one data load:

1. Why does CIC 15-class QSVM sit at the random baseline while EMBER 15-class
   does not? Compare separability of the SAME projected features both
   frameworks consume.
2. Are the 1000-row quantum subsamples faithful to their full populations?
   Assigned in Week 2 Day 2, never executed; every quantum result in the
   project rests on it.

Run: uv run python scripts/compare_class_geometry.py
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.decomposition import PCA  # noqa: F401  (documents the pipeline's tail)
from sklearn.metrics import silhouette_score

from common import data, geometry
from common.preprocess import build_feature_pipeline

DATASETS = ("cic", "ember")
TASK = "multiclass"
NCS = (1, 3, 6, 8)
SEED = 42

# EMBER's reference population is the 18,014-row quantum subset, NOT the full
# 200,000-row test parquet. Two reasons, both hard requirements:
#   - `data/splits/ember_quantum_sample_idx_multiclass.json` was computed against
#     the balanced family pool that `ember_family_xy` derives from THIS file.
#     `ember_family_xy` re-downsamples every kept family to the smallest kept
#     count at load time, so a different input frame yields a differently sized
#     and differently ordered pool and the persisted indices would select the
#     wrong rows.
#   - the full parquet is a single row group of 200,000 x 2,384 (~1.9 GB
#     resident) that cannot be streamed; see w4_consolidated_report.md section 5.
EMBER_PARQUET = "data/ember/ember2018_quantum_subset.parquet"
CIC_CSV = "data/cic_malmem/Obfuscated-MalMem2022.csv"


def _load(dataset):
    if dataset == "ember":
        df = data.load_ember(EMBER_PARQUET)
    else:
        df = data.load_cic_malmem(CIC_CSV)
    X, y = data.task_xy(df, TASK, dataset=dataset)
    sample_path, folds_path = data.split_paths(dataset, TASK)
    sample_idx = data.load_sample_idx(sample_path)
    folds = data.load_folds(folds_path)
    return X, y, sample_idx, folds


def representativeness(X, y, sample_idx):
    """Subsample vs. full population — proportions, moments, per-feature KS."""
    X_full = X.to_numpy()
    y_full = y.to_numpy()
    return {
        **geometry.class_proportion_drift(y_full, y_full[sample_idx]),
        **geometry.feature_drift(X_full, X_full[sample_idx]),
        **geometry.ks_reject_count(X_full, X_full[sample_idx]),
    }


def geometry_by_nc(X, y, sample_idx, folds):
    """Separability per n_components, averaged over the outer folds.

    The pipeline is fit on TRAIN-fold rows only — same leakage discipline the
    models use, so the geometry measured is the geometry they actually saw.
    """
    Xs = X.to_numpy()[sample_idx]
    ys = y.to_numpy()[sample_idx]
    out = {}
    for nc in NCS:
        per_fold = []
        for train_idx, _ in folds:
            pipe = build_feature_pipeline(n_components=nc, seed=SEED)
            Ztr = pipe.fit_transform(Xs[train_idx])
            ytr = ys[train_idx]
            ratios = geometry.fisher_ratio(Ztr, ytr)
            labels, D = geometry.centroid_distances(Ztr, ytr)
            offdiag = D[~np.eye(len(labels), dtype=bool)]
            pca = pipe.named_steps["pca"]
            per_fold.append({
                "mean_fisher_ratio": float(np.mean(list(ratios.values()))),
                "min_fisher_ratio": float(np.min(list(ratios.values()))),
                "silhouette": float(silhouette_score(Ztr, ytr)),
                "mean_centroid_distance": float(offdiag.mean()),
                "min_centroid_distance": float(offdiag.min()),
                "cumulative_explained_variance": float(
                    pca.explained_variance_ratio_.sum()
                ),
            })
        out[nc] = {
            k: {"mean": float(np.mean([f[k] for f in per_fold])),
                "std": float(np.std([f[k] for f in per_fold]))}
            for k in per_fold[0]
        }
    return out


def main() -> int:
    report = {}
    for dataset in DATASETS:
        X, y, sample_idx, folds = _load(dataset)
        counts = y.to_numpy()[sample_idx]
        report[dataset] = {
            "source": EMBER_PARQUET if dataset == "ember" else CIC_CSV,
            "n_full_rows": int(len(X)),
            "n_subsample_rows": int(len(sample_idx)),
            "family_composition": {
                str(c): int((counts == c).sum()) for c in np.unique(counts)
            },
            "representativeness": representativeness(X, y, sample_idx),
            "geometry": geometry_by_nc(X, y, sample_idx, folds),
        }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

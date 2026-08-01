"""Class-separability and subsample-representativeness statistics.

Pure functions over already-projected feature matrices, so they can be unit
tested without touching a dataset. `scripts/compare_class_geometry.py` is the
CLI that feeds real folds through them.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp


def fisher_ratio(Z, y) -> dict[str, float]:
    """One-vs-rest Fisher discriminant ratio per class.

    Between-class scatter (squared distance between the class mean and the
    mean of everything else) over within-class scatter (summed per-feature
    variance of both sides). Higher means the class is more linearly
    separable from the rest in this projection.
    """
    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y)
    out: dict[str, float] = {}
    for c in np.unique(y):
        mask = y == c
        Zc, Zr = Z[mask], Z[~mask]
        between = float(np.sum((Zc.mean(axis=0) - Zr.mean(axis=0)) ** 2))
        within = float(Zc.var(axis=0, ddof=0).sum() + Zr.var(axis=0, ddof=0).sum())
        out[str(c)] = between / within if within > 0 else float("inf")
    return out


def centroid_distances(Z, y) -> tuple[list[str], np.ndarray]:
    """Pairwise Euclidean distances between class centroids.

    Returns (labels, D) with D symmetric and zero-diagonal, so a reader can
    see which specific family pairs sit on top of each other.
    """
    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y)
    classes = np.unique(y)
    centroids = np.vstack([Z[y == c].mean(axis=0) for c in classes])
    diff = centroids[:, None, :] - centroids[None, :, :]
    D = np.sqrt((diff ** 2).sum(axis=-1))
    return [str(c) for c in classes], D


def class_proportion_drift(y_full, y_sample) -> dict[str, float]:
    """Largest absolute per-class proportion difference, subsample vs. full."""
    y_full = np.asarray(y_full)
    y_sample = np.asarray(y_sample)
    classes = np.unique(np.concatenate([y_full, y_sample]))
    p_full = np.array([(y_full == c).mean() for c in classes])
    p_samp = np.array([(y_sample == c).mean() for c in classes])
    return {
        "n_classes": float(len(classes)),
        "max_abs_proportion_diff": float(np.abs(p_full - p_samp).max()),
    }


def feature_drift(X_full, X_sample) -> dict[str, float]:
    """Per-feature mean/std drift of a subsample against the full population.

    Mean drift is expressed in full-population standard deviations so it is
    comparable across features of different scales. Zero-variance features get
    a denominator of 1.0 rather than being dropped, so the count of features
    stays honest.

    The std ratio is reported over NON-CONSTANT features only, and the constant
    ones are counted separately. A feature that is constant in the full
    population is also constant in any subsample of it, giving a ratio of 0/1 —
    a deviation of exactly 1.0, the maximum a well-behaved feature could show —
    so including it makes a perfectly preserved feature look like the worst
    drift in the dataset. Measured on CIC, where `pslist.nprocs64bit` is
    constant, that artifact alone set `max_abs_std_ratio_dev` to 1.0.

    `n_zero_variance_sample` is reported too: a feature that varies in the
    population but is constant in the subsample is real information loss, not
    an artifact.
    """
    X_full = np.asarray(X_full, dtype=float)
    X_sample = np.asarray(X_sample, dtype=float)
    sd_full = X_full.std(axis=0, ddof=0)
    sd_sample = X_sample.std(axis=0, ddof=0)
    safe_sd = np.where(sd_full == 0, 1.0, sd_full)
    mean_z = np.abs(X_sample.mean(axis=0) - X_full.mean(axis=0)) / safe_sd
    varying = sd_full > 0
    if varying.any():
        std_ratio = sd_sample[varying] / sd_full[varying]
        max_std_dev = float(np.abs(std_ratio - 1.0).max())
    else:
        max_std_dev = 0.0
    return {
        "n_features": float(X_full.shape[1]),
        "max_abs_mean_z": float(mean_z.max()),
        "max_abs_std_ratio_dev": max_std_dev,
        "n_zero_variance_full": float((~varying).sum()),
        "n_zero_variance_sample": float((sd_sample == 0).sum()),
    }


def ks_reject_count(X_full, X_sample, alpha: float = 0.05) -> dict[str, float]:
    """Two-sample KS test per feature, subsample vs. full population.

    Reports the rejection count alongside what chance alone would produce
    (alpha * n_features), because a bare count is uninterpretable — some
    rejections are expected even from a perfectly faithful subsample.
    """
    X_full = np.asarray(X_full, dtype=float)
    X_sample = np.asarray(X_sample, dtype=float)
    pvalues = np.array([
        ks_2samp(X_full[:, j], X_sample[:, j]).pvalue
        for j in range(X_full.shape[1])
    ])
    n_features = X_full.shape[1]
    return {
        "n_features": float(n_features),
        "n_reject": float((pvalues < alpha).sum()),
        "expected_rejects_under_null": float(alpha * n_features),
    }

"""Fold-safe preprocessing and feature-selection pipeline.

Every transformer learns only in fit(), so fitting inside a CV train fold keeps
the held-out fold unseen (no leakage).
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class DropCorrelated(BaseEstimator, TransformerMixin):
    """Drop one column from each pair whose |Pearson corr| exceeds `threshold`."""

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        corr = np.corrcoef(X, rowvar=False)
        corr = np.nan_to_num(corr)  # constant cols -> nan -> 0
        n = corr.shape[0]
        to_drop = set()
        for i in range(n):
            for j in range(i + 1, n):
                if j in to_drop or i in to_drop:
                    continue
                if abs(corr[i, j]) > self.threshold:
                    to_drop.add(j)
        self.keep_idx_ = np.array([c for c in range(n) if c not in to_drop])
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, self.keep_idx_]


def build_feature_pipeline(
    corr_threshold: float = 0.95, variance_threshold: float = 0.0
) -> Pipeline:
    """Unfitted pipeline: variance filter -> correlation filter -> standardize."""
    return Pipeline(
        steps=[
            ("variance", VarianceThreshold(threshold=variance_threshold)),
            ("decorrelate", DropCorrelated(threshold=corr_threshold)),
            ("scale", StandardScaler()),
        ]
    )

"""Encoding-specific scaling applied on top of the shared PCA output.

Separates dimensionality reduction (common.preprocess PCA, fit once) from
encoding preparation (here), so the QSVM never re-runs PCA. Fit on the train
fold only.
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MinMaxScaler


class EncodingScaler:
    def __init__(self, encoding: str, n_qubits: int):
        self.encoding = encoding
        self.n_qubits = n_qubits
        self._mm: MinMaxScaler | None = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if self.encoding in ("angle", "iqp"):
            self._mm = MinMaxScaler(feature_range=(0.0, np.pi)).fit(X)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if self.encoding in ("angle", "iqp"):
            return np.clip(self._mm.transform(X), 0.0, np.pi)
        if self.encoding == "amplitude":
            dim = 2 ** self.n_qubits
            if X.shape[1] < dim:
                X = np.hstack([X, np.zeros((X.shape[0], dim - X.shape[1]))])
            else:
                X = X[:, :dim]
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return X / norms
        raise ValueError(f"unknown encoding: {self.encoding}")

    def fit_transform(self, X):
        return self.fit(X).transform(X)

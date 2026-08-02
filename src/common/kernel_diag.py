"""Kernel-geometry diagnostics for fidelity vs. classical kernels.

Answers the question feature geometry cannot: given the SAME projected
features, why does the fidelity kernel fail where the RBF kernel does not?
"""
from __future__ import annotations

import numpy as np


def kernel_target_alignment(K, y) -> float:
    """Frobenius alignment between a Gram matrix and the ideal label kernel.

    <K, YY^T>_F / (||K||_F * ||YY^T||_F), where the ideal kernel is 1 for
    same-class pairs and 0 otherwise. Works unchanged for multiclass: the
    ideal kernel is a block-diagonal indicator, not a sign matrix. 1.0 means
    the kernel reproduces the label structure exactly; near 0 means it carries
    no label information.
    """
    K = np.asarray(K, dtype=float)
    y = np.asarray(y)
    ideal = (y[:, None] == y[None, :]).astype(float)
    denom = float(np.linalg.norm(K) * np.linalg.norm(ideal))
    if denom == 0.0:
        return 0.0
    return float(np.sum(K * ideal) / denom)


def offdiag_std(K) -> float:
    """Standard deviation of the strict upper-triangle entries of a Gram matrix.

    The concentration diagnostic: as off-diagonal entries collapse toward a
    constant, the Gram approaches a scaled identity and the SVM has nothing to
    separate on. Matches QSVM.gram_offdiag_std (src/quantum/qsvm.py:186), so a
    freshly built Gram can be cross-checked against the value MLflow logged.
    """
    K = np.asarray(K, dtype=float)
    off = K[np.triu_indices_from(K, k=1)]
    return float(off.std()) if off.size else 0.0

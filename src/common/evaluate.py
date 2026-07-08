"""Evaluation metrics, confusion-matrix plot, and timing helper."""
from __future__ import annotations

import time
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")  # headless: safe for scripts/CI
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_proba, task: str) -> dict:
    """Return the standard metric dict for one set of predictions."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    if task == "binary":
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    else:
        metrics["roc_auc"] = roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="macro"
        )
    return metrics


def confusion_matrix_figure(y_true, y_pred, labels=None):
    """Return a matplotlib Figure of the confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=labels, ax=ax)
    fig.tight_layout()
    return fig


def timed(fn: Callable, *args, **kwargs) -> tuple[Any, float]:
    """Call fn and return (result, elapsed_seconds)."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start


def softmax(S, axis: int = 1):
    """Numerically stable softmax."""
    S = np.asarray(S, dtype=float)
    S = S - S.max(axis=axis, keepdims=True)
    e = np.exp(S)
    return e / e.sum(axis=axis, keepdims=True)


def auc_scores(model, X, task: str):
    """Scores for roc_auc_score, honest about model type.

    Trees expose predict_proba; SVM/QSVM (probability disabled) expose only
    decision_function. sklearn SVC(probability=False) makes hasattr(model,
    "predict_proba") return False, so this branch is reliable. Multiclass SVM
    decision values are softmaxed so rows sum to 1, which roc_auc_score requires
    for multi_class="ovr".
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return proba[:, 1] if task == "binary" else proba
    scores = model.decision_function(X)
    if task == "binary":
        return scores
    return softmax(scores, axis=1)

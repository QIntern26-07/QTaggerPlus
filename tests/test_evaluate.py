import numpy as np
from common import evaluate


def test_compute_metrics_perfect_binary():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.2, 0.8])
    m = evaluate.compute_metrics(y_true, y_pred, y_proba, task="binary")
    assert set(m) == {"accuracy", "precision", "recall", "f1_macro", "mcc", "roc_auc"}
    assert m["accuracy"] == 1.0
    assert m["roc_auc"] == 1.0


def test_compute_metrics_multiclass_roc_auc_present():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 1])
    proba = np.eye(3)[y_pred]  # fake one-hot proba
    m = evaluate.compute_metrics(y_true, y_pred, proba, task="multiclass")
    assert 0.0 <= m["roc_auc"] <= 1.0
    assert 0.0 <= m["f1_macro"] <= 1.0


def test_timed_returns_result_and_positive_duration():
    result, secs = evaluate.timed(sum, [1, 2, 3])
    assert result == 6
    assert secs >= 0.0

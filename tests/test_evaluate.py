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


# Tests for softmax and auc_scores helpers
from common.evaluate import auc_scores, softmax


def test_softmax_rows_sum_to_one():
    S = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    P = softmax(S, axis=1)
    assert np.allclose(P.sum(axis=1), 1.0)


class _TreeLike:
    def predict_proba(self, X):
        n = len(X)
        return np.tile([0.3, 0.7], (n, 1))


class _SVMLikeBinary:
    # no predict_proba attribute at all
    def decision_function(self, X):
        return np.linspace(-1, 1, len(X))


class _SVMLikeMulti:
    def decision_function(self, X):
        return np.tile([0.1, 0.5, 0.4], (len(X), 1))


def test_auc_scores_tree_binary_returns_positive_class_column():
    s = auc_scores(_TreeLike(), np.zeros((4, 2)), "binary")
    assert s.ndim == 1 and np.allclose(s, 0.7)


def test_auc_scores_svm_binary_uses_decision_function():
    s = auc_scores(_SVMLikeBinary(), np.zeros((5, 2)), "binary")
    assert s.ndim == 1 and s[0] == -1.0


def test_auc_scores_svm_multiclass_softmaxed():
    P = auc_scores(_SVMLikeMulti(), np.zeros((3, 2)), "multiclass")
    assert P.shape == (3, 3) and np.allclose(P.sum(axis=1), 1.0)

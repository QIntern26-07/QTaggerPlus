import numpy as np
import pytest
from common import evaluate


def test_compute_metrics_perfect_binary():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.2, 0.8])
    m = evaluate.compute_metrics(y_true, y_pred, y_proba, task="binary")
    assert set(m) == {"accuracy", "precision", "recall", "f1_macro",
                      "f1_weighted", "mcc", "roc_auc"}
    assert m["accuracy"] == 1.0
    assert m["roc_auc"] == 1.0


def test_compute_metrics_includes_weighted_f1_distinct_from_macro():
    # Imbalanced on purpose: 3 samples of class 0, 1 of class 1, so the
    # weighted average is pulled toward class 0 and must differ from macro.
    y_true = np.array([0, 0, 0, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])
    m = evaluate.compute_metrics(y_true, y_pred, y_proba, task="binary")
    assert "f1_weighted" in m
    # class 0: P=1.0, R=2/3 -> F1=0.8 ; class 1: P=0.5, R=1.0 -> F1=2/3
    # macro = 0.7333..., weighted = (3*0.8 + 1*(2/3)) / 4 = 0.76666...
    assert m["f1_macro"] == pytest.approx(0.733333, abs=1e-5)
    assert m["f1_weighted"] == pytest.approx(0.766667, abs=1e-5)


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


# Tests for per_class_f1 and aggregate_metrics helpers
from common.evaluate import aggregate_metrics, per_class_f1


def test_per_class_f1_perfect_predictions():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = y_true.copy()
    scores = per_class_f1(y_true, y_pred)
    assert scores == {"0": 1.0, "1": 1.0, "2": 1.0}


def test_per_class_f1_respects_explicit_labels_even_when_absent_from_fold():
    # class "2" never appears in this fold's y_true/y_pred; an explicit label
    # set still reports a (zero) score for it instead of silently dropping the
    # key, so per-fold dicts stay uniform across a CV sweep.
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    scores = per_class_f1(y_true, y_pred, labels=[0, 1, 2])
    assert set(scores) == {"0", "1", "2"}
    assert scores["2"] == 0.0


def test_aggregate_metrics_mean_and_std():
    metrics_list = [{"f1_macro": 1.0}, {"f1_macro": 0.5}]
    agg = aggregate_metrics(metrics_list)
    assert agg["f1_macro_mean"] == 0.75
    assert agg["f1_macro_std"] == 0.25

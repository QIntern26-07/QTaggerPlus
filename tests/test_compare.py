import numpy as np
from classical import compare


def test_paired_ttest_detects_clear_difference():
    a = np.array([0.90, 0.91, 0.92, 0.93, 0.94])
    b = np.array([0.80, 0.81, 0.82, 0.83, 0.84])
    res = compare.paired_ttest(a, b)
    assert res["pvalue"] < 0.05


def test_wilcoxon_returns_pvalue():
    a = np.array([0.90, 0.91, 0.92, 0.93, 0.95])
    b = np.array([0.80, 0.82, 0.83, 0.81, 0.84])
    res = compare.wilcoxon(a, b)
    assert 0.0 <= res["pvalue"] <= 1.0


def test_mcnemar_identical_predictions_not_significant():
    y = np.array([0, 1, 0, 1, 1, 0])
    pred = np.array([0, 1, 1, 1, 0, 0])
    res = compare.mcnemar(y, pred, pred)  # same predictions -> no disagreement
    assert res["pvalue"] == 1.0

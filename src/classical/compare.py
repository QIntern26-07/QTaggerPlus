"""Paired statistical tests for classical-vs-quantum model comparison."""
from __future__ import annotations

import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar


def paired_ttest(scores_a, scores_b) -> dict:
    """Paired t-test across CV-fold scores of two models."""
    stat, p = stats.ttest_rel(scores_a, scores_b)
    return {"statistic": float(stat), "pvalue": float(p)}


def wilcoxon(scores_a, scores_b) -> dict:
    """Wilcoxon signed-rank test across CV-fold scores (non-parametric)."""
    stat, p = stats.wilcoxon(scores_a, scores_b)
    return {"statistic": float(stat), "pvalue": float(p)}


def mcnemar(y_true, pred_a, pred_b) -> dict:
    """McNemar's test on a shared test set: disagreement between two models."""
    y_true = np.asarray(y_true)
    correct_a = np.asarray(pred_a) == y_true
    correct_b = np.asarray(pred_b) == y_true
    # 2x2: [[both wrong, a wrong/b right],[a right/b wrong, both right]]
    n01 = int(np.sum(~correct_a & correct_b))
    n10 = int(np.sum(correct_a & ~correct_b))
    table = [[int(np.sum(~correct_a & ~correct_b)), n01],
             [n10, int(np.sum(correct_a & correct_b))]]
    result = sm_mcnemar(table, exact=True)
    return {"statistic": float(result.statistic), "pvalue": float(result.pvalue)}

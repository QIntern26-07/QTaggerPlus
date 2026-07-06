import numpy as np
import pytest
from classical import compare
from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar


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


def test_mcnemar_disagreement_with_reference():
    """Test mcnemar with genuine disagreement between models.

    This test catches table-orientation bugs (swapped n01/n10 or transposed rows/cols)
    that the identical-predictions test cannot detect.

    Setup:
      y_true = [1, 1, 1, 1, 0, 0, 0, 0]
      pred_a = [1, 1, 1, 0, 0, 0, 0, 0]  # 7 correct, 1 wrong
      pred_b = [0, 0, 0, 0, 1, 1, 1, 1]  # 4 correct, 4 wrong (opposite errors)

    Contingency table (after computing correct_a and correct_b):
      correct_a = [T, T, T, F, T, T, T, T]
      correct_b = [F, F, F, F, T, T, T, T]
      Discordant pairs:
        - Index 0: a correct, b wrong (n10)
        - Index 1: a correct, b wrong (n10)
        - Index 2: a correct, b wrong (n10)
        - Index 4: a correct, b wrong (n10)
        - Index 5: a correct, b wrong (n10)
        - Index 6: a correct, b wrong (n10)
        - Index 7: a correct, b wrong (n10)
      Concordant:
        - Index 3: a wrong, b wrong (n00)
      So: n10=7, n01=0, n00=1, n11=0
      Table: [[1, 0], [7, 0]]
    """
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    pred_a = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    pred_b = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    res = compare.mcnemar(y_true, pred_a, pred_b)

    # With genuine disagreement (n10 >> n01), pvalue should be significant
    assert 0.0 <= res["pvalue"] <= 1.0
    assert res["pvalue"] < 1.0

    # Verify against reference computation to catch row/column transposition
    # Build table manually: [[n00, n01], [n10, n11]]
    n00 = 1  # both wrong
    n01 = 0  # a wrong, b right
    n10 = 7  # a right, b wrong
    n11 = 0  # both right
    reference_table = [[n00, n01], [n10, n11]]
    reference_result = sm_mcnemar(reference_table, exact=True)

    # pvalue should match reference (within floating-point tolerance)
    assert res["pvalue"] == pytest.approx(reference_result.pvalue)

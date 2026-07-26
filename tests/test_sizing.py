import pytest

from common.sizing import solve_for_budget


def test_exact_power_law_inversion():
    # t(n) = t_ref * (n/n_ref)**exponent; at exponent 2, 4x the budget -> 2x n.
    assert solve_for_budget(budget_s=45.6, ref_n=400, ref_s=11.4, exponent=2.0) == 800


def test_smaller_budget_shrinks_n():
    assert solve_for_budget(budget_s=11.4, ref_n=400, ref_s=11.4, exponent=2.0) == 400
    assert solve_for_budget(budget_s=2.85, ref_n=400, ref_s=11.4, exponent=2.0) == 200


def test_rejects_nonpositive_budget():
    with pytest.raises(ValueError):
        solve_for_budget(budget_s=0, ref_n=400, ref_s=11.4, exponent=2.0)

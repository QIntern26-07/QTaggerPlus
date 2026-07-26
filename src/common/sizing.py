"""Solve the measured QSVM kernel scaling law for a target runtime budget.

The Day 1 profiling report fitted kernel build time against sample count as a
power law, t(n) = t_ref * (n / n_ref) ** exponent, with exponent ~2.0 (the
O(n^2) pair count) and t_ref ~11.4 s at n_ref=400, n_components=2. Inverting it
gives the largest subsample that fits a chosen per-Gram budget.
"""
from __future__ import annotations


def solve_for_budget(budget_s: float, ref_n: int, ref_s: float,
                     exponent: float) -> int:
    """Largest n whose predicted kernel build time stays within budget_s."""
    if budget_s <= 0:
        raise ValueError(f"budget_s must be positive, got {budget_s}")
    if ref_s <= 0 or ref_n <= 0 or exponent <= 0:
        raise ValueError("reference measurements and exponent must be positive")
    return int(ref_n * (budget_s / ref_s) ** (1.0 / exponent))

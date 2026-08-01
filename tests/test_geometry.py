import numpy as np
import pytest

from common import geometry


def test_fisher_ratio_high_for_separated_classes():
    rng = np.random.default_rng(0)
    Z = np.vstack([rng.normal(0.0, 0.1, (50, 2)),
                   rng.normal(10.0, 0.1, (50, 2))])
    y = np.array([0] * 50 + [1] * 50)
    ratios = geometry.fisher_ratio(Z, y)
    assert set(ratios) == {"0", "1"}
    assert ratios["0"] > 100.0


def test_fisher_ratio_near_zero_for_overlapping_classes():
    rng = np.random.default_rng(0)
    Z = rng.normal(0.0, 1.0, (200, 2))
    y = np.array([0, 1] * 100)  # labels carry no signal
    ratios = geometry.fisher_ratio(Z, y)
    assert ratios["0"] < 0.1


def test_centroid_distances_symmetric_with_zero_diagonal():
    Z = np.array([[0.0, 0.0], [0.0, 0.0], [3.0, 4.0], [3.0, 4.0]])
    y = np.array([0, 0, 1, 1])
    labels, D = geometry.centroid_distances(Z, y)
    assert labels == ["0", "1"]
    assert D.shape == (2, 2)
    assert np.allclose(D, D.T)
    assert np.allclose(np.diag(D), 0.0)
    assert D[0, 1] == pytest.approx(5.0)


def test_class_proportion_drift_zero_for_a_perfectly_stratified_sample():
    y_full = np.array([0] * 80 + [1] * 20)
    y_sample = np.array([0] * 8 + [1] * 2)
    out = geometry.class_proportion_drift(y_full, y_sample)
    assert out["max_abs_proportion_diff"] == pytest.approx(0.0, abs=1e-12)


def test_class_proportion_drift_detects_a_skewed_sample():
    y_full = np.array([0] * 80 + [1] * 20)
    y_sample = np.array([0] * 2 + [1] * 8)  # inverted
    out = geometry.class_proportion_drift(y_full, y_sample)
    assert out["max_abs_proportion_diff"] == pytest.approx(0.6, abs=1e-12)


def test_feature_drift_small_for_a_random_subsample():
    rng = np.random.default_rng(0)
    X_full = rng.normal(0.0, 1.0, (5000, 4))
    X_sample = X_full[rng.choice(5000, 1000, replace=False)]
    out = geometry.feature_drift(X_full, X_sample)
    assert out["max_abs_mean_z"] < 0.2
    assert out["max_abs_std_ratio_dev"] < 0.2


def test_feature_drift_ignores_constant_features_in_the_std_ratio():
    # Column 1 is constant in the population, so it is constant in any subsample
    # too. Counting it would report a deviation of exactly 1.0 -- a perfectly
    # preserved feature looking like the worst drift present.
    rng = np.random.default_rng(0)
    X_full = np.column_stack([rng.normal(0.0, 1.0, 5000), np.full(5000, 7.0)])
    X_sample = X_full[rng.choice(5000, 1000, replace=False)]
    out = geometry.feature_drift(X_full, X_sample)
    assert out["n_zero_variance_full"] == 1.0
    assert out["max_abs_std_ratio_dev"] < 0.2


def test_feature_drift_counts_features_the_subsample_flattened():
    # Varies in the population but constant in this subsample: real information
    # loss, not an artifact, so it must be counted and not silently absorbed.
    X_full = np.zeros((100, 2))
    X_full[:, 0] = np.arange(100)
    X_full[97:, 1] = 1.0            # non-zero only in the tail
    X_sample = X_full[:50]          # tail excluded -> column 1 flat
    out = geometry.feature_drift(X_full, X_sample)
    assert out["n_zero_variance_full"] == 0.0
    assert out["n_zero_variance_sample"] == 1.0


def test_ks_reject_count_reports_the_null_expectation():
    rng = np.random.default_rng(0)
    X_full = rng.normal(0.0, 1.0, (2000, 20))
    X_sample = X_full[rng.choice(2000, 500, replace=False)]
    out = geometry.ks_reject_count(X_full, X_sample, alpha=0.05)
    assert out["n_features"] == 20
    assert out["expected_rejects_under_null"] == pytest.approx(1.0)
    assert out["n_reject"] <= 5  # a faithful subsample should not blow past chance

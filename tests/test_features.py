import numpy as np
from classical.features import DropCorrelated, build_feature_pipeline


def test_drop_correlated_removes_duplicate_column():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(100, 1))
    X = np.hstack([base, base * 2.0, rng.normal(size=(100, 1))])  # col0,col1 collinear
    dc = DropCorrelated(threshold=0.95).fit(X)
    Xt = dc.transform(X)
    assert Xt.shape[1] == 2  # one of the collinear pair dropped


def test_drop_correlated_learns_on_fit_not_transform():
    rng = np.random.default_rng(1)
    X_train = np.hstack([rng.normal(size=(50, 1))] * 2)  # perfectly correlated
    dc = DropCorrelated(threshold=0.95).fit(X_train)
    X_test = rng.normal(size=(10, 2))
    # transform must use columns learned at fit time, so output width is stable
    assert dc.transform(X_test).shape[1] == 1


def test_drop_correlated_handles_constant_column():
    """Constant columns (all same value) produce NaN correlation; should not be dropped."""
    rng = np.random.default_rng(3)
    # Create array: constant col, varying col1, varying col2
    constant = np.full((100, 1), 5.0)  # all 5.0
    varying1 = rng.normal(size=(100, 1))
    varying2 = rng.normal(size=(100, 1))
    X = np.hstack([constant, varying1, varying2])
    # Fit with threshold 0.95; constant col's correlations are NaN -> 0
    # so it won't be dropped for being correlated with other cols
    dc = DropCorrelated(threshold=0.95).fit(X)
    Xt = dc.transform(X)
    # All 3 columns should survive (constant col not dropped)
    assert Xt.shape[1] == 3


def test_pipeline_output_is_standardized():
    rng = np.random.default_rng(2)
    X = rng.normal(loc=5.0, scale=3.0, size=(200, 4))
    pipe = build_feature_pipeline()
    Xt = pipe.fit_transform(X)
    # after StandardScaler, remaining columns have ~0 mean, ~1 std
    assert np.allclose(Xt.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(Xt.std(axis=0), 1, atol=1e-6)

import numpy as np
import pandas as pd
from classical import data


def _y():
    # 40 samples, imbalanced 3-class to exercise stratification
    return pd.Series([0] * 20 + [1] * 12 + [2] * 8)


def test_folds_are_a_partition_of_all_indices():
    y = _y()
    folds = data.make_outer_folds(y, n_splits=5, seed=42)
    assert len(folds) == 5
    test_union = np.concatenate([test for _, test in folds])
    assert sorted(test_union.tolist()) == list(range(len(y)))


def test_folds_are_deterministic():
    y = _y()
    a = data.make_outer_folds(y, n_splits=5, seed=42)
    b = data.make_outer_folds(y, n_splits=5, seed=42)
    for (_, ta), (_, tb) in zip(a, b):
        assert np.array_equal(ta, tb)


def test_folds_round_trip_through_disk(tmp_path):
    y = _y()
    folds = data.make_outer_folds(y, n_splits=5, seed=42)
    path = tmp_path / "folds.json"
    data.save_folds(folds, str(path))
    loaded = data.load_folds(str(path))
    for (tr, te), (ltr, lte) in zip(folds, loaded):
        assert np.array_equal(tr, ltr)
        assert np.array_equal(te, lte)

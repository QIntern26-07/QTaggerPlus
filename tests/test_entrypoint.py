import numpy as np

from classical import data
from classical.__main__ import aggregate_records


def test_save_predictions_round_trips_through_disk(tmp_path):
    records = [
        {
            "fold": 0,
            "test_idx": np.array([1, 3, 5]),
            "y_true": np.array([0, 1, 1]),
            "y_pred": np.array([0, 1, 0]),
        },
        {
            "fold": 1,
            "test_idx": np.array([2, 4]),
            "y_true": np.array([1, 0]),
            "y_pred": np.array([1, 0]),
        },
    ]
    path = tmp_path / "random_forest_binary_predictions.npz"
    data.save_predictions(records, str(path))
    loaded = data.load_predictions(str(path))

    expected_fold_ids = np.array([0, 0, 0, 1, 1])
    expected_test_idx = np.array([1, 3, 5, 2, 4])
    expected_y_true = np.array([0, 1, 1, 1, 0])
    expected_y_pred = np.array([0, 1, 0, 1, 0])

    assert np.array_equal(loaded["fold_ids"], expected_fold_ids)
    assert np.array_equal(loaded["test_idx"], expected_test_idx)
    assert np.array_equal(loaded["y_true"], expected_y_true)
    assert np.array_equal(loaded["y_pred"], expected_y_pred)


def test_aggregate_records_computes_mean_and_std():
    records = [
        {"model": "random_forest", "task": "binary",
         "metrics": {"accuracy": 0.90, "f1_macro": 0.88}},
        {"model": "random_forest", "task": "binary",
         "metrics": {"accuracy": 0.94, "f1_macro": 0.92}},
    ]
    agg = aggregate_records(records)
    assert abs(agg["accuracy_mean"] - 0.92) < 1e-9
    assert abs(agg["f1_macro_mean"] - 0.90) < 1e-9
    assert agg["accuracy_std"] > 0

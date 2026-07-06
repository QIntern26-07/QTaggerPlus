import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from classical import data, run


def _toy_xy(task):
    n_classes = 2 if task == "binary" else 3
    X, y = make_classification(
        n_samples=120, n_features=8, n_informative=5, n_redundant=2,
        n_classes=n_classes, random_state=42,
    )
    return pd.DataFrame(X), pd.Series(y)


def test_run_nested_cv_binary_smoke():
    X, y = _toy_xy("binary")
    folds = data.make_outer_folds(y, n_splits=3, seed=42)
    records = run.run_nested_cv(
        X, y, task="binary", name="random_forest",
        folds=folds, n_trials=2, seed=42, use_wandb=False,
    )
    assert len(records) == 3
    for rec in records:
        assert set(["accuracy", "f1_macro", "roc_auc"]).issubset(rec["metrics"])
        assert "train_time_sec" in rec and "inference_time_sec" in rec
        assert len(rec["y_pred"]) == len(rec["test_idx"])

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


def test_tune_and_fit_forces_single_threaded_model_during_inner_cv(monkeypatch):
    """The inner-CV objective must build models with n_jobs=1 (cross_val_score
    already parallelizes across inner_splits folds); only the single final refit
    should use the caller's configured n_jobs value."""
    calls = []
    real_make_model = run.make_model

    def spy_make_model(name, params, task, seed=42, n_jobs=-1):
        calls.append(n_jobs)
        return real_make_model(name, params, task, seed=seed, n_jobs=n_jobs)

    monkeypatch.setattr(run, "make_model", spy_make_model)

    X, y = _toy_xy("binary")
    run.tune_and_fit(
        "random_forest", "binary", X.values, y.values,
        n_trials=2, inner_splits=2, seed=42, n_jobs=4,
    )

    assert len(calls) >= 2  # at least the inner-objective calls plus the final refit
    assert calls[:-1] == [1] * (len(calls) - 1)  # every inner-CV model: n_jobs=1
    assert calls[-1] == 4  # the single final refit: the configured n_jobs


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


def test_run_nested_cv_logs_confusion_matrix_and_predictions_to_wandb_offline(
    tmp_path, monkeypatch
):
    """use_wandb=True should not crash and should exercise the confusion-matrix
    image + predictions table logging path, using WANDB_MODE=offline so no
    network/credentials are required."""
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    X, y = _toy_xy("binary")
    folds = data.make_outer_folds(y, n_splits=2, seed=42)
    records = run.run_nested_cv(
        X, y, task="binary", name="random_forest",
        folds=folds, n_trials=1, seed=42, use_wandb=True,
        dataset_name="unit-test-dataset",
    )
    assert len(records) == 2
    # offline mode still writes local run directories under wandb/
    assert (tmp_path / "wandb").exists()

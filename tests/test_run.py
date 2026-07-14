import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from common import data
from classical import run


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
        folds=folds, n_trials=2, seed=42, use_mlflow=False,
    )
    assert len(records) == 3
    for rec in records:
        assert set(["accuracy", "f1_macro", "roc_auc"]).issubset(rec["metrics"])
        assert "fit_time_sec" in rec and "tune_time_sec" in rec
        assert "inference_time_sec" in rec
        assert len(rec["y_pred"]) == len(rec["test_idx"])


def test_evaluate_fold_reports_separate_tune_and_fit_times():
    import numpy as np
    from classical.run import evaluate_fold
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 8))
    y = (X[:, 0] > 0).astype(int)
    folds_idx = (np.arange(0, 45), np.arange(45, 60))
    rec = evaluate_fold(
        "random_forest", "binary", X, y, folds_idx[0], folds_idx[1],
        n_trials=2, seed=0, inner_splits=2, n_jobs=1,
    )
    assert "fit_time_sec" in rec and "tune_time_sec" in rec
    assert "inference_time_sec" in rec
    assert rec["fit_time_sec"] <= rec["tune_time_sec"] + rec["fit_time_sec"]


def test_run_nested_cv_logs_to_mlflow(tmp_path):
    import numpy as np, mlflow
    from classical.run import run_nested_cv
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 6)); y = (X[:, 0] > 0).astype(int)
    folds = [(np.arange(0, 45), np.arange(45, 60))]
    uri = f"file:{tmp_path / 'mlruns'}"
    run_nested_cv(
        X, y, "binary", "random_forest", folds, n_trials=2, seed=0,
        use_mlflow=True, tracking_uri=uri, inner_splits=2, n_jobs=1,
        extra_params={"framework": "classical", "n_components": 6},
    )
    mlflow.set_tracking_uri(uri)
    exp = mlflow.get_experiment_by_name("qtaggerplus")
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    assert len(runs) == 2  # one sweep-level parent run + one nested per-fold run
    child = runs[runs["tags.mlflow.parentRunId"].notna()].iloc[0]
    parent = runs[runs["tags.mlflow.parentRunId"].isna()].iloc[0]
    assert child["params.framework"] == "classical"
    assert "metrics.fit_time_sec" in runs.columns
    assert "metrics.f1_macro_mean" in runs.columns
    assert pd.notna(parent["metrics.f1_macro_mean"])
    assert any(c.startswith("metrics.f1_class_") for c in runs.columns)

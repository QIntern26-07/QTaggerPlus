import mlflow
from common.tracking import run


def test_run_logs_params_and_metrics_to_local_store(tmp_path):
    uri = f"file:{tmp_path / 'mlruns'}"
    with run("test-exp", "r1", {"framework": "classical", "n_components": 1},
             tracking_uri=uri) as rec:
        rec.log_metrics({"f1_macro": 0.9, "fit_time_sec": 0.01})

    mlflow.set_tracking_uri(uri)
    exp = mlflow.get_experiment_by_name("test-exp")
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    assert len(runs) == 1
    assert runs.iloc[0]["params.framework"] == "classical"
    assert runs.iloc[0]["metrics.f1_macro"] == 0.9


def test_run_logs_tags(tmp_path):
    uri = f"file:{tmp_path / 'mlruns'}"
    with run("test-exp", "r1", {"framework": "classical"}, tracking_uri=uri,
             tags={"sweep": "random_forest-binary-nc6"}):
        pass

    mlflow.set_tracking_uri(uri)
    exp = mlflow.get_experiment_by_name("test-exp")
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    assert runs.iloc[0]["tags.sweep"] == "random_forest-binary-nc6"


def test_run_nested_creates_child_under_active_parent(tmp_path):
    uri = f"file:{tmp_path / 'mlruns'}"
    with run("test-exp", "parent", {"framework": "classical"}, tracking_uri=uri) as parent_log:
        with run("test-exp", "child", {"framework": "classical"}, tracking_uri=uri,
                 nested=True) as child_log:
            child_log.log_metrics({"f1_macro": 0.8})
        parent_log.log_metrics({"f1_macro_mean": 0.8})

    mlflow.set_tracking_uri(uri)
    exp = mlflow.get_experiment_by_name("test-exp")
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    assert len(runs) == 2
    child = runs[runs["tags.mlflow.parentRunId"].notna()].iloc[0]
    parent = runs[runs["tags.mlflow.parentRunId"].isna()].iloc[0]
    assert child["metrics.f1_macro"] == 0.8
    assert parent["metrics.f1_macro_mean"] == 0.8

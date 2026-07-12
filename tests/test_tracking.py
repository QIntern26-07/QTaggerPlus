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

import numpy as np
from quantum.run import run_quantum_cv, evaluate_fold_quantum, DEFAULT_GRID


def _data(n=40, d=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] > 0).astype(int)
    return X, y


def test_evaluate_fold_quantum_returns_timing_and_metrics():
    X, y = _data()
    tr, te = np.arange(0, 30), np.arange(30, 40)
    grid = {"encoding": ["angle"], "bandwidth": [None], "C": [1.0],
            "class_weight": [None]}
    rec = evaluate_fold_quantum(X, y, "binary", tr, te, n_components=1,
                                grid=grid, seed=0)
    for k in ("fit_time_sec", "tune_time_sec", "inference_time_sec",
              "kernel_build_train_s", "kernel_build_test_s"):
        assert k in rec
    assert "f1_macro" in rec["metrics"]


def _multiclass_data(n=45, d=5, n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = rng.integers(0, n_classes, size=n)
    return X, y


def test_evaluate_fold_quantum_multiclass_returns_valid_metrics():
    # Covers the auc_scores softmax path x compute_metrics(multi_class="ovr")
    # combination at the CV-integration level (evaluate_fold_quantum/
    # run_quantum_cv), which every other test in this file skips by only
    # exercising task="binary". This is the seam Finding 3's stratification
    # fix (binary vs. multiclass label) most needs covered.
    X, y = _multiclass_data()
    tr, te = np.arange(0, 35), np.arange(35, 45)
    grid = {"encoding": ["angle"], "bandwidth": [None], "C": [1.0],
            "class_weight": [None]}
    rec = evaluate_fold_quantum(X, y, "multiclass", tr, te, n_components=1,
                                grid=grid, seed=0)
    assert "f1_macro" in rec["metrics"]


def test_run_quantum_cv_logs_to_mlflow(tmp_path):
    import mlflow
    X, y = _data()
    folds = [(np.arange(0, 30), np.arange(30, 40))]
    uri = f"file:{tmp_path / 'mlruns'}"
    grid = {"encoding": ["angle"], "bandwidth": [None], "C": [1.0],
            "class_weight": [None]}
    run_quantum_cv(X, y, "binary", folds, n_components=1, grid=grid, seed=0,
                   use_mlflow=True, tracking_uri=uri)
    mlflow.set_tracking_uri(uri)
    exp = mlflow.get_experiment_by_name("qtaggerplus")
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    assert len(runs) == 2  # one sweep-level parent run + one nested per-fold run
    assert (runs["params.framework"] == "quantum").all()
    child = runs[runs["tags.mlflow.parentRunId"].notna()].iloc[0]
    assert child["params.n_qubits"] == "1"
    assert "metrics.fit_time_sec" in runs.columns
    assert "metrics.f1_macro_mean" in runs.columns


def test_run_quantum_cv_logs_encoding_param_single_encoding(tmp_path):
    import mlflow
    X, y = _data()
    folds = [(np.arange(0, 30), np.arange(30, 40))]
    uri = f"file:{tmp_path / 'mlruns'}"
    grid = {"encoding": ["angle"], "bandwidth": [None], "C": [1.0],
            "class_weight": [None]}
    run_quantum_cv(X, y, "binary", folds, n_components=1, grid=grid, seed=0,
                   use_mlflow=True, tracking_uri=uri, n_jobs=2)
    mlflow.set_tracking_uri(uri)
    exp = mlflow.get_experiment_by_name("qtaggerplus")
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    parent = runs[runs["tags.mlflow.parentRunId"].isna()].iloc[0]
    assert parent["params.encoding"] == "angle"
    assert parent["params.n_jobs"] == "2"
    assert parent["tags.sweep"] == "qsvm-binary-nc1-angle"


def test_run_quantum_cv_logs_encoding_param_joint_grid(tmp_path):
    import mlflow
    X, y = _data()
    folds = [(np.arange(0, 30), np.arange(30, 40))]
    uri = f"file:{tmp_path / 'mlruns'}"
    grid = {"encoding": ["angle", "iqp"], "bandwidth": [None], "C": [1.0],
            "class_weight": [None]}
    run_quantum_cv(X, y, "binary", folds, n_components=1, grid=grid, seed=0,
                   use_mlflow=True, tracking_uri=uri)
    mlflow.set_tracking_uri(uri)
    exp = mlflow.get_experiment_by_name("qtaggerplus")
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    parent = runs[runs["tags.mlflow.parentRunId"].isna()].iloc[0]
    assert parent["params.encoding"] == "joint"
    assert parent["tags.sweep"] == "qsvm-binary-nc1-joint"


def test_timing_probe_single_fit_reports_kernel_time():
    import numpy as np
    from quantum.run import timing_probe
    rng = np.random.default_rng(0)
    X = rng.normal(size=(24, 4)); y = (X[:, 0] > 0).astype(int)
    rec = timing_probe(X, y, "binary", n_components=1, encoding="angle", seed=0)
    assert rec["kernel_build_train_s"] >= 0.0
    assert "fit_time_sec" in rec and "inference_time_sec" in rec


def test_quantum_predictions_path_encodes_task_nc_and_encoding():
    from quantum.__main__ import predictions_path

    assert predictions_path("ember", "binary", 3, ["iqp"]) == (
        "results/ember/qsvm_binary_nc3_iqp_predictions.npz"
    )
    # several encodings in one sweep -> "joint", matching run.py's tag rule
    assert predictions_path("cic", "multiclass", 6, ["angle", "iqp"]) == (
        "results/cic/qsvm_multiclass_nc6_joint_predictions.npz"
    )

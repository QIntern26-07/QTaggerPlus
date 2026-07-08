def test_package_imports():
    import classical
    assert classical.__doc__ is not None


def test_classical_and_quantum_run_same_fold_n_components_1(tmp_path):
    import numpy as np
    from classical.run import run_nested_cv
    from quantum.run import run_quantum_cv
    rng = np.random.default_rng(0)
    X = rng.normal(size=(48, 5)); y = (X[:, 0] > 0).astype(int)
    folds = [(np.arange(0, 36), np.arange(36, 48))]
    c = run_nested_cv(X, y, "binary", "random_forest", folds, n_trials=2,
                      seed=0, inner_splits=2, n_jobs=1, n_components=1)
    q = run_quantum_cv(X, y, "binary", folds, n_components=1,
                       grid={"encoding": ["angle"], "bandwidth": [None],
                             "C": [1.0], "class_weight": [None]}, seed=0)
    assert c[0]["metrics"]["f1_macro"] >= 0.0
    assert q[0]["metrics"]["f1_macro"] >= 0.0
    # both scored the identical held-out indices
    assert list(c[0]["test_idx"]) == list(q[0]["test_idx"])

import numpy as np
from quantum.qsvm import QSVM


def _toy(n=24, d=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] + 0.3 * rng.normal(size=n) > 0).astype(int)
    # scale into [0, pi] as the pipeline would
    X = (X - X.min(0)) / (np.ptp(X, axis=0) + 1e-9) * np.pi
    return X, y


def test_gram_is_symmetric_psd_unit_diagonal():
    X, _ = _toy()
    m = QSVM(encoding="angle", n_components=2)
    K = m.gram(X, X)
    assert np.allclose(K, K.T, atol=1e-6)
    assert np.allclose(np.diag(K), 1.0, atol=1e-6)
    eigs = np.linalg.eigvalsh(K)
    assert eigs.min() > -1e-6  # PSD


def test_fit_predict_runs_and_has_no_proba():
    X, y = _toy()
    m = QSVM(encoding="angle", n_components=2).fit(X, y)
    preds = m.predict(X)
    assert set(np.unique(preds)).issubset({0, 1})
    assert not hasattr(m, "predict_proba")
    scores = m.decision_function(X)
    assert scores.shape[0] == len(X)


def test_test_gram_is_cached_between_predict_and_decision_function():
    X, y = _toy()
    m = QSVM(encoding="angle", n_components=2).fit(X, y)
    Xte = X[:5]
    m.predict(Xte)
    evals_after_predict = m.kernel_evals
    m.decision_function(Xte)  # same object -> should reuse cached gram
    assert m.kernel_evals == evals_after_predict


def test_multiclass_task_fits():
    rng = np.random.default_rng(1)
    X = rng.uniform(0, np.pi, size=(30, 2))
    y = rng.integers(0, 3, size=30)
    m = QSVM(encoding="angle", n_components=2, task="multiclass").fit(X, y)
    assert m.decision_function(X).shape == (30, 3)

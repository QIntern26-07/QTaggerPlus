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


def test_test_gram_is_cached_between_predict_and_decision_function_float32():
    # Regression test: np.asarray(X, dtype=float) allocates a NEW array when X
    # isn't already float64 (e.g. float32), so an id()-after-conversion cache
    # key would miss even though the caller passed the same object to
    # predict() then decision_function(). The cache must key off the caller's
    # original object identity, captured before any dtype conversion.
    X, y = _toy()
    m = QSVM(encoding="angle", n_components=2).fit(X, y)
    Xte = X[:5].astype(np.float32)
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


def test_gram_matches_across_batch_sizes():
    # Batched execution (day1_quantum_profiling_report.md) chunks pairs by
    # batch_size instead of one Python-loop iteration per pair. Chunking must
    # not change the result: force multiple chunks (batch_size=3) against a
    # single-shot batch (batch_size large) and require identical Grams.
    X, _ = _toy(n=10)
    m_chunked = QSVM(encoding="angle", n_components=2, batch_size=3)
    m_single = QSVM(encoding="angle", n_components=2, batch_size=10_000)
    K_chunked = m_chunked.gram(X, X)
    K_single = m_single.gram(X, X)
    assert np.allclose(K_chunked, K_single, atol=1e-10)


def test_gram_sym_matches_across_batch_sizes():
    X, _ = _toy(n=10)
    m_chunked = QSVM(encoding="angle", n_components=2, batch_size=3)
    m_single = QSVM(encoding="angle", n_components=2, batch_size=10_000)
    K_chunked = m_chunked._gram_sym(X)
    K_single = m_single._gram_sym(X)
    assert np.allclose(K_chunked, K_single, atol=1e-10)


def test_kernel_evals_counts_pairs_regardless_of_batch_size():
    # kernel_evals is used as an MLflow metric (quantum/run.py); it should
    # count logical pairs, not QNode dispatch calls, so it stays comparable
    # across batch_size choices.
    X, _ = _toy(n=8)
    m = QSVM(encoding="angle", n_components=2, batch_size=3)
    m._gram_sym(X)
    n = len(X)
    assert m.kernel_evals == n * (n - 1) // 2


def test_gram_single_row_does_not_crash():
    X, _ = _toy(n=1)
    m = QSVM(encoding="angle", n_components=2)
    K = m.gram(X, X)
    assert K.shape == (1, 1)

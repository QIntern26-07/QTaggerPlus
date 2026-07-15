"""Fidelity-kernel quantum SVM.

kappa(x, x') = |<phi(x')|phi(x)>|^2, realized as feature_map then
adjoint(feature_map) and reading probs[0] (R4: PennyLane
tutorial_kernel_based_training). The kernel is exactly PSD under analytic
expectation, so SVC(kernel="precomputed") is valid.

Design choices for honest timing:
- Train and test Gram matrices are cached; predict and decision_function reuse
  one test-Gram computation (no double kernel eval).
- probability=False; AUC uses decision_function upstream (common.evaluate).
- One warmup kernel eval before any timed Gram build, so lightning.qubit graph
  construction does not pollute small-n timings.
- Gram pairs are evaluated via QNode parameter broadcasting in chunks of
  `batch_size`, not a Python double loop: one QNode dispatch per chunk instead
  of one per pair, which is where the per-call Python/tape-construction
  overhead was concentrated (day1_quantum_profiling_report.md measured ~O(n^2)
  wall time on the unbatched loop; batching cuts the per-pair constant, not
  the O(n^2) pair count itself).
"""
from __future__ import annotations

import time

import numpy as np
import pennylane as qml
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.svm import SVC

from quantum.encoding import default_bandwidth, feature_map, n_qubits_for

# Below this many pairs, the process-spawn overhead of parallel evaluation
# outweighs the gain, so small Grams (tests, tiny inner-CV folds) stay serial.
_PARALLEL_MIN_PAIRS = 20_000


def _resolve_device(preferred: str = "lightning.qubit") -> str:
    try:
        qml.device(preferred, wires=1)
        return preferred
    except Exception:
        return "default.qubit"


def _eval_kernel_chunk(X1, X2, encoding, n_qubits, bandwidth, batch_size, device_name):
    """Evaluate fidelity kernel[0] for paired rows of X1, X2 in a worker process.

    Builds its own device + QNode so nothing PennyLane-stateful has to be pickled
    across the process boundary — only plain arrays and scalars are sent. The
    kernel is deterministic, so this returns exactly what the serial path would
    for the same rows.
    """
    dev = qml.device(device_name, wires=n_qubits)
    wires = list(range(n_qubits))

    @qml.qnode(dev)
    def kernel(x1, x2):
        feature_map(x1, wires, encoding, bandwidth)
        qml.adjoint(feature_map)(x2, wires, encoding, bandwidth)
        return qml.probs(wires=wires)

    n = len(X1)
    out = np.empty(n)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        probs = kernel(X1[start:end], X2[start:end])
        out[start:end] = np.asarray(probs)[:, 0]
    return out


class QSVM:
    def __init__(self, encoding="angle", n_components=None, bandwidth=None,
                 C=1.0, class_weight=None, task="binary", device=None, seed=42,
                 batch_size=4096, n_jobs=-1, parallel_min_pairs=_PARALLEL_MIN_PAIRS):
        self.encoding = encoding
        self.n_components = n_components
        self.C = C
        self.class_weight = class_weight
        self.task = task
        self.seed = seed
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.parallel_min_pairs = parallel_min_pairs
        n_features = n_components if n_components is not None else 1
        self.n_qubits = n_qubits_for(encoding, n_features)
        self.bandwidth = (
            default_bandwidth(self.n_qubits) if bandwidth is None else bandwidth
        )
        self._wires = list(range(self.n_qubits))
        self._dev_name = device or _resolve_device()
        self._dev = qml.device(self._dev_name, wires=self.n_qubits)
        self._svc = SVC(
            kernel="precomputed", C=C, class_weight=class_weight,
            decision_function_shape="ovr", random_state=seed,
        )
        self._X_train = None
        self._cache_X = None
        self._cache_gram = None
        self.kernel_build_train_s = 0.0
        self.kernel_build_test_s = 0.0
        self.kernel_evals = 0

        @qml.qnode(self._dev)
        def kernel(x1, x2):
            feature_map(x1, self._wires, encoding, self.bandwidth)
            qml.adjoint(feature_map)(x2, self._wires, encoding, self.bandwidth)
            return qml.probs(wires=self._wires)

        self._kernel = kernel
        self._warm = False

    def _warmup(self, x):
        if not self._warm:
            _ = self._kernel(x, x)
            self._warm = True

    def _kernel_pairs(self, X1, X2):
        """Evaluate kernel[0] for paired rows of X1, X2.

        Small Grams run serially (one QNode dispatch per `batch_size` chunk).
        Large Grams (>= parallel_min_pairs) are split into `n_jobs` contiguous
        slices evaluated in worker processes, since lightning.qubit on these
        small circuits is single-threaded — the wall-clock cost is the sheer
        O(n^2) pair count, which parallelizes cleanly across cores. Results are
        concatenated in order, so the output is identical to the serial path.
        """
        n = len(X1)
        self.kernel_evals += n
        n_workers = effective_n_jobs(self.n_jobs)
        if n_workers == 1 or n < self.parallel_min_pairs:
            out = np.empty(n)
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                probs = self._kernel(X1[start:end], X2[start:end])
                out[start:end] = np.asarray(probs)[:, 0]
            return out

        bounds = np.linspace(0, n, n_workers + 1).astype(int)
        slices = [
            (bounds[i], bounds[i + 1])
            for i in range(n_workers)
            if bounds[i] < bounds[i + 1]
        ]
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_eval_kernel_chunk)(
                X1[s:e], X2[s:e], self.encoding, self.n_qubits, self.bandwidth,
                self.batch_size, self._dev_name,
            )
            for s, e in slices
        )
        return np.concatenate(results)

    def gram(self, A, B):
        """Full (non-symmetric) Gram matrix between rows of A and B."""
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        self._warmup(A[0])
        ia, ib = np.meshgrid(np.arange(len(A)), np.arange(len(B)), indexing="ij")
        ia, ib = ia.ravel(), ib.ravel()
        K = np.zeros((len(A), len(B)))
        if ia.size:
            K[:, :] = self._kernel_pairs(A[ia], B[ib]).reshape(len(A), len(B))
        return K

    def _gram_sym(self, A):
        A = np.asarray(A, dtype=float)
        self._warmup(A[0])
        n = len(A)
        K = np.eye(n)
        iu, ju = np.triu_indices(n, k=1)
        if iu.size:
            vals = self._kernel_pairs(A[iu], A[ju])
            K[iu, ju] = vals
            K[ju, iu] = vals
        return K

    def fit(self, X, y):
        self._X_train = np.asarray(X, dtype=float)
        self._warmup(self._X_train[0])
        t0 = time.perf_counter()
        K = self._gram_sym(self._X_train)
        self.kernel_build_train_s = time.perf_counter() - t0
        # Concentration health-check (R3): if off-diagonal spread -> 0, the Gram
        # approaches the identity and the SVM cannot learn. Log this per fold.
        off = K[np.triu_indices_from(K, k=1)]
        self.gram_offdiag_std = float(off.std()) if off.size else 0.0
        self._svc.fit(K, y)
        return self

    def _test_gram(self, X):
        if X is self._cache_X and self._cache_gram is not None:
            return self._cache_gram
        X_arr = np.asarray(X, dtype=float)
        t0 = time.perf_counter()
        K = self.gram(X_arr, self._X_train)
        self.kernel_build_test_s = time.perf_counter() - t0
        self._cache_X = X
        self._cache_gram = K
        return K

    def predict(self, X):
        return self._svc.predict(self._test_gram(X))

    def decision_function(self, X):
        return self._svc.decision_function(self._test_gram(X))

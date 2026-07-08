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
"""
from __future__ import annotations

import time

import numpy as np
import pennylane as qml
from sklearn.svm import SVC

from quantum.encoding import default_bandwidth, feature_map, n_qubits_for


def _resolve_device(preferred: str = "lightning.qubit") -> str:
    try:
        qml.device(preferred, wires=1)
        return preferred
    except Exception:
        return "default.qubit"


class QSVM:
    def __init__(self, encoding="angle", n_components=None, bandwidth=None,
                 C=1.0, class_weight=None, task="binary", device=None, seed=42):
        self.encoding = encoding
        self.n_components = n_components
        self.C = C
        self.class_weight = class_weight
        self.task = task
        self.seed = seed
        n_features = n_components if n_components is not None else 1
        self.n_qubits = n_qubits_for(encoding, n_features)
        self.bandwidth = (
            default_bandwidth(self.n_qubits) if bandwidth is None else bandwidth
        )
        self._wires = list(range(self.n_qubits))
        dev_name = device or _resolve_device()
        self._dev = qml.device(dev_name, wires=self.n_qubits)
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

    def gram(self, A, B):
        """Full (non-symmetric) Gram matrix between rows of A and B."""
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        self._warmup(A[0])
        K = np.empty((len(A), len(B)))
        for i in range(len(A)):
            for j in range(len(B)):
                K[i, j] = float(self._kernel(A[i], B[j])[0])
                self.kernel_evals += 1
        return K

    def _gram_sym(self, A):
        A = np.asarray(A, dtype=float)
        self._warmup(A[0])
        n = len(A)
        K = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                v = float(self._kernel(A[i], A[j])[0])
                K[i, j] = K[j, i] = v
                self.kernel_evals += 1
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

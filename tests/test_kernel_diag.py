import numpy as np
import pytest

from common import kernel_diag


def test_alignment_is_one_for_the_ideal_kernel():
    y = np.array([0, 0, 1, 1])
    ideal = (y[:, None] == y[None, :]).astype(float)
    assert kernel_diag.kernel_target_alignment(ideal, y) == pytest.approx(1.0)


def test_alignment_lower_for_a_constant_kernel_than_the_ideal_one():
    y = np.array([0, 0, 1, 1])
    ideal = (y[:, None] == y[None, :]).astype(float)
    constant = np.ones((4, 4))
    assert kernel_diag.kernel_target_alignment(constant, y) < \
        kernel_diag.kernel_target_alignment(ideal, y)


def test_alignment_handles_a_multiclass_label_set():
    y = np.array([0, 1, 2, 0, 1, 2])
    ideal = (y[:, None] == y[None, :]).astype(float)
    assert kernel_diag.kernel_target_alignment(ideal, y) == pytest.approx(1.0)


def test_alignment_zero_for_an_all_zero_kernel():
    y = np.array([0, 1])
    assert kernel_diag.kernel_target_alignment(np.zeros((2, 2)), y) == 0.0


def test_offdiag_std_ignores_the_diagonal():
    # Diagonal varies wildly, off-diagonal is constant -> std must be 0.
    K = np.array([[5.0, 0.3, 0.3],
                  [0.3, 99.0, 0.3],
                  [0.3, 0.3, -7.0]])
    assert kernel_diag.offdiag_std(K) == pytest.approx(0.0)


def test_offdiag_std_matches_numpy_on_the_upper_triangle():
    rng = np.random.default_rng(0)
    K = rng.normal(size=(6, 6))
    K = (K + K.T) / 2
    expected = K[np.triu_indices_from(K, k=1)].std()
    assert kernel_diag.offdiag_std(K) == pytest.approx(expected)

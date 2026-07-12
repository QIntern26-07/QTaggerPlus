import numpy as np
from quantum.preprocess import EncodingScaler


def test_angle_scaler_maps_into_0_pi():
    X = np.array([[-2.0, 5.0], [0.0, 1.0], [2.0, -5.0]])
    sc = EncodingScaler("angle", n_qubits=2).fit(X)
    Xt = sc.transform(X)
    assert Xt.min() >= 0.0 and Xt.max() <= np.pi + 1e-9


def test_amplitude_scaler_pads_and_normalizes():
    X = np.array([[1.0, 0.0], [0.0, 2.0]])  # 2 features -> pad to 2**2 = 4
    sc = EncodingScaler("amplitude", n_qubits=2).fit(X)
    Xt = sc.transform(X)
    assert Xt.shape == (2, 4)
    assert np.allclose(np.linalg.norm(Xt, axis=1), 1.0)


def test_transform_clips_test_values_into_range():
    Xtr = np.array([[0.0], [1.0]])
    sc = EncodingScaler("angle", n_qubits=1).fit(Xtr)
    Xte = sc.transform(np.array([[10.0], [-10.0]]))
    assert Xte.min() >= 0.0 and Xte.max() <= np.pi + 1e-9

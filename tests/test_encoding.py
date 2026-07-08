import numpy as np
import pennylane as qml
from quantum.encoding import ENCODINGS, feature_map, n_qubits_for, default_bandwidth


def test_n_qubits_for_alignment():
    assert n_qubits_for("angle", 6) == 6
    assert n_qubits_for("iqp", 6) == 6
    assert n_qubits_for("amplitude", 4) == 2  # ceil(log2(4))


def test_default_bandwidth_shrinks_with_qubits():
    assert default_bandwidth(4) < default_bandwidth(1)


def test_fidelity_of_identical_inputs_is_one():
    for enc in ENCODINGS:
        n = n_qubits_for(enc, 4)
        dev = qml.device("default.qubit", wires=n)
        wires = list(range(n))

        @qml.qnode(dev)
        def kernel(x1, x2, enc=enc, wires=wires):
            feature_map(x1, wires, enc)
            qml.adjoint(feature_map)(x2, wires, enc)
            return qml.probs(wires=wires)

        x = np.array([0.2, 0.4, 0.6, 0.8])
        assert np.isclose(kernel(x, x)[0], 1.0, atol=1e-6)

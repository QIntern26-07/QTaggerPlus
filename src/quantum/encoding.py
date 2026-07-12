"""Quantum data encodings (feature maps) for fidelity-kernel QSVM.

Three candidates, all sharing one `feature_map` so the QSVM kernel is
encoding-agnostic:

- angle    : one feature per qubit, low entanglement. Simple baseline; the
             "geometric redundancy" reference point (R2).
- iqp      : ZZ-interaction feature map (qml.IQPEmbedding). Entangled,
             conjectured classically hard; empirically best on malware (R1).
- amplitude: packs 2**n features into n qubits; the fidelity kernel reduces to
             ~|cos|^2, a near-classical contrast (watch rank collapse, R2).

Basis encoding is intentionally excluded (Hamming-geometry mismatch on
continuous PCA features, R2).

`bandwidth` scales inputs before rotation. Fidelity/angle kernels exponentially
concentrate as qubits grow; bandwidth ~ n_qubits**-0.5 counteracts this (R3:
arXiv:2206.06686). R4: PennyLane tutorial_kernel_based_training.

References:
- R1: arXiv:2510.06803 — IQP/ZZ feature maps empirically best on malware.
- R2: Sci. Reports s41598-026-39392-9 — geometric redundancy / rank collapse
  in angle and amplitude encodings; basis encoding Hamming-geometry mismatch.
- R3: arXiv:2206.06686 — "Bandwidth Enables Generalization in Quantum Kernel
  Models"; bandwidth scaling counteracts kernel concentration.
"""
from __future__ import annotations

import math

import numpy as np
import pennylane as qml

ENCODINGS = ("angle", "iqp", "amplitude")


def n_qubits_for(encoding: str, n_components: int) -> int:
    """Qubit count for a given post-PCA dimensionality."""
    if encoding == "amplitude":
        return max(1, math.ceil(math.log2(max(1, n_components))))
    return n_components


def default_bandwidth(n_qubits: int) -> float:
    """Bandwidth scaling c = n_qubits**-0.5 (R3, alpha = 1/2)."""
    return float(n_qubits) ** -0.5


def feature_map(x, wires, encoding: str = "angle", bandwidth: float = 1.0) -> None:
    """Apply the chosen embedding in-place inside a QNode."""
    x = np.asarray(x, dtype=float)
    if encoding == "angle":
        qml.AngleEmbedding(bandwidth * x, wires=wires, rotation="Y")
        for i in range(len(wires) - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])
        qml.AngleEmbedding(bandwidth * x, wires=wires, rotation="Z")
    elif encoding == "iqp":
        qml.IQPEmbedding(bandwidth * x, wires=wires, n_repeats=1)
    elif encoding == "amplitude":
        qml.AmplitudeEmbedding(x, wires=wires, normalize=True, pad_with=0.0)
    else:
        raise ValueError(f"unknown encoding: {encoding}")

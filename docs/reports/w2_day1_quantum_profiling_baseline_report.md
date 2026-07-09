# Day 1 Quantum Profiling Report -- Pre-Batching Baseline

Frozen snapshot from before the batched-execution fix (`qsvm.py::gram`/`_gram_sym`
still using a Python double loop, one QNode dispatch per kernel pair). Kept for
comparison against [w2_day1_quantum_profiling_report.md](w2_day1_quantum_profiling_report.md),
which reflects the same sweep after batching was implemented.

## Findings

- **angle**: gate count scales roughly as qubits^1.2 (4 gates @ 1 qubits -> 46 gates @ 8 qubits).
- **iqp**: gate count scales roughly as qubits^1.5 (4 gates @ 1 qubits -> 88 gates @ 8 qubits).
- **amplitude**: gate count scales roughly as qubits^2.1 (1 gates @ 1 qubits -> 14 gates @ 3 qubits).
- `amplitude` embedding is left undecomposed by the simulator (1 `AmplitudeEmbedding` op) because state-vector simulators execute state prep natively; on real QPU hardware this would require an exponential-in-qubits gate decomposition, so simulator gate counts understate its true hardware cost.

- `lightning.qubit` is selected by `_resolve_device()` and is confirmed faster than `default.qubit` (1.50x speedup on 200 kernel pairs) -- the fallback is not silently picking the slower backend.

- Kernel-matrix train build time scales roughly as n_samples^2.1 across 50-400 samples (fit exponent; ~2.0 is the theoretical O(n^2) expectation for a symmetric Gram of uncached, non-batched kernel evals). This is the direct justification for subsample sizes in Day 2's EMBER/SOREL-20M dimensionality-reduction work, and the baseline batched execution (Day 1 efficiency fix) should reduce against.

## Circuit depth / gate count per encoding

| encoding | n_components | n_qubits | depth | n_gates | gate_types |
|---|---|---|---|---|---|
| angle | 1 | 1 | 4 | 4 | RY:1, RZ:1, Adjoint(RZ):1, Adjoint(RY):1 |
| angle | 2 | 2 | 6 | 10 | RY:2, CNOT:1, RZ:2, Adjoint(RZ):2, Adjoint(CNOT):1, Adjoint(RY):2 |
| angle | 4 | 4 | 10 | 22 | RY:4, CNOT:3, RZ:4, Adjoint(RZ):4, Adjoint(CNOT):3, Adjoint(RY):4 |
| angle | 8 | 8 | 18 | 46 | RY:8, CNOT:7, RZ:8, Adjoint(RZ):8, Adjoint(CNOT):7, Adjoint(RY):8 |
| iqp | 1 | 1 | 4 | 4 | Hadamard:1, RZ:1, Adjoint(RZ):1, Adjoint(Hadamard):1 |
| iqp | 2 | 2 | 6 | 10 | Hadamard:2, RZ:2, MultiRZ:1, Adjoint(MultiRZ):1, Adjoint(RZ):2, Adjoint(Hadamard):2 |
| iqp | 4 | 4 | 14 | 28 | Hadamard:4, RZ:4, MultiRZ:6, Adjoint(MultiRZ):6, Adjoint(RZ):4, Adjoint(Hadamard):4 |
| iqp | 8 | 8 | 30 | 88 | Hadamard:8, RZ:8, MultiRZ:28, Adjoint(MultiRZ):28, Adjoint(RZ):8, Adjoint(Hadamard):8 |
| amplitude | 1 | 1 | 1 | 1 | AmplitudeEmbedding:1 |
| amplitude | 2 | 1 | 2 | 2 | AmplitudeEmbedding:1, Adjoint(RY):1 |
| amplitude | 4 | 2 | 5 | 6 | AmplitudeEmbedding:1, Adjoint(CNOT):2, Adjoint(RY):3 |
| amplitude | 8 | 3 | 12 | 14 | AmplitudeEmbedding:1, Adjoint(CNOT):6, Adjoint(RY):7 |

## Simulator backend comparison

| backend | available | mean_s | std_s | n_pairs |
|---|---|---|---|---|
| lightning.qubit | yes | 0.2377 | 0.0041 | 200 |
| default.qubit | yes | 0.3566 | 0.0015 | 200 |

## Kernel-matrix cost vs. sample size (angle, n_components=2)

| n_samples | kernel_build_train_s | kernel_build_test_s | kernel_evals |
|---|---|---|---|
| 50 | 0.7369 | 0.3677 | 1180 |
| 100 | 2.9233 | 1.5189 | 4760 |
| 200 | 15.2676 | 7.5571 | 19120 |
| 400 | 60.6874 | 30.4825 | 76640 |

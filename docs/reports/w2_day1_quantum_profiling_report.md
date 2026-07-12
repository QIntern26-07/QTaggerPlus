# Day 1 Quantum Profiling Report

Post-batching measurements (`qsvm.py::gram`/`_gram_sym` now dispatch the kernel QNode once per chunk via parameter broadcasting instead of once per pair). Compare against the frozen pre-batching numbers in [w2_day1_quantum_profiling_baseline_report.md](w2_day1_quantum_profiling_baseline_report.md).

## Findings

- **angle**: gate count scales roughly as qubits^1.2 (4 gates @ 1 qubits -> 46 gates @ 8 qubits).
- **iqp**: gate count scales roughly as qubits^1.5 (4 gates @ 1 qubits -> 88 gates @ 8 qubits).
- **amplitude**: gate count scales roughly as qubits^2.1 (1 gates @ 1 qubits -> 14 gates @ 3 qubits).
- `amplitude` embedding is left undecomposed by the simulator (1 `AmplitudeEmbedding` op) because state-vector simulators execute state prep natively; on real QPU hardware this would require an exponential-in-qubits gate decomposition, so simulator gate counts understate its true hardware cost.

- `lightning.qubit` is selected by `_resolve_device()` and is confirmed faster than `default.qubit` (1.48x speedup on 200 kernel pairs) -- the fallback is not silently picking the slower backend.

- **angle** kernel-matrix train build time (batched execution, batch_size=4096 default) scales roughly as n_samples^2.2 across 50-400 samples (0.13s -> 13.85s).
- **iqp** kernel-matrix train build time (batched execution, batch_size=4096 default) scales roughly as n_samples^1.9 across 50-400 samples (0.22s -> 10.74s).
- Batching reduces the per-pair constant (measured ~4.4x on this machine at n=400, angle encoding, vs. the pre-batching double-loop), not the O(n^2) pair count itself -- the exponent stays ~2.0 as expected, since the number of kernel evaluations is unchanged. This scaling is the direct justification for subsample sizes in Day 2's EMBER/SOREL-20M work (sizing decision itself deferred).
- At n_components=2 (n_qubits=2), `angle` and `iqp` cost almost identically (both 10 gates per the circuit-spec table above, ~11.4s at n_samples=400 for either) -- `iqp`'s quadratic `MultiRZ` gate-count penalty only shows up at higher qubit counts (28 vs 6 gates at n_qubits=4; 88 vs 46 at n_qubits=8). If Day 2/3 multiclass work moves beyond n_components=2, re-run this sweep at the actual target n_components before picking an encoding on cost grounds alone.

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
| lightning.qubit | yes | 0.2384 | 0.0012 | 200 |
| default.qubit | yes | 0.3526 | 0.0010 | 200 |

## Kernel-matrix cost vs. sample size (n_components=2)

| encoding | n_samples | kernel_build_train_s | kernel_build_test_s | kernel_evals |
|---|---|---|---|---|
| angle | 50 | 0.1277 | 0.0655 | 1180 |
| angle | 100 | 0.6759 | 0.3394 | 4760 |
| angle | 200 | 3.0294 | 1.7414 | 19120 |
| angle | 400 | 13.8459 | 6.7858 | 76640 |
| iqp | 50 | 0.2170 | 0.0699 | 1180 |
| iqp | 100 | 0.6391 | 0.4289 | 4760 |
| iqp | 200 | 3.4187 | 1.3521 | 19120 |
| iqp | 400 | 10.7419 | 5.3689 | 76640 |

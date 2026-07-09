# Quantum (Team C) TODO

Deferred/open items that fell out of Week 2 Day 1 work but were explicitly
postponed. Check this file at the start of a session before assuming Day 1/2/3
quantum work is complete -- items here are known-open, not forgotten.

## Open

- [ ] **Day 2 EMBER/SOREL-20M subsample sizing decision.** We have the
  kernel-cost-vs-n_samples scaling data
  (`docs/reports/w2_day1_quantum_profiling_report.md`, both `angle` and `iqp`
  at n_components=2: exponent ~2.0, ~11.4s at n=400) but have NOT yet turned
  it into a concrete recommended max sample count for a runtime budget. Needs:
  pick a target per-fold runtime (e.g. under 5 min), solve the fitted power
  law for n, cross-check against class-balance requirements for the smallest
  malware family.
- [ ] **MLflow logging gaps** (queued for Day 3, see conversation on
  `src/common/tracking.py` / `src/quantum/run.py` / `src/classical/run.py`):
  - per-class F1 in `common/evaluate.py::compute_metrics` (currently only
    macro F1 is logged)
  - log `n_qubits` as an MLflow param in `quantum/run.py::_log_quantum_fold`
    (computed already, just not passed through)
  - tags support in `common/tracking.py::run()` (add `tags: dict | None`,
    `mlflow.set_tags`) so sweep runs can be grouped/filtered
  - nested parent/child MLflow runs so there's one aggregate mean/std row per
    CV sweep instead of only per-fold rows
- [ ] **VQC is not implemented and not planned to be implemented by this
  contributor.** Only QSVM exists in `src/quantum/`. Flag to the team/mentors
  that Week 2's "extend the existing binary QSVM/VQC architecture toward
  multi-class" (Day 2) is QSVM-only unless someone else picks this up.
- [ ] **Multiclass QSVM run — not yet done.** Everything run so far (Jul 9,
  `docs/reports/w2_Jul-9_experiments.md`) is `task=binary` only. Day 2's
  "extend the existing binary QSVM/VQC architecture toward multi-class
  (one-vs-rest or native multi-class circuit design)" and Day 3's "train the
  multi-class QSVM/VQC on the validated minimal representative subset and
  record first-pass results" are both still open. Note: `QSVM` already
  supports `task="multiclass"` via `SVC(decision_function_shape="ovr")` (see
  `tests/test_qsvm.py::test_multiclass_task_fits`) — the CLI just hasn't been
  run with `--tasks multiclass` against CIC-MalMem's real multiclass labels
  yet. Do this before claiming Day 2/3 multiclass work complete.

## Decided (no action needed, recorded for reference)

- Evaluation protocol skeleton for Team A's behavioral dataset drafted:
  `docs/day1_evaluation_protocol_skeleton.md`. Marked placeholder sections
  still need reconciling once Team A shares their real schema at the end of
  their Day 3 -- see that doc's "Open reconciliation points" section.

- `lightning.qubit` confirmed as the local-run simulator backend (faster than
  `default.qubit` by ~1.5x, see `_resolve_device()` in `src/quantum/qsvm.py`
  and the backend-comparison table in the w2 profiling report). No further
  backend re-benchmarking planned for now.
- Batched execution (kernel-matrix pair evaluation via QNode parameter
  broadcasting) is implemented in `src/quantum/qsvm.py::gram`/`_gram_sym`,
  tested in `tests/test_qsvm.py`, and measured (~4.4x speedup at n=400,
  angle encoding) in `docs/reports/w2_day1_quantum_profiling_report.md`.

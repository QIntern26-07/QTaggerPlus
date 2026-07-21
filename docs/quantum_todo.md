# Quantum (Team C) TODO

Deferred/open items that fell out of Week 2 Day 1 work but were explicitly
postponed. Check this file at the start of a session before assuming Day 1/2/3
quantum work is complete -- items here are known-open, not forgotten.

## Open

- [ ] **EMBER 2018 and SOREL-20M — not started at all.** `day6_7_classical_baselines_plan.md`
  scoped classical baselines (and, by extension, the quantum comparison) across
  three datasets, but every run so far (Week 1 classical baseline, all of Week
  2's quantum/classical sweeps) has only ever touched CIC-MalMem-2022. No
  loader exists in `src/common/data.py` for either dataset, and neither has
  been downloaded. This is the largest remaining gap against the original
  scope, not a small follow-up.
- [ ] **Day 2 EMBER/SOREL-20M subsample sizing decision.** Blocked on the item
  above — this was meant to size the quantum subsample for those datasets
  specifically. We have the kernel-cost-vs-n_samples scaling data
  (`docs/reports/w2_day1_quantum_profiling_report.md`, both `angle` and `iqp`
  at n_components=2: exponent ~2.0, ~11.4s at n=400) but have NOT yet turned
  it into a concrete recommended max sample count for a runtime budget. Needs:
  pick a target per-fold runtime (e.g. under 5 min), solve the fitted power
  law for n, cross-check against class-balance requirements for the smallest
  malware family (harder for SOREL/EMBER's own label distributions once known).
- [ ] **VQC is not implemented and not planned to be implemented by this
  contributor.** Only QSVM exists in `src/quantum/`. Flag to the team/mentors
  that Week 2's "extend the existing binary QSVM/VQC architecture toward
  multi-class" (Day 2) is QSVM-only unless someone else picks this up.

## Decided (no action needed, recorded for reference)

- **MLflow logging gaps — done** (2026-07-14, see
  `docs/reports/w2_jul-14_6-components_multi.md`'s intro): per-class F1
  (`common/evaluate.py::per_class_f1`), `n_qubits` logged as a quantum param,
  `tags` support in `common/tracking.py::run()`, and nested parent/child
  MLflow runs (one aggregate mean/std row per sweep). Unit-tested
  (`tests/test_evaluate.py`, `tests/test_tracking.py`,
  `tests/test_run.py`/`tests/test_quantum_run.py`) and exercised against the
  real `mlflow.db` via a classical smoke test before the full sweep.
- **Multiclass QSVM run — done** (2026-07-14, `docs/reports/w2_jul-14_6-components_multi.md`):
  ran `task=multiclass` (16-class `Category` family) end-to-end on CIC-MalMem
  at `n_components` 1-6, both 200- and 1000-sample subsamples, quantum-then-classical
  on shared splits. Finding: at 1000 samples both frameworks show a real,
  non-noise quality gap (classical clearly ahead of QSVM) and `iqp` overtakes
  `angle` monotonically as qubit count grows — worth carrying into any
  write-up. The 200-sample stage is a pipeline/logging validation run, not a
  result (rarest class had only ~5 members, per-fold F1 for it was close to a
  coin flip).

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

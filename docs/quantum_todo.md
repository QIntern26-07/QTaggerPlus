# Quantum (Team C) TODO

Deferred/open items that fell out of Week 2 Day 1 work but were explicitly
postponed. Check this file at the start of a session before assuming Day 1/2/3
quantum work is complete -- items here are known-open, not forgotten.

## Open

- [ ] **VQC is not implemented and not planned to be implemented by this
  contributor.** Only QSVM exists in `src/quantum/`. Flag to the team/mentors
  that Week 2's "extend the existing binary QSVM/VQC architecture toward
  multi-class" (Day 2) is QSVM-only unless someone else picks this up.

## Decided (no action needed, recorded for reference)

- **SOREL-20M — CLOSED, not deferred** (2026-08-01,
  `docs/reports/w5_day30_sorel_closeout.md`). The multiclass labelling design is
  delivered: dominant-tag argmax over the 11 raw behaviour-tag counts, ties broken
  by declared column order, all-zero-tag rows dropped, validated over 19.7M rows
  from `meta.db` (`docs/reports/w4_sorel_labelling_decision.md`). The feature store
  was never fetched and **no SOREL sweep was ever run** — `results/mlflow_runs.csv`
  holds zero `sorel-20m` rows. Do not read the labelling report as partial results.
  Neither disk (227 GiB free vs 71.6 GiB needed) nor RAM (LMDB is memory-mapped;
  reads page through the OS cache) is the blocker — it is the fixed 71.6 GiB
  download against the time available, unavoidable because LMDB has no key-level
  remote access. Resume conditions, source URI and the already-wired CLI surface
  are documented in the close-out; nothing needs re-investigating.

- **EMBER experiments — done** (2026-07-25/26,
  `docs/reports/w4_jul-25_ember_binary.md`, `docs/reports/w4_jul-26_ember_multiclass.md`).
  Full sweeps (n=1000, `angle`+`iqp` as separate single-encoding invocations, n_components
  1/3/6, quantum first then classical SVM replayed on the identical `--load-quantum-splits`
  rows/folds) ran for both binary and 15-class multiclass. Binary: classical SVM wins at
  every nc (0.5138 vs 0.6724 macro-F1 at nc=6) — overturns the Week 3 CIC reading of
  "quantum ties classical on binary," which held only because CIC's binary task is
  near-ceiling for every model. 15-class: classical SVM wins by the largest gap measured in
  the project (0.27 at nc=6), but QSVM does not collapse the way it did on CIC 15-class
  (recognizes 8-9/15 families even at nc=1, 14-15/15 from nc=3 up) — since EMBER's families
  are exactly balanced (955/class) and CIC's are not, this rules out class imbalance as the
  sole cause of CIC's collapse, though the actual cause remains unidentified.
- **Subsample sizing decision — done** (2026-07-25,
  `docs/reports/w4_subsample_sizing_decision.md`). Inverted the Day 1
  kernel-cost-vs-n_samples power law (`t(n) = t_ref * (n/n_ref)**exponent`,
  exponent~2.0, ref (400, 11.4s) at n_components=2) via
  `common.sizing.solve_for_budget` / `scripts/solve_sample_size.py`: 917/2051/2901
  max n at 1/5/10-minute per-Gram-build budgets. Cross-checked against the
  15-class balance floor (n >= 750, re-derived independently) — the floor only
  binds below a ~40s budget, so for the budgets actually considered the
  runtime ceiling binds, not the class floor. Recommendation: **n=900** for
  future EMBER/SOREL-20M sweeps at n_components=2, assuming a 60s/Gram-build
  budget; re-probe before reusing this number at a materially different qubit
  count.

- **EMBER 2018 support — done, binary + multiclass** (2026-07-20). Source:
  official EMBER tarball `ember_dataset_2018_2.tar.bz2`, test split
  `test_features.jsonl` (200k rows, 100k malware/100k benign, with sha256 and
  avclass labels). Vectorized via NEW vendored LIEF-free extractor
  `src/common/ember_vectorize.py` (feature version 2, 2381 dims) → cached
  parquet `data/ember/ember2018_test.parquet`. Binary: 100k/100k balanced.
  Multiclass: `data.ember_family_xy` with balanced top-15 avclass families
  (min_per_class=500, cap max_families=15, downsample-to-balance); smallest
  kept family wapomi=955. Mirrors CIC's malware-only design. Implemented:
  `common/data.py::load_ember`, `common/ember_vectorize.py` (vendored
  extractor), `task_xy(..., dataset=)` supports both binary and multiclass,
  `split_paths()` (ember files dataset-prefixed), `--dataset {cic,ember}` on
  both CLIs. End-to-end multiclass probe passed (~1.1s kernel at n_components=1,
  120 samples). Only full sweeps remain.

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

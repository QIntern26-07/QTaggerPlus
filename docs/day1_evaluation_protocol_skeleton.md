# Day 1 Evaluation Protocol Skeleton -- Team A Behavioral Dataset

Draft protocol for evaluating QSVM (and, if implemented, VQC) on Team A's
ransomware behavioral dataset (RANSAP/RANSMAP-style, built per the GUIDE-MLRan
methodology). Written against Team A's Day 1 schema lock -- **placeholder
fields are marked below and must be reconciled once Team A shares the
finalized schema at the end of their Day 3.**

## 1. Input contract (PLACEHOLDER -- reconcile against Team A's Day 3 schema)

- One row per executed ransomware sample.
- Feature groups expected per the Day 1 schema lock: API call sequences,
  registry operations, file-system operations, process activity, network
  activity. Exact column names/encodings (raw sequences vs. aggregated
  counts/n-grams) are Team A's call and not yet known -- this pipeline assumes
  a flat numeric feature vector per sample (i.e. sequences already
  vectorized/aggregated upstream by Team A, analogous to CIC-MalMem's
  pre-extracted numeric columns).
- Label column: malware family (multiclass) and/or malware/benign (binary),
  mirroring `common.data.build_xy`'s `y_binary`/`y_multiclass` split for
  CIC-MalMem.
- Row count and class balance are unknown until Team A's Day 3 capture
  statistics land; the sampling/fold strategy below (Section 3) assumes
  StratifiedKFold remains viable, i.e. every family has enough members for the
  chosen fold count -- re-check once real counts are known (this exact
  problem already forced a stratify-on-multiclass-not-binary fix in
  `quantum/__main__.py`, see the comment there).

## 2. Preprocessing chain

Reuse the existing shared pipeline unchanged -- do not build a
dataset-specific preprocessing path:

1. `common.preprocess.build_feature_pipeline`: variance filter -> correlation
   filter (drop |corr| > 0.95) -> `StandardScaler` -> optional PCA, fit on the
   train fold only (leakage-safe).
2. PCA `n_components` sets the quantum qubit budget directly
   (`quantum.encoding.n_qubits_for`) -- this is the single alignment point
   between classical and quantum models, so classical and quantum consume
   identical post-PCA features.
3. `quantum.preprocess.EncodingScaler`: encoding-specific scaling on top of
   PCA output (angle/iqp -> `MinMaxScaler` into `[0, pi]`; amplitude ->
   pad-and-normalize to unit norm), fit on train fold only.

Per the w2 Day 1 profiling report, `n_components=2` is the value actually
measured for kernel-cost scaling; if Team A's data needs more components to
retain signal, re-run `scripts/profile_quantum_day1.py`'s kernel-scaling sweep
at that target `n_components` before committing to a sample-size budget (this
is why `iqp` vs. `angle` cost parity at n_components=2 does NOT necessarily
hold at higher component counts, per that report's findings).

## 3. Split strategy

- `common.data.make_outer_folds`: stratified K-fold (default 5 splits,
  `shuffle=True`, fixed seed), stratified on whichever label (`binary` or
  `multiclass`) is being evaluated.
- Same fold indices must be reused across classical and quantum runs for a
  fair comparison -- mirrors the existing `data/splits/*_folds.json` +
  `data/splits/quantum_sample_idx.json` mechanism for CIC-MalMem
  (`quantum/__main__.py` persists the row subsample once, folds per task).
- If row count is large enough that QSVM's `O(n^2)` kernel cost is infeasible
  even after batching, apply the same stratified subsampling `quantum/__main__.py`
  already does for CIC-MalMem (`--max-samples`), sized per the Day 2 sizing
  decision (currently deferred, tracked in `docs/quantum_todo.md`).

## 4. Metrics

Reuse `common.evaluate.compute_metrics` and `confusion_matrix_figure`
unchanged:

- `accuracy`, `precision` (macro), `recall` (macro), `f1_macro`, `mcc`
- `roc_auc` (binary: direct; multiclass: OVR macro via `auc_scores`, which
  softmaxes QSVM's `decision_function` output since `probability=False`)
- confusion matrix figure, logged as an MLflow artifact
- Per-class F1 is a known current gap (not yet in `compute_metrics`), tracked
  as a Day 3 item in `docs/quantum_todo.md` -- add it before this protocol is
  used for real, since a 5-class comparison without per-class F1 hides which
  families the model actually fails on.

## 5. Timing / runtime metrics

Already instrumented and should be logged per run, unchanged:
`fit_time_sec`, `tune_time_sec`, `inference_time_sec`, `kernel_build_train_s`,
`kernel_build_test_s`, `kernel_evals`, `gram_offdiag_std` (Gram-collapse health
check). These are what Day 3's "benchmark the runtime improvement" and the
classical-vs-quantum comparison both depend on.

## 6. Multiclass handling

One-vs-rest via `SVC(decision_function_shape="ovr")`, matching the existing
`task="multiclass"` path in `quantum.qsvm.QSVM` and `quantum.run`. No native
multiclass quantum circuit design planned -- OVR is the adopted approach
unless Day 2's multiclass extension work decides otherwise.

## 7. Comparison baseline

Every quantum run must be compared against classical models (RF/XGBoost/
LightGBM/SVM per `src/classical`) trained on the **same** folds and the
**same** post-PCA features (i.e. classical also runs through
`build_feature_pipeline` with the same `n_components`), consistent with the
existing `--load-quantum-splits` / shared-sample-idx mechanism used for
CIC-MalMem.

## 8. Logging / run naming

MLflow experiment `"qtaggerplus"`, run name pattern
`{model}-{task}-nc{n_components}-fold{fold}` (existing convention in
`quantum/run.py::_log_quantum_fold` and `classical/run.py`). Params should
additionally include a `dataset` tag identifying this as Team A's behavioral
dataset (vs. `cic-malmem`), plus `n_qubits` and sweep-grouping tags once the
Day 3 MLflow logging fixes land (`docs/quantum_todo.md`).

## Open reconciliation points once Team A delivers their Day 3 schema

- Exact feature-column names/types and whether sequences arrive pre-vectorized
  or need vectorization in this pipeline.
- Real row counts and per-family class balance (determines fold count
  feasibility and whether subsampling is required before PCA).
- Whether "5 malware families" from the original project brief matches what
  Team A's sandbox runs actually produce.

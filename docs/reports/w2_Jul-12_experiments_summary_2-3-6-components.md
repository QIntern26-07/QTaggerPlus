# Week 2 Summary — n_components = 2, 3, 6 (binary task)

Consolidates the raw per-fold console logs already recorded in
[`w2_Jul-11_experiments_2-components.md`](./w2_Jul-11_experiments_2-components.md),
[`w2_Jul-11_experiments_3-components.md`](./w2_Jul-11_experiments_3-components.md), and
[`w2_Jul-12-experiments_6-components.md`](./w2_Jul-12-experiments_6-components.md) into one
comparison table, pulled from MLflow (`qtaggerplus` experiment, `sqlite:///mlflow.db`) rather than
the terminal logs, since MLflow also has `roc_auc`, `accuracy`, `mcc`, timing breakdown, and (for
QSVM) which encoding won each fold's inner-CV tuning — none of which prints to console.

All runs: binary task, 5-fold stratified outer CV, `--encodings angle iqp`, classical run via
`--load-quantum-splits` against the row subsample/folds the paired quantum run persisted (same
protocol as [`w2_Jul-9_experiments.md`](./w2_Jul-9_experiments.md)).

## Full results (mean across 5 outer folds)

### n_samples = 200

| n_components | model | f1_macro | roc_auc | accuracy | mcc | fit_time (s) | tune_time (s) | infer_time (s) |
|---|---|---|---|---|---|---|---|---|
| 2 | qsvm | 0.990 ± 0.020 | 0.981 | 0.990 | 0.980 | 3.16 | 5.29 | 1.73 |
| 2 | random_forest | 0.990 ± 0.020 | 0.987 | 0.990 | 0.980 | 0.48 | 14.05 | 0.047 |
| 2 | xgboost | 0.990 ± 0.020 | 0.982 | 0.990 | 0.980 | 1.03 | 1.79 | 0.004 |
| 2 | lightgbm | 0.990 ± 0.020 | 0.982 | 0.990 | 0.980 | 0.35 | 1.43 | 0.002 |
| 2 | svm | 0.985 ± 0.020 | 0.981 | 0.985 | 0.970 | 0.003 | 0.48 | 0.0003 |
| 3 | qsvm | 0.985 ± 0.020 | 0.982 | 0.985 | 0.970 | 4.43 | 8.31 | 2.19 |
| 3 | random_forest | 0.990 ± 0.020 | 0.990 | 0.990 | 0.980 | 0.44 | 15.65 | 0.043 |
| 3 | xgboost | 0.990 ± 0.020 | 0.994 | 0.990 | 0.980 | 0.61 | 2.04 | 0.006 |
| 3 | lightgbm | 0.985 ± 0.020 | 0.995 | 0.985 | 0.970 | 0.27 | 1.67 | 0.002 |
| 3 | svm | 0.980 ± 0.019 | 0.981 | 0.980 | 0.961 | 0.002 | 0.41 | 0.0003 |
| 6 | qsvm | 0.985 ± 0.030 | 0.990 | 0.985 | 0.970 | 7.72 | 18.32 | 3.97 |
| 6 | random_forest | 0.990 ± 0.020 | 0.998 | 0.990 | 0.980 | 0.43 | 11.21 | 0.043 |
| 6 | xgboost | 0.990 ± 0.020 | 0.998 | 0.990 | 0.980 | 0.13 | 2.18 | 0.006 |
| 6 | lightgbm | 0.985 ± 0.020 | 0.998 | 0.985 | 0.970 | 0.17 | 1.12 | 0.002 |
| 6 | svm | 0.985 ± 0.020 | 0.982 | 0.985 | 0.970 | 0.001 | 0.34 | 0.0003 |

### n_samples = 1000

| n_components | model | f1_macro | roc_auc | accuracy | mcc | fit_time (s) | tune_time (s) | infer_time (s) |
|---|---|---|---|---|---|---|---|---|
| 2 | qsvm | 0.990 ± 0.006 | 0.993 | 0.990 | 0.980 | 71.30 | 125.17 | 35.49 |
| 2 | random_forest | 0.992 ± 0.007 | 0.998 | 0.992 | 0.984 | 0.67 | 14.24 | 0.064 |
| 2 | xgboost | 0.992 ± 0.007 | 0.998 | 0.992 | 0.984 | 0.68 | 2.39 | 0.005 |
| 2 | lightgbm | 0.990 ± 0.003 | 0.998 | 0.990 | 0.980 | 0.74 | 2.60 | 0.002 |
| 2 | svm | 0.990 ± 0.006 | 0.997 | 0.990 | 0.980 | 0.004 | 0.53 | 0.001 |
| 3 | qsvm | 0.991 ± 0.006 | 0.991 | 0.991 | 0.982 | 114.77 | 209.42 | 54.93 |
| 3 | random_forest | 0.994 ± 0.007 | 0.999 | 0.994 | 0.988 | 0.72 | 16.69 | 0.072 |
| 3 | xgboost | 0.992 ± 0.007 | 0.997 | 0.992 | 0.984 | **83.61** ⚠ | 3.57 | 0.033 |
| 3 | lightgbm | 0.992 ± 0.007 | 0.997 | 0.992 | 0.984 | 0.95 | 3.13 | 0.003 |
| 3 | svm | 0.992 ± 0.006 | 0.998 | 0.992 | 0.984 | 0.004 | 0.55 | 0.001 |
| 6 | qsvm | 0.992 ± 0.007 | 0.994 | 0.992 | 0.984 | 168.96 | 465.11 | 85.38 |
| 6 | random_forest | 0.993 ± 0.005 | 0.999 | 0.993 | 0.986 | 0.73 | 15.52 | 0.071 |
| 6 | xgboost | 0.991 ± 0.007 | 1.000 | 0.991 | 0.982 | 0.13 | 2.44 | 0.006 |
| 6 | lightgbm | 0.993 ± 0.007 | 1.000 | 0.993 | 0.986 | 0.46 | 2.92 | 0.002 |
| 6 | svm | 0.989 ± 0.004 | 0.991 | 0.989 | 0.978 | 0.004 | 0.52 | 0.001 |

### QSVM encoding selected per fold (inner-CV winner, `angle` vs `iqp`)

| n_components | n_samples | fold0 | fold1 | fold2 | fold3 | fold4 |
|---|---|---|---|---|---|---|
| 2 | 200 | angle | angle | angle | angle | angle |
| 2 | 1000 | angle | angle | angle | angle | **iqp** |
| 3 | 200 | angle | angle | angle | angle | angle |
| 3 | 1000 | angle | angle | angle | angle | angle |
| 6 | 200 | angle | angle | angle | angle | **iqp** |
| 6 | 1000 | angle | angle | angle | angle | angle |

`iqp` wins only **2 of 30** folds across n_components ∈ {2, 3, 6} (both times the *last* fold of a
run, both times at 200 or 1000 but never consistently) — barely different from the 0/10 seen at
`n_components=1` in the Jul 9 report.

## Notable observations

- **`iqp` still shows no clear advantage even at 6 qubits.** The hypothesis raised in the Jul 9
  report ("re-run at `n_components=2` so `iqp` gets a fair shot, since it needs ≥2 qubits to have
  a feature pair to entangle") is **not strongly confirmed** — `iqp` wins only 2/30 folds,
  scattered across nc=2 and nc=6, with no trend toward winning more often as qubit count grows.
  This may be because the encoding tuning selects per fold by inner-CV macro-F1 on a very small
  set (200-1000 samples), so the gap between the two encodings isn't stable enough for `iqp`'s
  entangling structure to show a clear statistical edge; more data (more folds or larger samples)
  is needed before concluding `iqp` isn't useful here.
- **QSVM stays neck-and-neck with classical on quality, but its runtime gap widens fast with
  n_components, not just n_samples.** At 1000 samples, QSVM `fit_time` grows 71s (nc=2) → 115s
  (nc=3) → 169s (nc=6), and `tune_time` grows even steeper: 125s → 209s → 465s (~3.7x from nc=2 to
  nc=6). Meanwhile all four classical models stay under ~1s fit_time regardless of n_components —
  reaffirming the Jul 9 conclusion that the O(n²) cost of the quantum kernel, compounded by
  per-qubit circuit cost, is the real scaling bottleneck, not classification quality.
- **⚠ Anomaly: `xgboost` at `n_components=3, n_samples=1000` has a mean `fit_time_sec` of
  83.6s** — over 100x higher than the same model at nc=2 (0.68s) and nc=6 (0.13s) at the same
  1000 samples, even though `tune_time_sec` (3.57s) is entirely normal (this is a single final
  refit, not the tuning loop). Cross-referencing the log timestamps (`16:33:57`–`16:39:32`, Jul
  11) lines up exactly with the machine running low on RAM/swap (checked via `free -h` earlier in
  the session — swap sitting at 3.3-3.4Gi/4Gi) while `n_jobs=-1` was forking multiple
  `joblib.loky` workers concurrently. This is most likely a **system resource-contention artifact**
  (thrashing), not a genuine algorithmic effect of XGBoost at nc=3. Recommendation: don't use this
  number to compare XGBoost runtime across n_components; re-run the XGBoost/nc=3/1000-sample cell
  in isolation (with `--n-jobs` capped, on an otherwise idle machine) to get a clean number before
  it goes into any formal write-up.
- **Classification quality for both classical and quantum is essentially saturated near the
  ceiling (f1_macro 0.98–0.994) across the whole nc=2/3/6 range** — increasing n_components makes
  little difference to accuracy/f1/mcc for this binary task (the spread across nc is only about
  ±0.01, within 1 std of each model itself). This suggests the CIC-MalMem binary task is already
  "easy" even at 1-2 PCA dimensions, so to see clearer separation across n_components or between
  quantum/classical, moving to the multiclass task (harder) may be more informative than pushing
  n_components further on binary.
- **`svm` dips slightly and most noticeably at 1000 samples, nc=6** (f1_macro 0.989, roc_auc
  0.991) — a bit lower than itself at nc=2 (0.990) and nc=3 (0.992), and also lower than the other
  models at the same nc=6. The gap is small (within 1 std) so it may not be statistically
  meaningful, but it's worth watching if the trend repeats in future runs — possibly the default
  `C`/`gamma` search space isn't tuning as well as feature dimensionality grows.

## Methodology notes

- Data was pulled directly from MLflow (`mlflow.db`), not the console logs — console only prints
  `f1_macro`, so the `roc_auc`/`accuracy`/`mcc`/timing/encoding columns above are more complete
  than the raw log files.
- Because `data/splits/quantum_sample_idx.json` and `data/splits/cic_binary_quantum_folds.json`
  get overwritten on every quantum run, the runs in the tables above **do not share the same
  sample/fold set across different n_components** (each nc has its own 200- and 1000-sample
  subsample, freshly stratified-sampled each time) — comparisons are valid *within* each
  (quantum, classical) pair at a given nc, but comparisons *across* nc should be read as a trend
  estimate, not a strict apples-to-apples comparison on identical data.

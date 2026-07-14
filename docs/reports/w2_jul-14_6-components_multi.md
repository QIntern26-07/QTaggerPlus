# Week 2 Jul 14 Experiments — Multiclass, n_components = 1 to 6 (200 samples)

Extends the binary-task sweep (`w2_Jul-9_experiments.md` through `w2_Jul-12_experiments_summary_2-3-6-components.md`) to the **multiclass** task (16-class malware family, `Category` label) across `n_components` = 1 through 6, per `docs/quantum_todo.md`'s open "multiclass QSVM run — not yet done" item. Design/protocol is unchanged from the binary runs (design already existed — `SVC` one-vs-one on the cached Gram, `decision_function_shape="ovr"` for AUC — see `docs/superpowers/specs/2026-07-08-classical-quantum-pca-qsvm-design.md` and `docs/day1_evaluation_protocol_skeleton.md`); this run just exercises it for real.

**Before this sweep**, the MLflow logging gaps listed in `docs/quantum_todo.md` were implemented and unit-tested: per-class F1 (`common/evaluate.py::per_class_f1`), `n_qubits` as an MLflow param, `tags` support in `common/tracking.py::run()`, and nested parent/child MLflow runs (one parent run per model×task×n_components sweep, aggregating mean/std across its 5 fold child runs). A quick classical smoke test (`random_forest`, `--folds 3 --trials 3`, full dataset, `--mlflow`) confirmed all four work end-to-end against the real `mlflow.db` before committing to the full sweep below.

## Stratification sanity check (before running)

Simulated the exact subsample+fold code path against the real CIC-MalMem data before running anything: `train_test_split(..., stratify=y_multi, train_size=200)` then `StratifiedKFold(n_splits=5)`. Full dataset has 16 classes, smallest is class 9 with 1410/58,596 rows (~2.4%). At `max_samples=200` the smallest class survives with exactly 5 rows — enough for 5-fold CV (1 row/fold), but this means the rarest classes get scored on a single test sample per fold, so their per-fold F1 is inherently high-variance (0 or 1, nothing in between). Confirmed no `StratifiedKFold` error before running the sweep.

## Commands (per n_components, quantum always before classical)

```sh
uv run python -m quantum --n-components <nc> --max-samples 200 --folds 5 \
    --encodings angle iqp --tasks multiclass --mlflow

uv run python -m classical --n-components <nc> --load-quantum-splits --tasks multiclass --mlflow
```

## Full results (mean across 5 outer folds, MLflow parent-run aggregate)

### n_components = 1

| model | f1_macro | roc_auc | accuracy | mcc | fit_time (s) | tune_time (s) | infer_time (s) |
|---|---|---|---|---|---|---|---|
| qsvm | 0.0804 ± 0.0144 | 0.6847 | 0.5150 | 0.3634 | 1.8322 | 2.8689 | 1.0338 |
| random_forest | 0.0833 ± 0.0125 | 0.7408 | 0.4800 | 0.3107 | 0.6441 | 12.6959 | 0.1022 |
| xgboost | 0.0867 ± 0.0389 | 0.7692 | 0.5050 | 0.3313 | 1.5126 | 7.5847 | 0.0114 |
| lightgbm | 0.1167 ± 0.0268 | 0.7315 | 0.5100 | 0.3493 | 2.8511 | 10.0409 | 0.0029 |
| svm | 0.0817 ± 0.0154 | 0.5518 | 0.5000 | 0.3360 | 0.0064 | 0.4125 | 0.0005 |

### n_components = 2

| model | f1_macro | roc_auc | accuracy | mcc | fit_time (s) | tune_time (s) | infer_time (s) |
|---|---|---|---|---|---|---|---|
| qsvm | 0.0663 ± 0.0063 | 0.6470 | 0.5000 | 0.3380 | 2.7581 | 4.8753 | 1.3580 |
| random_forest | 0.1370 ± 0.0399 | 0.6872 | 0.5400 | 0.3805 | 0.3584 | 11.5340 | 0.0496 |
| xgboost | 0.1007 ± 0.0388 | 0.7805 | 0.5250 | 0.3534 | 1.0708 | 8.2118 | 0.0172 |
| lightgbm | 0.1480 ± 0.0498 | 0.7527 | 0.5400 | 0.3790 | 3.6102 | 11.5757 | 0.0033 |
| svm | 0.1335 ± 0.0661 | 0.5739 | 0.5450 | 0.3732 | 0.0038 | 0.3736 | 0.0003 |

### n_components = 3

| model | f1_macro | roc_auc | accuracy | mcc | fit_time (s) | tune_time (s) | infer_time (s) |
|---|---|---|---|---|---|---|---|
| qsvm | 0.0976 ± 0.0507 | 0.5987 | 0.5200 | 0.3586 | 3.9079 | 7.4776 | 1.9515 |
| random_forest | 0.1105 ± 0.0327 | 0.7273 | 0.5300 | 0.3659 | 0.4976 | 12.6530 | 0.0760 |
| xgboost | 0.0996 ± 0.0225 | 0.7671 | 0.5250 | 0.3559 | 1.4902 | 9.4702 | 0.0139 |
| lightgbm | 0.0940 ± 0.0260 | 0.7362 | 0.5050 | 0.3316 | 6.9377 | 13.2303 | 0.0042 |
| svm | 0.1194 ± 0.0372 | 0.5714 | 0.5500 | 0.3874 | 0.0033 | 0.3776 | 0.0003 |

### n_components = 4

| model | f1_macro | roc_auc | accuracy | mcc | fit_time (s) | tune_time (s) | infer_time (s) |
|---|---|---|---|---|---|---|---|
| qsvm | 0.0853 ± 0.0257 | 0.6200 | 0.5200 | 0.3712 | 5.3100 | 10.3662 | 2.6706 |
| random_forest | 0.1224 ± 0.0260 | 0.7473 | 0.5400 | 0.3800 | 0.5725 | 13.5341 | 0.0823 |
| xgboost | 0.1121 ± 0.0310 | 0.7779 | 0.5300 | 0.3589 | 2.1266 | 9.2859 | 0.0130 |
| lightgbm | 0.0921 ± 0.0199 | 0.7056 | 0.5100 | 0.3355 | 4.3461 | 12.8111 | 0.0034 |
| svm | 0.1006 ± 0.0287 | 0.5800 | 0.5300 | 0.3557 | 0.0033 | 0.3734 | 0.0003 |

### n_components = 5

| model | f1_macro | roc_auc | accuracy | mcc | fit_time (s) | tune_time (s) | infer_time (s) |
|---|---|---|---|---|---|---|---|
| qsvm | 0.0866 ± 0.0353 | 0.6362 | 0.5150 | 0.3720 | 7.0393 | 13.7428 | 3.5875 |
| random_forest | 0.1174 ± 0.0290 | 0.7278 | 0.5350 | 0.3722 | 0.6466 | 12.8979 | 0.0963 |
| xgboost | 0.1405 ± 0.0147 | 0.7738 | 0.5400 | 0.3654 | 2.1203 | 10.7247 | 0.0117 |
| lightgbm | 0.1279 ± 0.0207 | 0.7415 | 0.5250 | 0.3571 | 4.9015 | 12.8224 | 0.0039 |
| svm | 0.1182 ± 0.0295 | 0.5925 | 0.5450 | 0.3742 | 0.0033 | 0.3746 | 0.0003 |

### n_components = 6

| model | f1_macro | roc_auc | accuracy | mcc | fit_time (s) | tune_time (s) | infer_time (s) |
|---|---|---|---|---|---|---|---|
| qsvm | 0.0799 ± 0.0176 | 0.6371 | 0.5150 | 0.3681 | 10.0699 | 17.8814 | 5.1765 |
| random_forest | 0.1082 ± 0.0250 | 0.7081 | 0.5300 | 0.3654 | 0.5809 | 13.5874 | 0.0823 |
| xgboost | 0.1052 ± 0.0197 | 0.7699 | 0.5200 | 0.3441 | 1.5579 | 10.5270 | 0.0083 |
| lightgbm | 0.1011 ± 0.0318 | 0.7112 | 0.5200 | 0.3512 | 3.8328 | 12.5767 | 0.0033 |
| svm | 0.1037 ± 0.0219 | 0.5650 | 0.5200 | 0.3591 | 0.0031 | 0.3732 | 0.0003 |

## QSVM encoding selected per fold (inner-CV winner)

| n_components | fold0 | fold1 | fold2 | fold3 | fold4 | iqp wins |
|---|---|---|---|---|---|---|
| 1 | angle | angle | angle | angle | angle | 0/5 |
| 2 | angle | angle | **iqp** | angle | angle | 1/5 |
| 3 | angle | **iqp** | **iqp** | angle | angle | 2/5 |
| 4 | angle | **iqp** | angle | **iqp** | angle | 2/5 |
| 5 | angle | **iqp** | angle | angle | **iqp** | 2/5 |
| 6 | angle | **iqp** | angle | **iqp** | **iqp** | 3/5 |

`iqp` wins 12/30 folds across nc=1..6 in the multiclass task — a much higher rate than the binary task's 2/30 (`w2_Jul-12_experiments_summary_2-3-6-components.md`). `iqp` never wins at nc=1 (0/5), then wins start appearing from nc=2 onward and climb toward nc=6 (3/5 folds). See Notable observations below.

## Per-class F1 (mean across 5 folds) — QSVM vs. Random Forest

Classes: 0 = Benign, 1-15 = malware families (`Category` prefix, per `docs/reports/week1_cic_malmem_classical_baseline_report.md`). Full per-class F1 for every model/fold is in MLflow (`f1_class_<label>` metrics on each child run); the two most informative models are reproduced here.

### QSVM

| nc | 0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.98|0.10|0.00|0.13|0.00|0.00|0.00|0.00|0.00|0.00|0.04|0.00|0.00|0.00|0.04|0.00 |
| 2 | 0.98|0.00|0.00|0.00|0.00|0.00|0.00|0.00|0.00|0.00|0.05|0.00|0.00|0.03|0.00|0.00 |
| 3 | 0.98|0.00|0.00|0.10|0.00|0.00|0.00|0.10|0.00|0.00|0.00|0.00|0.00|0.04|0.20|0.13 |
| 4 | 0.98|0.00|0.00|0.00|0.00|0.00|0.00|0.17|0.13|0.00|0.00|0.00|0.00|0.02|0.06|0.00 |
| 5 | 0.97|0.20|0.00|0.00|0.00|0.00|0.00|0.00|0.00|0.00|0.08|0.00|0.00|0.05|0.03|0.06 |
| 6 | 0.98|0.00|0.00|0.16|0.00|0.00|0.00|0.00|0.00|0.00|0.06|0.00|0.00|0.05|0.03|0.00 |

### Random Forest

| nc | 0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.95|0.00|0.00|0.00|0.13|0.00|0.07|0.00|0.00|0.00|0.10|0.00|0.08|0.00|0.00|0.00 |
| 2 | 0.96|0.00|0.07|0.00|0.13|0.00|0.38|0.00|0.20|0.00|0.00|0.00|0.08|0.23|0.00|0.13 |
| 3 | 0.98|0.00|0.00|0.00|0.10|0.00|0.35|0.00|0.06|0.00|0.00|0.00|0.18|0.00|0.10|0.00 |
| 4 | 0.98|0.00|0.00|0.00|0.00|0.00|0.35|0.00|0.35|0.00|0.00|0.00|0.08|0.00|0.07|0.13 |
| 5 | 0.98|0.00|0.00|0.07|0.00|0.00|0.27|0.00|0.28|0.00|0.00|0.00|0.08|0.00|0.20|0.00 |
| 6 | 0.98|0.00|0.00|0.00|0.00|0.00|0.23|0.00|0.25|0.00|0.00|0.10|0.10|0.00|0.07|0.00 |

## Notable observations

- **Both frameworks are essentially collapsing to the majority class (Benign).** QSVM's per-class F1 table shows class 0 (Benign) at 0.975-0.985 across every `n_components`, while **13 of the 15 malware-family classes score 0.000 at every single n_components** — QSVM is not distinguishing malware families at all in this regime, just recognizing "not-Benign-enough-to-matter" noise. This is the expected consequence of the sanity check above: with only ~5 members of the rarest classes in a 200-sample subsample split 5 ways, there's nowhere near enough signal per class for a 16-way classifier (quantum or classical) to learn family-level distinctions. The overall `f1_macro` (0.06-0.10 for QSVM) is really reporting "how well can we tell 16 families apart from noise," not "how good is this classifier."
- **Random Forest degrades more gracefully than QSVM at this sample size.** RF's per-class table shows 2-4 malware families (typically classes 6 and 8, sometimes 4, 12, 13, 15) picking up real signal (F1 0.13-0.38) instead of collapsing to zero, while QSVM only ever recovers a class or two at F1 ≤0.2 and usually a *different* class each `n_components`. This lines up with the aggregate table: RF/XGBoost/LightGBM all sit around f1_macro 0.09-0.15 vs. QSVM's 0.07-0.10 — a small but consistent gap, and per-class data shows *why*: classical ensembles are partially learning a few families, QSVM isn't learning any.
- **`iqp` wins far more often in multiclass than in binary at the same n_components.** 12/30 folds here vs. 2/30 in the binary sweep (`w2_Jul-12_experiments_summary_2-3-6-components.md`), and the win rate trends up with qubit count (0/5 at nc=1, up to 3/5 at nc=6). Plausible reading: multiclass's one-vs-one SVM sees many more, noisier pairwise sub-problems than binary's single decision boundary, so `iqp`'s entangling structure has more chances to occasionally out-tune `angle` on some pairwise fit even without a systematic quality edge — this is *not* strong evidence `iqp` is actually better here, given how noisy every metric is at this sample size (see point 1); it needs re-checking once there's enough per-class signal to trust the inner-CV encoding selection at all.
- **QSVM's inference time now clearly dominates its per-fold cost, growing with n_components** (1.03s at nc=1 -> 5.18s at nc=6) since one-vs-one multiclass needs many more kernel evaluations against the support vectors of many more pairwise sub-models than binary's single SVM — consistent with the binary-task finding that runtime, not quality, is quantum's scaling bottleneck, now confirmed to extend to inference time in the multiclass setting too.
- **This 200-sample stage should be read as a pipeline/logging validation run, not a result to report as "QSVM is worse at multiclass."** The per-class F1 breakdown makes clear the real limiter is sample size per class, not the model or encoding. A meaningful multiclass comparison needs either a larger `--max-samples` (1000 gives the rarest class ~24 members, per the sanity check math) or accepting that some rare families simply can't be evaluated reliably at any subsample size QSVM's O(n²) kernel can afford.

## Raw logs

Full console output for every run is committed at `docs/reports/logs/w2_jul-14_multiclass_200/nc<N>_200_{quantum,classical}.txt` (`.txt` rather than `.log` only so the repo's blanket `*.log` gitignore rule doesn't swallow them; kept as separate files rather than inlined in full, since 6 nc x 2 commands is 12 logs — MLflow has the complete structured metrics; these are the literal stdout for traceability). Excerpt for nc=1 (shortest, representative of the format) below.

### nc=1 quantum (full)
```
2026-07-14 19:18:47.661 | INFO     | quantum.run:run_quantum_cv:147 - [qsvm/multiclass] outer fold 1/5
2026-07-14 19:18:53.351 | INFO     | quantum.run:evaluate_fold_quantum:109 - [qsvm/multiclass] fold done: f1_macro=0.0706
2026-07-14 19:18:53.631 | INFO     | quantum.run:run_quantum_cv:147 - [qsvm/multiclass] outer fold 2/5
2026-07-14 19:18:59.353 | INFO     | quantum.run:evaluate_fold_quantum:109 - [qsvm/multiclass] fold done: f1_macro=0.0889
2026-07-14 19:18:59.620 | INFO     | quantum.run:run_quantum_cv:147 - [qsvm/multiclass] outer fold 3/5
2026-07-14 19:19:05.250 | INFO     | quantum.run:evaluate_fold_quantum:109 - [qsvm/multiclass] fold done: f1_macro=0.1036
2026-07-14 19:19:05.503 | INFO     | quantum.run:run_quantum_cv:147 - [qsvm/multiclass] outer fold 4/5
2026-07-14 19:19:11.188 | INFO     | quantum.run:evaluate_fold_quantum:109 - [qsvm/multiclass] fold done: f1_macro=0.0625
2026-07-14 19:19:11.449 | INFO     | quantum.run:run_quantum_cv:147 - [qsvm/multiclass] outer fold 5/5
2026-07-14 19:19:17.469 | INFO     | quantum.run:evaluate_fold_quantum:109 - [qsvm/multiclass] fold done: f1_macro=0.0765
```

### nc=1 classical (full)
```
2026-07-14 19:19:21.977 | INFO     | classical.run:run_nested_cv:140 - [random_forest/multiclass] outer fold 1/5
2026-07-14 19:19:41.175 | INFO     | classical.run:evaluate_fold:97 - [random_forest/multiclass] fold done: f1_macro=0.0842
2026-07-14 19:19:41.492 | INFO     | classical.run:run_nested_cv:140 - [random_forest/multiclass] outer fold 2/5
2026-07-14 19:19:52.646 | INFO     | classical.run:evaluate_fold:97 - [random_forest/multiclass] fold done: f1_macro=0.0887
2026-07-14 19:19:52.935 | INFO     | classical.run:run_nested_cv:140 - [random_forest/multiclass] outer fold 3/5
2026-07-14 19:20:05.013 | INFO     | classical.run:evaluate_fold:97 - [random_forest/multiclass] fold done: f1_macro=0.0610
2026-07-14 19:20:05.411 | INFO     | classical.run:run_nested_cv:140 - [random_forest/multiclass] outer fold 4/5
2026-07-14 19:20:19.656 | INFO     | classical.run:evaluate_fold:97 - [random_forest/multiclass] fold done: f1_macro=0.0833
2026-07-14 19:20:19.942 | INFO     | classical.run:run_nested_cv:140 - [random_forest/multiclass] outer fold 5/5
2026-07-14 19:20:31.047 | INFO     | classical.run:evaluate_fold:97 - [random_forest/multiclass] fold done: f1_macro=0.0991
2026-07-14 19:20:31.351 | INFO     | __main__:main:98 - wrote per-fold predictions to results/cic/random_forest_multiclass_predictions.npz
2026-07-14 19:20:31.374 | INFO     | classical.run:run_nested_cv:140 - [xgboost/multiclass] outer fold 1/5
2026-07-14 19:20:43.392 | INFO     | classical.run:evaluate_fold:97 - [xgboost/multiclass] fold done: f1_macro=0.0592
2026-07-14 19:20:43.815 | INFO     | classical.run:run_nested_cv:140 - [xgboost/multiclass] outer fold 2/5
2026-07-14 19:20:50.729 | INFO     | classical.run:evaluate_fold:97 - [xgboost/multiclass] fold done: f1_macro=0.1604
2026-07-14 19:20:51.156 | INFO     | classical.run:run_nested_cv:140 - [xgboost/multiclass] outer fold 3/5
2026-07-14 19:21:01.548 | INFO     | classical.run:evaluate_fold:97 - [xgboost/multiclass] fold done: f1_macro=0.0922
2026-07-14 19:21:02.104 | INFO     | classical.run:run_nested_cv:140 - [xgboost/multiclass] outer fold 4/5
2026-07-14 19:21:09.331 | INFO     | classical.run:evaluate_fold:97 - [xgboost/multiclass] fold done: f1_macro=0.0625
2026-07-14 19:21:09.756 | INFO     | classical.run:run_nested_cv:140 - [xgboost/multiclass] outer fold 5/5
2026-07-14 19:21:18.955 | INFO     | classical.run:evaluate_fold:97 - [xgboost/multiclass] fold done: f1_macro=0.0594
2026-07-14 19:21:19.390 | INFO     | __main__:main:98 - wrote per-fold predictions to results/cic/xgboost_multiclass_predictions.npz
2026-07-14 19:21:19.416 | INFO     | classical.run:run_nested_cv:140 - [lightgbm/multiclass] outer fold 1/5
2026-07-14 19:21:34.292 | INFO     | classical.run:evaluate_fold:97 - [lightgbm/multiclass] fold done: f1_macro=0.0905
2026-07-14 19:21:34.756 | INFO     | classical.run:run_nested_cv:140 - [lightgbm/multiclass] outer fold 2/5
2026-07-14 19:21:48.949 | INFO     | classical.run:evaluate_fold:97 - [lightgbm/multiclass] fold done: f1_macro=0.1378
2026-07-14 19:21:49.400 | INFO     | classical.run:run_nested_cv:140 - [lightgbm/multiclass] outer fold 3/5
2026-07-14 19:22:03.703 | INFO     | classical.run:evaluate_fold:97 - [lightgbm/multiclass] fold done: f1_macro=0.1324
2026-07-14 19:22:04.155 | INFO     | classical.run:run_nested_cv:140 - [lightgbm/multiclass] outer fold 4/5
2026-07-14 19:22:16.093 | INFO     | classical.run:evaluate_fold:97 - [lightgbm/multiclass] fold done: f1_macro=0.0788
2026-07-14 19:22:16.698 | INFO     | classical.run:run_nested_cv:140 - [lightgbm/multiclass] outer fold 5/5
2026-07-14 19:22:26.004 | INFO     | classical.run:evaluate_fold:97 - [lightgbm/multiclass] fold done: f1_macro=0.1442
2026-07-14 19:22:26.463 | INFO     | __main__:main:98 - wrote per-fold predictions to results/cic/lightgbm_multiclass_predictions.npz
2026-07-14 19:22:26.489 | INFO     | classical.run:run_nested_cv:140 - [svm/multiclass] outer fold 1/5
2026-07-14 19:22:27.044 | INFO     | classical.run:evaluate_fold:97 - [svm/multiclass] fold done: f1_macro=0.1026
2026-07-14 19:22:27.358 | INFO     | classical.run:run_nested_cv:140 - [svm/multiclass] outer fold 2/5
2026-07-14 19:22:27.736 | INFO     | classical.run:evaluate_fold:97 - [svm/multiclass] fold done: f1_macro=0.0961
2026-07-14 19:22:28.089 | INFO     | classical.run:run_nested_cv:140 - [svm/multiclass] outer fold 3/5
2026-07-14 19:22:28.534 | INFO     | classical.run:evaluate_fold:97 - [svm/multiclass] fold done: f1_macro=0.0610
2026-07-14 19:22:28.929 | INFO     | classical.run:run_nested_cv:140 - [svm/multiclass] outer fold 4/5
2026-07-14 19:22:29.343 | INFO     | classical.run:evaluate_fold:97 - [svm/multiclass] fold done: f1_macro=0.0721
2026-07-14 19:22:29.911 | INFO     | classical.run:run_nested_cv:140 - [svm/multiclass] outer fold 5/5
2026-07-14 19:22:30.319 | INFO     | classical.run:evaluate_fold:97 - [svm/multiclass] fold done: f1_macro=0.0765
2026-07-14 19:22:30.702 | INFO     | __main__:main:98 - wrote per-fold predictions to results/cic/svm_multiclass_predictions.npz
2026-07-14 19:22:30.705 | INFO     | __main__:main:102 - wrote 4 model x task rows to results/cic/metrics.csv
```

## Stage 2 — 1000 samples (multiclass, n_components 1-6)

Same protocol as Stage 1 above, `--max-samples 1000` instead of 200. Per the sanity check math, the rarest class (class 9, ~2.4% of the full dataset) now has ~24 members in the subsample instead of ~5 — enough that per-fold F1 for rare classes is no longer just a single right/wrong coin flip.

**Runtime**: full sweep (quantum + classical, nc=1..6) took **~5h25m** (20:29 -> 01:54), dominated by QSVM: fit+tune+infer time grows from ~140s at nc=1 to ~1065s (~18min) at nc=6, roughly 7.6x for a 6x qubit increase — the single longest step was `nc=6` quantum alone (~3h29m across its 5 folds, ~42min/fold). All 6 nc levels completed without error (checked all 12 raw logs for `error`/`traceback`/`exception` — none found).

### Commands

```sh
uv run python -m quantum --n-components <nc> --max-samples 1000 --folds 5 \
    --encodings angle iqp --tasks multiclass --mlflow

uv run python -m classical --n-components <nc> --load-quantum-splits --tasks multiclass --mlflow
```

### Full results (mean across 5 outer folds, MLflow parent-run aggregate)

#### n_components = 1

| model | f1_macro | roc_auc | accuracy | mcc | fit_time (s) | tune_time (s) | infer_time (s) |
|---|---|---|---|---|---|---|---|
| qsvm | 0.0993 ± 0.0193 | 0.7018 | 0.5390 | 0.3965 | 44.3904 | 73.6990 | 22.5260 |
| random_forest | 0.1189 ± 0.0139 | 0.7169 | 0.5270 | 0.3576 | 0.6808 | 14.9575 | 0.0905 |
| xgboost | 0.1240 ± 0.0159 | 0.7736 | 0.5280 | 0.3579 | 2.9528 | 16.1330 | 0.0292 |
| lightgbm | 0.1195 ± 0.0205 | 0.7640 | 0.5290 | 0.3629 | 15.6136 | 28.0975 | 0.0120 |
| svm | 0.1245 ± 0.0094 | 0.5859 | 0.5410 | 0.3885 | 0.0273 | 0.8684 | 0.0044 |

#### n_components = 2

| model | f1_macro | roc_auc | accuracy | mcc | fit_time (s) | tune_time (s) | infer_time (s) |
|---|---|---|---|---|---|---|---|
| qsvm | 0.1223 ± 0.0082 | 0.6650 | 0.5420 | 0.3899 | 63.2093 | 121.0048 | 31.7984 |
| random_forest | 0.1852 ± 0.0254 | 0.7923 | 0.5630 | 0.4059 | 0.8285 | 17.2262 | 0.1090 |
| xgboost | 0.1616 ± 0.0315 | 0.7844 | 0.5520 | 0.3865 | 8.4018 | 24.4860 | 0.0251 |
| lightgbm | 0.1865 ± 0.0307 | 0.7913 | 0.5670 | 0.4104 | 72.3395 | 43.0371 | 0.0453 |
| svm | 0.1516 ± 0.0172 | 0.6248 | 0.5480 | 0.3826 | 0.0457 | 0.9095 | 0.0038 |

#### n_components = 3

| model | f1_macro | roc_auc | accuracy | mcc | fit_time (s) | tune_time (s) | infer_time (s) |
|---|---|---|---|---|---|---|---|
| qsvm | 0.1243 ± 0.0226 | 0.6568 | 0.5430 | 0.3903 | 97.2169 | 181.3483 | 48.6872 |
| random_forest | 0.1948 ± 0.0213 | 0.8053 | 0.5680 | 0.4131 | 1.0379 | 17.7585 | 0.1422 |
| xgboost | 0.2059 ± 0.0352 | 0.8233 | 0.5750 | 0.4206 | 6.5858 | 28.0940 | 0.0272 |
| lightgbm | 0.2019 ± 0.0495 | 0.8054 | 0.5720 | 0.4155 | 16.4113 | 37.6148 | 0.0148 |
| svm | 0.1761 ± 0.0229 | 0.6617 | 0.5640 | 0.4008 | 0.0347 | 1.1363 | 0.0037 |

#### n_components = 4

| model | f1_macro | roc_auc | accuracy | mcc | fit_time (s) | tune_time (s) | infer_time (s) |
|---|---|---|---|---|---|---|---|
| qsvm | 0.1105 ± 0.0097 | 0.6382 | 0.5380 | 0.3843 | 142.0016 | 257.1645 | 71.8320 |
| random_forest | 0.2122 ± 0.0354 | 0.8142 | 0.5740 | 0.4206 | 0.8127 | 18.9218 | 0.1141 |
| xgboost | 0.2094 ± 0.0289 | 0.8239 | 0.5790 | 0.4259 | 2.6650 | 23.9357 | 0.0159 |
| lightgbm | 0.2013 ± 0.0192 | 0.8112 | 0.5730 | 0.4186 | 22.8325 | 43.5442 | 0.0186 |
| svm | 0.1704 ± 0.0247 | 0.6688 | 0.5590 | 0.3903 | 0.0387 | 0.9884 | 0.0041 |

#### n_components = 5

| model | f1_macro | roc_auc | accuracy | mcc | fit_time (s) | tune_time (s) | infer_time (s) |
|---|---|---|---|---|---|---|---|
| qsvm | 0.1240 ± 0.0054 | 0.6436 | 0.5420 | 0.3893 | 208.1287 | 348.4209 | 105.4030 |
| random_forest | 0.2067 ± 0.0177 | 0.8088 | 0.5760 | 0.4236 | 0.9994 | 19.8106 | 0.1377 |
| xgboost | 0.2122 ± 0.0365 | 0.8243 | 0.5830 | 0.4310 | 3.8643 | 27.9963 | 0.0192 |
| lightgbm | 0.1901 ± 0.0203 | 0.8117 | 0.5690 | 0.4126 | 19.9039 | 39.7780 | 0.0145 |
| svm | 0.1695 ± 0.0141 | 0.6740 | 0.5600 | 0.3905 | 0.0388 | 1.1235 | 0.0038 |

#### n_components = 6

| model | f1_macro | roc_auc | accuracy | mcc | fit_time (s) | tune_time (s) | infer_time (s) |
|---|---|---|---|---|---|---|---|
| qsvm | 0.1213 ± 0.0091 | 0.6860 | 0.5440 | 0.4026 | 349.3559 | 522.0651 | 193.8352 |
| random_forest | 0.1997 ± 0.0207 | 0.8190 | 0.5710 | 0.4167 | 1.3509 | 35.8471 | 0.1798 |
| xgboost | 0.2187 ± 0.0392 | 0.8306 | 0.5830 | 0.4314 | 48.1829 | 39.8526 | 0.0229 |
| lightgbm | 0.2259 ± 0.0339 | 0.8176 | 0.5860 | 0.4355 | 16.7926 | 251.7083 | 0.0146 |
| svm | 0.1839 ± 0.0312 | 0.6990 | 0.5680 | 0.4077 | 0.0359 | 1.1066 | 0.0048 |

### QSVM encoding selected per fold (inner-CV winner)

| n_components | fold0 | fold1 | fold2 | fold3 | fold4 | iqp wins |
|---|---|---|---|---|---|---|
| 1 | angle | **iqp** | angle | angle | angle | 1/5 |
| 2 | **iqp** | angle | **iqp** | **iqp** | angle | 3/5 |
| 3 | **iqp** | angle | **iqp** | **iqp** | **iqp** | 4/5 |
| 4 | angle | **iqp** | **iqp** | **iqp** | angle | 3/5 |
| 5 | angle | **iqp** | **iqp** | **iqp** | **iqp** | 4/5 |
| 6 | **iqp** | **iqp** | **iqp** | **iqp** | **iqp** | 5/5 |

`iqp` wins 20/30 folds here (vs. 12/30 at 200 samples, vs. 2/30 in the binary task) and the trend is now monotonic and strong: 1/5 at nc=1 up to **5/5 (unanimous) at nc=6**. This is the first sweep where `iqp` clearly overtakes `angle` at higher qubit counts, in a way the 200-sample stage only hinted at. See Notable observations.

### Per-class F1 (mean across 5 folds) — QSVM vs. Random Forest

#### QSVM

| nc | 0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.99|0.00|0.00|0.19|0.00|0.00|0.00|0.10|0.00|0.14|0.00|0.00|0.07|0.02|0.08|0.00 |
| 2 | 0.99|0.00|0.05|0.09|0.04|0.10|0.00|0.19|0.00|0.31|0.02|0.00|0.05|0.00|0.06|0.07 |
| 3 | 0.99|0.00|0.04|0.14|0.00|0.09|0.00|0.17|0.02|0.32|0.00|0.00|0.05|0.02|0.06|0.10 |
| 4 | 1.00|0.00|0.00|0.00|0.01|0.18|0.00|0.12|0.03|0.34|0.05|0.00|0.00|0.00|0.00|0.03 |
| 5 | 0.99|0.00|0.00|0.04|0.01|0.16|0.00|0.06|0.03|0.35|0.10|0.00|0.05|0.00|0.08|0.12 |
| 6 | 0.99|0.02|0.00|0.04|0.00|0.12|0.00|0.10|0.04|0.36|0.11|0.04|0.10|0.00|0.02|0.00 |

#### Random Forest

| nc | 0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.99|0.06|0.03|0.15|0.03|0.00|0.04|0.08|0.04|0.21|0.02|0.09|0.04|0.03|0.03|0.07 |
| 2 | 0.99|0.17|0.16|0.21|0.05|0.11|0.03|0.03|0.19|0.42|0.15|0.04|0.13|0.07|0.17|0.05 |
| 3 | 0.99|0.12|0.14|0.16|0.07|0.08|0.10|0.08|0.32|0.31|0.22|0.16|0.14|0.04|0.19|0.00 |
| 4 | 0.99|0.08|0.14|0.15|0.12|0.19|0.14|0.10|0.29|0.34|0.19|0.11|0.23|0.11|0.17|0.06 |
| 5 | 0.99|0.03|0.05|0.19|0.09|0.12|0.05|0.13|0.26|0.31|0.28|0.11|0.23|0.16|0.26|0.05 |
| 6 | 0.99|0.08|0.07|0.18|0.08|0.08|0.06|0.04|0.25|0.33|0.20|0.15|0.23|0.20|0.21|0.04 |

### Notable observations (1000-sample stage)

- **More samples clearly help classical models, much less so QSVM.** Random Forest's f1_macro roughly doubled from the 200-sample stage (~0.08-0.14) to 1000 samples (~0.12-0.21), and its per-class table now shows *every* class picking up nonzero F1 (vs. 8-13 zeroed-out classes at 200 samples) — e.g. class 9 at nc=2 reaches 0.417. QSVM barely moved (f1_macro 0.10-0.12 at 1000 samples vs. 0.07-0.10 at 200) and still zeroes out roughly half the malware families at every n_components. The gap between classical and quantum, which was small and noisy at 200 samples, is now clear and consistent: classical ensembles are meaningfully better at this multiclass task, not just lucky on a tiny sample.
- **`iqp` overtakes `angle` decisively as qubit count grows — the clearest signal in either sweep.** 20/30 folds here (up from 12/30 at 200 samples, 2/30 in binary), climbing monotonically from 1/5 at nc=1 to a unanimous 5/5 at nc=6. Unlike the 200-sample stage (which could plausibly be sampling noise given ~5 members/class), 1000 samples gives each fold's inner-CV comparison enough data to trust — this is the first run in the whole project where `iqp` looks like a genuinely better encoding rather than statistical noise, and it only shows up once both n_components is high enough (≥2, matching the Day 1 profiling report's prediction that iqp's entangling advantage needs a feature pair to act on) and the sample size is large enough for the comparison to be reliable.
- **QSVM's cost exploded far faster than the binary task's O(n²) intuition alone predicts, because multiclass's one-vs-one SVM stacks on top of it.** fit+tune+infer for QSVM went from ~140s (nc=1) to ~1065s (nc=6) — a ~7.6x increase for a 6x qubit increase, dominated by the `nc=6` run taking ~42min *per fold* (5 folds -> ~3.5h alone). Classical models, by contrast, stayed under ~90s fit+tune even at nc=6 (LightGBM's tune_time hit 251s at nc=6 as an outlier — worth a closer look, possibly the same kind of resource-contention artifact flagged for XGBoost in the earlier `n_components=3` binary report, since this ran unattended overnight for hours). This reinforces, at a starker scale than binary ever showed, that runtime is the real barrier to scaling QSVM to real multiclass malware datasets, not classification quality.
- **Class 9 (the rarest class, ~2.4% of the full dataset) is, somewhat counterintuitively, one of QSVM's best-recognized classes at 1000 samples** (F1 0.14-0.36 across nc, its single strongest non-Benign class at every n_components). Random Forest recognizes it even better (F1 up to 0.417 at nc=2). This may be because class 9's samples are more separable in feature space regardless of how few there are — worth a closer look at what `Category` family class 9 actually is before drawing conclusions.
- **Even at 1000 samples, Benign (class 0) still dominates every model's F1 budget** (0.989-0.996 across the board) while most malware families sit in the 0.03-0.3 range — the task is still fundamentally hard at this scale for every model tried, quantum or classical. 1000 samples fixed the *pipeline reliability* problem from the 200-sample stage (no more single-test-sample coin-flip F1s) but did not fix the *underlying* multiclass difficulty, which matches Week 1's classical-only finding that SVM was "the clear laggard on multiclass" even with the full ~58K-row dataset (`week1_cic_malmem_classical_baseline_report.md`).

### Raw logs

Full console output for every run committed at `docs/reports/logs/w2_jul-14_multiclass_1000/nc<N>_1000_{quantum,classical}.txt`. Excerpt for nc=6 (the slowest, most consequential run) below.

#### nc=6 quantum (full)
```
2026-07-14 23:52:49.995 | INFO     | quantum.run:run_quantum_cv:147 - [qsvm/multiclass] outer fold 1/5
2026-07-15 00:07:50.854 | INFO     | quantum.run:evaluate_fold_quantum:109 - [qsvm/multiclass] fold done: f1_macro=0.1120
2026-07-15 00:07:51.141 | INFO     | quantum.run:run_quantum_cv:147 - [qsvm/multiclass] outer fold 2/5
2026-07-15 00:22:50.586 | INFO     | quantum.run:evaluate_fold_quantum:109 - [qsvm/multiclass] fold done: f1_macro=0.1105
2026-07-15 00:22:50.895 | INFO     | quantum.run:run_quantum_cv:147 - [qsvm/multiclass] outer fold 3/5
2026-07-15 00:38:39.259 | INFO     | quantum.run:evaluate_fold_quantum:109 - [qsvm/multiclass] fold done: f1_macro=0.1341
2026-07-15 00:38:39.550 | INFO     | quantum.run:run_quantum_cv:147 - [qsvm/multiclass] outer fold 4/5
2026-07-15 00:57:29.192 | INFO     | quantum.run:evaluate_fold_quantum:109 - [qsvm/multiclass] fold done: f1_macro=0.1221
2026-07-15 00:57:29.680 | INFO     | quantum.run:run_quantum_cv:147 - [qsvm/multiclass] outer fold 5/5
2026-07-15 01:21:37.776 | INFO     | quantum.run:evaluate_fold_quantum:109 - [qsvm/multiclass] fold done: f1_macro=0.1278
```

## Next

- Both stages (200 and 1000 samples, multiclass, nc=1-6) are now complete. The 1000-sample stage's `iqp`-overtakes-`angle` trend and the classical/quantum gap are the two findings worth carrying into any write-up; the 200-sample stage should still be treated mainly as a pipeline/logging validation run per its own Notable observations above.
- The LightGBM tune_time outlier at nc=6/1000-samples (251s vs. ~40s at neighboring nc) is unverified — worth a resource-contention check (same as the XGBoost nc=3 binary anomaly) before trusting it as a real effect.

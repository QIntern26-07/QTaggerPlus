# Week 2 Jul 16 — Multiclass reframed to 15-class malware-only (1000 samples, n_components 1/3/6)

## What changed since the last multiclass report

The previous multiclass sweep (`w2_jul-14_6-components_multi.md`) ran a **16-class**
task (Benign + 15 malware families). This report re-runs multiclass after three
deliberate changes to how the task and the imbalance are handled. Read this first —
the results below are **not** comparable to the Jul-14 numbers, because the task
itself is different.

1. **Multiclass reframed to 15-class, malware-only.** Benign rows are dropped
   entirely; only the 15 malware families (`Category` prefix, e.g. `Ransomware-Ako`)
   remain, re-encoded 0..14. Rationale:
   - **Imbalance:** in the 16-class framing, Benign (29,298 rows) was ~21× larger
     than any single family (~1,410–2,410). The Week-1 EDA's "mild ~1.7× imbalance"
     figure only measured *between families* and silently ignored the Benign
     majority. Dropping Benign leaves the 15 families internally near-balanced
     (~1.7×), which is genuinely mild.
   - **Redundancy:** benign-vs-malware *detection* is already the binary task, solved
     at F1 ~0.99. Re-testing it inside multiclass added no information.
   - **Metric confound:** Benign was an easy class scoring ~0.98, inflating macro-F1
     (1/16 of the average sat near the ceiling) and masking how weak family
     attribution actually is. The 15-class macro-F1 now reflects *only* the hard
     part — telling malware families apart.
   - Implemented in `common/data.py::malware_family_xy`, routed through
     `common/data.py::task_xy` so the classical and quantum CLIs share one
     definition of the task and cannot drift on it.

2. **Per-task subsample splits.** Because binary and multiclass now operate on
   *different row sets* (all rows vs. malware-only), the single shared
   `quantum_sample_idx.json` is replaced by per-task files
   (`quantum_sample_idx_<task>.json`). A multiclass subsample now indexes into the
   malware-only frame.

3. **XGBoost imbalance handling (gap (a) closed).** `XGBClassifier` has no
   `class_weight`, so it was the only model with *zero* imbalance handling
   (RF/LightGBM hard-code `balanced`; SVM/QSVM tune it). `BalancedXGBClassifier`
   now injects `compute_sample_weight("balanced")` on every `fit()` — computed per
   fold from that fold's own labels, so no leakage. At the ~1.7× post-reframe
   imbalance its numerical effect is small, but it removes a real methodological
   inconsistency in a project whose whole point is a *fair* model comparison.

MLflow logging added in the prior PR (per-class F1 `f1_class_<label>`, `n_qubits`
param, `sweep` tags, nested parent/child runs with mean/std aggregates) is used
throughout, so every table below is pulled from `mlflow.db`, not console logs.

## Experiment design

- **Task:** 15-class malware-family attribution, CIC-MalMem-2022.
- **n_components (= qubit budget):** 1, 3, 6 — low/mid/high across the range. The
  Jul-14 sweep showed the `iqp`-vs-`angle` trend and the classical/quantum gap are
  monotonic in n_components, so three points suffice to re-establish them under the
  new task without paying for all six.
- **Sample size:** 1000 (stratified on the 15 family labels → ~66 rows/class, ~13
  per test fold — enough that per-class F1 is a real estimate, not a coin flip).
  The 200-sample stage is **skipped**: it existed only to validate the pipeline,
  which is now validated.
- **Protocol:** QSVM first (persists the malware-only subsample + folds), then
  classical with `--load-quantum-splits` on the identical rows/folds — the standard
  apples-to-apples setup. 5-fold outer CV, `--encodings angle iqp`.
- **Models:** QSVM vs. Random Forest / XGBoost / LightGBM / SVM.

### Commands

```sh
# per nc ∈ {1, 3, 6}
uv run python -m quantum --n-components <nc> --max-samples 1000 --folds 5 \
    --encodings angle iqp --tasks multiclass --mlflow

uv run python -m classical --n-components <nc> --load-quantum-splits --tasks multiclass --mlflow
```

## Full results (mean across 5 outer folds, MLflow parent-run aggregate)

Random baselines for a 15-way balanced task: accuracy 1/15 = **0.067**; macro-F1 of a one-class-always predictor ≈ 0.01, of uniform-random ≈ 0.067. Keep these in mind — several models sit close to them.

### n_components = 1

| model | f1_macro | roc_auc | accuracy | mcc | fit (s) | tune (s) | infer (s) |
|---|---|---|---|---|---|---|---|
| qsvm | 0.0577 ± 0.0076 | 0.5525 | 0.0980 | 0.0439 | 43.81 | 77.61 | 22.28 |
| random_forest | 0.1283 ± 0.0359 | 0.5780 | 0.1280 | 0.0666 | 1.17 | 21.35 | 0.13 |
| xgboost | 0.1010 ± 0.0162 | 0.5667 | 0.1020 | 0.0388 | 10.66 | 34.99 | 0.03 |
| lightgbm | 0.1094 ± 0.0226 | 0.5675 | 0.1100 | 0.0476 | 24.80 | 36.76 | 0.02 |
| svm | 0.1075 ± 0.0164 | 0.5507 | 0.1110 | 0.0494 | 0.05 | 1.12 | 0.01 |

### n_components = 3

| model | f1_macro | roc_auc | accuracy | mcc | fit (s) | tune (s) | infer (s) |
|---|---|---|---|---|---|---|---|
| qsvm | 0.0585 ± 0.0044 | 0.5447 | 0.1010 | 0.0498 | 100.62 | 185.71 | 49.73 |
| random_forest | 0.1398 ± 0.0130 | 0.6064 | 0.1380 | 0.0759 | 0.90 | 25.70 | 0.10 |
| xgboost | 0.1270 ± 0.0087 | 0.5890 | 0.1260 | 0.0631 | 75.58 | 43.52 | 0.03 |
| lightgbm | 0.1185 ± 0.0164 | 0.5840 | 0.1140 | 0.0497 | 24.12 | 37.04 | 0.02 |
| svm | 0.1309 ± 0.0265 | 0.5794 | 0.1290 | 0.0679 | 0.10 | 2.06 | 0.01 |

### n_components = 6

| model | f1_macro | roc_auc | accuracy | mcc | fit (s) | tune (s) | infer (s) |
|---|---|---|---|---|---|---|---|
| qsvm | 0.0708 ± 0.0141 | 0.5782 | 0.1060 | 0.0513 | 102.90 | 183.12 | 50.31 |
| random_forest | 0.1510 ± 0.0262 | 0.6285 | 0.1510 | 0.0901 | 0.86 | 26.66 | 0.10 |
| xgboost | 0.1494 ± 0.0249 | 0.6208 | 0.1480 | 0.0864 | 8.61 | 46.90 | 0.03 |
| lightgbm | 0.1571 ± 0.0221 | 0.6038 | 0.1510 | 0.0895 | 20.50 | 45.78 | 0.02 |
| svm | 0.1647 ± 0.0233 | 0.6183 | 0.1700 | 0.1118 | 0.07 | 1.25 | 0.01 |

## QSVM encoding selected per fold (inner-CV winner)

| n_components | fold0 | fold1 | fold2 | fold3 | fold4 | iqp wins |
|---|---|---|---|---|---|---|
| 1 | angle | **iqp** | angle | **iqp** | angle | 2/5 |
| 3 | **iqp** | **iqp** | angle | **iqp** | angle | 3/5 |
| 6 | **iqp** | **iqp** | angle | angle | **iqp** | 3/5 |

`iqp` wins **8/15** folds overall (2/5, 3/5, 3/5). That is well above the binary task's 2/30 but, unlike the earlier *16-class* 1000-sample run (which climbed to a unanimous 5/5 at nc=6), there is no strong monotone dominance here — `iqp` and `angle` are roughly co-competitive on the malware-only task. See analysis.

## Per-class F1 (mean across 5 folds)

Family legend (LabelEncoder order, malware-only): 0=Ransomware-Ako, 1=Ransomware-Conti, 2=Ransomware-Maze, 3=Ransomware-Pysa, 4=Ransomware-Shade, 5=Spyware-180solutions, 6=Spyware-CWS, 7=Spyware-Gator, 8=Spyware-TIBS, 9=Spyware-Transponder, 10=Trojan-Emotet, 11=Trojan-Reconyc, 12=Trojan-Refroso, 13=Trojan-Scar, 14=Trojan-Zeus.

### QSVM

| nc | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.00 | 0.00 | 0.02 | 0.08 | 0.00 | 0.02 | 0.09 | 0.00 | 0.44 | 0.00 | 0.00 | 0.05 | 0.00 | 0.06 | 0.10 |
| 3 | 0.00 | 0.03 | 0.00 | 0.02 | 0.06 | 0.02 | 0.06 | 0.04 | 0.47 | 0.07 | 0.00 | 0.05 | 0.00 | 0.00 | 0.05 |
| 6 | 0.00 | 0.00 | 0.02 | 0.00 | 0.07 | 0.05 | 0.08 | 0.13 | 0.46 | 0.04 | 0.00 | 0.05 | 0.04 | 0.08 | 0.03 |

### Random Forest

| nc | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.10 | 0.07 | 0.10 | 0.12 | 0.10 | 0.12 | 0.12 | 0.09 | 0.45 | 0.13 | 0.06 | 0.14 | 0.11 | 0.09 | 0.12 |
| 3 | 0.07 | 0.06 | 0.08 | 0.07 | 0.20 | 0.07 | 0.22 | 0.20 | 0.43 | 0.10 | 0.11 | 0.11 | 0.15 | 0.11 | 0.13 |
| 6 | 0.09 | 0.04 | 0.11 | 0.07 | 0.23 | 0.03 | 0.24 | 0.15 | 0.45 | 0.15 | 0.14 | 0.10 | 0.18 | 0.09 | 0.20 |

### SVM

| nc | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.04 | 0.08 | 0.04 | 0.04 | 0.12 | 0.12 | 0.09 | 0.03 | 0.52 | 0.11 | 0.06 | 0.12 | 0.09 | 0.01 | 0.15 |
| 3 | 0.06 | 0.03 | 0.10 | 0.09 | 0.17 | 0.12 | 0.14 | 0.22 | 0.40 | 0.12 | 0.09 | 0.10 | 0.09 | 0.11 | 0.12 |
| 6 | 0.08 | 0.03 | 0.16 | 0.04 | 0.21 | 0.02 | 0.15 | 0.28 | 0.47 | 0.09 | 0.16 | 0.16 | 0.24 | 0.19 | 0.19 |

## Analysis

- **QSVM is barely above chance on 15-class family attribution.** Its accuracy (0.098–0.106) is only ~1.5× the 0.067 random floor, and its macro-F1 (0.058–0.071) sits right on the random baseline. The per-class table explains why: QSVM scores **0.00 on roughly half the families at every n_components** and recognizes essentially **one** family well — class 8, Spyware-TIBS (F1 0.44–0.47). It is, in effect, a one-family detector plus noise.

- **Classical models extract real, if modest, signal across many families — and clearly beat QSVM at every n_components.** Random Forest / SVM / XGBoost / LightGBM reach macro-F1 0.10–0.16 with *nonzero* F1 spread across most of the 15 classes (e.g. RF nc=6: Spyware-CWS 0.24, Ransomware-Shade 0.23, Trojan-Zeus 0.20). This is the reframe doing its job: with Benign removed, the macro-F1 now honestly measures family attribution, and on that honest metric the classical models are meaningfully ahead of QSVM rather than tied.

- **Spyware-TIBS (class 8) is the universally separable family — despite being the smallest (1,410 rows).** Every model, quantum and classical, scores it 0.40–0.52 while struggling on the rest. Its memory-dump signature must be distinctive enough that even a 1-qubit fidelity kernel isolates it. This mirrors the Jul-14 16-class run, where one specific family (then class 9) similarly stood out — separability here is about a family's intrinsic feature signature, not its sample count.

- **Every model improves with n_components (qubits/features), so the task is dimensionality-starved, not saturated.** macro-F1 rises nc=1→6 for all: QSVM 0.058→0.071, RF 0.128→0.151, SVM 0.108→**0.165** (SVM overtakes RF at nc=6). Unlike the binary task (near-perfect at 1 component), 15-class attribution genuinely needs more of the PCA signal — and had we run beyond nc=6, the still-rising curves suggest it would keep helping.

- **`iqp` is co-competitive with `angle`, not dominant** (8/15 folds). The reframe visibly changed encoding selection versus the 16-class run's strong iqp trend — plausibly because the one-vs-one sub-problems now exclude the large, easy Benign-vs-family boundaries that the 16-class OvO had, leaving a noisier, more balanced set of pairwise fits where neither encoding systematically wins. This is weak evidence at best; encoding choice is not the lever that matters here.

- **Parallelization made this run practical.** The nc=6 QSVM sweep (tuning + 5-fold CV, both encodings) finished in **~28 min** wall-clock with `--n-jobs 12`, versus the ~3.5 h the equivalent 16-class nc=6 quantum run took single-core on Jul-14 — an effective ~7.5× for the full run (higher than the 2.7× single-Gram micro-benchmark, because the warm loky pool amortizes worker spawn across the many Grams a tuning sweep builds). Kernel values are unchanged (parallel Gram == serial, unit-tested).

- **Caveat — XGBoost fit_time at nc=3 (75.6 s) is an artifact, not signal.** It is ~9× XGBoost's fit at nc=1/nc=6 (8–11 s). The nc=3 classical run trained during the window when a stray parallel benchmark oversubscribed the machine (and crashed the IDE); this is the same resource-contention footprint flagged for XGBoost/nc=3 in the binary report, not an algorithmic effect. Do not read into it.

## Key findings

1. **On honest 15-class malware-family attribution, QSVM performs at roughly the random baseline** (macro-F1 ≈ 0.06, accuracy ≈ 0.10 vs. 0.067 chance), learning only a single separable family (Spyware-TIBS) and nothing else. This is the clearest negative result for QSVM so far — and it was partly hidden in the 16-class framing, where an easy, macro-F1-inflating Benign class masked it.

2. **Classical baselines are consistently and meaningfully better** (macro-F1 0.10–0.16, rising to SVM 0.165 at nc=6), extracting partial signal across most families. The classical–quantum gap, ambiguous at 200 samples in the old framing, is now unambiguous.

3. **The task is hard for everyone and dimensionality-limited** — all models improve monotonically with n_components and none exceeds macro-F1 0.17. 15-way memory-family attribution from ≤6 PCA components is near the edge of what any of these models can do at 1000 samples.

4. **Family separability is intrinsic, not sample-driven:** the rarest family (Spyware-TIBS, 1,410 rows) is the easiest to classify for every model, while larger families stay near zero.

5. **The QSVM kernel Gram parallelization delivered ~7.5× wall-clock on the nc=6 run** (28 min vs. ~3.5 h), correctness-preserving, making 1000-sample QSVM sweeps routine rather than overnight — a prerequisite for the heavier EMBER/SOREL work ahead. It must be run with a capped `--n-jobs` (we used 12/20): grabbing all cores starves the desktop and, as happened once during this session, can crash the IDE.

## Raw logs

All six runs inlined below (nc ∈ {1,3,6} × {quantum, classical}); also committed under `docs/reports/logs/w2_jul-16_multiclass_15class_1000/`.

### nc=1 quantum
```
2026-07-16 01:38:14.485 | INFO     | quantum.run:run_quantum_cv:147 - [qsvm/multiclass] outer fold 1/5
2026-07-16 01:40:52.138 | INFO     | quantum.run:evaluate_fold_quantum:109 - [qsvm/multiclass] fold done: f1_macro=0.0520
2026-07-16 01:40:52.425 | INFO     | quantum.run:run_quantum_cv:147 - [qsvm/multiclass] outer fold 2/5
2026-07-16 01:42:53.890 | INFO     | quantum.run:evaluate_fold_quantum:109 - [qsvm/multiclass] fold done: f1_macro=0.0567
2026-07-16 01:42:54.169 | INFO     | quantum.run:run_quantum_cv:147 - [qsvm/multiclass] outer fold 3/5
2026-07-16 01:45:22.516 | INFO     | quantum.run:evaluate_fold_quantum:109 - [qsvm/multiclass] fold done: f1_macro=0.0696
2026-07-16 01:45:22.796 | INFO     | quantum.run:run_quantum_cv:147 - [qsvm/multiclass] outer fold 4/5
2026-07-16 01:47:25.651 | INFO     | quantum.run:evaluate_fold_quantum:109 - [qsvm/multiclass] fold done: f1_macro=0.0481
2026-07-16 01:47:26.086 | INFO     | quantum.run:run_quantum_cv:147 - [qsvm/multiclass] outer fold 5/5
2026-07-16 01:50:14.401 | INFO     | quantum.run:evaluate_fold_quantum:109 - [qsvm/multiclass] fold done: f1_macro=0.0620
```

### nc=1 classical
```
2026-07-16 01:50:18.569 | INFO     | classical.run:run_nested_cv:140 - [random_forest/multiclass] outer fold 1/5
2026-07-16 01:50:50.460 | INFO     | classical.run:evaluate_fold:97 - [random_forest/multiclass] fold done: f1_macro=0.0866
2026-07-16 01:50:50.833 | INFO     | classical.run:run_nested_cv:140 - [random_forest/multiclass] outer fold 2/5
2026-07-16 01:51:14.931 | INFO     | classical.run:evaluate_fold:97 - [random_forest/multiclass] fold done: f1_macro=0.1103
2026-07-16 01:51:15.551 | INFO     | classical.run:run_nested_cv:140 - [random_forest/multiclass] outer fold 3/5
2026-07-16 01:51:36.162 | INFO     | classical.run:evaluate_fold:97 - [random_forest/multiclass] fold done: f1_macro=0.1914
2026-07-16 01:51:36.531 | INFO     | classical.run:run_nested_cv:140 - [random_forest/multiclass] outer fold 4/5
2026-07-16 01:51:51.980 | INFO     | classical.run:evaluate_fold:97 - [random_forest/multiclass] fold done: f1_macro=0.1408
2026-07-16 01:51:52.337 | INFO     | classical.run:run_nested_cv:140 - [random_forest/multiclass] outer fold 5/5
2026-07-16 01:52:14.393 | INFO     | classical.run:evaluate_fold:97 - [random_forest/multiclass] fold done: f1_macro=0.1124
2026-07-16 01:52:14.979 | INFO     | __main__:main:97 - wrote per-fold predictions to results/cic/random_forest_multiclass_predictions.npz
2026-07-16 01:52:15.026 | INFO     | classical.run:run_nested_cv:140 - [xgboost/multiclass] outer fold 1/5
2026-07-16 01:52:57.122 | INFO     | classical.run:evaluate_fold:97 - [xgboost/multiclass] fold done: f1_macro=0.1057
2026-07-16 01:52:57.625 | INFO     | classical.run:run_nested_cv:140 - [xgboost/multiclass] outer fold 2/5
2026-07-16 01:53:39.496 | INFO     | classical.run:evaluate_fold:97 - [xgboost/multiclass] fold done: f1_macro=0.0724
2026-07-16 01:53:39.987 | INFO     | classical.run:run_nested_cv:140 - [xgboost/multiclass] outer fold 3/5
2026-07-16 01:54:16.791 | INFO     | classical.run:evaluate_fold:97 - [xgboost/multiclass] fold done: f1_macro=0.1221
2026-07-16 01:54:17.199 | INFO     | classical.run:run_nested_cv:140 - [xgboost/multiclass] outer fold 4/5
2026-07-16 01:55:09.993 | INFO     | classical.run:evaluate_fold:97 - [xgboost/multiclass] fold done: f1_macro=0.0996
2026-07-16 01:55:10.653 | INFO     | classical.run:run_nested_cv:140 - [xgboost/multiclass] outer fold 5/5
2026-07-16 01:56:05.881 | INFO     | classical.run:evaluate_fold:97 - [xgboost/multiclass] fold done: f1_macro=0.1053
2026-07-16 01:56:06.406 | INFO     | __main__:main:97 - wrote per-fold predictions to results/cic/xgboost_multiclass_predictions.npz
2026-07-16 01:56:06.441 | INFO     | classical.run:run_nested_cv:140 - [lightgbm/multiclass] outer fold 1/5
2026-07-16 01:56:51.602 | INFO     | classical.run:evaluate_fold:97 - [lightgbm/multiclass] fold done: f1_macro=0.0858
2026-07-16 01:56:52.081 | INFO     | classical.run:run_nested_cv:140 - [lightgbm/multiclass] outer fold 2/5
2026-07-16 01:57:58.343 | INFO     | classical.run:evaluate_fold:97 - [lightgbm/multiclass] fold done: f1_macro=0.0814
2026-07-16 01:57:58.827 | INFO     | classical.run:run_nested_cv:140 - [lightgbm/multiclass] outer fold 3/5
2026-07-16 01:59:25.856 | INFO     | classical.run:evaluate_fold:97 - [lightgbm/multiclass] fold done: f1_macro=0.1126
2026-07-16 01:59:26.284 | INFO     | classical.run:run_nested_cv:140 - [lightgbm/multiclass] outer fold 4/5
2026-07-16 02:00:21.871 | INFO     | classical.run:evaluate_fold:97 - [lightgbm/multiclass] fold done: f1_macro=0.1371
2026-07-16 02:00:22.332 | INFO     | classical.run:run_nested_cv:140 - [lightgbm/multiclass] outer fold 5/5
2026-07-16 02:01:16.410 | INFO     | classical.run:evaluate_fold:97 - [lightgbm/multiclass] fold done: f1_macro=0.1303
2026-07-16 02:01:17.072 | INFO     | __main__:main:97 - wrote per-fold predictions to results/cic/lightgbm_multiclass_predictions.npz
2026-07-16 02:01:17.101 | INFO     | classical.run:run_nested_cv:140 - [svm/multiclass] outer fold 1/5
2026-07-16 02:01:18.324 | INFO     | classical.run:evaluate_fold:97 - [svm/multiclass] fold done: f1_macro=0.0997
2026-07-16 02:01:18.653 | INFO     | classical.run:run_nested_cv:140 - [svm/multiclass] outer fold 2/5
2026-07-16 02:01:19.824 | INFO     | classical.run:evaluate_fold:97 - [svm/multiclass] fold done: f1_macro=0.0979
2026-07-16 02:01:20.156 | INFO     | classical.run:run_nested_cv:140 - [svm/multiclass] outer fold 3/5
2026-07-16 02:01:21.329 | INFO     | classical.run:evaluate_fold:97 - [svm/multiclass] fold done: f1_macro=0.1402
2026-07-16 02:01:21.646 | INFO     | classical.run:run_nested_cv:140 - [svm/multiclass] outer fold 4/5
2026-07-16 02:01:22.838 | INFO     | classical.run:evaluate_fold:97 - [svm/multiclass] fold done: f1_macro=0.0984
2026-07-16 02:01:23.157 | INFO     | classical.run:run_nested_cv:140 - [svm/multiclass] outer fold 5/5
2026-07-16 02:01:24.421 | INFO     | classical.run:evaluate_fold:97 - [svm/multiclass] fold done: f1_macro=0.1010
2026-07-16 02:01:24.764 | INFO     | __main__:main:97 - wrote per-fold predictions to results/cic/svm_multiclass_predictions.npz
2026-07-16 02:01:24.781 | INFO     | __main__:main:101 - wrote 4 model x task rows to results/cic/metrics.csv
```

### nc=3 quantum
```
2026-07-16 02:01:32.062 | INFO     | quantum.run:run_quantum_cv:147 - [qsvm/multiclass] outer fold 1/5
2026-07-16 02:07:01.768 | INFO     | quantum.run:evaluate_fold_quantum:109 - [qsvm/multiclass] fold done: f1_macro=0.0620
2026-07-16 02:07:02.178 | INFO     | quantum.run:run_quantum_cv:147 - [qsvm/multiclass] outer fold 2/5
2026-07-16 02:12:30.825 | INFO     | quantum.run:evaluate_fold_quantum:109 - [qsvm/multiclass] fold done: f1_macro=0.0630
2026-07-16 02:12:31.088 | INFO     | quantum.run:run_quantum_cv:147 - [qsvm/multiclass] outer fold 3/5
2026-07-16 02:17:57.429 | INFO     | quantum.run:evaluate_fold_quantum:109 - [qsvm/multiclass] fold done: f1_macro=0.0505
2026-07-16 02:17:57.691 | INFO     | quantum.run:run_quantum_cv:147 - [qsvm/multiclass] outer fold 4/5
2026-07-16 02:23:32.146 | INFO     | quantum.run:evaluate_fold_quantum:109 - [qsvm/multiclass] fold done: f1_macro=0.0582
2026-07-16 02:23:32.408 | INFO     | quantum.run:run_quantum_cv:147 - [qsvm/multiclass] outer fold 5/5
2026-07-16 02:29:33.652 | INFO     | quantum.run:evaluate_fold_quantum:109 - [qsvm/multiclass] fold done: f1_macro=0.0588
```

### nc=3 classical
```
2026-07-16 02:29:39.267 | INFO     | classical.run:run_nested_cv:140 - [random_forest/multiclass] outer fold 1/5
2026-07-16 02:30:19.641 | INFO     | classical.run:evaluate_fold:97 - [random_forest/multiclass] fold done: f1_macro=0.1225
2026-07-16 02:30:20.031 | INFO     | classical.run:run_nested_cv:140 - [random_forest/multiclass] outer fold 2/5
2026-07-16 02:30:34.833 | INFO     | classical.run:evaluate_fold:97 - [random_forest/multiclass] fold done: f1_macro=0.1262
2026-07-16 02:30:35.201 | INFO     | classical.run:run_nested_cv:140 - [random_forest/multiclass] outer fold 3/5
2026-07-16 02:30:59.707 | INFO     | classical.run:evaluate_fold:97 - [random_forest/multiclass] fold done: f1_macro=0.1452
2026-07-16 02:31:00.057 | INFO     | classical.run:run_nested_cv:140 - [random_forest/multiclass] outer fold 4/5
2026-07-16 02:31:23.118 | INFO     | classical.run:evaluate_fold:97 - [random_forest/multiclass] fold done: f1_macro=0.1519
2026-07-16 02:31:23.496 | INFO     | classical.run:run_nested_cv:140 - [random_forest/multiclass] outer fold 5/5
2026-07-16 02:31:54.868 | INFO     | classical.run:evaluate_fold:97 - [random_forest/multiclass] fold done: f1_macro=0.1534
2026-07-16 02:31:55.266 | INFO     | __main__:main:97 - wrote per-fold predictions to results/cic/random_forest_multiclass_predictions.npz
2026-07-16 02:31:55.295 | INFO     | classical.run:run_nested_cv:140 - [xgboost/multiclass] outer fold 1/5
2026-07-16 02:32:50.822 | INFO     | classical.run:evaluate_fold:97 - [xgboost/multiclass] fold done: f1_macro=0.1350
2026-07-16 02:32:51.272 | INFO     | classical.run:run_nested_cv:140 - [xgboost/multiclass] outer fold 2/5
2026-07-16 02:39:38.212 | INFO     | classical.run:evaluate_fold:97 - [xgboost/multiclass] fold done: f1_macro=0.1161
2026-07-16 02:39:39.017 | INFO     | classical.run:run_nested_cv:140 - [xgboost/multiclass] outer fold 3/5
2026-07-16 02:40:37.254 | INFO     | classical.run:evaluate_fold:97 - [xgboost/multiclass] fold done: f1_macro=0.1391
2026-07-16 02:40:37.719 | INFO     | classical.run:run_nested_cv:140 - [xgboost/multiclass] outer fold 4/5
2026-07-16 02:41:13.033 | INFO     | classical.run:evaluate_fold:97 - [xgboost/multiclass] fold done: f1_macro=0.1247
2026-07-16 02:41:14.812 | INFO     | classical.run:run_nested_cv:140 - [xgboost/multiclass] outer fold 5/5
2026-07-16 02:41:54.904 | INFO     | classical.run:evaluate_fold:97 - [xgboost/multiclass] fold done: f1_macro=0.1201
2026-07-16 02:41:55.395 | INFO     | __main__:main:97 - wrote per-fold predictions to results/cic/xgboost_multiclass_predictions.npz
2026-07-16 02:41:55.427 | INFO     | classical.run:run_nested_cv:140 - [lightgbm/multiclass] outer fold 1/5
2026-07-16 02:42:36.580 | INFO     | classical.run:evaluate_fold:97 - [lightgbm/multiclass] fold done: f1_macro=0.1425
2026-07-16 02:42:37.065 | INFO     | classical.run:run_nested_cv:140 - [lightgbm/multiclass] outer fold 2/5
2026-07-16 02:43:54.205 | INFO     | classical.run:evaluate_fold:97 - [lightgbm/multiclass] fold done: f1_macro=0.1007
2026-07-16 02:43:54.704 | INFO     | classical.run:run_nested_cv:140 - [lightgbm/multiclass] outer fold 3/5
2026-07-16 02:45:16.965 | INFO     | classical.run:evaluate_fold:97 - [lightgbm/multiclass] fold done: f1_macro=0.1190
2026-07-16 02:45:17.448 | INFO     | classical.run:run_nested_cv:140 - [lightgbm/multiclass] outer fold 4/5
2026-07-16 02:46:09.116 | INFO     | classical.run:evaluate_fold:97 - [lightgbm/multiclass] fold done: f1_macro=0.1005
2026-07-16 02:46:09.699 | INFO     | classical.run:run_nested_cv:140 - [lightgbm/multiclass] outer fold 5/5
2026-07-16 02:47:03.818 | INFO     | classical.run:evaluate_fold:97 - [lightgbm/multiclass] fold done: f1_macro=0.1300
2026-07-16 02:47:04.976 | INFO     | __main__:main:97 - wrote per-fold predictions to results/cic/lightgbm_multiclass_predictions.npz
2026-07-16 02:47:05.039 | INFO     | classical.run:run_nested_cv:140 - [svm/multiclass] outer fold 1/5
2026-07-16 02:47:07.697 | INFO     | classical.run:evaluate_fold:97 - [svm/multiclass] fold done: f1_macro=0.1340
2026-07-16 02:47:08.553 | INFO     | classical.run:run_nested_cv:140 - [svm/multiclass] outer fold 2/5
2026-07-16 02:47:11.124 | INFO     | classical.run:evaluate_fold:97 - [svm/multiclass] fold done: f1_macro=0.0832
2026-07-16 02:47:11.766 | INFO     | classical.run:run_nested_cv:140 - [svm/multiclass] outer fold 3/5
2026-07-16 02:47:14.365 | INFO     | classical.run:evaluate_fold:97 - [svm/multiclass] fold done: f1_macro=0.1350
2026-07-16 02:47:14.949 | INFO     | classical.run:run_nested_cv:140 - [svm/multiclass] outer fold 4/5
2026-07-16 02:47:16.682 | INFO     | classical.run:evaluate_fold:97 - [svm/multiclass] fold done: f1_macro=0.1650
2026-07-16 02:47:17.137 | INFO     | classical.run:run_nested_cv:140 - [svm/multiclass] outer fold 5/5
2026-07-16 02:47:18.745 | INFO     | classical.run:evaluate_fold:97 - [svm/multiclass] fold done: f1_macro=0.1371
2026-07-16 02:47:19.230 | INFO     | __main__:main:97 - wrote per-fold predictions to results/cic/svm_multiclass_predictions.npz
2026-07-16 02:47:19.241 | INFO     | __main__:main:101 - wrote 4 model x task rows to results/cic/metrics.csv
```

### nc=6 quantum
```
2026-07-16 02:58:21.784 | INFO     | quantum.run:run_quantum_cv:150 - [qsvm/multiclass] outer fold 1/5
2026-07-16 03:05:46.221 | INFO     | quantum.run:evaluate_fold_quantum:111 - [qsvm/multiclass] fold done: f1_macro=0.0836
2026-07-16 03:05:46.790 | INFO     | quantum.run:run_quantum_cv:150 - [qsvm/multiclass] outer fold 2/5
2026-07-16 03:12:55.120 | INFO     | quantum.run:evaluate_fold_quantum:111 - [qsvm/multiclass] fold done: f1_macro=0.0497
2026-07-16 03:12:55.567 | INFO     | quantum.run:run_quantum_cv:150 - [qsvm/multiclass] outer fold 3/5
2026-07-16 03:16:59.345 | INFO     | quantum.run:evaluate_fold_quantum:111 - [qsvm/multiclass] fold done: f1_macro=0.0581
2026-07-16 03:16:59.799 | INFO     | quantum.run:run_quantum_cv:150 - [qsvm/multiclass] outer fold 4/5
2026-07-16 03:21:08.242 | INFO     | quantum.run:evaluate_fold_quantum:111 - [qsvm/multiclass] fold done: f1_macro=0.0804
2026-07-16 03:21:08.632 | INFO     | quantum.run:run_quantum_cv:150 - [qsvm/multiclass] outer fold 5/5
2026-07-16 03:26:25.445 | INFO     | quantum.run:evaluate_fold_quantum:111 - [qsvm/multiclass] fold done: f1_macro=0.0821
```

### nc=6 classical
```
2026-07-16 03:26:30.558 | INFO     | classical.run:run_nested_cv:140 - [random_forest/multiclass] outer fold 1/5
2026-07-16 03:27:04.719 | INFO     | classical.run:evaluate_fold:97 - [random_forest/multiclass] fold done: f1_macro=0.1722
2026-07-16 03:27:05.170 | INFO     | classical.run:run_nested_cv:140 - [random_forest/multiclass] outer fold 2/5
2026-07-16 03:27:34.328 | INFO     | classical.run:evaluate_fold:97 - [random_forest/multiclass] fold done: f1_macro=0.1134
2026-07-16 03:27:34.701 | INFO     | classical.run:run_nested_cv:140 - [random_forest/multiclass] outer fold 3/5
2026-07-16 03:28:01.590 | INFO     | classical.run:evaluate_fold:97 - [random_forest/multiclass] fold done: f1_macro=0.1721
2026-07-16 03:28:01.946 | INFO     | classical.run:run_nested_cv:140 - [random_forest/multiclass] outer fold 4/5
2026-07-16 03:28:17.899 | INFO     | classical.run:evaluate_fold:97 - [random_forest/multiclass] fold done: f1_macro=0.1251
2026-07-16 03:28:18.256 | INFO     | classical.run:run_nested_cv:140 - [random_forest/multiclass] outer fold 5/5
2026-07-16 03:28:50.754 | INFO     | classical.run:evaluate_fold:97 - [random_forest/multiclass] fold done: f1_macro=0.1721
2026-07-16 03:28:51.143 | INFO     | __main__:main:97 - wrote per-fold predictions to results/cic/random_forest_multiclass_predictions.npz
2026-07-16 03:28:51.172 | INFO     | classical.run:run_nested_cv:140 - [xgboost/multiclass] outer fold 1/5
2026-07-16 03:29:50.254 | INFO     | classical.run:evaluate_fold:97 - [xgboost/multiclass] fold done: f1_macro=0.1710
2026-07-16 03:29:50.674 | INFO     | classical.run:run_nested_cv:140 - [xgboost/multiclass] outer fold 2/5
2026-07-16 03:30:33.515 | INFO     | classical.run:evaluate_fold:97 - [xgboost/multiclass] fold done: f1_macro=0.1014
2026-07-16 03:30:33.942 | INFO     | classical.run:run_nested_cv:140 - [xgboost/multiclass] outer fold 3/5
2026-07-16 03:31:24.741 | INFO     | classical.run:evaluate_fold:97 - [xgboost/multiclass] fold done: f1_macro=0.1504
2026-07-16 03:31:25.194 | INFO     | classical.run:run_nested_cv:140 - [xgboost/multiclass] outer fold 4/5
2026-07-16 03:32:36.880 | INFO     | classical.run:evaluate_fold:97 - [xgboost/multiclass] fold done: f1_macro=0.1623
2026-07-16 03:32:37.454 | INFO     | classical.run:run_nested_cv:140 - [xgboost/multiclass] outer fold 5/5
2026-07-16 03:33:31.119 | INFO     | classical.run:evaluate_fold:97 - [xgboost/multiclass] fold done: f1_macro=0.1618
2026-07-16 03:33:31.575 | INFO     | __main__:main:97 - wrote per-fold predictions to results/cic/xgboost_multiclass_predictions.npz
2026-07-16 03:33:31.607 | INFO     | classical.run:run_nested_cv:140 - [lightgbm/multiclass] outer fold 1/5
2026-07-16 03:34:46.522 | INFO     | classical.run:evaluate_fold:97 - [lightgbm/multiclass] fold done: f1_macro=0.1480
2026-07-16 03:34:46.981 | INFO     | classical.run:run_nested_cv:140 - [lightgbm/multiclass] outer fold 2/5
2026-07-16 03:35:51.432 | INFO     | classical.run:evaluate_fold:97 - [lightgbm/multiclass] fold done: f1_macro=0.1395
2026-07-16 03:35:51.923 | INFO     | classical.run:run_nested_cv:140 - [lightgbm/multiclass] outer fold 3/5
2026-07-16 03:37:24.275 | INFO     | classical.run:evaluate_fold:97 - [lightgbm/multiclass] fold done: f1_macro=0.1364
2026-07-16 03:37:24.737 | INFO     | classical.run:run_nested_cv:140 - [lightgbm/multiclass] outer fold 4/5
2026-07-16 03:38:09.672 | INFO     | classical.run:evaluate_fold:97 - [lightgbm/multiclass] fold done: f1_macro=0.1653
2026-07-16 03:38:10.133 | INFO     | classical.run:run_nested_cv:140 - [lightgbm/multiclass] outer fold 5/5
2026-07-16 03:39:05.239 | INFO     | classical.run:evaluate_fold:97 - [lightgbm/multiclass] fold done: f1_macro=0.1965
2026-07-16 03:39:05.886 | INFO     | __main__:main:97 - wrote per-fold predictions to results/cic/lightgbm_multiclass_predictions.npz
2026-07-16 03:39:05.915 | INFO     | classical.run:run_nested_cv:140 - [svm/multiclass] outer fold 1/5
2026-07-16 03:39:07.383 | INFO     | classical.run:evaluate_fold:97 - [svm/multiclass] fold done: f1_macro=0.1456
2026-07-16 03:39:07.719 | INFO     | classical.run:run_nested_cv:140 - [svm/multiclass] outer fold 2/5
2026-07-16 03:39:09.127 | INFO     | classical.run:evaluate_fold:97 - [svm/multiclass] fold done: f1_macro=0.1385
2026-07-16 03:39:09.455 | INFO     | classical.run:run_nested_cv:140 - [svm/multiclass] outer fold 3/5
2026-07-16 03:39:10.763 | INFO     | classical.run:evaluate_fold:97 - [svm/multiclass] fold done: f1_macro=0.1569
2026-07-16 03:39:11.100 | INFO     | classical.run:run_nested_cv:140 - [svm/multiclass] outer fold 4/5
2026-07-16 03:39:12.374 | INFO     | classical.run:evaluate_fold:97 - [svm/multiclass] fold done: f1_macro=0.2010
2026-07-16 03:39:12.716 | INFO     | classical.run:run_nested_cv:140 - [svm/multiclass] outer fold 5/5
2026-07-16 03:39:14.043 | INFO     | classical.run:evaluate_fold:97 - [svm/multiclass] fold done: f1_macro=0.1817
2026-07-16 03:39:14.403 | INFO     | __main__:main:97 - wrote per-fold predictions to results/cic/svm_multiclass_predictions.npz
2026-07-16 03:39:14.405 | INFO     | __main__:main:101 - wrote 4 model x task rows to results/cic/metrics.csv
```


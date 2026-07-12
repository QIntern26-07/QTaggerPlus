# Week 2 Jul 9 Experiments

## Overview: architecture and experiment design

**Architecture updates going into this run** (Day 1 work, branch `feature/quantum-day1-profiling`):

- **Batched kernel evaluation** — `QSVM.gram`/`_gram_sym` (`src/quantum/qsvm.py`) now flatten
  pair indices and dispatch the fidelity-kernel QNode once per chunk (`batch_size=4096`) via
  PennyLane parameter broadcasting, instead of one QNode call per pair in a Python double loop.
  Measured ~4.4x wall-clock speedup at n=400 vs. the pre-batching baseline
  (`docs/reports/w2_day1_quantum_profiling_baseline_report.md` vs.
  `w2_day1_quantum_profiling_report.md`).
- **`lightning.qubit`** confirmed as the local simulator backend (`_resolve_device()`), ~1.5x
  faster than `default.qubit` on this machine.
- **Shared preprocessing pipeline** (`common.preprocess.build_feature_pipeline`): variance filter
  → correlation filter → `StandardScaler` → PCA to `n_components`, fit on the train fold only.
  PCA output dimensionality is the single alignment point between classical and quantum models —
  both consume the identical post-PCA features, and `n_components` sets the qubit budget directly
  (`quantum.encoding.n_qubits_for`).
- **Two-tier tuning** (`quantum.run.tune_and_fit_qsvm`): the kernel Gram depends only on
  `(encoding, bandwidth)`, not on `C`/`class_weight`, so the inner train/val Gram is built once per
  encoding candidate and `C`/`class_weight` are swept cheaply on the cached Gram — `encoding` is
  therefore a **tuned hyperparameter**, selected per outer fold by inner-CV macro-F1, not something
  run and reported separately per encoding.
- **Cosmetic fixes**: removed an sklearn `FutureWarning` from `SVC(probability=False)` (now omits
  the kwarg entirely, sklearn's default is already disabled), and suppressed a benign upstream
  LightGBM/sklearn "feature names" warning that fires even with plain numpy input.

**Experiment design for this run:**

- **Dataset**: CIC-MalMem-2022, binary task (malware vs. benign).
- **Sample-size ramp**: 200 samples first (cheap correctness check), then 1000 samples, rather
  than jumping straight to a large run — 1000 samples is ~25x more kernel pairs than 200
  (`O(n^2)`), so validating the pipeline cheaply first avoids wasting time on a broken run at scale.
- **`n_components=1`** (1 qubit) — the smallest possible quantum feature space, chosen deliberately
  as the starting point per the plan ("test from 1 compressed dim or qubit first").
- **Encodings**: both `angle` and `iqp` given to the CLI (`--encodings angle iqp`); the QSVM
  two-tier tuning picks the winner per fold — it is not a forced/fixed choice.
- **5-fold stratified CV**, same fold indices shared between quantum and classical via
  `data/splits/quantum_sample_idx.json` + `--load-quantum-splits`, so the comparison is
  apples-to-apples rather than two independently-sampled runs.
- **Classical baselines**: Random Forest, XGBoost, LightGBM, SVM (`src/classical`), run on the
  identical folds and `n_components=1` PCA features as the quantum run, via nested-CV
  (Optuna-tuned inner loop, single held-out outer test fold per iteration).
- All runs logged to the local MLflow experiment `qtaggerplus` (`sqlite:///mlflow.db` +
  `./mlruns` artifacts).

## Stage 0. - Sanity probe

Before committing to any CV/tuning run, confirm the pipeline executes end-to-end at n_components=1 
and see a single-fit timing number:

`--probe` skips CV/tuning entirely (one untuned 80/20 fit), so this just catches broken pipelines or 
wildly-off timing before you spend minutes on a full sweep.


```sh
uv run python -m quantum --n-components 1 --max-samples 200 --encodings angle --probe
2026-07-09 18:02:55.756 | INFO     | __main__:main:73 - [probe] angle nc=1 kernel_train=1.760s fit=1.765s infer=0.987s

uv run python -m quantum --n-components 1 --max-samples 200 --encodings iqp --probe
2026-07-09 18:03:25.163 | INFO     | __main__:main:73 - [probe] iqp nc=1 kernel_train=1.125s fit=1.129s infer=0.570s

uv run python -m quantum --n-components 1 --max-samples 1000 --encodings angle --probe
2026-07-09 18:28:23.878 | INFO     | __main__:main:73 - [probe] angle nc=1 kernel_train=48.489s fit=48.497s infer=24.270s

uv run python -m quantum --n-components 1 --max-samples 1000 --encodings iqp --probe
2026-07-09 18:30:00.910 | INFO     | __main__:main:73 - [probe] iqp nc=1 kernel_train=32.285s fit=32.294s infer=18.216s
```

Result: The E2E pipeline ran successfully.

## Stage 1. 200 samples

### Stage 1.1. Quantum


```sh
uv run python -m quantum --n-components 1 --max-samples 200 --folds 5 \
    --encodings angle iqp --tasks binary --mlflow
2026-07-09 18:09:51.577 | INFO     | quantum.run:run_quantum_cv:120 - [qsvm/binary] outer fold 1/5
2026-07-09 18:09:57.479 | INFO     | quantum.run:evaluate_fold_quantum:100 - [qsvm/binary] fold done: f1_macro=1.0000
2026/07/09 18:09:58 INFO mlflow.store.db.utils: Creating initial MLflow database tables...
2026/07/09 18:09:58 INFO mlflow.store.db.utils: Updating database tables
2026/07/09 18:09:59 INFO mlflow.tracking.fluent: Experiment with name 'qtaggerplus' does not exist. Creating a new experiment.
2026-07-09 18:09:59.670 | INFO     | quantum.run:run_quantum_cv:120 - [qsvm/binary] outer fold 2/5
2026-07-09 18:10:05.571 | INFO     | quantum.run:evaluate_fold_quantum:100 - [qsvm/binary] fold done: f1_macro=0.9500
2026-07-09 18:10:05.662 | INFO     | quantum.run:run_quantum_cv:120 - [qsvm/binary] outer fold 3/5
2026-07-09 18:10:11.751 | INFO     | quantum.run:evaluate_fold_quantum:100 - [qsvm/binary] fold done: f1_macro=1.0000
2026-07-09 18:10:11.898 | INFO     | quantum.run:run_quantum_cv:120 - [qsvm/binary] outer fold 4/5
2026-07-09 18:10:18.358 | INFO     | quantum.run:evaluate_fold_quantum:100 - [qsvm/binary] fold done: f1_macro=0.9499
2026-07-09 18:10:18.502 | INFO     | quantum.run:run_quantum_cv:120 - [qsvm/binary] outer fold 5/5
2026-07-09 18:10:25.877 | INFO     | quantum.run:evaluate_fold_quantum:100 - [qsvm/binary] fold done: f1_macro=1.0000
```

### Stage 1.2. Classical

```sh
uv run python -m classical --n-components 1 --load-quantum-splits --tasks binary --mlflow
2026-07-09 18:12:59.395 | INFO     | classical.run:run_nested_cv:99 - [random_forest/binary] outer fold 1/5
2026-07-09 18:13:13.270 | INFO     | classical.run:evaluate_fold:76 - [random_forest/binary] fold done: f1_macro=1.0000
2026-07-09 18:13:13.939 | INFO     | classical.run:run_nested_cv:99 - [random_forest/binary] outer fold 2/5
2026-07-09 18:13:23.962 | INFO     | classical.run:evaluate_fold:76 - [random_forest/binary] fold done: f1_macro=0.9500
2026-07-09 18:13:24.081 | INFO     | classical.run:run_nested_cv:99 - [random_forest/binary] outer fold 3/5
2026-07-09 18:13:34.948 | INFO     | classical.run:evaluate_fold:76 - [random_forest/binary] fold done: f1_macro=1.0000
2026-07-09 18:13:35.065 | INFO     | classical.run:run_nested_cv:99 - [random_forest/binary] outer fold 4/5
2026-07-09 18:13:46.168 | INFO     | classical.run:evaluate_fold:76 - [random_forest/binary] fold done: f1_macro=1.0000
2026-07-09 18:13:46.285 | INFO     | classical.run:run_nested_cv:99 - [random_forest/binary] outer fold 5/5
2026-07-09 18:13:56.324 | INFO     | classical.run:evaluate_fold:76 - [random_forest/binary] fold done: f1_macro=1.0000
2026-07-09 18:13:56.436 | INFO     | __main__:main:102 - wrote per-fold predictions to results/cic/random_forest_binary_predictions.npz
2026-07-09 18:13:56.436 | INFO     | classical.run:run_nested_cv:99 - [xgboost/binary] outer fold 1/5
2026-07-09 18:14:00.803 | INFO     | classical.run:evaluate_fold:76 - [xgboost/binary] fold done: f1_macro=1.0000
2026-07-09 18:14:00.931 | INFO     | classical.run:run_nested_cv:99 - [xgboost/binary] outer fold 2/5
2026-07-09 18:14:01.912 | INFO     | classical.run:evaluate_fold:76 - [xgboost/binary] fold done: f1_macro=0.9500
2026-07-09 18:14:02.056 | INFO     | classical.run:run_nested_cv:99 - [xgboost/binary] outer fold 3/5
2026-07-09 18:14:02.981 | INFO     | classical.run:evaluate_fold:76 - [xgboost/binary] fold done: f1_macro=1.0000
2026-07-09 18:14:03.113 | INFO     | classical.run:run_nested_cv:99 - [xgboost/binary] outer fold 4/5
2026-07-09 18:14:03.979 | INFO     | classical.run:evaluate_fold:76 - [xgboost/binary] fold done: f1_macro=1.0000
2026-07-09 18:14:04.110 | INFO     | classical.run:run_nested_cv:99 - [xgboost/binary] outer fold 5/5
2026-07-09 18:14:04.996 | INFO     | classical.run:evaluate_fold:76 - [xgboost/binary] fold done: f1_macro=1.0000
2026-07-09 18:14:05.127 | INFO     | __main__:main:102 - wrote per-fold predictions to results/cic/xgboost_binary_predictions.npz
2026-07-09 18:14:05.127 | INFO     | classical.run:run_nested_cv:99 - [lightgbm/binary] outer fold 1/5
2026-07-09 18:14:07.534 | INFO     | classical.run:evaluate_fold:76 - [lightgbm/binary] fold done: f1_macro=1.0000
2026-07-09 18:14:07.707 | INFO     | classical.run:run_nested_cv:99 - [lightgbm/binary] outer fold 2/5
2026-07-09 18:14:08.618 | INFO     | classical.run:evaluate_fold:76 - [lightgbm/binary] fold done: f1_macro=0.9500
2026-07-09 18:14:08.784 | INFO     | classical.run:run_nested_cv:99 - [lightgbm/binary] outer fold 3/5
2026-07-09 18:14:09.917 | INFO     | classical.run:evaluate_fold:76 - [lightgbm/binary] fold done: f1_macro=1.0000
2026-07-09 18:14:10.110 | INFO     | classical.run:run_nested_cv:99 - [lightgbm/binary] outer fold 4/5
2026-07-09 18:14:11.382 | INFO     | classical.run:evaluate_fold:76 - [lightgbm/binary] fold done: f1_macro=1.0000
2026-07-09 18:14:11.571 | INFO     | classical.run:run_nested_cv:99 - [lightgbm/binary] outer fold 5/5
2026-07-09 18:14:12.682 | INFO     | classical.run:evaluate_fold:76 - [lightgbm/binary] fold done: f1_macro=1.0000
2026-07-09 18:14:13.007 | INFO     | __main__:main:102 - wrote per-fold predictions to results/cic/lightgbm_binary_predictions.npz
2026-07-09 18:14:13.007 | INFO     | classical.run:run_nested_cv:99 - [svm/binary] outer fold 1/5
2026-07-09 18:14:13.407 | INFO     | classical.run:evaluate_fold:76 - [svm/binary] fold done: f1_macro=1.0000
2026-07-09 18:14:13.569 | INFO     | classical.run:run_nested_cv:99 - [svm/binary] outer fold 2/5
2026-07-09 18:14:13.970 | INFO     | classical.run:evaluate_fold:76 - [svm/binary] fold done: f1_macro=0.9500
2026-07-09 18:14:14.131 | INFO     | classical.run:run_nested_cv:99 - [svm/binary] outer fold 3/5
2026-07-09 18:14:14.534 | INFO     | classical.run:evaluate_fold:76 - [svm/binary] fold done: f1_macro=1.0000
2026-07-09 18:14:14.693 | INFO     | classical.run:run_nested_cv:99 - [svm/binary] outer fold 4/5
2026-07-09 18:14:15.094 | INFO     | classical.run:evaluate_fold:76 - [svm/binary] fold done: f1_macro=1.0000
2026-07-09 18:14:15.254 | INFO     | classical.run:run_nested_cv:99 - [svm/binary] outer fold 5/5
2026-07-09 18:14:15.657 | INFO     | classical.run:evaluate_fold:76 - [svm/binary] fold done: f1_macro=1.0000
2026-07-09 18:14:15.816 | INFO     | __main__:main:102 - wrote per-fold predictions to results/cic/svm_binary_predictions.npz
2026-07-09 18:14:15.821 | INFO     | __main__:main:106 - wrote 4 model x task rows to results/cic/metrics.csv
```

## Stage 2. 1000 samples

### Stage 2.1. Quantum

```sh
uv run python -m quantum --n-components 1 --max-samples 1000 --folds 5 \
    --encodings angle iqp --tasks binary --mlflow
2026-07-09 18:31:38.156 | INFO     | quantum.run:run_quantum_cv:120 - [qsvm/binary] outer fold 1/5
2026-07-09 18:34:05.941 | INFO     | quantum.run:evaluate_fold_quantum:100 - [qsvm/binary] fold done: f1_macro=0.9650
2026-07-09 18:34:06.728 | INFO     | quantum.run:run_quantum_cv:120 - [qsvm/binary] outer fold 2/5
2026-07-09 18:36:48.250 | INFO     | quantum.run:evaluate_fold_quantum:100 - [qsvm/binary] fold done: f1_macro=0.9900
2026-07-09 18:36:48.345 | INFO     | quantum.run:run_quantum_cv:120 - [qsvm/binary] outer fold 3/5
2026-07-09 18:39:37.702 | INFO     | quantum.run:evaluate_fold_quantum:100 - [qsvm/binary] fold done: f1_macro=1.0000
2026-07-09 18:39:37.797 | INFO     | quantum.run:run_quantum_cv:120 - [qsvm/binary] outer fold 4/5
2026-07-09 18:42:11.680 | INFO     | quantum.run:evaluate_fold_quantum:100 - [qsvm/binary] fold done: f1_macro=0.9950
2026-07-09 18:42:11.798 | INFO     | quantum.run:run_quantum_cv:120 - [qsvm/binary] outer fold 5/5
2026-07-09 18:45:06.251 | INFO     | quantum.run:evaluate_fold_quantum:100 - [qsvm/binary] fold done: f1_macro=0.9398
```

### Stage 2.2. Classical

```sh
uv run python -m classical --n-components 1 --load-quantum-splits --tasks binary --mlflow
2026-07-09 18:46:28.338 | INFO     | classical.run:run_nested_cv:112 - [random_forest/binary] outer fold 1/5
2026-07-09 18:46:45.698 | INFO     | classical.run:evaluate_fold:89 - [random_forest/binary] fold done: f1_macro=0.9800
2026-07-09 18:46:47.053 | INFO     | classical.run:run_nested_cv:112 - [random_forest/binary] outer fold 2/5
2026-07-09 18:47:02.214 | INFO     | classical.run:evaluate_fold:89 - [random_forest/binary] fold done: f1_macro=0.9950
2026-07-09 18:47:02.335 | INFO     | classical.run:run_nested_cv:112 - [random_forest/binary] outer fold 3/5
2026-07-09 18:47:13.628 | INFO     | classical.run:evaluate_fold:89 - [random_forest/binary] fold done: f1_macro=1.0000
2026-07-09 18:47:13.784 | INFO     | classical.run:run_nested_cv:112 - [random_forest/binary] outer fold 4/5
^[	2026-07-09 18:47:25.446 | INFO     | classical.run:evaluate_fold:89 - [random_forest/binary] fold done: f1_macro=0.9950
2026-07-09 18:47:25.570 | INFO     | classical.run:run_nested_cv:112 - [random_forest/binary] outer fold 5/5
2026-07-09 18:47:37.480 | INFO     | classical.run:evaluate_fold:89 - [random_forest/binary] fold done: f1_macro=0.9900
2026-07-09 18:47:37.608 | INFO     | __main__:main:102 - wrote per-fold predictions to results/cic/random_forest_binary_predictions.npz
2026-07-09 18:47:37.609 | INFO     | classical.run:run_nested_cv:112 - [xgboost/binary] outer fold 1/5
2026-07-09 18:47:43.004 | INFO     | classical.run:evaluate_fold:89 - [xgboost/binary] fold done: f1_macro=0.9800
2026-07-09 18:47:43.170 | INFO     | classical.run:run_nested_cv:112 - [xgboost/binary] outer fold 2/5
2026-07-09 18:47:44.713 | INFO     | classical.run:evaluate_fold:89 - [xgboost/binary] fold done: f1_macro=0.9950
2026-07-09 18:47:44.851 | INFO     | classical.run:run_nested_cv:112 - [xgboost/binary] outer fold 3/5
2026-07-09 18:47:46.554 | INFO     | classical.run:evaluate_fold:89 - [xgboost/binary] fold done: f1_macro=1.0000
2026-07-09 18:47:46.705 | INFO     | classical.run:run_nested_cv:112 - [xgboost/binary] outer fold 4/5
2026-07-09 18:47:51.126 | INFO     | classical.run:evaluate_fold:89 - [xgboost/binary] fold done: f1_macro=0.9950
2026-07-09 18:47:51.300 | INFO     | classical.run:run_nested_cv:112 - [xgboost/binary] outer fold 5/5
2026-07-09 18:47:52.847 | INFO     | classical.run:evaluate_fold:89 - [xgboost/binary] fold done: f1_macro=0.9900
2026-07-09 18:47:52.977 | INFO     | __main__:main:102 - wrote per-fold predictions to results/cic/xgboost_binary_predictions.npz
2026-07-09 18:47:52.977 | INFO     | classical.run:run_nested_cv:112 - [lightgbm/binary] outer fold 1/5
2026-07-09 18:47:55.946 | INFO     | classical.run:evaluate_fold:89 - [lightgbm/binary] fold done: f1_macro=0.9800
2026-07-09 18:47:56.078 | INFO     | classical.run:run_nested_cv:112 - [lightgbm/binary] outer fold 2/5
2026-07-09 18:47:57.702 | INFO     | classical.run:evaluate_fold:89 - [lightgbm/binary] fold done: f1_macro=0.9900
2026-07-09 18:47:57.845 | INFO     | classical.run:run_nested_cv:112 - [lightgbm/binary] outer fold 3/5
2026-07-09 18:47:59.889 | INFO     | classical.run:evaluate_fold:89 - [lightgbm/binary] fold done: f1_macro=1.0000
2026-07-09 18:48:00.156 | INFO     | classical.run:run_nested_cv:112 - [lightgbm/binary] outer fold 4/5
2026-07-09 18:48:02.267 | INFO     | classical.run:evaluate_fold:89 - [lightgbm/binary] fold done: f1_macro=0.9950
2026-07-09 18:48:02.422 | INFO     | classical.run:run_nested_cv:112 - [lightgbm/binary] outer fold 5/5
2026-07-09 18:48:04.206 | INFO     | classical.run:evaluate_fold:89 - [lightgbm/binary] fold done: f1_macro=0.9900
2026-07-09 18:48:04.338 | INFO     | __main__:main:102 - wrote per-fold predictions to results/cic/lightgbm_binary_predictions.npz
2026-07-09 18:48:04.338 | INFO     | classical.run:run_nested_cv:112 - [svm/binary] outer fold 1/5
2026-07-09 18:48:04.850 | INFO     | classical.run:evaluate_fold:89 - [svm/binary] fold done: f1_macro=0.9800
2026-07-09 18:48:04.978 | INFO     | classical.run:run_nested_cv:112 - [svm/binary] outer fold 2/5
2026-07-09 18:48:05.538 | INFO     | classical.run:evaluate_fold:89 - [svm/binary] fold done: f1_macro=0.9950
2026-07-09 18:48:05.668 | INFO     | classical.run:run_nested_cv:112 - [svm/binary] outer fold 3/5
2026-07-09 18:48:06.182 | INFO     | classical.run:evaluate_fold:89 - [svm/binary] fold done: f1_macro=1.0000
2026-07-09 18:48:06.301 | INFO     | classical.run:run_nested_cv:112 - [svm/binary] outer fold 4/5
2026-07-09 18:48:06.829 | INFO     | classical.run:evaluate_fold:89 - [svm/binary] fold done: f1_macro=0.9950
2026-07-09 18:48:06.952 | INFO     | classical.run:run_nested_cv:112 - [svm/binary] outer fold 5/5
2026-07-09 18:48:07.466 | INFO     | classical.run:evaluate_fold:89 - [svm/binary] fold done: f1_macro=0.9950
2026-07-09 18:48:07.593 | INFO     | __main__:main:102 - wrote per-fold predictions to results/cic/svm_binary_predictions.npz
2026-07-09 18:48:07.595 | INFO     | __main__:main:106 - wrote 4 model x task rows to results/cic/metrics.csv
```

## Results

Pulled from MLflow (`qtaggerplus` experiment) rather than terminal logs, since the console only
prints `f1_macro` per fold — the table below adds ROC-AUC, accuracy, MCC, timing, and (for QSVM)
which encoding won each fold's inner-CV tuning.

### Comparison table (mean ± std across 5 outer folds, binary task, n_components=1)

| n_samples | model | f1_macro | roc_auc | accuracy | mcc | fit_time (s) | tune_time (s) | infer_time (s) |
|---|---|---|---|---|---|---|---|---|
| 200 | qsvm (angle) | 0.980 ± 0.027 | 0.981 | 0.980 | 0.961 | 2.03 | 3.22 | 1.09 |
| 200 | random_forest | 0.990 ± 0.022 | 0.990 | 0.990 | 0.980 | 0.41 | 10.68 | 0.04 |
| 200 | xgboost | 0.990 ± 0.022 | 0.990 | 0.990 | 0.980 | 0.11 | 1.48 | 0.001 |
| 200 | lightgbm | 0.990 ± 0.022 | 0.994 | 0.990 | 0.980 | 0.17 | 1.18 | 0.002 |
| 200 | svm | 0.990 ± 0.022 | 0.981 | 0.990 | 0.980 | 0.003 | 0.39 | 0.0003 |
| 1000 | qsvm (angle) | 0.978 ± 0.025 | 0.992 | 0.978 | 0.957 | 52.94 | 83.51 | 24.94 |
| 1000 | random_forest | 0.992 ± 0.008 | 0.995 | 0.992 | 0.984 | 0.57 | 12.77 | 0.06 |
| 1000 | xgboost | 0.992 ± 0.008 | 0.997 | 0.992 | 0.984 | 0.68 | 2.21 | 0.003 |
| 1000 | lightgbm | 0.991 ± 0.007 | 0.996 | 0.991 | 0.982 | 0.34 | 1.74 | 0.002 |
| 1000 | svm | 0.993 ± 0.008 | 0.994 | 0.993 | 0.986 | 0.004 | 0.50 | 0.0005 |

### QSVM per-fold detail (encoding tuning outcome)

| n_samples | fold | encoding selected | f1_macro | roc_auc | kernel_build_train_s | kernel_build_test_s |
|---|---|---|---|---|---|---|
| 200 | 0 | angle | 1.000 | 1.000 | 1.99 | 0.89 |
| 200 | 1 | angle | 0.950 | 0.905 | 1.81 | 1.02 |
| 200 | 2 | angle | 1.000 | 1.000 | 1.86 | 1.15 |
| 200 | 3 | angle | 0.950 | 1.000 | 2.27 | 1.23 |
| 200 | 4 | angle | 1.000 | 1.000 | 2.20 | 1.18 |
| 1000 | 0 | angle | 0.965 | 0.981 | 48.81 | 24.14 |
| 1000 | 1 | angle | 0.990 | 0.999 | 50.30 | 24.46 |
| 1000 | 2 | angle | 1.000 | 1.000 | 57.47 | 25.24 |
| 1000 | 3 | angle | 0.995 | 0.999 | 51.74 | 26.21 |
| 1000 | 4 | angle | 0.940 | 0.982 | 56.34 | 24.65 |

`iqp` was included in the tuning grid for every fold (`--encodings angle iqp`) but **never won** —
`angle` was selected in all 10/10 folds across both sample sizes.

### Analysis

- **`angle` swept every fold; `iqp` never won a single one.** At `n_components=1` (1 qubit),
  this is expected rather than surprising: `iqp`'s entangling `MultiRZ` terms operate on *pairs* of
  encoded features (per the Day 1 circuit-spec profiling, `iqp`'s extra cost only appears at
  `n_qubits ≥ 2`) — with only one feature, there is no pair to correlate, so `iqp` degenerates to
  something functionally very close to `angle` with no structural advantage to exploit. The
  literature reference motivating `iqp` (R1, arXiv:2510.06803) reports gains at higher-dimensional
  encodings, not at 1 qubit — this result doesn't contradict that, it just hasn't tested the regime
  where `iqp` could plausibly win yet.
- **Quantum accuracy is competitive but not superior to classical at this scale.** QSVM's
  `f1_macro` (0.978–0.980) sits slightly below every classical model (0.990–0.993) at both sample
  sizes, though within roughly one std of the classical spread — not a large gap, but also no
  quantum advantage observed yet.
- **Runtime is the real story.** QSVM's fit+tune+infer time exploded from ~6.3s combined at
  n=200 to ~161s at n=1000 (~26x for a 5x sample increase), consistent with the `O(n^2)`
  kernel-matrix cost characterized in the Day 1 profiling report. Every classical model, by
  contrast, stayed under ~15s total even at n=1000. This gap will only widen as sample size grows
  toward the 1000+ scales needed for the EMBER/SOREL-20M work — reinforcing that the Day 2 sizing
  decision (still deferred, `docs/quantum_todo.md`) is the binding constraint on how far this can
  scale, not classification quality.
- **Next step implied by this result**: re-run at `n_components=2` before drawing conclusions
  about `angle` vs. `iqp` — 1 qubit structurally can't give `iqp` a fair test, and the Day 1
  profiling report already shows `angle`/`iqp` cost parity holds at `n_components=2` too, so that
  step doesn't cost anything extra to find out.

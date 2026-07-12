# Week 2 Jul 11 Experiments - 3 components

Today's work focusing on the result and benchmark of number qubits from 2 to 6.

## Overview: architecture and experiment design

Same as in [week 2 first report](./w2_Jul-9_experiments.md).

## Stage 0. - Sanity probe

```sh
uv run python -m quantum --n-components 3 --max-samples 200 --encodings angle --probe
2026-07-11 15:43:13.484 | INFO     | __main__:main:73 - [probe] angle nc=3 kernel_train=3.525s fit=3.531s infer=1.647s

uv run python -m quantum --n-components 3 --max-samples 200 --encodings iqp --probe
2026-07-11 15:43:45.889 | INFO     | __main__:main:73 - [probe] iqp nc=3 kernel_train=3.644s fit=3.648s infer=2.212s

uv run python -m quantum --n-components 2 --max-samples 1000 --encodings angle --probe
2026-07-11 15:47:04.172 | INFO     | __main__:main:73 - [probe] angle nc=3 kernel_train=98.961s fit=98.970s infer=52.715s

uv run python -m quantum --n-components 2 --max-samples 1000 --encodings iqp --probe
2026-07-11 15:50:46.458 | INFO     | __main__:main:73 - [probe] iqp nc=3 kernel_train=107.191s fit=107.198s infer=54.948s
```

## Stage 1. 200 samples

### 1.1. Quantum

```sh
uv run python -m quantum --n-components 3 --max-samples 200 --folds 5 \
    --encodings angle iqp --tasks binary --mlflow
2026-07-11 15:52:44.441 | INFO     | quantum.run:run_quantum_cv:120 - [qsvm/binary] outer fold 1/5
2026-07-11 15:52:56.564 | INFO     | quantum.run:evaluate_fold_quantum:100 - [qsvm/binary] fold done: f1_macro=1.0000
2026-07-11 15:52:57.326 | INFO     | quantum.run:run_quantum_cv:120 - [qsvm/binary] outer fold 2/5
2026-07-11 15:53:11.338 | INFO     | quantum.run:evaluate_fold_quantum:100 - [qsvm/binary] fold done: f1_macro=0.9500
2026-07-11 15:53:11.464 | INFO     | quantum.run:run_quantum_cv:120 - [qsvm/binary] outer fold 3/5
2026-07-11 15:53:27.874 | INFO     | quantum.run:evaluate_fold_quantum:100 - [qsvm/binary] fold done: f1_macro=0.9750
2026-07-11 15:53:28.040 | INFO     | quantum.run:run_quantum_cv:120 - [qsvm/binary] outer fold 4/5
2026-07-11 15:53:44.712 | INFO     | quantum.run:evaluate_fold_quantum:100 - [qsvm/binary] fold done: f1_macro=1.0000
2026-07-11 15:53:44.836 | INFO     | quantum.run:run_quantum_cv:120 - [qsvm/binary] outer fold 5/5
2026-07-11 15:54:00.304 | INFO     | quantum.run:evaluate_fold_quantum:100 - [qsvm/binary] fold done: f1_macro=1.0000
```

### 1.2. Classicals

```sh
uv run python -m classical --n-components 3 --load-quantum-splits --tasks binary --mlflow
2026-07-11 15:56:11.592 | INFO     | classical.run:run_nested_cv:112 - [random_forest/binary] outer fold 1/5
2026-07-11 15:56:36.137 | INFO     | classical.run:evaluate_fold:89 - [random_forest/binary] fold done: f1_macro=1.0000
2026-07-11 15:56:37.375 | INFO     | classical.run:run_nested_cv:112 - [random_forest/binary] outer fold 2/5
2026-07-11 15:56:53.239 | INFO     | classical.run:evaluate_fold:89 - [random_forest/binary] fold done: f1_macro=0.9500
2026-07-11 15:56:53.403 | INFO     | classical.run:run_nested_cv:112 - [random_forest/binary] outer fold 3/5
2026-07-11 15:57:06.515 | INFO     | classical.run:evaluate_fold:89 - [random_forest/binary] fold done: f1_macro=1.0000
2026-07-11 15:57:06.676 | INFO     | classical.run:run_nested_cv:112 - [random_forest/binary] outer fold 4/5
2026-07-11 15:57:20.068 | INFO     | classical.run:evaluate_fold:89 - [random_forest/binary] fold done: f1_macro=1.0000
2026-07-11 15:57:20.218 | INFO     | classical.run:run_nested_cv:112 - [random_forest/binary] outer fold 5/5
2026-07-11 15:57:34.236 | INFO     | classical.run:evaluate_fold:89 - [random_forest/binary] fold done: f1_macro=1.0000
2026-07-11 15:57:34.387 | INFO     | __main__:main:102 - wrote per-fold predictions to results/cic/random_forest_binary_predictions.npz
2026-07-11 15:57:34.387 | INFO     | classical.run:run_nested_cv:112 - [xgboost/binary] outer fold 1/5
2026-07-11 15:57:39.780 | INFO     | classical.run:evaluate_fold:89 - [xgboost/binary] fold done: f1_macro=1.0000
2026-07-11 15:57:39.969 | INFO     | classical.run:run_nested_cv:112 - [xgboost/binary] outer fold 2/5
2026-07-11 15:57:41.964 | INFO     | classical.run:evaluate_fold:89 - [xgboost/binary] fold done: f1_macro=0.9500
2026-07-11 15:57:42.213 | INFO     | classical.run:run_nested_cv:112 - [xgboost/binary] outer fold 3/5
2026-07-11 15:57:43.830 | INFO     | classical.run:evaluate_fold:89 - [xgboost/binary] fold done: f1_macro=1.0000
2026-07-11 15:57:44.024 | INFO     | classical.run:run_nested_cv:112 - [xgboost/binary] outer fold 4/5
2026-07-11 15:57:45.800 | INFO     | classical.run:evaluate_fold:89 - [xgboost/binary] fold done: f1_macro=1.0000
2026-07-11 15:57:46.020 | INFO     | classical.run:run_nested_cv:112 - [xgboost/binary] outer fold 5/5
2026-07-11 15:57:48.608 | INFO     | classical.run:evaluate_fold:89 - [xgboost/binary] fold done: f1_macro=1.0000
2026-07-11 15:57:48.831 | INFO     | __main__:main:102 - wrote per-fold predictions to results/cic/xgboost_binary_predictions.npz
2026-07-11 15:57:48.831 | INFO     | classical.run:run_nested_cv:112 - [lightgbm/binary] outer fold 1/5
2026-07-11 15:57:52.752 | INFO     | classical.run:evaluate_fold:89 - [lightgbm/binary] fold done: f1_macro=1.0000
2026-07-11 15:57:52.955 | INFO     | classical.run:run_nested_cv:112 - [lightgbm/binary] outer fold 2/5
2026-07-11 15:57:54.201 | INFO     | classical.run:evaluate_fold:89 - [lightgbm/binary] fold done: f1_macro=0.9500
2026-07-11 15:57:54.368 | INFO     | classical.run:run_nested_cv:112 - [lightgbm/binary] outer fold 3/5
2026-07-11 15:57:55.782 | INFO     | classical.run:evaluate_fold:89 - [lightgbm/binary] fold done: f1_macro=0.9750
2026-07-11 15:57:55.984 | INFO     | classical.run:run_nested_cv:112 - [lightgbm/binary] outer fold 4/5
2026-07-11 15:57:57.416 | INFO     | classical.run:evaluate_fold:89 - [lightgbm/binary] fold done: f1_macro=1.0000
2026-07-11 15:57:57.603 | INFO     | classical.run:run_nested_cv:112 - [lightgbm/binary] outer fold 5/5
2026-07-11 15:57:59.368 | INFO     | classical.run:evaluate_fold:89 - [lightgbm/binary] fold done: f1_macro=1.0000
2026-07-11 15:57:59.570 | INFO     | __main__:main:102 - wrote per-fold predictions to results/cic/lightgbm_binary_predictions.npz
2026-07-11 15:57:59.570 | INFO     | classical.run:run_nested_cv:112 - [svm/binary] outer fold 1/5
2026-07-11 15:58:00.028 | INFO     | classical.run:evaluate_fold:89 - [svm/binary] fold done: f1_macro=1.0000
2026-07-11 15:58:00.221 | INFO     | classical.run:run_nested_cv:112 - [svm/binary] outer fold 2/5
2026-07-11 15:58:00.649 | INFO     | classical.run:evaluate_fold:89 - [svm/binary] fold done: f1_macro=0.9500
2026-07-11 15:58:00.808 | INFO     | classical.run:run_nested_cv:112 - [svm/binary] outer fold 3/5
2026-07-11 15:58:01.213 | INFO     | classical.run:evaluate_fold:89 - [svm/binary] fold done: f1_macro=0.9750
2026-07-11 15:58:01.371 | INFO     | classical.run:run_nested_cv:112 - [svm/binary] outer fold 4/5
2026-07-11 15:58:01.772 | INFO     | classical.run:evaluate_fold:89 - [svm/binary] fold done: f1_macro=1.0000
2026-07-11 15:58:01.927 | INFO     | classical.run:run_nested_cv:112 - [svm/binary] outer fold 5/5
2026-07-11 15:58:02.361 | INFO     | classical.run:evaluate_fold:89 - [svm/binary] fold done: f1_macro=0.9750
2026-07-11 15:58:02.535 | INFO     | __main__:main:102 - wrote per-fold predictions to results/cic/svm_binary_predictions.npz
2026-07-11 15:58:02.537 | INFO     | __main__:main:106 - wrote 4 model x task rows to results/cic/metrics.csv
```

## Stage 2. 1000 samples

### 2.1. Quantum

```sh
uv run python -m quantum --n-components 3 --max-samples 1000 --folds 5 \
    --encodings angle iqp --tasks binary --mlflow
2026-07-11 15:59:27.154 | INFO     | quantum.run:run_quantum_cv:120 - [qsvm/binary] outer fold 1/5
2026-07-11 16:05:54.083 | INFO     | quantum.run:evaluate_fold_quantum:100 - [qsvm/binary] fold done: f1_macro=0.9800
2026-07-11 16:05:54.987 | INFO     | quantum.run:run_quantum_cv:120 - [qsvm/binary] outer fold 2/5
2026-07-11 16:12:04.312 | INFO     | quantum.run:evaluate_fold_quantum:100 - [qsvm/binary] fold done: f1_macro=0.9950
2026-07-11 16:12:04.421 | INFO     | quantum.run:run_quantum_cv:120 - [qsvm/binary] outer fold 3/5
2026-07-11 16:18:14.421 | INFO     | quantum.run:evaluate_fold_quantum:100 - [qsvm/binary] fold done: f1_macro=0.9950
2026-07-11 16:18:14.543 | INFO     | quantum.run:run_quantum_cv:120 - [qsvm/binary] outer fold 4/5
2026-07-11 16:24:32.422 | INFO     | quantum.run:evaluate_fold_quantum:100 - [qsvm/binary] fold done: f1_macro=0.9950
2026-07-11 16:24:32.547 | INFO     | quantum.run:run_quantum_cv:120 - [qsvm/binary] outer fold 5/5
2026-07-11 16:31:04.086 | INFO     | quantum.run:evaluate_fold_quantum:100 - [qsvm/binary] fold done: f1_macro=0.9900
```

### 2.2. Classical

```sh
uv run python -m classical --n-components 3 --load-quantum-splits --tasks binary --mlflow
2026-07-11 16:33:36.535 | INFO     | classical.run:run_nested_cv:112 - [random_forest/binary] outer fold 1/5
2026-07-11 16:33:56.935 | INFO     | classical.run:evaluate_fold:89 - [random_forest/binary] fold done: f1_macro=0.9800
2026-07-11 16:33:57.796 | INFO     | classical.run:run_nested_cv:112 - [random_forest/binary] outer fold 2/5
2026-07-11 16:34:14.702 | INFO     | classical.run:evaluate_fold:89 - [random_forest/binary] fold done: f1_macro=1.0000
2026-07-11 16:34:14.820 | INFO     | classical.run:run_nested_cv:112 - [random_forest/binary] outer fold 3/5
2026-07-11 16:34:32.231 | INFO     | classical.run:evaluate_fold:89 - [random_forest/binary] fold done: f1_macro=1.0000
2026-07-11 16:34:32.364 | INFO     | classical.run:run_nested_cv:112 - [random_forest/binary] outer fold 4/5
2026-07-11 16:34:50.161 | INFO     | classical.run:evaluate_fold:89 - [random_forest/binary] fold done: f1_macro=0.9950
2026-07-11 16:34:50.278 | INFO     | classical.run:run_nested_cv:112 - [random_forest/binary] outer fold 5/5
2026-07-11 16:35:05.651 | INFO     | classical.run:evaluate_fold:89 - [random_forest/binary] fold done: f1_macro=0.9950
2026-07-11 16:35:05.766 | INFO     | __main__:main:102 - wrote per-fold predictions to results/cic/random_forest_binary_predictions.npz
2026-07-11 16:35:05.766 | INFO     | classical.run:run_nested_cv:112 - [xgboost/binary] outer fold 1/5
2026-07-11 16:39:32.099 | INFO     | classical.run:evaluate_fold:89 - [xgboost/binary] fold done: f1_macro=0.9800
2026-07-11 16:39:33.270 | INFO     | classical.run:run_nested_cv:112 - [xgboost/binary] outer fold 2/5
2026-07-11 16:42:13.982 | INFO     | classical.run:evaluate_fold:89 - [xgboost/binary] fold done: f1_macro=0.9950
2026-07-11 16:42:14.208 | INFO     | classical.run:run_nested_cv:112 - [xgboost/binary] outer fold 3/5
2026-07-11 16:42:16.222 | INFO     | classical.run:evaluate_fold:89 - [xgboost/binary] fold done: f1_macro=1.0000
2026-07-11 16:42:16.372 | INFO     | classical.run:run_nested_cv:112 - [xgboost/binary] outer fold 4/5
2026-07-11 16:42:22.016 | INFO     | classical.run:evaluate_fold:89 - [xgboost/binary] fold done: f1_macro=0.9950
2026-07-11 16:42:22.216 | INFO     | classical.run:run_nested_cv:112 - [xgboost/binary] outer fold 5/5
2026-07-11 16:42:23.993 | INFO     | classical.run:evaluate_fold:89 - [xgboost/binary] fold done: f1_macro=0.9900
2026-07-11 16:42:24.144 | INFO     | __main__:main:102 - wrote per-fold predictions to results/cic/xgboost_binary_predictions.npz
2026-07-11 16:42:24.144 | INFO     | classical.run:run_nested_cv:112 - [lightgbm/binary] outer fold 1/5
2026-07-11 16:42:32.597 | INFO     | classical.run:evaluate_fold:89 - [lightgbm/binary] fold done: f1_macro=0.9800
2026-07-11 16:42:32.798 | INFO     | classical.run:run_nested_cv:112 - [lightgbm/binary] outer fold 2/5
2026-07-11 16:42:36.417 | INFO     | classical.run:evaluate_fold:89 - [lightgbm/binary] fold done: f1_macro=0.9950
2026-07-11 16:42:36.589 | INFO     | classical.run:run_nested_cv:112 - [lightgbm/binary] outer fold 3/5
2026-07-11 16:42:39.483 | INFO     | classical.run:evaluate_fold:89 - [lightgbm/binary] fold done: f1_macro=1.0000
2026-07-11 16:42:39.637 | INFO     | classical.run:run_nested_cv:112 - [lightgbm/binary] outer fold 4/5
2026-07-11 16:42:42.254 | INFO     | classical.run:evaluate_fold:89 - [lightgbm/binary] fold done: f1_macro=0.9900
2026-07-11 16:42:42.419 | INFO     | classical.run:run_nested_cv:112 - [lightgbm/binary] outer fold 5/5
2026-07-11 16:42:45.371 | INFO     | classical.run:evaluate_fold:89 - [lightgbm/binary] fold done: f1_macro=0.9950
2026-07-11 16:42:45.539 | INFO     | __main__:main:102 - wrote per-fold predictions to results/cic/lightgbm_binary_predictions.npz
2026-07-11 16:42:45.540 | INFO     | classical.run:run_nested_cv:112 - [svm/binary] outer fold 1/5
2026-07-11 16:42:46.148 | INFO     | classical.run:evaluate_fold:89 - [svm/binary] fold done: f1_macro=0.9800
2026-07-11 16:42:46.286 | INFO     | classical.run:run_nested_cv:112 - [svm/binary] outer fold 2/5
2026-07-11 16:42:46.853 | INFO     | classical.run:evaluate_fold:89 - [svm/binary] fold done: f1_macro=0.9950
2026-07-11 16:42:47.143 | INFO     | classical.run:run_nested_cv:112 - [svm/binary] outer fold 3/5
2026-07-11 16:42:47.695 | INFO     | classical.run:evaluate_fold:89 - [svm/binary] fold done: f1_macro=0.9950
2026-07-11 16:42:47.844 | INFO     | classical.run:run_nested_cv:112 - [svm/binary] outer fold 4/5
2026-07-11 16:42:48.425 | INFO     | classical.run:evaluate_fold:89 - [svm/binary] fold done: f1_macro=0.9950
2026-07-11 16:42:48.561 | INFO     | classical.run:run_nested_cv:112 - [svm/binary] outer fold 5/5
2026-07-11 16:42:49.137 | INFO     | classical.run:evaluate_fold:89 - [svm/binary] fold done: f1_macro=0.9950
2026-07-11 16:42:49.259 | INFO     | __main__:main:102 - wrote per-fold predictions to results/cic/svm_binary_predictions.npz
2026-07-11 16:42:49.266 | INFO     | __main__:main:106 - wrote 4 model x task rows to results/cic/metrics.csv
```

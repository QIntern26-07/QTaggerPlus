# Week 2 Jul 12 Experiments - 6 components

Today's work focusing on the result and benchmark of number qubits from 2 to 6.

## Overview: architecture and experiment design

Same as in [week 2 first report](./w2_Jul-9_experiments.md).

## Stage 0. - Sanity probe

```sh
uv run python -m quantum --n-components 6 --max-samples 200 --encodings angle --probe
2026-07-12 12:54:50.596 | INFO     | __main__:main:73 - [probe] angle nc=6 kernel_train=8.556s fit=8.563s infer=4.291s

uv run python -m quantum --n-components 6 --max-samples 200 --encodings iqp --probe
2026-07-12 12:55:48.108 | INFO     | __main__:main:73 - [probe] iqp nc=6 kernel_train=17.414s fit=17.421s infer=8.562s

uv run python -m quantum --n-components 6 --max-samples 1000 --encodings angle --probe
2026-07-12 13:00:21.225 | INFO     | __main__:main:73 - [probe] angle nc=6 kernel_train=169.429s fit=169.439s infer=84.126s

uv run python -m quantum --n-components 6 --max-samples 1000 --encodings iqp --probe
2026-07-12 13:12:17.127 | INFO     | __main__:main:73 - [probe] iqp nc=6 kernel_train=311.576s fit=311.585s infer=158.726s
```

## Stage 1. 200 samples

### 1.1. Quantum

```sh
uv run python -m quantum --n-components 6 --max-samples 200 --folds 5 \
    --encodings angle iqp --tasks binary --mlflow
2026-07-12 13:13:15.900 | INFO     | quantum.run:run_quantum_cv:120 - [qsvm/binary] outer fold 1/5
2026-07-12 13:13:42.134 | INFO     | quantum.run:evaluate_fold_quantum:100 - [qsvm/binary] fold done: f1_macro=1.0000
2026-07-12 13:13:42.927 | INFO     | quantum.run:run_quantum_cv:120 - [qsvm/binary] outer fold 2/5
2026-07-12 13:14:12.037 | INFO     | quantum.run:evaluate_fold_quantum:100 - [qsvm/binary] fold done: f1_macro=0.9250
2026-07-12 13:14:12.140 | INFO     | quantum.run:run_quantum_cv:120 - [qsvm/binary] outer fold 3/5
2026-07-12 13:14:40.601 | INFO     | quantum.run:evaluate_fold_quantum:100 - [qsvm/binary] fold done: f1_macro=1.0000
2026-07-12 13:14:40.714 | INFO     | quantum.run:run_quantum_cv:120 - [qsvm/binary] outer fold 4/5
2026-07-12 13:15:09.834 | INFO     | quantum.run:evaluate_fold_quantum:100 - [qsvm/binary] fold done: f1_macro=1.0000
2026-07-12 13:15:10.082 | INFO     | quantum.run:run_quantum_cv:120 - [qsvm/binary] outer fold 5/5
2026-07-12 13:15:47.236 | INFO     | quantum.run:evaluate_fold_quantum:100 - [qsvm/binary] fold done: f1_macro=1.0000
```

### 1.2. Classicals

```sh
uv run python -m classical --n-components 6 --load-quantum-splits --tasks binary --mlflow
2026-07-12 13:16:57.206 | INFO     | classical.run:run_nested_cv:112 - [random_forest/binary] outer fold 1/5
2026-07-12 13:17:11.396 | INFO     | classical.run:evaluate_fold:89 - [random_forest/binary] fold done: f1_macro=1.0000
2026-07-12 13:17:12.060 | INFO     | classical.run:run_nested_cv:112 - [random_forest/binary] outer fold 2/5
2026-07-12 13:17:24.489 | INFO     | classical.run:evaluate_fold:89 - [random_forest/binary] fold done: f1_macro=0.9500
2026-07-12 13:17:24.605 | INFO     | classical.run:run_nested_cv:112 - [random_forest/binary] outer fold 3/5
2026-07-12 13:17:35.606 | INFO     | classical.run:evaluate_fold:89 - [random_forest/binary] fold done: f1_macro=1.0000
2026-07-12 13:17:35.743 | INFO     | classical.run:run_nested_cv:112 - [random_forest/binary] outer fold 4/5
2026-07-12 13:17:46.318 | INFO     | classical.run:evaluate_fold:89 - [random_forest/binary] fold done: f1_macro=1.0000
2026-07-12 13:17:46.426 | INFO     | classical.run:run_nested_cv:112 - [random_forest/binary] outer fold 5/5
2026-07-12 13:17:56.925 | INFO     | classical.run:evaluate_fold:89 - [random_forest/binary] fold done: f1_macro=1.0000
2026-07-12 13:17:57.052 | INFO     | __main__:main:102 - wrote per-fold predictions to results/cic/random_forest_binary_predictions.npz
2026-07-12 13:17:57.052 | INFO     | classical.run:run_nested_cv:112 - [xgboost/binary] outer fold 1/5
2026-07-12 13:18:04.103 | INFO     | classical.run:evaluate_fold:89 - [xgboost/binary] fold done: f1_macro=1.0000
2026-07-12 13:18:04.271 | INFO     | classical.run:run_nested_cv:112 - [xgboost/binary] outer fold 2/5
2026-07-12 13:18:05.323 | INFO     | classical.run:evaluate_fold:89 - [xgboost/binary] fold done: f1_macro=0.9500
2026-07-12 13:18:05.465 | INFO     | classical.run:run_nested_cv:112 - [xgboost/binary] outer fold 3/5
2026-07-12 13:18:06.573 | INFO     | classical.run:evaluate_fold:89 - [xgboost/binary] fold done: f1_macro=1.0000
2026-07-12 13:18:06.714 | INFO     | classical.run:run_nested_cv:112 - [xgboost/binary] outer fold 4/5
2026-07-12 13:18:08.052 | INFO     | classical.run:evaluate_fold:89 - [xgboost/binary] fold done: f1_macro=1.0000
2026-07-12 13:18:08.205 | INFO     | classical.run:run_nested_cv:112 - [xgboost/binary] outer fold 5/5
2026-07-12 13:18:09.276 | INFO     | classical.run:evaluate_fold:89 - [xgboost/binary] fold done: f1_macro=1.0000
2026-07-12 13:18:09.413 | INFO     | __main__:main:102 - wrote per-fold predictions to results/cic/xgboost_binary_predictions.npz
2026-07-12 13:18:09.413 | INFO     | classical.run:run_nested_cv:112 - [lightgbm/binary] outer fold 1/5
2026-07-12 13:18:11.728 | INFO     | classical.run:evaluate_fold:89 - [lightgbm/binary] fold done: f1_macro=0.9750
2026-07-12 13:18:11.876 | INFO     | classical.run:run_nested_cv:112 - [lightgbm/binary] outer fold 2/5
2026-07-12 13:18:12.754 | INFO     | classical.run:evaluate_fold:89 - [lightgbm/binary] fold done: f1_macro=0.9500
2026-07-12 13:18:12.896 | INFO     | classical.run:run_nested_cv:112 - [lightgbm/binary] outer fold 3/5
2026-07-12 13:18:13.969 | INFO     | classical.run:evaluate_fold:89 - [lightgbm/binary] fold done: f1_macro=1.0000
2026-07-12 13:18:14.264 | INFO     | classical.run:run_nested_cv:112 - [lightgbm/binary] outer fold 4/5
2026-07-12 13:18:15.441 | INFO     | classical.run:evaluate_fold:89 - [lightgbm/binary] fold done: f1_macro=1.0000
2026-07-12 13:18:15.612 | INFO     | classical.run:run_nested_cv:112 - [lightgbm/binary] outer fold 5/5
2026-07-12 13:18:16.695 | INFO     | classical.run:evaluate_fold:89 - [lightgbm/binary] fold done: f1_macro=1.0000
2026-07-12 13:18:16.838 | INFO     | __main__:main:102 - wrote per-fold predictions to results/cic/lightgbm_binary_predictions.npz
2026-07-12 13:18:16.838 | INFO     | classical.run:run_nested_cv:112 - [svm/binary] outer fold 1/5
2026-07-12 13:18:17.205 | INFO     | classical.run:evaluate_fold:89 - [svm/binary] fold done: f1_macro=0.9750
2026-07-12 13:18:17.333 | INFO     | classical.run:run_nested_cv:112 - [svm/binary] outer fold 2/5
2026-07-12 13:18:17.684 | INFO     | classical.run:evaluate_fold:89 - [svm/binary] fold done: f1_macro=0.9500
2026-07-12 13:18:17.803 | INFO     | classical.run:run_nested_cv:112 - [svm/binary] outer fold 3/5
2026-07-12 13:18:18.155 | INFO     | classical.run:evaluate_fold:89 - [svm/binary] fold done: f1_macro=1.0000
2026-07-12 13:18:18.271 | INFO     | classical.run:run_nested_cv:112 - [svm/binary] outer fold 4/5
2026-07-12 13:18:18.621 | INFO     | classical.run:evaluate_fold:89 - [svm/binary] fold done: f1_macro=1.0000
2026-07-12 13:18:18.731 | INFO     | classical.run:run_nested_cv:112 - [svm/binary] outer fold 5/5
2026-07-12 13:18:19.081 | INFO     | classical.run:evaluate_fold:89 - [svm/binary] fold done: f1_macro=1.0000
2026-07-12 13:18:19.190 | INFO     | __main__:main:102 - wrote per-fold predictions to results/cic/svm_binary_predictions.npz
2026-07-12 13:18:19.192 | INFO     | __main__:main:106 - wrote 4 model x task rows to results/cic/metrics.csv
```

## Stage 2. 1000 samples

### 2.1. Quantum

```sh
uv run python -m quantum --n-components 6 --max-samples 1000 --folds 5 \
    --encodings angle iqp --tasks binary --mlflow
2026-07-12 13:19:55.659 | INFO     | quantum.run:run_quantum_cv:120 - [qsvm/binary] outer fold 1/5
2026-07-12 13:31:39.674 | INFO     | quantum.run:evaluate_fold_quantum:100 - [qsvm/binary] fold done: f1_macro=0.9850
2026-07-12 13:31:40.546 | INFO     | quantum.run:run_quantum_cv:120 - [qsvm/binary] outer fold 2/5
2026-07-12 13:43:31.167 | INFO     | quantum.run:evaluate_fold_quantum:100 - [qsvm/binary] fold done: f1_macro=0.9850
2026-07-12 13:43:31.290 | INFO     | quantum.run:run_quantum_cv:120 - [qsvm/binary] outer fold 3/5
2026-07-12 13:55:20.050 | INFO     | quantum.run:evaluate_fold_quantum:100 - [qsvm/binary] fold done: f1_macro=1.0000
2026-07-12 13:55:20.149 | INFO     | quantum.run:run_quantum_cv:120 - [qsvm/binary] outer fold 4/5
2026-07-12 14:07:43.321 | INFO     | quantum.run:evaluate_fold_quantum:100 - [qsvm/binary] fold done: f1_macro=1.0000
2026-07-12 14:07:43.446 | INFO     | quantum.run:run_quantum_cv:120 - [qsvm/binary] outer fold 5/5
2026-07-12 14:19:53.345 | INFO     | quantum.run:evaluate_fold_quantum:100 - [qsvm/binary] fold done: f1_macro=0.9900
```

### 2.2. Classical

```sh
uv run python -m classical --n-components 6 --load-quantum-splits --tasks binary --mlflow
2026-07-12 14:20:55.716 | INFO     | classical.run:run_nested_cv:112 - [random_forest/binary] outer fold 1/5
2026-07-12 14:21:17.046 | INFO     | classical.run:evaluate_fold:89 - [random_forest/binary] fold done: f1_macro=0.9850
2026-07-12 14:21:17.832 | INFO     | classical.run:run_nested_cv:112 - [random_forest/binary] outer fold 2/5
2026-07-12 14:21:34.089 | INFO     | classical.run:evaluate_fold:89 - [random_forest/binary] fold done: f1_macro=0.9950
2026-07-12 14:21:34.213 | INFO     | classical.run:run_nested_cv:112 - [random_forest/binary] outer fold 3/5
2026-07-12 14:21:50.320 | INFO     | classical.run:evaluate_fold:89 - [random_forest/binary] fold done: f1_macro=1.0000
2026-07-12 14:21:50.424 | INFO     | classical.run:run_nested_cv:112 - [random_forest/binary] outer fold 4/5
2026-07-12 14:22:05.739 | INFO     | classical.run:evaluate_fold:89 - [random_forest/binary] fold done: f1_macro=0.9950
2026-07-12 14:22:05.870 | INFO     | classical.run:run_nested_cv:112 - [random_forest/binary] outer fold 5/5
2026-07-12 14:22:18.921 | INFO     | classical.run:evaluate_fold:89 - [random_forest/binary] fold done: f1_macro=0.9900
2026-07-12 14:22:19.051 | INFO     | __main__:main:102 - wrote per-fold predictions to results/cic/random_forest_binary_predictions.npz
2026-07-12 14:22:19.051 | INFO     | classical.run:run_nested_cv:112 - [xgboost/binary] outer fold 1/5
2026-07-12 14:22:23.885 | INFO     | classical.run:evaluate_fold:89 - [xgboost/binary] fold done: f1_macro=0.9800
2026-07-12 14:22:24.036 | INFO     | classical.run:run_nested_cv:112 - [xgboost/binary] outer fold 2/5
2026-07-12 14:22:26.220 | INFO     | classical.run:evaluate_fold:89 - [xgboost/binary] fold done: f1_macro=0.9950
2026-07-12 14:22:26.378 | INFO     | classical.run:run_nested_cv:112 - [xgboost/binary] outer fold 3/5
2026-07-12 14:22:28.465 | INFO     | classical.run:evaluate_fold:89 - [xgboost/binary] fold done: f1_macro=1.0000
2026-07-12 14:22:28.608 | INFO     | classical.run:run_nested_cv:112 - [xgboost/binary] outer fold 4/5
2026-07-12 14:22:30.456 | INFO     | classical.run:evaluate_fold:89 - [xgboost/binary] fold done: f1_macro=0.9950
2026-07-12 14:22:30.613 | INFO     | classical.run:run_nested_cv:112 - [xgboost/binary] outer fold 5/5
2026-07-12 14:22:32.652 | INFO     | classical.run:evaluate_fold:89 - [xgboost/binary] fold done: f1_macro=0.9850
2026-07-12 14:22:32.795 | INFO     | __main__:main:102 - wrote per-fold predictions to results/cic/xgboost_binary_predictions.npz
2026-07-12 14:22:32.795 | INFO     | classical.run:run_nested_cv:112 - [lightgbm/binary] outer fold 1/5
2026-07-12 14:22:37.691 | INFO     | classical.run:evaluate_fold:89 - [lightgbm/binary] fold done: f1_macro=0.9800
2026-07-12 14:22:37.850 | INFO     | classical.run:run_nested_cv:112 - [lightgbm/binary] outer fold 2/5
2026-07-12 14:22:40.606 | INFO     | classical.run:evaluate_fold:89 - [lightgbm/binary] fold done: f1_macro=0.9950
2026-07-12 14:22:40.773 | INFO     | classical.run:run_nested_cv:112 - [lightgbm/binary] outer fold 3/5
2026-07-12 14:22:43.687 | INFO     | classical.run:evaluate_fold:89 - [lightgbm/binary] fold done: f1_macro=1.0000
2026-07-12 14:22:43.855 | INFO     | classical.run:run_nested_cv:112 - [lightgbm/binary] outer fold 4/5
2026-07-12 14:22:46.606 | INFO     | classical.run:evaluate_fold:89 - [lightgbm/binary] fold done: f1_macro=0.9950
2026-07-12 14:22:46.747 | INFO     | classical.run:run_nested_cv:112 - [lightgbm/binary] outer fold 5/5
2026-07-12 14:22:50.449 | INFO     | classical.run:evaluate_fold:89 - [lightgbm/binary] fold done: f1_macro=0.9950
2026-07-12 14:22:50.616 | INFO     | __main__:main:102 - wrote per-fold predictions to results/cic/lightgbm_binary_predictions.npz
2026-07-12 14:22:50.617 | INFO     | classical.run:run_nested_cv:112 - [svm/binary] outer fold 1/5
2026-07-12 14:22:51.201 | INFO     | classical.run:evaluate_fold:89 - [svm/binary] fold done: f1_macro=0.9850
2026-07-12 14:22:51.337 | INFO     | classical.run:run_nested_cv:112 - [svm/binary] outer fold 2/5
2026-07-12 14:22:51.877 | INFO     | classical.run:evaluate_fold:89 - [svm/binary] fold done: f1_macro=0.9900
2026-07-12 14:22:51.998 | INFO     | classical.run:run_nested_cv:112 - [svm/binary] outer fold 3/5
2026-07-12 14:22:52.572 | INFO     | classical.run:evaluate_fold:89 - [svm/binary] fold done: f1_macro=0.9900
2026-07-12 14:22:52.819 | INFO     | classical.run:run_nested_cv:112 - [svm/binary] outer fold 4/5
2026-07-12 14:22:53.319 | INFO     | classical.run:evaluate_fold:89 - [svm/binary] fold done: f1_macro=0.9950
2026-07-12 14:22:53.441 | INFO     | classical.run:run_nested_cv:112 - [svm/binary] outer fold 5/5
2026-07-12 14:22:53.963 | INFO     | classical.run:evaluate_fold:89 - [svm/binary] fold done: f1_macro=0.9850
2026-07-12 14:22:54.090 | INFO     | __main__:main:102 - wrote per-fold predictions to results/cic/svm_binary_predictions.npz
2026-07-12 14:22:54.092 | INFO     | __main__:main:106 - wrote 4 model x task rows to results/cic/metrics.csv
```

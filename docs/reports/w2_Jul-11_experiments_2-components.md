# Week 2 Jul 11 Experiments - 3 qubits

Today's work focusing on the result and benchmark of number qubits from 2 to 6.

## Overview: architecture and experiment design

Same as in [week 2 first report](./w2_Jul-9_experiments.md).

## Stage 0. - Sanity probe

```sh
uv run python -m quantum --n-components 2 --max-samples 200 --encodings angle --probe
2026-07-11 14:29:29.679 | INFO     | __main__:main:73 - [probe] angle nc=2 kernel_train=3.247s fit=3.253s infer=1.716s

uv run python -m quantum --n-components 2 --max-samples 200 --encodings iqp --probe
2026-07-11 14:29:45.677 | INFO     | __main__:main:73 - [probe] iqp nc=2 kernel_train=2.391s fit=2.396s infer=1.369s

uv run python -m quantum --n-components 2 --max-samples 1000 --encodings angle --probe
2026-07-11 14:35:54.220 | INFO     | __main__:main:73 - [probe] angle nc=2 kernel_train=82.245s fit=82.253s infer=41.034s

uv run python -m quantum --n-components 2 --max-samples 1000 --encodings iqp --probe
2026-07-11 14:39:44.253 | INFO     | __main__:main:73 - [probe] iqp nc=2 kernel_train=63.022s fit=63.031s infer=31.232s
```

## Stage 1. 200 samples

### 1.1. Quantum

```sh
uv run python -m quantum --n-components 2 --max-samples 200 --folds 5 \
    --encodings angle iqp --tasks binary --mlflow
2026-07-11 14:44:52.603 | INFO     | quantum.run:run_quantum_cv:120 - [qsvm/binary] outer fold 1/5
2026-07-11 14:45:02.605 | INFO     | quantum.run:evaluate_fold_quantum:100 - [qsvm/binary] fold done: f1_macro=1.0000
2026-07-11 14:45:03.743 | INFO     | quantum.run:run_quantum_cv:120 - [qsvm/binary] outer fold 2/5
2026-07-11 14:45:13.853 | INFO     | quantum.run:evaluate_fold_quantum:100 - [qsvm/binary] fold done: f1_macro=0.9500
2026-07-11 14:45:13.985 | INFO     | quantum.run:run_quantum_cv:120 - [qsvm/binary] outer fold 3/5
2026-07-11 14:45:24.555 | INFO     | quantum.run:evaluate_fold_quantum:100 - [qsvm/binary] fold done: f1_macro=1.0000
2026-07-11 14:45:24.664 | INFO     | quantum.run:run_quantum_cv:120 - [qsvm/binary] outer fold 4/5
2026-07-11 14:45:34.617 | INFO     | quantum.run:evaluate_fold_quantum:100 - [qsvm/binary] fold done: f1_macro=1.0000
2026-07-11 14:45:34.729 | INFO     | quantum.run:run_quantum_cv:120 - [qsvm/binary] outer fold 5/5
2026-07-11 14:45:45.007 | INFO     | quantum.run:evaluate_fold_quantum:100 - [qsvm/binary] fold done: f1_macro=1.0000
```

### 1.2. Classicals

```sh
uv run python -m classical --n-components 2 --load-quantum-splits --tasks b
inary --mlflow
2026-07-11 14:55:44.255 | INFO     | classical.run:run_nested_cv:112 - [random_forest/binary] outer fold 1/5
2026-07-11 14:56:06.050 | INFO     | classical.run:evaluate_fold:89 - [random_forest/binary] fold done: f1_macro=1.0000
2026-07-11 14:56:06.984 | INFO     | classical.run:run_nested_cv:112 - [random_forest/binary] outer fold 2/5
2026-07-11 14:56:21.116 | INFO     | classical.run:evaluate_fold:89 - [random_forest/binary] fold done: f1_macro=0.9500
2026-07-11 14:56:21.274 | INFO     | classical.run:run_nested_cv:112 - [random_forest/binary] outer fold 3/5
2026-07-11 14:56:33.225 | INFO     | classical.run:evaluate_fold:89 - [random_forest/binary] fold done: f1_macro=1.0000
2026-07-11 14:56:33.350 | INFO     | classical.run:run_nested_cv:112 - [random_forest/binary] outer fold 4/5
2026-07-11 14:56:46.191 | INFO     | classical.run:evaluate_fold:89 - [random_forest/binary] fold done: f1_macro=1.0000
2026-07-11 14:56:46.321 | INFO     | classical.run:run_nested_cv:112 - [random_forest/binary] outer fold 5/5
2026-07-11 14:56:58.774 | INFO     | classical.run:evaluate_fold:89 - [random_forest/binary] fold done: f1_macro=1.0000
2026-07-11 14:56:58.899 | INFO     | __main__:main:102 - wrote per-fold predictions to results/cic/random_forest_binary_predictions.npz
2026-07-11 14:56:58.900 | INFO     | classical.run:run_nested_cv:112 - [xgboost/binary] outer fold 1/5
2026-07-11 14:57:08.773 | INFO     | classical.run:evaluate_fold:89 - [xgboost/binary] fold done: f1_macro=1.0000
2026-07-11 14:57:08.987 | INFO     | classical.run:run_nested_cv:112 - [xgboost/binary] outer fold 2/5
2026-07-11 14:57:10.191 | INFO     | classical.run:evaluate_fold:89 - [xgboost/binary] fold done: f1_macro=0.9500
2026-07-11 14:57:10.338 | INFO     | classical.run:run_nested_cv:112 - [xgboost/binary] outer fold 3/5
2026-07-11 14:57:11.459 | INFO     | classical.run:evaluate_fold:89 - [xgboost/binary] fold done: f1_macro=1.0000
2026-07-11 14:57:11.595 | INFO     | classical.run:run_nested_cv:112 - [xgboost/binary] outer fold 4/5
2026-07-11 14:57:12.611 | INFO     | classical.run:evaluate_fold:89 - [xgboost/binary] fold done: f1_macro=1.0000
2026-07-11 14:57:12.754 | INFO     | classical.run:run_nested_cv:112 - [xgboost/binary] outer fold 5/5
2026-07-11 14:57:13.768 | INFO     | classical.run:evaluate_fold:89 - [xgboost/binary] fold done: f1_macro=1.0000
2026-07-11 14:57:13.903 | INFO     | __main__:main:102 - wrote per-fold predictions to results/cic/xgboost_binary_predictions.npz
2026-07-11 14:57:13.904 | INFO     | classical.run:run_nested_cv:112 - [lightgbm/binary] outer fold 1/5
2026-07-11 14:57:16.783 | INFO     | classical.run:evaluate_fold:89 - [lightgbm/binary] fold done: f1_macro=1.0000
2026-07-11 14:57:16.997 | INFO     | classical.run:run_nested_cv:112 - [lightgbm/binary] outer fold 2/5
2026-07-11 14:57:18.374 | INFO     | classical.run:evaluate_fold:89 - [lightgbm/binary] fold done: f1_macro=0.9500
2026-07-11 14:57:18.778 | INFO     | classical.run:run_nested_cv:112 - [lightgbm/binary] outer fold 3/5
2026-07-11 14:57:20.002 | INFO     | classical.run:evaluate_fold:89 - [lightgbm/binary] fold done: f1_macro=1.0000
2026-07-11 14:57:20.192 | INFO     | classical.run:run_nested_cv:112 - [lightgbm/binary] outer fold 4/5
2026-07-11 14:57:21.792 | INFO     | classical.run:evaluate_fold:89 - [lightgbm/binary] fold done: f1_macro=1.0000
2026-07-11 14:57:21.992 | INFO     | classical.run:run_nested_cv:112 - [lightgbm/binary] outer fold 5/5
2026-07-11 14:57:23.879 | INFO     | classical.run:evaluate_fold:89 - [lightgbm/binary] fold done: f1_macro=1.0000
2026-07-11 14:57:24.156 | INFO     | __main__:main:102 - wrote per-fold predictions to results/cic/lightgbm_binary_predictions.npz
2026-07-11 14:57:24.157 | INFO     | classical.run:run_nested_cv:112 - [svm/binary] outer fold 1/5
2026-07-11 14:57:24.734 | INFO     | classical.run:evaluate_fold:89 - [svm/binary] fold done: f1_macro=1.0000
2026-07-11 14:57:24.936 | INFO     | classical.run:run_nested_cv:112 - [svm/binary] outer fold 2/5
2026-07-11 14:57:25.517 | INFO     | classical.run:evaluate_fold:89 - [svm/binary] fold done: f1_macro=0.9500
2026-07-11 14:57:25.699 | INFO     | classical.run:run_nested_cv:112 - [svm/binary] outer fold 3/5
2026-07-11 14:57:26.154 | INFO     | classical.run:evaluate_fold:89 - [svm/binary] fold done: f1_macro=0.9750
2026-07-11 14:57:26.344 | INFO     | classical.run:run_nested_cv:112 - [svm/binary] outer fold 4/5
2026-07-11 14:57:26.791 | INFO     | classical.run:evaluate_fold:89 - [svm/binary] fold done: f1_macro=1.0000
2026-07-11 14:57:26.973 | INFO     | classical.run:run_nested_cv:112 - [svm/binary] outer fold 5/5
2026-07-11 14:57:27.393 | INFO     | classical.run:evaluate_fold:89 - [svm/binary] fold done: f1_macro=1.0000
2026-07-11 14:57:27.567 | INFO     | __main__:main:102 - wrote per-fold predictions to results/cic/svm_binary_predictions.npz
2026-07-11 14:57:27.570 | INFO     | __main__:main:106 - wrote 4 model x task rows to results/cic/metrics.csv
```

## Stage 2. 1000 samples

### 2.1. Quantum

```sh
uv run python -m quantum --n-components 2 --max-samples 1000 --folds 5 \
    --encodings angle iqp --tasks binary --mlflow
2026-07-11 15:06:30.035 | INFO     | quantum.run:run_quantum_cv:120 - [qsvm/binary] outer fold 1/5
2026-07-11 15:10:26.099 | INFO     | quantum.run:evaluate_fold_quantum:100 - [qsvm/binary] fold done: f1_macro=0.9800
2026-07-11 15:10:26.890 | INFO     | quantum.run:run_quantum_cv:120 - [qsvm/binary] outer fold 2/5
2026-07-11 15:14:24.585 | INFO     | quantum.run:evaluate_fold_quantum:100 - [qsvm/binary] fold done: f1_macro=0.9900
2026-07-11 15:14:24.678 | INFO     | quantum.run:run_quantum_cv:120 - [qsvm/binary] outer fold 3/5
2026-07-11 15:18:21.927 | INFO     | quantum.run:evaluate_fold_quantum:100 - [qsvm/binary] fold done: f1_macro=0.9950
2026-07-11 15:18:22.047 | INFO     | quantum.run:run_quantum_cv:120 - [qsvm/binary] outer fold 4/5
2026-07-11 15:22:18.271 | INFO     | quantum.run:evaluate_fold_quantum:100 - [qsvm/binary] fold done: f1_macro=0.9950
2026-07-11 15:22:18.368 | INFO     | quantum.run:run_quantum_cv:120 - [qsvm/binary] outer fold 5/5
2026-07-11 15:25:51.010 | INFO     | quantum.run:evaluate_fold_quantum:100 - [qsvm/binary] fold done: f1_macro=0.9900
```

### 2.2. Classical

```sh
uv run python -m classical --n-components 2 --load-quantum-splits --tasks b
inary --mlflow
2026-07-11 15:35:06.499 | INFO     | classical.run:run_nested_cv:112 - [random_forest/binary] outer fold 1/5
2026-07-11 15:35:22.373 | INFO     | classical.run:evaluate_fold:89 - [random_forest/binary] fold done: f1_macro=0.9800
2026-07-11 15:35:23.158 | INFO     | classical.run:run_nested_cv:112 - [random_forest/binary] outer fold 2/5
2026-07-11 15:35:37.025 | INFO     | classical.run:evaluate_fold:89 - [random_forest/binary] fold done: f1_macro=0.9950
2026-07-11 15:35:37.170 | INFO     | classical.run:run_nested_cv:112 - [random_forest/binary] outer fold 3/5
2026-07-11 15:35:53.236 | INFO     | classical.run:evaluate_fold:89 - [random_forest/binary] fold done: f1_macro=1.0000
2026-07-11 15:35:53.353 | INFO     | classical.run:run_nested_cv:112 - [random_forest/binary] outer fold 4/5
2026-07-11 15:36:08.595 | INFO     | classical.run:evaluate_fold:89 - [random_forest/binary] fold done: f1_macro=0.9950
2026-07-11 15:36:08.731 | INFO     | classical.run:run_nested_cv:112 - [random_forest/binary] outer fold 5/5
2026-07-11 15:36:23.001 | INFO     | classical.run:evaluate_fold:89 - [random_forest/binary] fold done: f1_macro=0.9900
2026-07-11 15:36:23.114 | INFO     | __main__:main:102 - wrote per-fold predictions to results/cic/random_forest_binary_predictions.npz
2026-07-11 15:36:23.114 | INFO     | classical.run:run_nested_cv:112 - [xgboost/binary] outer fold 1/5
2026-07-11 15:36:28.247 | INFO     | classical.run:evaluate_fold:89 - [xgboost/binary] fold done: f1_macro=0.9800
2026-07-11 15:36:28.382 | INFO     | classical.run:run_nested_cv:112 - [xgboost/binary] outer fold 2/5
2026-07-11 15:36:30.443 | INFO     | classical.run:evaluate_fold:89 - [xgboost/binary] fold done: f1_macro=0.9950
2026-07-11 15:36:30.610 | INFO     | classical.run:run_nested_cv:112 - [xgboost/binary] outer fold 3/5
2026-07-11 15:36:35.575 | INFO     | classical.run:evaluate_fold:89 - [xgboost/binary] fold done: f1_macro=1.0000
2026-07-11 15:36:35.768 | INFO     | classical.run:run_nested_cv:112 - [xgboost/binary] outer fold 4/5
2026-07-11 15:36:37.550 | INFO     | classical.run:evaluate_fold:89 - [xgboost/binary] fold done: f1_macro=0.9950
2026-07-11 15:36:37.695 | INFO     | classical.run:run_nested_cv:112 - [xgboost/binary] outer fold 5/5
2026-07-11 15:36:39.295 | INFO     | classical.run:evaluate_fold:89 - [xgboost/binary] fold done: f1_macro=0.9900
2026-07-11 15:36:39.453 | INFO     | __main__:main:102 - wrote per-fold predictions to results/cic/xgboost_binary_predictions.npz
2026-07-11 15:36:39.453 | INFO     | classical.run:run_nested_cv:112 - [lightgbm/binary] outer fold 1/5
2026-07-11 15:36:43.602 | INFO     | classical.run:evaluate_fold:89 - [lightgbm/binary] fold done: f1_macro=0.9850
2026-07-11 15:36:43.760 | INFO     | classical.run:run_nested_cv:112 - [lightgbm/binary] outer fold 2/5
2026-07-11 15:36:45.980 | INFO     | classical.run:evaluate_fold:89 - [lightgbm/binary] fold done: f1_macro=0.9950
2026-07-11 15:36:46.135 | INFO     | classical.run:run_nested_cv:112 - [lightgbm/binary] outer fold 3/5
2026-07-11 15:36:49.165 | INFO     | classical.run:evaluate_fold:89 - [lightgbm/binary] fold done: f1_macro=0.9900
2026-07-11 15:36:49.318 | INFO     | classical.run:run_nested_cv:112 - [lightgbm/binary] outer fold 4/5
2026-07-11 15:36:51.576 | INFO     | classical.run:evaluate_fold:89 - [lightgbm/binary] fold done: f1_macro=0.9900
2026-07-11 15:36:51.721 | INFO     | classical.run:run_nested_cv:112 - [lightgbm/binary] outer fold 5/5
2026-07-11 15:36:56.877 | INFO     | classical.run:evaluate_fold:89 - [lightgbm/binary] fold done: f1_macro=0.9900
2026-07-11 15:36:57.064 | INFO     | __main__:main:102 - wrote per-fold predictions to results/cic/lightgbm_binary_predictions.npz
2026-07-11 15:36:57.065 | INFO     | classical.run:run_nested_cv:112 - [svm/binary] outer fold 1/5
2026-07-11 15:36:57.694 | INFO     | classical.run:evaluate_fold:89 - [svm/binary] fold done: f1_macro=0.9800
2026-07-11 15:36:57.843 | INFO     | classical.run:run_nested_cv:112 - [svm/binary] outer fold 2/5
2026-07-11 15:36:58.378 | INFO     | classical.run:evaluate_fold:89 - [svm/binary] fold done: f1_macro=0.9950
2026-07-11 15:36:58.508 | INFO     | classical.run:run_nested_cv:112 - [svm/binary] outer fold 3/5
2026-07-11 15:36:59.098 | INFO     | classical.run:evaluate_fold:89 - [svm/binary] fold done: f1_macro=0.9950
2026-07-11 15:36:59.250 | INFO     | classical.run:run_nested_cv:112 - [svm/binary] outer fold 4/5
2026-07-11 15:36:59.773 | INFO     | classical.run:evaluate_fold:89 - [svm/binary] fold done: f1_macro=0.9950
2026-07-11 15:37:00.041 | INFO     | classical.run:run_nested_cv:112 - [svm/binary] outer fold 5/5
2026-07-11 15:37:00.559 | INFO     | classical.run:evaluate_fold:89 - [svm/binary] fold done: f1_macro=0.9850
2026-07-11 15:37:00.683 | INFO     | __main__:main:102 - wrote per-fold predictions to results/cic/svm_binary_predictions.npz
2026-07-11 15:37:00.685 | INFO     | __main__:main:106 - wrote 4 model x task rows to results/cic/metrics.csv
```
# Week 2 Jul 11 Experiments - 4 components

Today's work focusing on the result and benchmark of number qubits from 2 to 6.

## Overview: architecture and experiment design

Same as in [week 2 first report](./w2_Jul-9_experiments.md).

## Stage 0. - Sanity probe

```sh
uv run python -m quantum --n-components 4 --max-samples 200 --encodings angle --probe
2026-07-11 16:55:01.700 | INFO     | __main__:main:73 - [probe] angle nc=4 kernel_train=4.269s fit=4.274s infer=2.321s

uv run python -m quantum --n-components 4 --max-samples 200 --encodings iqp --probe
2026-07-11 16:55:31.362 | INFO     | __main__:main:73 - [probe] iqp nc=4 kernel_train=5.901s fit=5.906s infer=3.645s

uv run python -m quantum --n-components 4 --max-samples 1000 --encodings angle --probe
2026-07-11 16:59:00.537 | INFO     | __main__:main:73 - [probe] angle nc=4 kernel_train=121.989s fit=121.999s infer=65.860s

uv run python -m quantum --n-components 4 --max-samples 1000 --encodings iqp --probe
2026-07-11 17:04:36.342 | INFO     | __main__:main:73 - [probe] iqp nc=4 kernel_train=158.199s fit=158.207s infer=89.611s
```

## Stage 1. 200 samples

### 1.1. Quantum

```sh
uv run python -m quantum --n-components 4 --max-samples 200 --folds 5 \
    --encodings angle iqp --tasks binary --mlflow
2026-07-11 17:05:09.871 | INFO     | quantum.run:run_quantum_cv:120 - [qsvm/binary] outer fold 1/5
2026-07-11 17:05:28.426 | INFO     | quantum.run:evaluate_fold_quantum:100 - [qsvm/binary] fold done: f1_macro=1.0000
2026-07-11 17:05:29.649 | INFO     | quantum.run:run_quantum_cv:120 - [qsvm/binary] outer fold 2/5
2026-07-11 17:05:47.984 | INFO     | quantum.run:evaluate_fold_quantum:100 - [qsvm/binary] fold done: f1_macro=0.9250
2026-07-11 17:05:48.113 | INFO     | quantum.run:run_quantum_cv:120 - [qsvm/binary] outer fold 3/5
2026-07-11 17:06:06.772 | INFO     | quantum.run:evaluate_fold_quantum:100 - [qsvm/binary] fold done: f1_macro=1.0000
2026-07-11 17:06:06.902 | INFO     | quantum.run:run_quantum_cv:120 - [qsvm/binary] outer fold 4/5
2026-07-11 17:06:25.313 | INFO     | quantum.run:evaluate_fold_quantum:100 - [qsvm/binary] fold done: f1_macro=1.0000
2026-07-11 17:06:25.427 | INFO     | quantum.run:run_quantum_cv:120 - [qsvm/binary] outer fold 5/5
2026-07-11 17:06:46.036 | INFO     | quantum.run:evaluate_fold_quantum:100 - [qsvm/binary] fold done: f1_macro=1.0000
```

### 1.2. Classicals

```sh
uv run python -m classical --n-components 4 --load-quantum-splits --tasks binary --mlflow
2026-07-11 17:07:33.787 | INFO     | classical.run:run_nested_cv:112 - [random_forest/binary] outer fold 1/5
2026-07-11 17:07:53.306 | INFO     | classical.run:evaluate_fold:89 - [random_forest/binary] fold done: f1_macro=1.0000
2026-07-11 17:07:54.105 | INFO     | classical.run:run_nested_cv:112 - [random_forest/binary] outer fold 2/5
2026-07-11 17:08:06.133 | INFO     | classical.run:evaluate_fold:89 - [random_forest/binary] fold done: f1_macro=0.9500
2026-07-11 17:08:06.245 | INFO     | classical.run:run_nested_cv:112 - [random_forest/binary] outer fold 3/5
2026-07-11 17:08:17.355 | INFO     | classical.run:evaluate_fold:89 - [random_forest/binary] fold done: f1_macro=1.0000
2026-07-11 17:08:17.481 | INFO     | classical.run:run_nested_cv:112 - [random_forest/binary] outer fold 4/5
2026-07-11 17:08:27.777 | INFO     | classical.run:evaluate_fold:89 - [random_forest/binary] fold done: f1_macro=1.0000
2026-07-11 17:08:27.896 | INFO     | classical.run:run_nested_cv:112 - [random_forest/binary] outer fold 5/5
2026-07-11 17:08:39.090 | INFO     | classical.run:evaluate_fold:89 - [random_forest/binary] fold done: f1_macro=1.0000
2026-07-11 17:08:39.214 | INFO     | __main__:main:102 - wrote per-fold predictions to results/cic/random_forest_binary_predictions.npz
2026-07-11 17:08:39.214 | INFO     | classical.run:run_nested_cv:112 - [xgboost/binary] outer fold 1/5
2026-07-11 17:08:43.991 | INFO     | classical.run:evaluate_fold:89 - [xgboost/binary] fold done: f1_macro=1.0000
2026-07-11 17:08:44.147 | INFO     | classical.run:run_nested_cv:112 - [xgboost/binary] outer fold 2/5
2026-07-11 17:08:45.327 | INFO     | classical.run:evaluate_fold:89 - [xgboost/binary] fold done: f1_macro=0.9500
2026-07-11 17:08:45.508 | INFO     | classical.run:run_nested_cv:112 - [xgboost/binary] outer fold 3/5
2026-07-11 17:08:46.747 | INFO     | classical.run:evaluate_fold:89 - [xgboost/binary] fold done: f1_macro=1.0000
2026-07-11 17:08:46.900 | INFO     | classical.run:run_nested_cv:112 - [xgboost/binary] outer fold 4/5
2026-07-11 17:08:47.964 | INFO     | classical.run:evaluate_fold:89 - [xgboost/binary] fold done: f1_macro=1.0000
2026-07-11 17:08:48.108 | INFO     | classical.run:run_nested_cv:112 - [xgboost/binary] outer fold 5/5
2026-07-11 17:08:49.174 | INFO     | classical.run:evaluate_fold:89 - [xgboost/binary] fold done: f1_macro=1.0000
2026-07-11 17:08:49.466 | INFO     | __main__:main:102 - wrote per-fold predictions to results/cic/xgboost_binary_predictions.npz
2026-07-11 17:08:49.466 | INFO     | classical.run:run_nested_cv:112 - [lightgbm/binary] outer fold 1/5
2026-07-11 17:08:51.759 | INFO     | classical.run:evaluate_fold:89 - [lightgbm/binary] fold done: f1_macro=0.9750
2026-07-11 17:08:51.901 | INFO     | classical.run:run_nested_cv:112 - [lightgbm/binary] outer fold 2/5
2026-07-11 17:08:52.787 | INFO     | classical.run:evaluate_fold:89 - [lightgbm/binary] fold done: f1_macro=0.9500
2026-07-11 17:08:52.919 | INFO     | classical.run:run_nested_cv:112 - [lightgbm/binary] outer fold 3/5
2026-07-11 17:08:53.908 | INFO     | classical.run:evaluate_fold:89 - [lightgbm/binary] fold done: f1_macro=1.0000
2026-07-11 17:08:54.047 | INFO     | classical.run:run_nested_cv:112 - [lightgbm/binary] outer fold 4/5
2026-07-11 17:08:55.120 | INFO     | classical.run:evaluate_fold:89 - [lightgbm/binary] fold done: f1_macro=1.0000
2026-07-11 17:08:55.277 | INFO     | classical.run:run_nested_cv:112 - [lightgbm/binary] outer fold 5/5
2026-07-11 17:08:56.556 | INFO     | classical.run:evaluate_fold:89 - [lightgbm/binary] fold done: f1_macro=1.0000
2026-07-11 17:08:56.719 | INFO     | __main__:main:102 - wrote per-fold predictions to results/cic/lightgbm_binary_predictions.npz
2026-07-11 17:08:56.719 | INFO     | classical.run:run_nested_cv:112 - [svm/binary] outer fold 1/5
2026-07-11 17:08:57.091 | INFO     | classical.run:evaluate_fold:89 - [svm/binary] fold done: f1_macro=1.0000
2026-07-11 17:08:57.218 | INFO     | classical.run:run_nested_cv:112 - [svm/binary] outer fold 2/5
2026-07-11 17:08:57.577 | INFO     | classical.run:evaluate_fold:89 - [svm/binary] fold done: f1_macro=0.9500
2026-07-11 17:08:57.695 | INFO     | classical.run:run_nested_cv:112 - [svm/binary] outer fold 3/5
2026-07-11 17:08:58.044 | INFO     | classical.run:evaluate_fold:89 - [svm/binary] fold done: f1_macro=0.9750
2026-07-11 17:08:58.156 | INFO     | classical.run:run_nested_cv:112 - [svm/binary] outer fold 4/5
2026-07-11 17:08:58.506 | INFO     | classical.run:evaluate_fold:89 - [svm/binary] fold done: f1_macro=1.0000
2026-07-11 17:08:58.615 | INFO     | classical.run:run_nested_cv:112 - [svm/binary] outer fold 5/5
2026-07-11 17:08:58.959 | INFO     | classical.run:evaluate_fold:89 - [svm/binary] fold done: f1_macro=1.0000
2026-07-11 17:08:59.065 | INFO     | __main__:main:102 - wrote per-fold predictions to results/cic/svm_binary_predictions.npz
2026-07-11 17:08:59.067 | INFO     | __main__:main:106 - wrote 4 model x task rows to results/cic/metrics.csv
```

## Stage 2. 1000 samples

### 2.1. Quantum

```sh
uv run python -m quantum --n-components 4 --max-samples 1000 --folds 5 \
    --encodings angle iqp --tasks binary --mlflow
    
```

### 2.2. Classical

```sh
uv run python -m classical --n-components 4 --load-quantum-splits --tasks binary --mlflow

```

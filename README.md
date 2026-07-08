# QTagger+

* **Project Name**: QTagger+: Quantum Machine Learning for Ransomware Tagging and Classification with Generative Data
  Augmentation
* **Mentors**: Dr. Simranjit Singh, Dr. Mohit Sajwan.
* **Team**: C
* **Focus**: Quantum Machine Learning
* **Members**:
    * Elizabeth Quah
    * Zhongyuan Ge
    * Duy Anh

## Project structure

```
QTaggerPlus/
├── src/classical/          # classical ML baseline package
│   ├── data.py             # load CSV, build labels, CV fold indices, prediction I/O
│   ├── features.py         # leakage-free preprocessing (variance/correlation filter + scaler)
│   ├── models.py           # model factories (RF/XGBoost/LightGBM/SVM) + Optuna search spaces
│   ├── evaluate.py         # metrics, confusion matrix, timing
│   ├── compare.py          # paired significance tests (t-test / Wilcoxon / McNemar)
│   ├── run.py              # nested-CV orchestration + W&B logging
│   └── __main__.py         # CLI entrypoint (`python -m classical`)
├── scripts/
│   └── download_cic.py     # Kaggle dataset download helper
├── notebooks/
│   └── 01_cic_eda.ipynb    # exploratory data analysis (run against the real dataset)
├── tests/                  # pytest suite, mirrors src/classical/
├── data/
│   ├── cic_malmem/         # raw dataset (gitignored, downloaded via scripts/download_cic.py)
│   └── splits/             # committed CV fold indices (shared with the quantum team)
├── results/cic/            # generated: metrics.csv, per-fold predictions (gitignored)
├── docs/
│   ├── day6_7_classical_baselines_plan.md
│   └── reports/            # week-by-week deliverable reports
└── README.md
```

## Classical baseline — CIC-MalMem-2022

A classical ML baseline (Random Forest, XGBoost, LightGBM, SVM) over the
CIC-MalMem-2022 obfuscated-malware-memory dataset, with nested stratified CV, Optuna
hyperparameter tuning, and paired significance testing. This baseline exists so the
quantum pipeline has a fair, reproducible point of comparison (same folds, same metrics).

### 1. Kaggle authentication

The dataset is downloaded via the `kaggle` CLI, which (as of v2.2.3) uses the modern
token format, **not** the legacy `~/.kaggle/kaggle.json` file:

1. Generate a token at [kaggle.com/settings/api](https://www.kaggle.com/settings/api).
2. Save it to `~/.kaggle/access_token` (and `chmod 600 ~/.kaggle/access_token`), or
   export it directly as the `KAGGLE_API_TOKEN` environment variable.

### 2. Download the dataset

```sh
uv run python scripts/download_cic.py
```

This fetches `Obfuscated-MalMem2022.csv` into `data/cic_malmem/` (gitignored — raw data
is never committed).

### 3. (Optional) Log in to Weights & Biases

Only needed if you pass `--wandb` to log per-model x task x fold runs:

```sh
wandb login
```

### 4. Run the baseline

Full run (all four models, both binary and multiclass tasks, 5-fold nested CV,
Optuna tuning, logged to W&B):

```sh
uv run python -m classical --wandb
```

Fast smoke test (one model, one task, fewer folds/trials — useful for verifying the
environment is set up correctly before committing to a full run):

```sh
uv run python -m classical --models random_forest --tasks binary --folds 3 --trials 3
```

### 5. Results (full run, 5-fold nested CV)

**Binary (Benign vs. Malware)** — near-perfect across all four models (the
dataset has a few features that individually reach >0.99 univariate AUC, per
the EDA notebook, so treat this as close to a ceiling rather than "solved"):

| Model | F1 (macro) | ROC-AUC |
|---|---|---|
| Random Forest | 0.99995 | 0.999999 |
| LightGBM | 0.99990 | 0.9999998 |
| SVM | 0.99986 | 0.999996 |
| XGBoost | 0.99985 | 0.999993 |

**Multiclass (16 true families, e.g. Ransomware-Ako vs. Ransomware-Conti)** —
the real differentiator. Tree models cluster together; SVM (one-vs-one over
16 classes) trails clearly and is by far the slowest to train:

| Model | F1 (macro) | ROC-AUC |
|---|---|---|
| **LightGBM** | **0.5738** | **0.9634** |
| Random Forest | 0.5726 | 0.9622 |
| XGBoost | 0.5676 | 0.9619 |
| SVM | 0.3853 | 0.9158 |

Full metrics, per-model analysis, methodology rationale, and a writeup of a
memory-exhaustion incident hit (and fixed) during a real multiclass run are in
[`docs/reports/week1_cic_malmem_classical_baseline_report.md`](docs/reports/week1_cic_malmem_classical_baseline_report.md).
Live per-fold results: https://wandb.ai/dduyanhhoang-fpt-university/qtaggerplus-classical

### 6. Outputs

* `results/cic/metrics.csv` — aggregated mean/std per metric for each model x task
  (accuracy, precision, recall, F1, ROC-AUC, and train/inference timings).
* `data/splits/*.json` — the outer CV fold indices, **committed** so the quantum
  team can reuse the exact same train/test splits for a fair comparison.
* `run.log` — rotating log file (loguru) capturing the full run.

## Quantum QSVM (aligned with classical via PCA)

Time a single 1-qubit run before committing to a sweep:
```bash
uv run python -m quantum --probe --n-components 1 --max-samples 120 --encodings angle iqp
```

Run the tuned QSVM CV and log to MLflow:
```bash
uv run python -m quantum --tasks binary --n-components 1 --folds 5 --mlflow
```

Run the classical baseline on the *same* PCA dimensionality for comparison:
```bash
uv run python -m classical --tasks binary --n-components 1 --mlflow
```

View results:
```bash
uv run mlflow ui   # then open http://127.0.0.1:5000
```

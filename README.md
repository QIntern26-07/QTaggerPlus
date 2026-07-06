# QTagger+

* **Project Name**: QTagger+: Quantum Machine Learning for Ransomware Tagging and Classification with Generative Data
  Augmentation
* **Mentors**: Dr. Simranjit Singh, Dr. Mohit Sajwan.
* **Team**: C
* **Focus**: Quantum Machine Learning
* **Members**:
    * Elizabeth
    * Zhongyuan Ge
    * Duy Anh

## Classical baseline — CIC-MalMem-2022

A classical ML baseline (Random Forest, SVM, Logistic Regression, XGBoost) over the
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

### 5. Outputs

* `results/cic/metrics.csv` — aggregated mean/std per metric for each model x task
  (accuracy, precision, recall, F1, ROC-AUC, and train/inference timings).
* `data/splits/*.json` — the outer CV fold indices, **committed** so the quantum
  team can reuse the exact same train/test splits for a fair comparison.
* `run.log` — rotating log file (loguru) capturing the full run.

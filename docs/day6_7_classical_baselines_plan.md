# Day 6–7 Plan — Classical Baselines (Team C)

Scope: my portion of Week 1 — classical ML baselines (RF, XGBoost, LightGBM, SVM) on
CIC-MalMem-2022, EMBER 2018, and SOREL-20M, to be compared against Team C's quantum
models (QSVM, VQC, Hybrid) on Day 6, then documented/packaged on Day 7.

## Task

Train classical classifiers to detect/classify malware on each dataset, evaluate with
Accuracy, Precision, Recall, F1, MCC, ROC-AUC (+ train/inference time), and log every
run so results are directly comparable to the quantum models run by teammates.

## Datasets

| Dataset | Size | Notes |
|---|---|---|
| CIC-MalMem-2022 | ~58K rows, 55 features | Memory-dump features. Small — full run, no caveats. |
| EMBER 2018 | ~1.1M rows, 2381 features | Static PE features. Full run OK for RF/XGBoost/LightGBM; use `LinearSVC` or subsample (~50K rows) for SVM instead of RBF `SVC`. |
| SOREL-20M | ~20M rows | Static PE features, large-scale. Use a stratified subsample (~50K–200K rows) for all models — same subsample Team C uses for quantum, to keep the Day 6 comparison apples-to-apples. |

## Environment additions needed

`pyproject.toml` currently only has `numpy`, `pennylane`, `qiskit`. Need to add:

```
scikit-learn
xgboost
lightgbm
pandas
wandb
loguru
```

Run `uv add scikit-learn xgboost lightgbm pandas wandb loguru` to update
`pyproject.toml`/`uv.lock`.

## Execution steps (Day 6)

1. **Setup** — `uv sync`, `wandb login` once (API key from wandb.ai/authorize; store as
   env var / Colab-Kaggle secret, never hardcode).
2. **CIC-MalMem-2022** — load full dataset, stratified 80/20 split, train RF/XGBoost/
   LightGBM/SVM on the full set (fast locally).
3. **EMBER 2018** — load vectorized features, train RF/XGBoost/LightGBM on full set;
   SVM on `LinearSVC` or a subsample.
4. **SOREL-20M** — load/stream a stratified subsample, train all four models on it only.
5. **If local resources are insufficient** (mainly full EMBER or larger SOREL-20M
   subsamples) — move that run to Colab or Kaggle (Kaggle has EMBER/SOREL hosted as
   existing datasets, avoids re-uploading). Same script, same logging.
6. **Log every run to W&B** — one `wandb.init()` per model × dataset combo:
   - config: model name, dataset, hyperparameters
   - metrics: accuracy, precision, recall, f1_macro, mcc, roc_auc, train_time_sec,
     inference_time_sec
   - artifacts: confusion matrix plot
7. **Use loguru alongside W&B** — loguru handles execution/debug logging (data
   loading progress, warnings, tracebacks, timing checkpoints written to `run.log`);
   W&B handles structured metric comparison. Not redundant — different jobs.
8. **Statistical comparison vs. quantum models** — once teammates' QSVM/VQC/Hybrid
   results are in, run paired significance tests (paired t-test / Wilcoxon across CV
   folds, or McNemar's on shared test sets) rather than comparing point estimates only.
9. **Pull the cross-model comparison table from the W&B dashboard** for Day 6
   figures/report.

## Day 7 deliverables

- Repo README updated with setup + how to reproduce each dataset's baseline run.
- Pipeline doc: dataset → preprocessing/split → model → evaluation → W&B logging.
- Comparison tables/figures (classical vs. quantum, all three datasets) committed
  under `results/` or similar.
- Weekly presentation summarizing baseline results and the classical-vs-quantum
  comparison.
- Draft evaluation protocol for Team A's behavioral dataset + MLRan dataset (spec
  only — executed in Weeks 3–5), covering: which models/metrics to reuse, how
  behavioral (API/registry/file/process/network) features will need different
  preprocessing than the static-feature datasets used here.

## Suggested repo layout

```
QTaggerPlus/
├── data/           # raw/downloaded datasets (gitignored)
├── src/
│   ├── classical/  # RF/XGBoost/LightGBM/SVM training scripts
│   └── quantum/    # teammates' QSVM/VQC/Hybrid code
├── results/        # exported comparison tables, figures, confusion matrices
├── docs/
│   └── day6_7_classical_baselines_plan.md   # this file
```

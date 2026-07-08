# Week 1 Report — Classical ML Baselines on CIC-MalMem-2022

**Author:** dduyanhhoang (Team C — Quantum, per `Simranjit_Singh-Week_1_Plan.md`)
**Project:** QTagger+ (`qi26_7.md`) — Quantum Machine Learning for Ransomware
Tagging and Classification
**Scope:** Day 6–7 deliverable — classical baselines (Random Forest, XGBoost,
LightGBM, SVM) on CIC-MalMem-2022, to be compared against Team C's quantum
models (QSVM, VQC, Hybrid) later in the project.
**Repo / results:** `QTaggerPlus` (branch `feature/cic-malmem-classical-baseline`);
W&B project: https://wandb.ai/dduyanhhoang-fpt-university/qtaggerplus-classical

---

## 1. Purpose and Fit in the Project

Per `Simranjit_Singh-Week_1_Plan.md`, Team C's Week 1 mandate is to build
classical ML baselines on CIC-MalMem-2022, EMBER 2018, and SOREL-20M, so that
Day 6 can compare them statistically against teammates' quantum models, and
Day 7 packages the pipeline and documentation. This report covers the
**CIC-MalMem-2022** portion in full; EMBER 2018 and SOREL-20M reuse the same
pipeline and are follow-on work.

This also lays groundwork for the broader QTagger+ roadmap (`qi26_7.md`): Week 3
calls for "baseline multi-class performance summary" and hybrid
classical/quantum model design, and Week 4 explicitly compares quantum results
against classical models across synthetic/SMOTE/original data conditions. The
paired significance-testing infrastructure built here (Section 6) exists
specifically to make that later comparison rigorous rather than a raw
point-estimate comparison.

---

## 2. Dataset

**CIC-MalMem-2022** (Canadian Institute for Cybersecurity) — memory-dump
features from malware execution, downloaded via Kaggle
(`luccagodoy/obfuscated-malware-memory-2022-cic`; the originally planned slug
did not exist and was corrected after verifying against the live Kaggle API).

- **Shape:** 58,596 rows × 57 columns (55 numeric memory-dump features +
  `Category` + `Class`).
- **Binary label (`Class`):** exactly balanced — 29,298 Benign / 29,298
  Malware (50.0% / 50.0%).
- **Multiclass label (`Category`):** the raw column encodes values like
  `Ransomware-Ako-<hash>-1.raw` or plain `Benign`. Joining the first **two**
  hyphen-separated tokens yields the true family granularity: **16 classes**
  (1 Benign + 5 Ransomware + 5 Spyware + 5 Trojan subfamilies):

  | Family | Count | | Family | Count |
  |---|---|---|---|---|
  | Benign | 29,298 | | Ransomware-Conti | 1,988 |
  | Spyware-Transponder | 2,410 | | Trojan-Emotet | 1,967 |
  | Spyware-Gator | 2,200 | | Ransomware-Maze | 1,958 |
  | Ransomware-Shade | 2,128 | | Trojan-Zeus | 1,950 |
  | Ransomware-Ako | 2,000 | | Ransomware-Pysa | 1,717 |
  | Spyware-180solutions | 2,000 | | Trojan-Reconyc | 1,570 |
  | Spyware-CWS | 2,000 | | Spyware-TIBS | 1,410 |
  | Trojan-Refroso | 2,000 | | | |
  | Trojan-Scar | 2,000 | | | |

  Malware-subfamily imbalance ratio is mild (~1.7×, max 2,410 vs. min 1,410),
  well under the ~10× threshold typically used to justify synthetic
  oversampling (SMOTE) — this directly informed a design decision (§5).

  This was a genuine bug caught mid-project: an earlier implementation of the
  label-extraction logic took only the **first** token, silently collapsing
  the label space to 4 classes (Benign/Spyware/Ransomware/Trojan). It was
  caught and fixed only after downloading the real dataset and inspecting raw
  `Category` values directly — a caveat worth flagging for any downstream
  consumer parsing this dataset's labels.

### Exploratory Data Analysis

An EDA notebook (`notebooks/01_cic_eda.ipynb`) was run end-to-end against the
real data (existing public Kaggle EDAs were surveyed first, including one
notably titled *"CICMalMem2022: Unlikely to be effective"*, which questions the
dataset's real-world classification validity — relevant context for
interpreting the near-perfect binary results below). Key findings that shaped
the pipeline:

- **No missing values.**
- **3 constant/zero-variance columns**: `pslist.nprocs64bit`,
  `handles.nport`, `svcscan.interactive_process_services` — removed via a
  variance filter.
- **Feature redundancy**: correlated-pair counts at four thresholds —
  9 pairs (|r|>0.99), 34 (>0.95), 57 (>0.90), 118 (>0.80). The 0.95 threshold
  was judged a reasonable cut (removes near-duplicates without discarding
  much information).
- **Leakage risk**: 6 features individually reach a univariate ROC-AUC >0.99
  against the binary label (e.g. `dlllist.avg_dlls_per_proc` at 0.9975). This
  is a real concern for interpreting binary results as "solved" — it means a
  single feature is nearly sufficient on its own, which is a known critique of
  memory-dump malware datasets.

---

## 3. Pipeline and Methodology

### 3.1 Task framing

Both **binary** (Benign/Malware) and **multiclass** (16-class family) tasks
were trained, per an explicit design decision: binary is directly comparable
to quantum baselines (QSVM/VQC are typically binary), while multiclass
exercises harder, more realistic classification and imbalance handling.

### 3.2 Validation: nested cross-validation

- **Outer loop:** 5-fold stratified CV for unbiased scoring. Fold indices are
  generated deterministically (`seed=42`) and **committed to
  `data/splits/cic_{binary,multiclass}_folds.json`** so Team C's quantum
  models can be evaluated on the *identical* splits — a prerequisite for
  the paired significance tests planned for Day 6.
- **Inner loop:** 3-fold CV inside each outer training fold, used only for
  hyperparameter search (never touches the outer test fold).
- **Why nested, not a single train/test split:** a single split's score
  depends on which rows happen to land in the test set; nested CV gives 5
  independent score estimates per model, which is exactly the paired data a
  t-test/Wilcoxon/McNemar's test needs, and is the standard rigorous choice
  for a result meant to be scientifically defensible.

### 3.3 Feature pipeline (leakage-free)

Per outer fold: `VarianceThreshold` (drop constant columns) → custom
`DropCorrelated` (drop one of each pair with |Pearson r| > 0.95) →
`StandardScaler`. This pipeline is **fit only on the training fold** and only
`.transform()`-applied to the held-out test fold — verified line-by-line
during code review, since this is the single most important correctness
property for an honest evaluation (get this wrong and every metric is
optimistically biased).

### 3.4 Models and why

Four classical models, matching the exact set Team C's plan specifies for
comparison against QSVM/VQC/Hybrid:

| Model | Why included |
|---|---|
| Random Forest | Strong, low-maintenance bagging baseline; robust to feature scale/redundancy. |
| XGBoost | Fast, regularized gradient boosting; usually a top classical performer. |
| LightGBM | Histogram-based gradient boosting; typically fastest at this scale, native multiclass support. |
| SVM (RBF) | Classical kernel method — the most direct classical analogue to QSVM, making it the most relevant comparison point for the quantum side. |

**Imbalance handling:** `class_weight="balanced"` on RF/LightGBM/SVM (XGBoost
has no such parameter; its loss is otherwise unweighted). SMOTE was
deliberately **not** used, per the EDA finding that multiclass imbalance is
mild (~1.7×) — class weighting is the simpler, sufficient choice, and SMOTE
was reserved as a fallback only if imbalance had proven severe.

### 3.5 Hyperparameter tuning

**Optuna** (Bayesian TPE sampler, `seed=42`, `MedianPruner`), 25 trials per
model × task × outer fold, optimizing inner-CV F1-macro. Chosen over grid/random
search for sample efficiency — Bayesian search converges on good regions of
the hyperparameter space with fewer evaluations, which matters given each
trial requires a full 3-fold inner CV.

### 3.6 Evaluation metrics

Accuracy, Precision (macro), Recall (macro), F1 (macro), MCC, ROC-AUC
(binary: direct; multiclass: one-vs-rest macro-averaged), plus train/inference
time per fold — reported as mean ± std across the 5 outer folds.

### 3.7 Logging and reproducibility infrastructure

- **W&B**: one run per model × task × fold (30 runs total), logging config,
  metrics, a confusion-matrix image, and a per-sample predictions table.
- **loguru**: execution/debug log (`run.log`) — fold timing, progress,
  warnings.
- **Per-fold predictions persisted to disk** (`results/cic/*_predictions.npz`)
  — `test_idx`, `y_true`, `y_pred` per fold, per model × task — specifically
  so the paired significance tests (§6) have real classical-model inputs to
  consume once quantum results exist. This was flagged and closed as a
  Critical gap during final review: metrics alone (mean/std in a CSV) are not
  enough to run a paired test against quantum results.
- **`--n-jobs` CLI control** for machine-specific parallelism tuning (§7).

All of this is orchestrated by a single CLI entrypoint:
```bash
uv run python -m classical --models <model> --tasks <binary|multiclass> --n-jobs <n> --wandb
```

---

## 4. Results

### 4.1 Binary (Benign vs. Malware)

| Model | Accuracy | Precision | Recall | F1 (macro) | MCC | ROC-AUC |
|---|---|---|---|---|---|---|
| Random Forest | 0.99995 ± 0.00007 | 0.99995 | 0.99995 | 0.99995 ± 0.00007 | 0.99990 ± 0.00014 | 0.999999 |
| LightGBM | 0.99990 ± 0.00008 | 0.99990 | 0.99990 | 0.99990 ± 0.00008 | 0.99980 ± 0.00017 | 0.9999998 |
| SVM (RBF) | 0.99986 ± 0.00014 | 0.99986 | 0.99986 | 0.99986 ± 0.00014 | 0.99973 ± 0.00028 | 0.999996 |
| XGBoost | 0.99985 ± 0.00015 | 0.99985 | 0.99985 | 0.99985 ± 0.00015 | 0.99969 ± 0.00029 | 0.999993 |

### 4.2 Multiclass (16-class family)

| Model | Accuracy | Precision | Recall | F1 (macro) | MCC | ROC-AUC |
|---|---|---|---|---|---|---|
| **LightGBM** | 0.7723 ± 0.0019 | 0.5757 ± 0.0041 | 0.5772 ± 0.0027 | **0.5738 ± 0.0030** | 0.6896 ± 0.0026 | 0.9634 ± 0.0004 |
| Random Forest | 0.7711 ± 0.0027 | 0.5741 ± 0.0061 | 0.5752 ± 0.0053 | 0.5726 ± 0.0060 | 0.6879 ± 0.0037 | 0.9622 ± 0.0010 |
| XGBoost | 0.7688 ± 0.0019 | 0.5722 ± 0.0043 | 0.5694 ± 0.0028 | 0.5676 ± 0.0044 | 0.6848 ± 0.0025 | 0.9619 ± 0.0005 |
| SVM (RBF) | 0.6736 ± 0.0022 | 0.4022 ± 0.0025 | 0.3911 ± 0.0045 | 0.3853 ± 0.0038 | 0.5558 ± 0.0028 | 0.9158 ± 0.0008 |

### 4.3 Training time (representative single fold, multiclass task)

| Model | Train time (1 fold, incl. Optuna tuning) |
|---|---|
| Random Forest | ~971 s (~16 min) |
| XGBoost | ~1,095 s (~18 min) |
| LightGBM | ~1,871 s (~31 min) |
| SVM (RBF) | ~4,302 s (~72 min) |

---

## 5. Analysis

**Binary task is effectively saturated** — all four models land within
0.0001–0.0003 of each other on every metric, with ROC-AUC ≥0.9999998 for the
best model. Read at face value this looks like "solved," but the EDA's
leakage-risk finding (§2) tempers that: with individual features already
reaching univariate AUC >0.99, near-perfect scores are close to what any
reasonable classifier would achieve, not necessarily evidence of subtle
signal extraction. This matches the skeptical framing found in prior public
analyses of this dataset ("Unlikely to be effective" for realistic
detection). **Practical implication for the quantum comparison**: binary
results here will likely not be a strong discriminator between classical
and quantum models — most of the four classical models plus, likely, most
quantum encodings will cluster near-perfect too. The multiclass task is
where a meaningful classical-vs-quantum performance gap is more likely to
show up.

**Multiclass is the real, differentiating problem.** All three tree ensembles
converge to a similar ceiling (F1-macro 0.568–0.574), suggesting this is a
genuine problem-difficulty ceiling (label noise from imbalanced/overlapping
subfamilies, not a model-specific weakness) rather than any one algorithm
under-fitting. The **accuracy-vs-F1-macro gap is the key signal**: ~0.77
accuracy but only ~0.57 F1-macro means the models do well in aggregate but
poorly on some individual classes — consistent with rarer subfamilies (e.g.
Spyware-TIBS at 1,410 rows, Trojan-Reconyc at 1,570) being harder to separate
from their siblings within the same top-level family (e.g. distinguishing
Ransomware-Ako from Ransomware-Conti is a much finer-grained decision than
distinguishing Ransomware from Benign).

**SVM is the clear laggard on multiclass** (F1-macro 0.385 vs. ~0.57 for the
tree models, and 72 minutes/fold vs. 16–31 for the others). This is expected:
scikit-learn's RBF `SVC` handles multiclass via one-vs-one, training 120
pairwise binary classifiers for 16 classes, each without the benefit of a
single, jointly-optimized decision boundary. This is also directly relevant
to the quantum comparison, since QSVM is the most natural quantum analogue to
this model — the classical SVM's multiclass ceiling here is a useful lower
reference point.

**Recommendation for Day 6 quantum comparison**: focus primary comparison on
the multiclass task (where classical models are genuinely differentiated and
plausibly beatable), use LightGBM as the strongest classical multiclass
baseline, and use per-fold predictions (already persisted) to run McNemar's
test against quantum predictions on the same folds once available, rather
than comparing point-estimate F1 scores alone.

---

## 6. Statistical Comparison Infrastructure (for Day 6)

A `src/classical/compare.py` module was built ahead of the quantum
comparison, implementing:
- **Paired t-test** and **Wilcoxon signed-rank test** — for comparing two
  models' per-fold score vectors (5 paired values each, from the persisted
  fold-level metrics).
- **McNemar's test** — for comparing two models' per-sample predictions on a
  shared test set. This required extra care: a two-sided exact McNemar test's
  p-value is mathematically symmetric under swapping the two discordant
  cells, so a naive implementation could have a swapped-cell bug that no
  p-value-based test would ever catch. The function now also returns the raw
  discordant counts (`n01`, `n10`) so this specific bug class is verifiable.

This module has no live quantum data to compare against yet — that's Day 6's
task once QSVM/VQC/Hybrid results exist on the same committed fold splits.

---

## 7. Engineering Note: A Real Memory Crash, Root Cause, and Fix

Partway through the multiclass Random Forest run, **the development machine
appeared to crash** — the browser and other applications died and the
desktop became unresponsive.

**Investigation** (via `journalctl`/`dmesg`, not guesswork): the machine had
**not actually rebooted** (`uptime` showed continuous operation since the
prior boot). What had actually happened was a genuine **kernel
out-of-memory (OOM) killer event**:

```
11:45:16  "Under memory pressure, flushing caches" (repeated, ~13 times)
11:51:46  OOM killer: killed process (Brave/Chrome, 1.5GB)
11:57:23  OOM killer: killed process (gnome-software)
```

This lined up exactly with the Random Forest/multiclass training window. The
machine has 15GB RAM total, and the training job pushed memory usage past
what the kernel could sustain alongside the browser — hence what felt like a
"crash" was actually the OS defensively killing processes to survive.

**Root cause (in the code, not the machine):** `models.py` hardcoded
`n_jobs=-1` (use all CPU cores) on every tree-model constructor, and the
inner-CV tuning loop in `run.py` wrapped that same model inside
`cross_val_score(..., n_jobs=-1)` — a **nested parallelism** bug. With 3 inner
CV folds, this meant up to 3 concurrent Random Forest fits, **each also**
trying to claim all 20 CPU cores and allocate its own copy of the training
data and tree structures — multiplying peak memory usage by roughly the
inner fold count, on top of whatever else was running.

**Fix:** `make_model` now accepts an explicit `n_jobs` parameter. Inside the
Optuna tuning objective, models are always built with `n_jobs=1` (since
`cross_val_score` already parallelizes across the inner folds — parallelizing
both levels at once is the oversubscription). Only the single, one-off final
refit after tuning uses the full configured parallelism. This is exposed on
the CLI as `--n-jobs` (default `-1`, unchanged behavior unless explicitly
lowered), giving direct, per-machine control since this pipeline currently
only runs on one developer machine.

**Verification:** after the fix, Random Forest/multiclass was re-run in full
with `--n-jobs 4`, and memory was monitored continuously (available RAM held
between 4–9GB throughout, with an automated low-memory alert armed at
<1.5GB). No further OOM events occurred across any of the subsequent
multiclass runs, including the heaviest one (SVM, 72 minutes/fold). A
regression test (`test_tune_and_fit_forces_single_threaded_model_during_inner_cv`)
now asserts the exact fix directly: every model built during inner-CV tuning
gets `n_jobs=1`, while the single final refit gets the caller's configured
value.

---

## 8. Reproducibility

- Environment: Python 3.12, managed via `uv` (`pyproject.toml` / `uv.lock`).
- All randomness seeded (`seed=42`): fold splits, Optuna's TPE sampler,
  every model constructor.
- Fold indices for both tasks committed to `data/splits/` for exact reuse by
  the quantum team.
- Full command to reproduce any single result:
  ```bash
  uv run python -m classical --models <random_forest|xgboost|lightgbm|svm> \
      --tasks <binary|multiclass> --csv data/cic_malmem/Obfuscated-MalMem2022.csv \
      --n-jobs 4 --wandb
  ```
- All 40 runs (4 models × 2 tasks × 5 folds) are live at
  https://wandb.ai/dduyanhhoang-fpt-university/qtaggerplus-classical

---

## 9. Deliverables Checklist (against Team C's Day 6–7 plan)

- [x] Classical baselines (RF, XGBoost, LightGBM, SVM) trained on CIC-MalMem-2022
- [x] Both binary and multiclass tasks
- [x] Accuracy, Precision, Recall, F1, MCC, ROC-AUC + train/inference time, all logged
- [x] W&B logging (config, metrics, confusion matrix, per-fold predictions) per run
- [x] loguru execution logging
- [x] EDA notebook, run against real data
- [x] Repo README with reproduction instructions
- [x] Statistical comparison module ready for Day 6 (pending quantum results)
- [ ] EMBER 2018 / SOREL-20M baselines (follow-on work, same pipeline)
- [ ] Day 6 quantum-vs-classical statistical comparison (blocked on quantum team's results)
- [ ] Evaluation protocol draft for Team A's behavioral/MLRan dataset (Weeks 3–5, spec-only)

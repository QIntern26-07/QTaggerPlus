# Week 5 / Day 34 — Master tables and paired significance tests

**Date:** 2026-08-01
**Drivers:** `scripts/run_significance_tests.py`, `common.significance.fold_scores`
**Raw output:** `docs/reports/logs/w5_day34/significance.json`,
`mcnemar_classical.json`

This closes **Week 1 Day 6**, assigned as *"Prepare cross-dataset comparison
tables and statistical significance tests."* The infrastructure —
`classical.compare.paired_ttest`, `wilcoxon`, `mcnemar` — was built and unit
tested in Week 1 and **had never been run**. Four weeks of "QSVM loses to
classical" rested entirely on differences in mean macro-F1, with no test that any
gap exceeded fold-to-fold noise. It does.

## 1. How a comparison is selected

Every number below comes from **one** cross-validation sweep, identified by
`parent_run_id`, whose parent is `FINISHED` with exactly 5 fold children. Where a
cell was swept more than once across the project's weeks, the most recently
started clean sweep wins.

This is not a formality. Filtering by `(dataset, task, n_components, model,
encoding)` — the obvious approach, and the one this task was originally specified
with — is wrong twice over, and both failures were measured rather than
anticipated (`w5_day29_ember_four_model.md` §7):

- the same cell has legitimately been swept several times, so a param-based
  groupby returns 10, 15 or 20 rows for a 5-fold sweep and averages distinct
  hyperparameter searches together;
- `encoding` is a **per-fold outcome** for QSVM, not a cell key — the two-tier
  tuner picks the winning encoding inside each fold, so one joint sweep appears
  as "4 angle folds + 1 iqp fold" and a groupby splits it in two.

## 2. Master table — binary

macro-F1, mean ± std over the 5 outer folds.

### EMBER 2018

| nc | random_forest | xgboost | lightgbm | svm | QSVM angle | QSVM iqp |
|---|---|---|---|---|---|---|
| 1 | **0.5582 ± 0.0242** | 0.5451 ± 0.0196 | 0.5093 ± 0.0335 | 0.5314 ± 0.0277 | 0.4532 ± 0.0277 | 0.4587 ± 0.0229 |
| 3 | **0.6575 ± 0.0544** | 0.6465 ± 0.0614 | 0.6434 ± 0.0160 | 0.6306 ± 0.0192 | 0.4924 ± 0.0548 | 0.5084 ± 0.0347 |
| 6 | **0.6835 ± 0.0741** | 0.6680 ± 0.0489 | 0.6593 ± 0.0708 | 0.6724 ± 0.0376 | 0.5138 ± 0.0084 | 0.5137 ± 0.0372 |

### CIC-MalMem

**No table. Not testable.** See §5 — this is the most consequential gap in this
report and it is not a quantum-specific one.

## 3. Master table — 15-class

### CIC-MalMem

| nc | random_forest | xgboost | lightgbm | svm | QSVM angle | QSVM iqp | QSVM joint | random |
|---|---|---|---|---|---|---|---|---|
| 1 | **0.1283 ± 0.0359** | 0.1010 ± 0.0162 | 0.1094 ± 0.0226 | 0.1075 ± 0.0164 | 0.0550 ± 0.0095 | 0.0559 ± 0.0077 | 0.0577 ± 0.0076 | 0.0667 |
| 3 | **0.1398 ± 0.0130** | 0.1270 ± 0.0087 | 0.1185 ± 0.0164 | 0.1309 ± 0.0265 | 0.0585 ± 0.0057 | 0.0707 ± 0.0176 | 0.0585 ± 0.0044 | 0.0667 |
| 6 | 0.1510 ± 0.0262 | 0.1494 ± 0.0249 | 0.1571 ± 0.0221 | **0.1647 ± 0.0233** | 0.0681 ± 0.0147 | 0.0794 ± 0.0171 | 0.0708 ± 0.0141 | 0.0667 |

Every QSVM cell straddles the 0.0667 random baseline. Every classical cell is
1.5–2.5× above it.

### EMBER 2018

| nc | random_forest | xgboost | lightgbm | svm | QSVM angle | QSVM iqp | random |
|---|---|---|---|---|---|---|---|
| 1 | **0.5562 ± 0.0268** | 0.5313 ± 0.0315 | 0.5410 ± 0.0414 | 0.5294 ± 0.0328 | 0.1688 ± 0.0313 | 0.1541 ± 0.0262 | 0.0667 |
| 3 | 0.7194 ± 0.0102 | 0.7089 ± 0.0162 | 0.7035 ± 0.0216 | **0.7389 ± 0.0061** | 0.3908 ± 0.0590 | 0.4797 ± 0.0277 | 0.0667 |
| 6 | 0.7903 ± 0.0238 | 0.7659 ± 0.0214 | 0.7709 ± 0.0166 | **0.7930 ± 0.0208** | 0.4232 ± 0.0751 | 0.5195 ± 0.0582 | 0.0667 |

## 4. Significance — 84 quantum-vs-classical pairs

Paired t-test and Wilcoxon signed-rank over the 5 outer-fold macro-F1 values.
Full per-pair output in `significance.json`.

**Every one of the 84 tested pairs favours classical, and every one reaches
p < 0.05 on the paired t-test.** The largest p-value across all 84 is **0.0241**
(EMBER binary nc=1, QSVM angle vs lightgbm); the smallest is 4 × 10⁻⁶ (EMBER
15-class nc=1, QSVM angle vs random_forest). Summary by cell, worst case shown:

| dataset | task | nc | delta range (classical − QSVM) | worst t-test p |
|---|---|---|---|---|
| CIC | 15-class | 1 | +0.0434 … +0.0733 | 0.0119 |
| CIC | 15-class | 3 | +0.0479 … +0.0814 | 0.0191 |
| CIC | 15-class | 6 | +0.0700 … +0.0966 | 0.0082 |
| EMBER | binary | 1 | +0.0506 … +0.1050 | 0.0241 |
| EMBER | binary | 3 | +0.1222 … +0.1651 | 0.0160 |
| EMBER | binary | 6 | +0.1456 … +0.1698 | 0.0223 |
| EMBER | 15-class | 1 | +0.3606 … +0.4021 | 0.0001 |
| EMBER | 15-class | 3 | +0.2238 … +0.3481 | 0.0007 |
| EMBER | 15-class | 6 | +0.2465 … +0.3699 | 0.0032 |

The project's headline claim now has evidence behind it rather than an
unexamined mean difference.

### 4.1 Wilcoxon returned 0.0625 for all 84 pairs — and that is not a result

With 5 paired samples, the Wilcoxon signed-rank test's **smallest attainable
two-sided p-value is 0.0625**. It is therefore mathematically incapable of
reaching p < 0.05 at this fold count, no matter how large the effect.

All 84 pairs returning exactly 0.0625 means every pair achieved the most extreme
rank configuration possible — all 5 folds favouring classical. That is the
strongest signal the test can emit here. **Reporting "Wilcoxon found nothing
significant" would be a serious misreading**, and reporting it as corroboration
of the t-test would be double-counting a test that had no room to disagree. It is
recorded and set aside.

Raising the fold count to 6 would put 0.03125 within reach. That is the fix if a
non-parametric test is wanted, not a different test.

## 5. What could not be tested, and why

| what | why | quantum-specific? |
|---|---|---|
| **All CIC binary comparisons** (9 cells × 4 models) | Every CIC binary run — classical *and* quantum, 45 fold rows per model — dates from 2026-07-09…07-12, before nested MLflow logging existed. No row carries a `parent_run_id` or a `tags.sweep`, so no sweep boundary is recoverable. | **No** — symmetric across frameworks |
| EMBER "joint" encoding sweeps | Week 4 deliberately ran EMBER one encoding per invocation, so no joint sweep exists to test. | No |
| McNemar, quantum vs classical | QSVM per-fold predictions were never persisted; `run_quantum_cv`'s return value was discarded. Fixed this week (`src/quantum/__main__.py`), but **no existing sweep was re-run**, so nothing is recoverable retroactively. | Yes |

On CIC binary specifically: the raw fold rows do exist and show QSVM at
0.92–1.00 macro-F1, consistent with Week 3's "quantum ties classical on binary"
reading — CIC's binary task is near-ceiling for every model. But the 45 rows per
model resolve into two distinct sweeps per `n_components` with different
configurations and no identifier separating them; grouping them by time-gap
clustering would be inventing a boundary the data does not record. **The
directional claim stands on the Week 3 report; it cannot be significance-tested
from what was logged, and no attempt was made to manufacture a grouping.**

## 6. McNemar — classical vs classical

Sample-level test on the 1000 held-out predictions. `test_idx` and `y_true` were
verified identical for every pair before testing; all 12 pairs share their test
set, so none was dropped.

| group | pair | n01 | n10 | p |
|---|---|---|---|---|
| EMBER nc=6, 15-class | random_forest vs xgboost | 19 | 44 | **0.0022** |
| EMBER nc=6, 15-class | lightgbm vs random_forest | 38 | 21 | **0.0363** |
| EMBER nc=6, 15-class | lightgbm vs xgboost | 17 | 25 | 0.2800 |
| EMBER nc=6, binary | lightgbm vs random_forest | 21 | 30 | 0.2624 |
| EMBER nc=6, binary | random_forest vs xgboost | 33 | 27 | 0.5190 |
| EMBER nc=6, binary | lightgbm vs xgboost | 23 | 26 | 0.7754 |
| CIC 15-class | svm vs xgboost | 57 | 79 | 0.0714 |
| CIC 15-class | random_forest vs svm | 77 | 58 | 0.1210 |
| CIC 15-class | lightgbm vs svm | 85 | 66 | 0.1427 |
| CIC 15-class | random_forest vs xgboost | 35 | 38 | 0.8151 |
| CIC 15-class | lightgbm vs xgboost | 38 | 41 | 0.8221 |
| CIC 15-class | lightgbm vs random_forest | 49 | 49 | 1.0000 |

Reading: on EMBER 15-class the classical models are **not** interchangeable —
random_forest beats xgboost at p = 0.0022. On CIC 15-class **no pair separates**;
the four models are statistically indistinguishable from each other, which is
consistent with Day 31's finding that CIC's projected features barely support
family discrimination at all.

Two caveats on the data feeding this table:

- **`svm` is absent from the EMBER nc=6 rows.** Week 4 wrote its predictions to
  `results/ember/svm_*_predictions.npz` without an `n_components` in the path,
  so which `nc` they belong to is not recoverable from the file.
- **The CIC files carry no `nc` either** (`results/cic/{model}_{task}_predictions.npz`).
  They are whatever the last CIC run wrote. The table therefore says "CIC
  15-class" without an `nc`, deliberately.

Both are the same live defect: `src/classical/__main__.py:119` names prediction
files `{model}_{task}_predictions.npz` with no `n_components`, so each new `nc`
silently overwrites the last. Day 29 worked around it with per-`nc`
`--predictions-dir`, which is why the EMBER nc=6 rows above are trustworthy. The
CLI itself is unchanged — fixing it is a one-line change to the path template and
is listed in the Week 5 backlog rather than done here, because changing it
invalidates the existing legacy files' interpretation and belongs with a
re-run.

## 7. Weighted F1

`f1_weighted` was added to `compute_metrics` this week (`src/common/evaluate.py`).
**No sweep in this report was run with it**, so no weighted-F1 column exists —
adding a metric cannot retroactively produce values for runs that finished before
it existed, and no sweep was re-run to obtain them.

It matters most where it is missing: the 15-class task. CIC's families run 48–82
rows in the subsample (1.71× imbalance), so macro-F1 gives a 48-row family the
same weight as an 82-row one. EMBER's are balanced at 66–67 (1.02×), where macro
and weighted F1 nearly coincide. Any future CIC 15-class sweep should report both.

## 8. Protocol integrity

Every quantum-vs-classical pair above was scored on the identical rows and folds:
the quantum CLI persists its subsample and folds, and the classical runs replay
them with `--load-quantum-splits`. Week 4 demonstrated the pipeline is bit-exact
reproducible across a three-week gap; that finding is carried forward, not
re-derived.

The one place this was verified afresh is EMBER Day 29, where the subsample
indices were checked to select the same `sha256` rows they were built against —
see `w5_day29_ember_four_model.md` §2.

## Artifacts

- `src/common/significance.py` — sweep-aware `fold_scores`.
- `tests/test_significance.py` — 7 tests, covering repeated sweeps, orphaned
  folds under an unfinished parent, and mixed per-fold encodings within one joint
  sweep.
- `scripts/run_significance_tests.py` — the driver.
- `docs/reports/logs/w5_day34/significance.json` — 84 comparisons, 15 skips.
- `docs/reports/logs/w5_day34/mcnemar_classical.json` — 12 pairs.

# Week 5 / Day 29 — EMBER 2018 four-model classical baseline

**Date:** 2026-08-01
**Scope:** close the EMBER classical-coverage gap flagged in `w4_consolidated_report.md` §8.1.
**Status:** complete. All 24 (task × model × `n_components`) cells now hold exactly 5 outer folds.

## 1. Why this was needed

Week 4 produced a full QSVM sweep on EMBER but only ever ran **`svm`** on the
classical side. Every EMBER quantum-classical margin recorded in
`w4_jul-25_ember_binary.md` and `w4_jul-26_ember_multiclass.md` is therefore a
QSVM-vs-SVM margin, not a QSVM-vs-best-classical margin. On CIC the four models
differ by several F1 points, so there was no reason to assume SVM was EMBER's
strongest baseline — and, as §4 shows, on the binary task it is not.

This day adds `random_forest`, `xgboost` and `lightgbm` at `n_components ∈ {1, 3, 6}`
for both tasks, bringing EMBER to the same coverage CIC already had.

## 2. Protocol

Every invocation replayed the persisted quantum splits, so the classical models
scored **the identical 1000 rows and the identical 5 outer folds** the QSVM was
scored on:

```bash
uv run python -m classical --dataset ember \
  --csv data/ember/ember2018_quantum_subset.parquet \
  --models random_forest xgboost lightgbm --n-jobs 4 \
  --tasks ${task} --n-components ${nc} --load-quantum-splits --mlflow \
  --out results/ember/metrics_${task}_nc${nc}.csv \
  --predictions-dir results/ember/nc${nc}
```

for `task ∈ {binary, multiclass}` × `nc ∈ {1, 3, 6}` — six invocations, each run
in the foreground, one per shell call.

Three flags carry non-obvious weight and must not be dropped:

- `--load-quantum-splits` — reads `data/splits/ember_quantum_sample_idx_{task}.json`
  (n = 1000) and `data/splits/ember_{task}_quantum_folds.json`. Without it the CLI
  computes fresh folds and the comparison against QSVM becomes meaningless.
- `--csv data/ember/ember2018_quantum_subset.parquet` — the default is the full
  200,000 × 2,384 `ember2018_test.parquet`, stored as a **single row group** so it
  cannot be streamed (~1.9 GB resident; `w4_consolidated_report.md` §5). It is also
  a correctness matter: the persisted sample indices were computed against the
  18,014-row subset, and the subset is *not* a prefix of the full frame — applying
  the same indices to the full frame selects rows with different `sha256` and
  different labels. Verified directly rather than assumed.
- `--n-jobs 4` — see §3.

`--predictions-dir results/ember/nc${nc}` is required because
`src/classical/__main__.py:119` names prediction files `{model}_{task}_predictions.npz`
with no `n_components` in the name; without a per-`nc` directory, nc=3 silently
overwrites nc=1. Likewise each cell gets its own `--out`, since `:124` rewrites the
metrics CSV wholesale on every invocation.

## 3. Cost, and the resource incident that preceded this run

A first attempt at this grid exhausted the machine and had to be abandoned. The
cause was **`--n-jobs` left at its default `-1`**: on a 20-core laptop that spawns
20 loky workers, each holding its own copy of the fold data. The crash log's
"26 leaked semlock objects / 50 leaked folder objects" is that pool dying. The
timeline shows `random_forest` binary nc1 completing normally, then the machine
failing one second later when `xgboost` started and requested a second 20-worker
pool.

> An earlier write-up of this incident blamed a missing `--csv` flag and claimed
> the completed run had scored the wrong rows. That was wrong and is retracted.
> The corrected re-run reproduced the crashed run's `f1_macro_mean = 0.558183651105556`
> to every recorded digit, across all five folds — only possible if both runs read
> the same rows. The flag had been passed; `--n-jobs` was the fault.

Capped at `--n-jobs 4`, the probe cell (`random_forest`, binary, nc=1) measured
**75.7 s wall, 301 % CPU, peak RSS 879 MB, zero swaps**. Full grid timings:

| task | nc | wall |
|---|---|---|
| binary | 1 | 2 m 02 s |
| binary | 3 | 2 m 30 s |
| binary | 6 | 2 m 35 s |
| multiclass | 1 | 8 m 04 s |
| multiclass | 3 | 7 m 20 s |
| multiclass | 6 | 9 m 14 s |

Total ≈ 32 min, no cell over the 10-minute escalation threshold, so the grid ran
at full width — no reduction to `nc ∈ {1, 6}` and no cut to `--trials` was needed.
The multiclass cells cost roughly 3× the binary ones, as expected for 15 classes.

## 4. Results — binary

`f1_macro`, mean ± std over the 5 outer folds. `svm` carried from Week 4; QSVM
columns carried from `w4_jul-25_ember_binary.md`.

| nc | random_forest | xgboost | lightgbm | svm | QSVM angle | QSVM iqp |
|---|---|---|---|---|---|---|
| 1 | **0.5582 ± 0.0242** | 0.5451 ± 0.0196 | 0.5093 ± 0.0335 | 0.5314 ± 0.0277 | 0.4532 ± 0.0277 | 0.4587 ± 0.0229 |
| 3 | **0.6575 ± 0.0544** | 0.6465 ± 0.0614 | 0.6434 ± 0.0160 | 0.6306 ± 0.0192 | 0.4924 ± 0.0548 | 0.5084 ± 0.0347 |
| 6 | **0.6835 ± 0.0741** | 0.6680 ± 0.0489 | 0.6593 ± 0.0708 | 0.6724 ± 0.0376 | 0.5138 ± 0.0084 | 0.5137 ± 0.0372 |

## 5. Results — multiclass

| nc | random_forest | xgboost | lightgbm | svm | QSVM angle | QSVM iqp |
|---|---|---|---|---|---|---|
| 1 | **0.5562 ± 0.0268** | 0.5313 ± 0.0315 | 0.5410 ± 0.0414 | 0.5294 ± 0.0328 | 0.1688 ± 0.0313 | 0.1541 ± 0.0262 |
| 3 | 0.7194 ± 0.0102 | 0.7089 ± 0.0162 | 0.7035 ± 0.0216 | **0.7389 ± 0.0061** | 0.3908 ± 0.0590 | 0.4797 ± 0.0277 |
| 6 | 0.7903 ± 0.0238 | 0.7659 ± 0.0214 | 0.7709 ± 0.0166 | **0.7930 ± 0.0208** | 0.4232 ± 0.0751 | 0.5195 ± 0.0582 |

## 6. Which classical model actually wins, and what it does to the recorded margins

**Binary: `random_forest` wins at every `n_components`, and SVM is never the
strongest.** Week 4's binary margins were therefore measured against a
sub-optimal baseline and are **understated**:

| task | nc | best classical | best QSVM | gap vs SVM (Week 4) | gap vs best classical | understated by |
|---|---|---|---|---|---|---|
| binary | 1 | random_forest 0.5582 | 0.4587 | +0.0727 | **+0.0995** | 0.0268 |
| binary | 3 | random_forest 0.6575 | 0.5084 | +0.1222 | **+0.1491** | 0.0269 |
| binary | 6 | random_forest 0.6835 | 0.5138 | +0.1587 | **+0.1697** | 0.0110 |
| multiclass | 1 | random_forest 0.5562 | 0.1688 | +0.3606 | **+0.3874** | 0.0268 |
| multiclass | 3 | svm 0.7389 | 0.4797 | +0.2592 | +0.2592 | 0 |
| multiclass | 6 | svm 0.7930 | 0.5195 | +0.2736 | +0.2736 | 0 |

**Multiclass: SVM does hold up** at nc ∈ {3, 6} — it is genuinely EMBER's best
multiclass baseline there, so those two Week 4 margins stand unchanged. At nc = 1
`random_forest` takes it and the gap widens by 0.0268.

Reading of the four rows that move: none of them changes a conclusion — QSVM was
already behind everywhere on EMBER — but each makes the deficit larger, so the
direction of the correction is uniformly against the quantum pipeline. The
honest one-line summary is that **Week 4 flattered QSVM on EMBER**, mildly on
binary and not at all on multiclass at higher `nc`.

Two patterns worth carrying into the Week 5 write-up:

- Every model, classical and quantum, improves monotonically with `n_components`.
  The quantum curve is far flatter on binary (0.4587 → 0.5138, +0.055 over
  1 → 6 components) than the classical one (0.5582 → 0.6835, +0.125), so the gap
  *widens* as dimensionality grows rather than closing.
- On multiclass at nc = 1, QSVM's 0.1688 against a 15-class problem is barely above
  chance-with-imbalance, while classical models reach 0.53–0.56 from the same
  single component. That is the single largest deficit anywhere in the project.

## 7. Data-integrity findings (incidental, but they affect Day 34)

Verifying "5 folds per cell" surfaced two problems in the existing MLflow store.
Neither was created by this day's run; both were already there.

1. **Cell params do not identify a CV run.** The same
   `(dataset, task, model, n_components)` cell has legitimately been swept more
   than once across the project's weeks. A naive groupby over those params returns
   10, 15 or 20 fold rows for a 5-fold sweep and averages distinct hyperparameter
   searches together. Runs must be grouped by **`parent_run_id`**, keeping only
   sweeps whose parent is `FINISHED` with exactly 5 children. Under that rule the
   store holds 125 nested sweeps, **123 of them clean**.
2. **`encoding` is a per-fold outcome for QSVM, not a cell key.** The two-tier
   tuner in `src/quantum/run.py` picks the winning encoding *inside* each fold and
   `_log_quantum_fold` logs that choice, so a single joint sweep can appear as
   "4 angle folds + 1 iqp fold". Grouping by encoding splits one sweep in two.
   (EMBER is unaffected — its Week 4 sweeps were run one encoding at a time.)

The two unclean sweeps:

| parent | status | folds | what it is |
|---|---|---|---|
| `669f467a` | RUNNING | 2 | EMBER binary nc1 angle, interrupted mid-sweep 2026-07-25, relaunched as `cd8e909f`; the 2 orphans duplicate folds 0–1 |
| `69647899` | FINISHED | 3 | CIC `random_forest` multiclass nc2, the truncated sweep from the memory-exhaustion incident in `w4_consolidated_report.md` §6 |

Neither has been deleted. The parent-status rule excludes `669f467a` automatically,
and `69647899`'s 3-fold count excludes it too, so filtering is preferable to
destroying history. This is also why the Week 4 angle/binary/nc1 figure in §4 is
**0.4532 ± 0.0277** (sweep `cd8e909f`) — a flat average over all 7 matching fold
rows gives 0.4480, which is wrong.

`scripts/export_mlflow_runs.py` previously dropped both `run_id` and
`tags.mlflow.*` (which carries `mlflow.parentRunId`), making per-sweep grouping
impossible from the CSV. It now emits `run_id`, `parent_run_id` and `status`.

## 8. Artifacts

- MLflow: 30 new fold runs + 6 sweep parents, `params.dataset = ember-2018`.
- `results/mlflow_runs.csv` — refreshed, 976 runs × 73 columns.
- `results/ember/metrics_{task}_nc{1,3,6}.csv` — per-cell aggregates.
- `results/ember/nc{1,3,6}/{model}_{task}_predictions.npz` — per-fold predictions,
  needed for the McNemar tests on Day 34.

## 9. Follow-ups this creates

- Day 34's `common.significance.fold_scores` must select folds by `parent_run_id`,
  not by cell params, and must not treat `encoding` as a cell key for quantum runs.
- The EMBER tables in `w4_jul-25_ember_binary.md` and `w4_consolidated_report.md`
  §8.1 describe SVM as *the* classical baseline. They are not wrong as written but
  are now incomplete; the Week 5 consolidated report should carry the corrected
  margins from §6.
- QSVM per-fold predictions are still not persisted (`src/quantum/__main__.py`
  discards `run_quantum_cv`'s return value), so McNemar cannot yet compare QSVM
  against these classical predictions. Tracked as Week 5 Task 4.

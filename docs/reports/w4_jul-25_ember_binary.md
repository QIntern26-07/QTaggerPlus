# Week 4 Jul 25 — EMBER 2018 binary QSVM vs SVM (n_components 1/3/6)

## What changed since the last report

This is the **first EMBER 2018 result** in the project. EMBER dataset support
(`src/common/data.py::load_ember`, `ember_family_xy`, `task_xy` routing) was built
earlier in Week 4 but never actually exercised end-to-end. This report closes that
gap for the **binary** task (malicious vs. benign), running the same QSVM-first /
classical-replay protocol already established on CIC-MalMem, on EMBER for the
first time.

Two things are new relative to prior reports:

1. **A pre-built quantum subset file.** `data/ember/ember2018_quantum_subset.parquet`
   (50 MB) was prepared earlier the same day from the 565 MB `ember2018_test.parquet`
   (single row group — loading it directly risks ~1.9 GB in memory on a machine with
   ~2.4 GB free at the time of this run). All commands below point at the subset
   file, never at the full test parquet.
2. **Held-out macro-F1 per encoding, not inner-CV win-count.** Every prior QSVM
   report (CIC binary/multiclass) tuned over `--encodings angle iqp` in one
   invocation and reported which encoding the inner-CV picked per fold. That
   approach conflates "which encoding wins the tuner's internal comparison" with
   "how good is each encoding's held-out generalization" — and the team retired it
   this week. Here, `angle` and `iqp` are run as **separate CLI invocations**, each
   producing its own 5-fold outer-CV macro-F1. The results table below reports both
   encodings' held-out numbers directly; there is no win-count table in this report.

## Experiment design

- **Task:** binary malicious-vs-benign classification, EMBER 2018.
- **Dataset:** `data/ember/ember2018_quantum_subset.parquet`. The full subset file
  has 18,014 rows at a base label ratio of 16,014 malicious (`label=1`) to 2,000
  benign (`label=0`), roughly 8:1.
- **Sample size:** 1,000 rows, stratified (see Step 1 below for why this size was
  kept rather than reduced). At ~8:1 stratification this yields ~889 malicious /
  ~111 benign in the 1000-row subsample, ~178/~22 per 200-row outer test fold.
- **n_components (= qubit budget):** 1, 3, 6 — the same three points used in prior
  CIC sweeps.
- **Protocol:** quantum first (persists sample_idx + folds), then classical SVM
  replayed with `--load-quantum-splits` on the identical 1000 rows / 5 folds /
  PCA-dimensionality. 5-fold outer CV throughout.
- **Encodings:** `angle`, `iqp` — each run as a **separate invocation** per the
  methodology note above.
- **Models:** QSVM (quantum) vs. **SVM only** (classical). Random Forest / XGBoost /
  LightGBM are out of scope for this task by explicit instruction — another team
  member owns those this week.
- **`--n-jobs 8`** on every quantum invocation (20-core machine, shared with an IDE;
  the CLI's own help text warns against `-1`).
- **`--mlflow`** on every run; local file-store backend, `dataset=ember-2018` tag.

## Step 1: probe and the max-samples decision

```sh
uv run python -m quantum --dataset ember --csv data/ember/ember2018_quantum_subset.parquet \
  --probe --tasks binary --n-components 1 --max-samples 200 \
  --encodings angle iqp --n-jobs 8
```

Result:

```
[probe] angle nc=1 kernel_train=1.642s fit=1.733s infer=0.811s
[probe] iqp   nc=1 kernel_train=1.117s fit=1.119s infer=0.544s
```

Projected to n=1000 (Gram build is O(n²), so multiply by (1000/200)² = 25):

- angle: 1.642s × 25 ≈ **41s**
- iqp: 1.117s × 25 ≈ **28s**

Both are far under the 10-minute stop-rule threshold, so the sweep proceeded at
the brief's default **`--max-samples 1000`** — no reduction to 750 was needed.
(For reference, actual nc=1 kernel-train time in the real sweep, visible via
`fit_time_sec_mean` in MLflow, landed at 18.1s/angle and 11.2s/iqp — well within
the projection, and the gap from the naive 25x scaling is expected since the
probe measures a single untuned fit while the real run tunes C/class_weight/
bandwidth over several Gram builds per fold.)

## Commands actually run

```sh
mkdir -p docs/reports/logs/w4_ember_binary

# Step 1 — probe
uv run python -m quantum --dataset ember --csv data/ember/ember2018_quantum_subset.parquet \
  --probe --tasks binary --n-components 1 --max-samples 200 \
  --encodings angle iqp --n-jobs 8

# Step 2/3 — quantum sweep, nc in {1,3,6}, one encoding per invocation
for nc in 1 3 6; do
  for enc in angle iqp; do
    uv run python -m quantum --dataset ember \
      --csv data/ember/ember2018_quantum_subset.parquet \
      --tasks binary --n-components $nc --max-samples 1000 --folds 5 \
      --encodings $enc --n-jobs 8 --mlflow \
      > docs/reports/logs/w4_ember_binary/nc${nc}_${enc}_quantum.txt 2>&1
  done
done

# Step 4 — classical SVM replay on the identical quantum splits, nc in {1,3,6}
for nc in 1 3 6; do
  uv run python -m classical --dataset ember \
    --csv data/ember/ember2018_quantum_subset.parquet \
    --models svm --tasks binary --n-components $nc \
    --load-quantum-splits --folds 5 --mlflow \
    2>&1 | tee docs/reports/logs/w4_ember_binary/nc${nc}_classical.txt
done

# Step 5 — export
uv run python scripts/export_mlflow_runs.py
```

### Wall-clock per invocation

| run | wall clock |
|---|---|
| probe (angle+iqp, n=200) | ~2s |
| nc=1 angle quantum | 3m 51s |
| nc=1 iqp quantum | 2m 28s |
| nc=3 angle quantum | 7m 31s |
| nc=3 iqp quantum | 7m 40s |
| nc=6 angle quantum | 12m 41s |
| nc=6 iqp quantum | 24m 15s |
| nc=1 classical (svm) | 13s |
| nc=3 classical (svm) | 19s |
| nc=6 classical (svm) | 19s |

Total quantum sweep: ~58 minutes wall-clock across 6 invocations. Classical SVM
replays are effectively free (< 1 minute combined) — expected, since SVM's
kernel is a closed-form RBF/linear fit on ≤6-dimensional PCA features versus
QSVM's per-pair fidelity circuit evaluation.

Note the nc=6 iqp run (24m 15s) is nearly 2x the nc=6 angle run (12m 41s) despite
identical sample size and qubit count — iqp's ZZ-interaction circuit is deeper
than angle's per-qubit rotations, and that cost compounds across the O(n²) Gram
pairs and the tuning sweep's repeated Gram builds.

## Full results per n_components (mean ± std across 5 outer folds, MLflow parent-run aggregate)

### n_components = 1

| model | encoding | f1_macro | roc_auc | accuracy | mcc | fit (s) | tune (s) | infer (s) |
|---|---|---|---|---|---|---|---|---|
| qsvm | angle | 0.4532 ± 0.0277 | 0.5316 | 0.7330 | 0.0351 | 18.10 | 18.69 | 8.80 |
| qsvm | iqp | 0.4587 ± 0.0229 | 0.5243 | 0.7410 | 0.0407 | 11.17 | 11.83 | 5.84 |
| svm | — | 0.5314 ± 0.0277 | 0.6779 | 0.6620 | 0.1801 | 0.01 | 2.06 | 0.002 |

### n_components = 3

| model | encoding | f1_macro | roc_auc | accuracy | mcc | fit (s) | tune (s) | infer (s) |
|---|---|---|---|---|---|---|---|---|
| qsvm | angle | 0.4924 ± 0.0548 | 0.6325 | 0.6020 | 0.1384 | 36.03 | 35.54 | 17.96 |
| qsvm | iqp | 0.5084 ± 0.0347 | 0.6701 | 0.6250 | 0.1589 | 37.19 | 35.52 | 18.59 |
| svm | — | 0.6306 ± 0.0192 | 0.7414 | 0.8200 | 0.2794 | 0.02 | 2.88 | 0.003 |

### n_components = 6

| model | encoding | f1_macro | roc_auc | accuracy | mcc | fit (s) | tune (s) | infer (s) |
|---|---|---|---|---|---|---|---|---|
| qsvm | angle | 0.5138 ± 0.0084 | 0.6611 | 0.6350 | 0.1614 | 61.82 | 58.79 | 30.97 |
| qsvm | iqp | 0.5137 ± 0.0372 | 0.6827 | 0.6360 | 0.1582 | 118.41 | 112.63 | 59.26 |
| svm | — | 0.6724 ± 0.0376 | 0.7801 | 0.8460 | 0.3565 | 0.02 | 2.95 | 0.002 |

## Held-out macro-F1 per encoding (the required table for this report)

| n_components | QSVM (angle) | QSVM (iqp) | SVM |
|---|---|---|---|
| 1 | 0.4532 | 0.4587 | **0.5314** |
| 3 | 0.4924 | 0.5084 | **0.6306** |
| 6 | 0.5138 | 0.5137 | **0.6724** |

Classical SVM beats QSVM at every n_components, on both encodings, by a wide and
growing margin (+0.073 at nc=1, up to +0.159 at nc=6). `iqp` edges out `angle` at
nc=1 and nc=3 (+0.0055, +0.0160) but the two are statistically indistinguishable
at nc=6 (0.5138 vs 0.5137 — a 0.0001 difference, well inside either encoding's
own fold-to-fold std of 0.008–0.037).

## Per-class F1 (mean across 5 folds)

Class 0 = benign (minority, ~11% of the 1000-row subsample), class 1 = malicious
(majority, ~89%).

| n_components | model | encoding | F1 class 0 (benign) | F1 class 1 (malicious) |
|---|---|---|---|---|
| 1 | qsvm | angle | 0.090 | 0.816 |
| 1 | qsvm | iqp | 0.093 | 0.824 |
| 1 | svm | — | 0.285 | 0.778 |
| 3 | qsvm | angle | 0.259 | 0.726 |
| 3 | qsvm | iqp | 0.269 | 0.747 |
| 3 | svm | — | 0.366 | 0.895 |
| 6 | qsvm | angle | 0.271 | 0.756 |
| 6 | qsvm | iqp | 0.271 | 0.757 |
| 6 | svm | — | 0.434 | 0.911 |

(angle nc=1 class-0 mean computed from raw per-fold values, in chronological
fold order: 0.000, 0.216, 0.235, 0.000, 0.000; iqp nc=1 from 0.000, 0.222,
0.243, 0.000, 0.000 — see raw logs. Class 0 goes to exactly 0.000 F1 in 3/5
angle folds and 3/5 iqp folds at nc=1: QSVM collapses to an always-malicious
predictor in those folds. See analysis.)

## Analysis

- **Classical SVM wins decisively at every n_components, and the gap widens with
  more components.** QSVM narrows the accuracy/roc_auc gap somewhat as
  n_components grows (roc_auc gap: 0.146 at nc=1 → 0.109 at nc=3 → 0.097 at nc=6)
  but the macro-F1 gap actually widens in absolute terms (0.078 → 0.138 → 0.159)
  because SVM's accuracy and per-class balance both improve faster than QSVM's as
  more PCA signal becomes available. This mirrors the CIC pattern (classical
  ahead of quantum, gap non-trivial) but the EMBER binary gap is noticeably larger
  than CIC's near-ceiling binary numbers — EMBER's malware detection task from
  static PE features is evidently harder for both pipelines than CIC's memory-dump
  binary task, and QSVM in particular struggles with EMBER's larger class
  imbalance (~8:1 vs. CIC's near-balanced binary split).

- **QSVM at nc=1 partially collapses into an always-malicious predictor.** Class-0
  (benign) F1 hits exactly 0.000 in 3 of 5 outer folds for both angle and iqp at
  nc=1 — with only ~22 benign rows in a 200-row test fold, a 1-qubit fidelity
  kernel evidently cannot separate the minority class from the majority at all in
  those folds, and the tuned SVC (even with `class_weight` search) ends up
  predicting label 1 everywhere. This drags the *macro*-F1 down substantially
  despite a deceptively high raw accuracy (0.73–0.74) — accuracy alone would have
  hidden this collapse, which is the whole reason this project reports macro-F1 as
  the headline metric.

- **QSVM's minority-class recovery with more components is real but incomplete.**
  Class-0 F1 climbs from ~0.09 (nc=1) to ~0.27 (nc=3, nc=6) for both encodings —
  a genuine improvement, but it plateaus between nc=3 and nc=6 (0.259→0.271 for
  angle, 0.269→0.271 for iqp) while SVM's class-0 F1 keeps climbing (0.285→0.366→
  0.434) over the same range. QSVM is not just behind, it appears to saturate
  earlier on the components tested.

- **`angle` vs `iqp` is a near-wash on this task**, unlike CIC-binary where `iqp`'s
  ZZ-interaction encoding showed a clearer edge in earlier reports. Here iqp wins
  by a small, plausibly-real margin at nc=1/nc=3 (+0.006, +0.016) but the two are
  identical within noise at nc=6. Given iqp's substantially higher wall-clock cost
  at nc=6 (24m vs. 13m for angle — see wall-clock table) with no accuracy benefit
  at that setting, **angle is the more cost-effective encoding for EMBER binary at
  higher qubit counts**, a finding worth carrying into any EMBER multiclass sweep.

- **Fit/tune/infer times scale as expected with n_components and dominate the
  quantum sweep's wall-clock.** QSVM fit time roughly doubles nc=1→nc=3 (11–18s→
  36–37s) then again nc=3→nc=6 for iqp specifically (37s→118s, a >3x jump versus
  angle's ~1.7x, 36s→62s) — consistent with iqp's deeper circuit making the
  per-pair fidelity evaluation costlier as qubit count grows, on top of the
  quadratic pair-count cost both encodings share.

## Key findings

1. **First EMBER result: classical SVM clearly beats QSVM on binary malware
   detection**, at every n_components tested (macro-F1 gap vs the better-scoring
   QSVM encoding at each nc: 0.073–0.159, widening with more PCA components) —
   consistent in direction with the CIC findings but a larger gap than CIC's
   near-ceiling binary task showed.

2. **QSVM partially collapses to an always-malicious predictor at nc=1** (class-0
   F1 = 0.000 in 3/5 folds, both encodings) on EMBER's ~8:1 imbalanced binary
   task. This is the clearest single negative result in this report and is masked
   by a superficially reasonable accuracy (~0.73–0.74) — another data point for
   why this project insists on macro-F1, not accuracy, as the primary metric.

3. **`angle` and `iqp` are statistically indistinguishable at nc=6** (0.5138 vs.
   0.5137) despite iqp costing ~2x the wall-clock at that setting — angle is the
   more efficient choice for EMBER binary once qubit count is not tiny.

4. **The 50 MB pre-built subset parquet worked as intended**: no memory pressure,
   no OOM risk, and per-probe timings (1.1–1.6s kernel-train at n=200) were small
   enough that the full n=1000 sweep needed no reduction from the brief's default.

## Anomalies / things I was unsure about

- **The exported MLflow CSV contains 2 extra child (fold-level) run rows for
  exactly one sweep — nc=1, angle — out of all nine nc x encoding x model
  combinations run.** 7 child rows are logged for that sweep where the stdout
  log shows exactly 5 "outer fold" iterations; every other sweep has exactly the
  expected number of child rows. **Root cause (confirmed, not code):** the nc=1
  angle sweep's first invocation was interrupted mid-run and relaunched. The two
  extra rows are orphaned folds from that first, interrupted launch — in
  `results/mlflow_runs.csv` they sit at UTC `03:33:13` (f1_macro=0.470899) and
  `03:33:59` (f1_macro=0.399101), and both **predate the parent run's own
  `start_time` of 03:34:36**. Nothing is wrong with `export_mlflow_runs.py` or
  `quantum/run.py`. The **parent aggregate metrics** (`f1_macro_mean`, `_std`,
  etc., used throughout the results tables above) match a hand-computed mean of
  the 5 real "fold done" lines in the raw stdout log exactly, to 6 decimal
  places, so nothing reported above is affected. **Filtering rule for a future
  reader** re-deriving fold-level statistics directly from the CSV: drop any
  child row whose `start_time` precedes its parent run's `start_time` — that
  discards exactly the orphaned rows from an interrupted-and-relaunched sweep
  and nothing else.
- I did not independently re-verify EMBER's label convention (`0`=benign,
  `1`=malicious) against EMBER's own documentation — I inferred it from the raw
  class counts (16,014 vs. 2,000, with 2,000 matching Endgame's published EMBER
  2018 benign-family count) and from `label` being an existing column in
  `common/data.py::task_xy`'s ember-binary branch, but did not open EMBER's
  original spec to confirm.

## Raw logs

Full stdout for all 9 runs (6 quantum + 3 classical) is committed at
`docs/reports/logs/w4_ember_binary/`. Fold-level results inlined below.

### nc=1 angle quantum
```
2026-07-25 10:34:36.343 | INFO | quantum.run:run_quantum_cv:150 - [qsvm/binary] outer fold 1/5
2026-07-25 10:35:26.002 | INFO | quantum.run:evaluate_fold_quantum:111 - [qsvm/binary] fold done: f1_macro=0.4709
2026-07-25 10:35:26.172 | INFO | quantum.run:run_quantum_cv:150 - [qsvm/binary] outer fold 2/5
2026-07-25 10:36:11.813 | INFO | quantum.run:evaluate_fold_quantum:111 - [qsvm/binary] fold done: f1_macro=0.3991
2026-07-25 10:36:11.953 | INFO | quantum.run:run_quantum_cv:150 - [qsvm/binary] outer fold 3/5
2026-07-25 10:36:57.481 | INFO | quantum.run:evaluate_fold_quantum:111 - [qsvm/binary] fold done: f1_macro=0.4557
2026-07-25 10:36:57.643 | INFO | quantum.run:run_quantum_cv:150 - [qsvm/binary] outer fold 4/5
2026-07-25 10:37:42.372 | INFO | quantum.run:evaluate_fold_quantum:111 - [qsvm/binary] fold done: f1_macro=0.4709
2026-07-25 10:37:42.562 | INFO | quantum.run:run_quantum_cv:150 - [qsvm/binary] outer fold 5/5
2026-07-25 10:38:27.709 | INFO | quantum.run:evaluate_fold_quantum:111 - [qsvm/binary] fold done: f1_macro=0.4695
```

### nc=1 iqp quantum
```
2026-07-25 10:39:09.929 | INFO | quantum.run:run_quantum_cv:150 - [qsvm/binary] outer fold 1/5
2026-07-25 10:39:36.355 | INFO | quantum.run:evaluate_fold_quantum:111 - [qsvm/binary] fold done: f1_macro=0.4709
2026-07-25 10:39:36.545 | INFO | quantum.run:run_quantum_cv:150 - [qsvm/binary] outer fold 2/5
2026-07-25 10:40:07.326 | INFO | quantum.run:evaluate_fold_quantum:111 - [qsvm/binary] fold done: f1_macro=0.4130
2026-07-25 10:40:07.467 | INFO | quantum.run:run_quantum_cv:150 - [qsvm/binary] outer fold 3/5
2026-07-25 10:40:37.419 | INFO | quantum.run:evaluate_fold_quantum:111 - [qsvm/binary] fold done: f1_macro=0.4691
2026-07-25 10:40:37.587 | INFO | quantum.run:run_quantum_cv:150 - [qsvm/binary] outer fold 4/5
2026-07-25 10:41:07.721 | INFO | quantum.run:evaluate_fold_quantum:111 - [qsvm/binary] fold done: f1_macro=0.4709
2026-07-25 10:41:07.902 | INFO | quantum.run:run_quantum_cv:150 - [qsvm/binary] outer fold 5/5
2026-07-25 10:41:37.702 | INFO | quantum.run:evaluate_fold_quantum:111 - [qsvm/binary] fold done: f1_macro=0.4695
```

### nc=3 angle quantum
```
2026-07-25 10:41:50.686 | INFO | quantum.run:run_quantum_cv:150 - [qsvm/binary] outer fold 1/5
2026-07-25 10:43:23.809 | INFO | quantum.run:evaluate_fold_quantum:111 - [qsvm/binary] fold done: f1_macro=0.4813
2026-07-25 10:43:23.962 | INFO | quantum.run:run_quantum_cv:150 - [qsvm/binary] outer fold 2/5
2026-07-25 10:44:52.903 | INFO | quantum.run:evaluate_fold_quantum:111 - [qsvm/binary] fold done: f1_macro=0.4390
2026-07-25 10:44:53.054 | INFO | quantum.run:run_quantum_cv:150 - [qsvm/binary] outer fold 3/5
2026-07-25 10:46:22.519 | INFO | quantum.run:evaluate_fold_quantum:111 - [qsvm/binary] fold done: f1_macro=0.4564
2026-07-25 10:46:22.637 | INFO | quantum.run:run_quantum_cv:150 - [qsvm/binary] outer fold 4/5
2026-07-25 10:47:52.146 | INFO | quantum.run:evaluate_fold_quantum:111 - [qsvm/binary] fold done: f1_macro=0.4893
2026-07-25 10:47:52.262 | INFO | quantum.run:run_quantum_cv:150 - [qsvm/binary] outer fold 5/5
2026-07-25 10:49:21.633 | INFO | quantum.run:evaluate_fold_quantum:111 - [qsvm/binary] fold done: f1_macro=0.5960
```

### nc=3 iqp quantum
```
2026-07-25 10:50:01.548 | INFO | quantum.run:run_quantum_cv:150 - [qsvm/binary] outer fold 1/5
2026-07-25 10:51:31.068 | INFO | quantum.run:evaluate_fold_quantum:111 - [qsvm/binary] fold done: f1_macro=0.5005
2026-07-25 10:51:31.229 | INFO | quantum.run:run_quantum_cv:150 - [qsvm/binary] outer fold 2/5
2026-07-25 10:53:04.370 | INFO | quantum.run:evaluate_fold_quantum:111 - [qsvm/binary] fold done: f1_macro=0.4900
2026-07-25 10:53:04.508 | INFO | quantum.run:run_quantum_cv:150 - [qsvm/binary] outer fold 3/5
2026-07-25 10:54:36.603 | INFO | quantum.run:evaluate_fold_quantum:111 - [qsvm/binary] fold done: f1_macro=0.4698
2026-07-25 10:54:36.717 | INFO | quantum.run:run_quantum_cv:150 - [qsvm/binary] outer fold 4/5
2026-07-25 10:56:08.532 | INFO | quantum.run:evaluate_fold_quantum:111 - [qsvm/binary] fold done: f1_macro=0.5088
2026-07-25 10:56:08.657 | INFO | quantum.run:run_quantum_cv:150 - [qsvm/binary] outer fold 5/5
2026-07-25 10:57:41.315 | INFO | quantum.run:evaluate_fold_quantum:111 - [qsvm/binary] fold done: f1_macro=0.5727
```

### nc=6 angle quantum
```
2026-07-25 10:58:21.085 | INFO | quantum.run:run_quantum_cv:150 - [qsvm/binary] outer fold 1/5
2026-07-25 11:00:52.181 | INFO | quantum.run:evaluate_fold_quantum:111 - [qsvm/binary] fold done: f1_macro=0.5062
2026-07-25 11:00:52.313 | INFO | quantum.run:run_quantum_cv:150 - [qsvm/binary] outer fold 2/5
2026-07-25 11:03:25.623 | INFO | quantum.run:evaluate_fold_quantum:111 - [qsvm/binary] fold done: f1_macro=0.5203
2026-07-25 11:03:25.741 | INFO | quantum.run:run_quantum_cv:150 - [qsvm/binary] outer fold 3/5
2026-07-25 11:05:57.499 | INFO | quantum.run:evaluate_fold_quantum:111 - [qsvm/binary] fold done: f1_macro=0.5066
2026-07-25 11:05:57.624 | INFO | quantum.run:run_quantum_cv:150 - [qsvm/binary] outer fold 4/5
2026-07-25 11:08:29.245 | INFO | quantum.run:evaluate_fold_quantum:111 - [qsvm/binary] fold done: f1_macro=0.5088
2026-07-25 11:08:29.379 | INFO | quantum.run:run_quantum_cv:150 - [qsvm/binary] outer fold 5/5
2026-07-25 11:11:02.153 | INFO | quantum.run:evaluate_fold_quantum:111 - [qsvm/binary] fold done: f1_macro=0.5270
```

### nc=6 iqp quantum
```
2026-07-25 11:11:22.992 | INFO | quantum.run:run_quantum_cv:150 - [qsvm/binary] outer fold 1/5
2026-07-25 11:16:14.697 | INFO | quantum.run:evaluate_fold_quantum:111 - [qsvm/binary] fold done: f1_macro=0.5165
2026-07-25 11:16:14.853 | INFO | quantum.run:run_quantum_cv:150 - [qsvm/binary] outer fold 2/5
2026-07-25 11:21:10.258 | INFO | quantum.run:evaluate_fold_quantum:111 - [qsvm/binary] fold done: f1_macro=0.4800
2026-07-25 11:21:10.376 | INFO | quantum.run:run_quantum_cv:150 - [qsvm/binary] outer fold 3/5
2026-07-25 11:25:57.225 | INFO | quantum.run:evaluate_fold_quantum:111 - [qsvm/binary] fold done: f1_macro=0.4950
2026-07-25 11:25:57.344 | INFO | quantum.run:run_quantum_cv:150 - [qsvm/binary] outer fold 4/5
2026-07-25 11:30:42.431 | INFO | quantum.run:evaluate_fold_quantum:111 - [qsvm/binary] fold done: f1_macro=0.4926
2026-07-25 11:30:42.598 | INFO | quantum.run:run_quantum_cv:150 - [qsvm/binary] outer fold 5/5
2026-07-25 11:35:37.600 | INFO | quantum.run:evaluate_fold_quantum:111 - [qsvm/binary] fold done: f1_macro=0.5842
```

### nc=1 classical (svm)
```
2026-07-25 11:36:17.349 | INFO | classical.run:run_nested_cv:140 - [svm/binary] outer fold 1/5
2026-07-25 11:36:25.854 | INFO | classical.run:evaluate_fold:97 - [svm/binary] fold done: f1_macro=0.5408
2026-07-25 11:36:25.976 | INFO | classical.run:run_nested_cv:140 - [svm/binary] outer fold 2/5
2026-07-25 11:36:27.083 | INFO | classical.run:evaluate_fold:97 - [svm/binary] fold done: f1_macro=0.4847
2026-07-25 11:36:27.177 | INFO | classical.run:run_nested_cv:140 - [svm/binary] outer fold 3/5
2026-07-25 11:36:28.192 | INFO | classical.run:evaluate_fold:97 - [svm/binary] fold done: f1_macro=0.5556
2026-07-25 11:36:28.293 | INFO | classical.run:run_nested_cv:140 - [svm/binary] outer fold 4/5
2026-07-25 11:36:29.283 | INFO | classical.run:evaluate_fold:97 - [svm/binary] fold done: f1_macro=0.5592
2026-07-25 11:36:29.378 | INFO | classical.run:run_nested_cv:140 - [svm/binary] outer fold 5/5
2026-07-25 11:36:30.400 | INFO | classical.run:evaluate_fold:97 - [svm/binary] fold done: f1_macro=0.5167
2026-07-25 11:36:30.609 | INFO | __main__:main:108 - wrote per-fold predictions to results/ember/svm_binary_predictions.npz
2026-07-25 11:36:30.611 | INFO | __main__:main:112 - wrote 1 model x task rows to results/ember/metrics.csv
```

### nc=3 classical (svm)
```
2026-07-25 11:36:34.932 | INFO | classical.run:run_nested_cv:140 - [svm/binary] outer fold 1/5
2026-07-25 11:36:46.975 | INFO | classical.run:evaluate_fold:97 - [svm/binary] fold done: f1_macro=0.6146
2026-07-25 11:36:47.143 | INFO | classical.run:run_nested_cv:140 - [svm/binary] outer fold 2/5
2026-07-25 11:36:48.460 | INFO | classical.run:evaluate_fold:97 - [svm/binary] fold done: f1_macro=0.6371
2026-07-25 11:36:48.604 | INFO | classical.run:run_nested_cv:140 - [svm/binary] outer fold 3/5
2026-07-25 11:36:49.909 | INFO | classical.run:evaluate_fold:97 - [svm/binary] fold done: f1_macro=0.6146
2026-07-25 11:36:50.060 | INFO | classical.run:run_nested_cv:140 - [svm/binary] outer fold 4/5
2026-07-25 11:36:51.781 | INFO | classical.run:evaluate_fold:97 - [svm/binary] fold done: f1_macro=0.6212
2026-07-25 11:36:51.933 | INFO | classical.run:run_nested_cv:140 - [svm/binary] outer fold 5/5
2026-07-25 11:36:53.318 | INFO | classical.run:evaluate_fold:97 - [svm/binary] fold done: f1_macro=0.6652
2026-07-25 11:36:53.475 | INFO | __main__:main:108 - wrote per-fold predictions to results/ember/svm_binary_predictions.npz
2026-07-25 11:36:53.477 | INFO | __main__:main:112 - wrote 1 model x task rows to results/ember/metrics.csv
```

### nc=6 classical (svm)
```
2026-07-25 11:36:58.050 | INFO | classical.run:run_nested_cv:140 - [svm/binary] outer fold 1/5
2026-07-25 11:37:10.549 | INFO | classical.run:evaluate_fold:97 - [svm/binary] fold done: f1_macro=0.6138
2026-07-25 11:37:10.718 | INFO | classical.run:run_nested_cv:140 - [svm/binary] outer fold 2/5
2026-07-25 11:37:12.123 | INFO | classical.run:evaluate_fold:97 - [svm/binary] fold done: f1_macro=0.6684
2026-07-25 11:37:12.270 | INFO | classical.run:run_nested_cv:140 - [svm/binary] outer fold 3/5
2026-07-25 11:37:13.613 | INFO | classical.run:evaluate_fold:97 - [svm/binary] fold done: f1_macro=0.6733
2026-07-25 11:37:13.755 | INFO | classical.run:run_nested_cv:140 - [svm/binary] outer fold 4/5
2026-07-25 11:37:15.153 | INFO | classical.run:evaluate_fold:97 - [svm/binary] fold done: f1_macro=0.7325
2026-07-25 11:37:15.306 | INFO | classical.run:run_nested_cv:140 - [svm/binary] outer fold 5/5
2026-07-25 11:37:16.720 | INFO | classical.run:evaluate_fold:97 - [svm/binary] fold done: f1_macro=0.6741
2026-07-25 11:37:16.891 | INFO | __main__:main:108 - wrote per-fold predictions to results/ember/svm_binary_predictions.npz
2026-07-25 11:37:16.893 | INFO | __main__:main:112 - wrote 1 model x task rows to results/ember/metrics.csv
```

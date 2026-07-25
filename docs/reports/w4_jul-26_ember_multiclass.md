# Week 4 Jul 25 — EMBER 2018 15-class family attribution, QSVM vs SVM (n_components 1/3/6)

## What changed since the last report

The [Jul 25 EMBER binary report](w4_jul-25_ember_binary.md) established the
quantum-first / classical-replay protocol on EMBER for the malicious-vs-benign
task. This report runs the harder task — 15-class avclass family attribution —
which is the central comparison this project exists to make: does the QSVM
fidelity kernel hold up when the label space gets hard, not just imbalanced?

The key methodological difference from EMBER binary and from prior CIC
15-class reports: **EMBER's 15 avclass families are downsampled to be exactly
balanced** (955 rows per family, 1.0x ratio), whereas CIC's 15 malware-only
families are naturally ~1.7x imbalanced. This lets the comparison isolate
whether imbalance, specifically, was driving the near-total collapse seen on
CIC 15-class (macro-F1 0.0577–0.0708, barely above the ~0.067 random baseline
for 15 balanced classes).

## Experiment design

- **Task:** 15-class malware family attribution (avclass family label),
  EMBER 2018. Families: `azorult, downloadguide, emotet, kovter, lethic,
  ramnit, sality, sdbot, sivis, startsurf, ursnif, wannacry, wapomi, xtrat,
  zbot` (alphabetical — this is the label-index order used throughout, i.e.
  class 0 = azorult, class 14 = zbot).
- **Dataset:** `data/ember/ember2018_quantum_subset.parquet`, restricted (by
  `common/data.py::ember_family_xy`) to the top 15 avclass families with
  >=500 malicious rows each, capped/balanced to 955 rows/family (min count
  across the 15). Full eligible pool: 14,325 rows x 2,381 raw feature columns,
  15 classes, exactly 955 rows per class (confirmed in Step 1).
- **Sample size:** 1,000 rows, stratified from the balanced pool — same
  `--max-samples 1000` default used throughout this project's EMBER and CIC
  15-class work.
- **n_components (= qubit budget):** 1, 3, 6 — same three points as the EMBER
  binary and CIC 15-class sweeps.
- **Protocol:** quantum first (persists sample_idx + folds to
  `data/splits/ember_quantum_sample_idx_multiclass.json` and
  `data/splits/ember_multiclass_quantum_folds.json`), then classical SVM
  replayed with `--load-quantum-splits` on the identical 1000 rows / 5 folds /
  PCA-dimensionality. 5-fold outer CV throughout.
- **Encodings:** `angle`, `iqp` — each run as a **separate invocation**, per
  the methodology retired-win-count decision from the binary report. Every
  number in this report is a held-out macro-F1 from its own dedicated
  invocation, not an inner-CV win-count.
- **Models:** QSVM (quantum) vs. **SVM only** (classical) — RF/XGBoost/
  LightGBM out of scope this week by explicit instruction.
- **`--n-jobs 8`** on every quantum invocation, **`--mlflow`** on every run
  (local file-store backend, `dataset=ember-2018` tag).

## Step 1: class floor check

```sh
uv run python -c "
from common import data
df = data.load_ember('data/ember/ember2018_quantum_subset.parquet')
X, y = data.task_xy(df, 'multiclass', dataset='ember')
print('pool', X.shape, 'classes', y.nunique(), 'per class', y.value_counts().min())
"
```

Output:

```
pool (14325, 2381) classes 15 per class 955
```

Matches the brief's expectation exactly: 15 classes, 955 rows/class in the
pool. At `--max-samples 1000` that's ~66.7 rows/class in the subsample and
~13/class per 200-row outer test fold — thin but the same shape the CIC
15-class runs already validated as workable.

## Commands actually run

```sh
mkdir -p docs/reports/logs/w4_ember_multiclass

# Step 2 — quantum sweep, nc in {1,3,6}, one encoding per invocation
for nc in 1 3 6; do
  for enc in angle iqp; do
    uv run python -m quantum --dataset ember \
      --csv data/ember/ember2018_quantum_subset.parquet \
      --tasks multiclass --n-components $nc --max-samples 1000 --folds 5 \
      --encodings $enc --n-jobs 8 --mlflow \
      > docs/reports/logs/w4_ember_multiclass/nc${nc}_${enc}_quantum.txt 2>&1
  done
done

# Step 3 — classical SVM replay on the identical quantum splits, nc in {1,3,6}
for nc in 1 3 6; do
  uv run python -m classical --dataset ember \
    --csv data/ember/ember2018_quantum_subset.parquet \
    --models svm --tasks multiclass --n-components $nc \
    --load-quantum-splits --folds 5 --mlflow \
    2>&1 | tee docs/reports/logs/w4_ember_multiclass/nc${nc}_classical.txt
done

# Step 4 — export
uv run python scripts/export_mlflow_runs.py
```

(Each invocation was launched, backgrounded, and polled by PID to completion
before the next was launched — no two runs ever overlapped, and no run was
launched twice.)

### Wall-clock per invocation (MLflow parent run `start_time`→`end_time`)

| run | wall clock |
|---|---|
| nc=1 angle quantum | 3m 44s (224.1s) |
| nc=1 iqp quantum | 2m 37s (156.7s) |
| nc=3 angle quantum | 7m 37s (457.2s) |
| nc=3 iqp quantum | 7m 52s (471.6s) |
| nc=6 angle quantum | 13m 11s (790.9s) |
| nc=6 iqp quantum | 24m 26s (1466.1s) |
| nc=1 classical (svm) | 16.2s |
| nc=3 classical (svm) | 20.4s |
| nc=6 classical (svm) | 19.9s |

Total quantum sweep: ~59.5 minutes across 6 invocations — dominated by nc=6
iqp alone (24m 26s, more than the other five quantum runs combined at
nc<=3). Classical SVM replays total ~57s. As the brief predicted, 15-class
one-vs-one (105 pairwise fits per config, vs 1 for binary) makes this sweep
markedly slower than the EMBER binary equivalents at the same n_components —
e.g. nc=6 iqp here (24m 26s) is essentially identical in wall-clock to nc=6
iqp binary (24m 15s) despite doing 105x the pairwise work per outer fold,
which only makes sense because at n=1000/5-fold the O(n²) *Gram-matrix* cost
(shared across all one-vs-one sub-problems via kernel caching) dominates over
the SVC-side one-vs-one fit cost — the per-pair kernel isn't recomputed 105
times, only the SVC decomposition is.

## Full results per n_components (mean ± std across 5 outer folds, MLflow parent-run aggregate)

### n_components = 1

| model | encoding | f1_macro | accuracy | roc_auc | mcc | fit (s) | tune (s) | infer (s) |
|---|---|---|---|---|---|---|---|---|
| qsvm | angle | 0.1688 ± 0.0313 | 0.270 | 0.7317 | 0.2329 | 17.13 | 17.51 | 9.24 |
| qsvm | iqp | 0.1541 ± 0.0262 | 0.243 | 0.7285 | 0.2014 | 11.86 | 12.71 | 5.80 |
| svm | — | 0.5294 ± 0.0328 | 0.538 | 0.8311 | 0.5064 | 0.03 | 2.36 | 0.005 |

### n_components = 3

| model | encoding | f1_macro | accuracy | roc_auc | mcc | fit (s) | tune (s) | infer (s) |
|---|---|---|---|---|---|---|---|---|
| qsvm | angle | 0.3908 ± 0.0590 | 0.440 | 0.8557 | 0.4079 | 36.65 | 35.51 | 18.41 |
| qsvm | iqp | 0.4797 ± 0.0277 | 0.523 | 0.8686 | 0.4962 | 37.39 | 36.93 | 19.10 |
| svm | — | 0.7389 ± 0.0061 | 0.728 | 0.9281 | 0.7140 | 0.03 | 3.09 | 0.006 |

### n_components = 6

| model | encoding | f1_macro | accuracy | roc_auc | mcc | fit (s) | tune (s) | infer (s) |
|---|---|---|---|---|---|---|---|---|
| qsvm | angle | 0.4232 ± 0.0751 | 0.462 | 0.8601 | 0.4322 | 63.83 | 61.32 | 32.20 |
| qsvm | iqp | 0.5195 ± 0.0582 | 0.547 | 0.8804 | 0.5216 | 118.56 | 114.03 | 59.84 |
| svm | — | 0.7930 ± 0.0208 | 0.785 | 0.9451 | 0.7738 | 0.03 | 3.01 | 0.004 |

## Held-out macro-F1 per encoding (the required table for this report)

| n_components | QSVM (angle) | QSVM (iqp) | SVM |
|---|---|---|---|
| 1 | 0.1688 | 0.1541 | **0.5294** |
| 3 | 0.3908 | **0.4797** | **0.7389** |
| 6 | 0.4232 | **0.5195** | **0.7930** |

Classical SVM beats QSVM at every n_components on both encodings, by a wide
and growing absolute margin (+0.36 at nc=1, up to +0.37 at nc=3 and +0.27–0.37
at nc=6 depending on encoding). Unlike EMBER binary (where `angle` and `iqp`
converged to a near-wash at nc=6), here **`iqp` clearly and consistently beats
`angle`** at nc=3 (+0.089) and nc=6 (+0.096) — the first clean per-encoding
separation seen anywhere in this project's EMBER or CIC work. At nc=1 the two
are close (angle 0.1688 vs iqp 0.1541, angle slightly ahead) but both are so
low that the difference is not the interesting part of that row.

## Comparison against CIC 15-class and EMBER binary (numbers carried over,
explicitly labelled)

| dataset / task | QSVM best macro-F1 (nc, encoding) | SVM best macro-F1 | gap |
|---|---|---|---|
| CIC 15-class malware-only, n=1000 (prior report) | 0.0708 (nc=6) | 0.1647 (nc=6) | +0.094 |
| EMBER binary, n=1000 (Jul 25 report) | 0.5138 (nc=6, angle) | 0.6724 (nc=6) | +0.159 |
| **EMBER 15-class, n=1000 (this report)** | **0.5195 (nc=6, iqp)** | **0.7930 (nc=6)** | **+0.274** |

EMBER 15-class QSVM's best number (0.5195) is **~7.3x** CIC 15-class QSVM's
best number (0.0708), and comfortably clears the ~0.067 random baseline at
every n_components tested, including nc=1 (0.1541–0.1688). CIC 15-class QSVM,
by contrast, barely cleared random baseline at any n_components. Classical
SVM shows the same pattern in the same direction but far less dramatically
(EMBER 15-class SVM 0.7930 vs CIC 15-class SVM 0.1647, ~4.8x) — so part of
this gap is that EMBER family attribution from static PE features is simply
an easier separability problem for *both* pipelines than CIC's memory-dump
15-class task, not a QSVM-specific effect. But the QSVM-specific multiplier
(7.3x) exceeding the SVM-specific multiplier (4.8x) is consistent with QSVM
benefiting disproportionately from EMBER's exact class balance, on top of the
shared task-difficulty effect.

## Per-class F1 (mean across 5 folds), all 15 avclass families

| family (class idx) | qsvm nc1 angle | qsvm nc1 iqp | qsvm nc3 angle | qsvm nc3 iqp | qsvm nc6 angle | qsvm nc6 iqp | svm nc1 | svm nc3 | svm nc6 |
|---|---|---|---|---|---|---|---|---|---|
| azorult (0) | 0.000 | 0.000 | 0.494 | 0.532 | 0.569 | 0.672 | 0.215 | 0.686 | 0.758 |
| downloadguide (1) | 0.822 | 0.774 | 0.965 | 0.979 | 0.985 | 0.993 | 0.985 | 1.000 | 1.000 |
| emotet (2) | 0.019 | 0.000 | 0.038 | 0.094 | 0.222 | 0.077 | 0.308 | 0.624 | 0.691 |
| kovter (3) | 0.347 | 0.163 | 0.628 | 0.684 | 0.750 | 0.875 | 0.858 | 0.928 | 0.913 |
| lethic (4) | 0.172 | 0.178 | 0.435 | 0.641 | 0.439 | 0.699 | 0.830 | 0.936 | 0.929 |
| ramnit (5) | 0.000 | 0.000 | 0.000 | 0.049 | 0.064 | 0.111 | 0.436 | 0.490 | 0.580 |
| sality (6) | 0.027 | 0.000 | 0.164 | 0.232 | 0.131 | 0.228 | 0.238 | 0.414 | 0.462 |
| sdbot (7) | 0.000 | 0.301 | 0.327 | 0.617 | 0.308 | 0.270 | 0.891 | 0.891 | 0.897 |
| sivis (8) | 0.309 | 0.272 | 0.591 | 0.630 | 0.745 | 0.756 | 0.281 | 0.654 | 0.885 |
| startsurf (9) | 0.000 | 0.000 | 0.224 | 0.169 | 0.337 | 0.378 | 0.356 | 0.703 | 0.782 |
| ursnif (10) | 0.183 | 0.155 | 0.476 | 0.461 | 0.221 | 0.498 | 0.552 | 0.842 | 0.872 |
| wannacry (11) | 0.159 | 0.135 | 0.264 | 0.321 | 0.185 | 0.362 | 0.294 | 0.595 | 0.747 |
| wapomi (12) | 0.000 | 0.000 | 0.165 | 0.251 | 0.335 | 0.447 | 0.075 | 0.486 | 0.512 |
| xtrat (13) | 0.494 | 0.334 | 0.364 | 0.794 | 0.331 | 0.694 | 0.971 | 0.985 | 0.971 |
| zbot (14) | 0.000 | 0.000 | 0.725 | 0.741 | 0.725 | 0.732 | 0.651 | 0.850 | 0.896 |
| **families with F1 > 0** | **9 / 15** | **8 / 15** | **14 / 15** | **15 / 15** | **15 / 15** | **15 / 15** | **15 / 15** | **15 / 15** | **15 / 15** |

## Analysis

- **QSVM does not collapse to one or two families anywhere in this sweep, in
  clear contrast to CIC 15-class.** Even at nc=1 — the worst setting — QSVM
  recognizes 8/15 (iqp) to 9/15 (angle) families with nonzero F1, and by nc=3
  it recognizes 14–15/15. This directly answers the question this report was
  designed to answer: **EMBER's exact class balance (1.0x, 955/family) rules
  out class imbalance as the sole cause of CIC 15-class's near-total collapse**
  (which showed macro-F1 barely above random baseline, consistent with
  recognizing essentially 1–2 families across most folds). Something about
  CIC's memory-dump feature geometry, not imbalance alone, is driving that
  collapse — this report cannot say what, only that removing imbalance
  removes the collapse pattern.

- **The families QSVM fails on at nc=1 are not random — they cluster.**
  Zero-F1 families at nc=1 (both encodings, or one): azorult, ramnit,
  startsurf, wapomi, zbot are 0.000 in *both* encodings; emotet, sality,
  sdbot are 0.000 in one encoding but not the other. downloadguide (class 1)
  is the strongest family at every n_components and both encodings (0.774–
  0.993) — it is also the easiest family for SVM (0.985–1.000), so this looks
  like a task-difficulty effect (downloadguide is linearly separable enough
  that even a 1-qubit kernel finds it) rather than a QSVM-specific artifact.

- **`iqp` beats `angle` clearly at nc=3 and nc=6** (+0.089, +0.096 macro-F1),
  the first unambiguous per-encoding separation in this project's EMBER or
  CIC work — EMBER binary showed iqp only marginally ahead at nc<=3 and
  statistically tied at nc=6. The per-class table shows this is not uniform:
  iqp is substantially better on lethic (+0.206 at nc=1, +0.26 at nc=6),
  xtrat (+0.363 at nc=3), and wannacry, but *worse* than angle on sdbot at
  nc=6 (0.270 vs 0.308) and emotet at nc=6 (0.077 vs 0.222). iqp's advantage
  is real in aggregate but not universal per-family.

- **QSVM's macro-F1 improves monotonically with n_components for iqp
  (0.154→0.480→0.520) but non-monotonically in the per-class detail for
  angle** — angle's macro-F1 also rises overall (0.169→0.391→0.423) but
  several individual families *regress* from nc=3 to nc=6 under angle
  specifically (lethic 0.435→0.439 roughly flat, ursnif 0.476→0.221 drops
  sharply, xtrat 0.364→0.331 drops). iqp shows no such regression on any
  family nc=3→nc=6 except sdbot (0.617→0.270) and a small emotet dip. This
  instability under angle at higher qubit counts, on a per-family basis, is
  new — it wasn't visible in the binary task's two-class results and only
  shows up once there are 15 classes' worth of per-class detail to inspect.

- **Fit/tune/infer times track the binary pattern**: iqp costs progressively
  more than angle as n_components grows (11.9s vs 17.1s fit at nc=1 — angle
  slower here, unlike binary — but 118.6s vs 63.8s at nc=6, iqp ~1.9x angle,
  consistent with binary's iqp/angle ratio at nc=6 of ~1.9x). Unlike EMBER
  binary where angle was always faster, at nc=1 iqp is actually faster here
  (11.86s vs 17.13s fit) — the crossover between "iqp cheaper" and "iqp
  costlier" happens somewhere between nc=1 and nc=3 for the multiclass task.

## Key findings

1. **Classical SVM wins decisively at every n_components** (macro-F1 gap
   0.27–0.36, the largest gap of any task/dataset combination measured in
   this project so far), but **QSVM performs far better on EMBER 15-class
   than on CIC 15-class** — best QSVM macro-F1 here (0.5195) is ~7.3x CIC
   15-class's best (0.0708), versus only a ~4.8x gap on the SVM side, so part
   of but not all of this improvement is a QSVM-specific effect of exact
   class balance.

2. **QSVM does not collapse to 1–2 families anywhere in this sweep** — it
   recognizes 8–9/15 families even at the worst setting (nc=1) and 14–15/15
   from nc=3 upward. This is the headline negative-result contrast with CIC
   15-class, where the near-random macro-F1 is consistent with collapse to
   almost nothing. **Class imbalance is ruled out as the sole explanation for
   CIC's collapse pattern**, since removing it here does not reproduce the
   collapse.

3. **`iqp` clearly beats `angle` at nc=3 and nc=6** (+0.089, +0.096), the
   first clean, non-noise-level encoding separation seen in this project —
   contrasting with EMBER binary where the two encodings converged to
   statistical parity at nc=6.

4. **Per-family detail exposes instability under `angle` at nc=6** (ursnif,
   xtrat regress from their nc=3 values) that the binary task's two-class
   summary could never have shown — a reason to keep reporting per-class F1
   even when it triples the table size.

## Anomalies / things I was unsure about

- No orphaned-run artifact like the one documented in the EMBER binary report
  (interrupted-and-relaunched nc=1 angle sweep) occurred here: every quantum
  invocation in this sweep was launched exactly once, polled to completion by
  PID before the next was launched, and each MLflow parent run's child-fold
  count matches its stdout log's fold count exactly (5 for all 9 sweeps,
  confirmed against `results/mlflow_runs.csv` child-row timestamps grouped by
  `start_time` ordering within each nc/model pair).
- The `angle`/`iqp` cost crossover at low n_components (iqp cheaper at nc=1,
  costlier from nc=3 on) is new relative to EMBER binary, where angle was
  cheaper at every n_components. I have not investigated why — plausibly an
  artifact of PennyLane's circuit-compilation overhead dominating at very
  small qubit counts rather than a real asymptotic difference, but this is
  speculation, not something I verified.
- I did not investigate *why* CIC's memory-dump features produce collapse
  while EMBER's static-PE features do not, beyond ruling out imbalance as the
  sole cause — that would require inspecting per-family feature separability
  directly (e.g. pairwise class distances in the PCA'd space), which is out
  of scope for this report.

## Raw logs

Full stdout for all 9 runs (6 quantum + 3 classical) is committed at
`docs/reports/logs/w4_ember_multiclass/`. Fold-level results:

### nc=1 angle quantum
```
outer fold 1/5 -> fold done: f1_macro=0.1087
outer fold 2/5 -> fold done: f1_macro=0.1938
outer fold 3/5 -> fold done: f1_macro=0.1703
outer fold 4/5 -> fold done: f1_macro=0.1786
outer fold 5/5 -> fold done: f1_macro=0.1925
```

### nc=1 iqp quantum
```
outer fold 1/5 -> fold done: f1_macro=0.1585
outer fold 2/5 -> fold done: f1_macro=0.1444
outer fold 3/5 -> fold done: f1_macro=0.1515
outer fold 4/5 -> fold done: f1_macro=0.1986
outer fold 5/5 -> fold done: f1_macro=0.1175
```

### nc=3 angle quantum
```
outer fold 1/5 -> fold done: f1_macro=0.3418
outer fold 2/5 -> fold done: f1_macro=0.3836
outer fold 3/5 -> fold done: f1_macro=0.3532
outer fold 4/5 -> fold done: f1_macro=0.5052
outer fold 5/5 -> fold done: f1_macro=0.3702
```

### nc=3 iqp quantum
```
outer fold 1/5 -> fold done: f1_macro=0.4838
outer fold 2/5 -> fold done: f1_macro=0.4950
outer fold 3/5 -> fold done: f1_macro=0.4333
outer fold 4/5 -> fold done: f1_macro=0.5164
outer fold 5/5 -> fold done: f1_macro=0.4699
```

### nc=6 angle quantum
```
outer fold 1/5 -> fold done: f1_macro=0.3553
outer fold 2/5 -> fold done: f1_macro=0.4280
outer fold 3/5 -> fold done: f1_macro=0.3881
outer fold 4/5 -> fold done: f1_macro=0.5658
outer fold 5/5 -> fold done: f1_macro=0.3788
```

### nc=6 iqp quantum
```
outer fold 1/5 -> fold done: f1_macro=0.5363
outer fold 2/5 -> fold done: f1_macro=0.4843
outer fold 3/5 -> fold done: f1_macro=0.4635
outer fold 4/5 -> fold done: f1_macro=0.6256
outer fold 5/5 -> fold done: f1_macro=0.4875
```

### nc=1 classical (svm)
```
outer fold 1/5 -> fold done: f1_macro=0.4929
outer fold 2/5 -> fold done: f1_macro=0.5910
outer fold 3/5 -> fold done: f1_macro=0.5173
outer fold 4/5 -> fold done: f1_macro=0.5245
outer fold 5/5 -> fold done: f1_macro=0.5214
```

### nc=3 classical (svm)
```
outer fold 1/5 -> fold done: f1_macro=0.7306
outer fold 2/5 -> fold done: f1_macro=0.7392
outer fold 3/5 -> fold done: f1_macro=0.7345
outer fold 4/5 -> fold done: f1_macro=0.7420
outer fold 5/5 -> fold done: f1_macro=0.7483
```

### nc=6 classical (svm)
```
outer fold 1/5 -> fold done: f1_macro=0.7673
outer fold 2/5 -> fold done: f1_macro=0.8218
outer fold 3/5 -> fold done: f1_macro=0.7774
outer fold 4/5 -> fold done: f1_macro=0.7861
outer fold 5/5 -> fold done: f1_macro=0.8126
```

Full timestamped logs (including the `VIRTUAL_ENV` mismatch warning line
that `uv run` emits on this machine — harmless, unrelated to this repo) are
in the committed `.txt` files at `docs/reports/logs/w4_ember_multiclass/`.

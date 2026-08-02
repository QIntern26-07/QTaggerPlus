# Week 5 / Day 31 — CIC vs EMBER 15-class separability, and subsample fidelity

**Date:** 2026-08-01
**Driver:** `scripts/compare_class_geometry.py` (~32 s)
**Raw output:** `docs/reports/logs/w5_day31/geometry.json`

Two questions share one data load. Section A closes an assignment from Week 2
Day 2 that was never executed; Section B takes up the open question
`w4_consolidated_report.md` ends on.

---

# Section A — Are the 1000-row quantum subsamples faithful?

Every quantum result in this project is computed on a 1000-row subsample. That
the subsample represents its population has been assumed for four weeks and
never demonstrated. Week 2 Day 2 assigned exactly this: *"Validate that each
subset preserves the class distribution and feature statistics of the full
data."*

## A.1 What "full population" means here

| dataset | reference population | rows | features |
|---|---|---|---|
| CIC-MalMem | `Obfuscated-MalMem2022.csv` after `task_xy(..., "multiclass")` — malware-only, 15 families | 29,298 | 55 |
| EMBER 2018 | `ember2018_quantum_subset.parquet` after `ember_family_xy` — balanced top-15 avclass pool | 14,325 | 2,381 |

EMBER's reference is the 18,014-row quantum subset, **not** the 200,000-row test
parquet. This is a hard requirement, not a shortcut:
`data/splits/ember_quantum_sample_idx_multiclass.json` was computed against the
pool `ember_family_xy` derives from the subset file, and that function
re-downsamples every kept family to the smallest kept count *at load time* — so
feeding a different input frame produces a differently sized and differently
ordered pool, and the persisted indices would select the wrong rows. (The full
parquet is also a single 200,000 × 2,384 row group that cannot be streamed,
~1.9 GB resident; `w4_consolidated_report.md` §5.)

The consequence to state plainly: this section validates 1,000 rows against
14,325, and **not** against the 200,000-row EMBER test set. The earlier
200K → subset step deliberately rebalances families from a heavy-tailed
distribution to an exactly balanced one, so drift against the true full EMBER is
by design and is not a defect. What is being checked is the step the models'
comparability actually depends on.

## A.2 Results

| statistic | CIC | EMBER | reads as |
|---|---|---|---|
| classes | 15 | 15 | |
| max abs class-proportion diff | **0.00044** | **0.00067** | stratification is essentially exact |
| max per-feature mean drift (population SDs) | **0.046** | **0.122** | no feature's mean moves an eighth of an SD |
| max abs std-ratio deviation | 1.000 | 2.783 | see A.3 — not what it looks like |
| KS rejections at α=0.05 | **0 of 55** | **1 of 2,381** | |
| KS rejections expected under the null | 2.75 | 119.05 | |

The KS line is the decisive one, and it is why the expected count is printed
next to it. A perfectly faithful subsample still produces about `α × n_features`
rejections by chance alone — 2.75 for CIC, 119 for EMBER. Both datasets come in
**far below chance**: CIC rejects nothing where ~3 were expected, EMBER rejects
one feature where ~119 were expected. A bare "1 rejection" would be
uninterpretable; against 119 expected it is conclusive.

**Verdict: both subsamples are faithful.** Class proportions, per-feature means
and per-feature distributions are preserved. The Week 2 Day 2 assignment is
closed, and every quantum result in the project rests on a foundation that has
now been measured rather than assumed.

## A.3 The one number that looks alarming and is not

`max_abs_std_ratio_dev` reaching 1.0 on CIC and 2.78 on EMBER deserves an
explanation rather than a footnote.

The first implementation of `feature_drift` gave every zero-variance feature a
denominator of 1.0. A feature that is constant in the population is constant in
any subsample of it, so its ratio is 0/1 — a deviation of exactly 1.0, the
largest a well-behaved feature could show. CIC's `pslist.nprocs64bit` is
constant, and that artifact alone drove the reported maximum. The function now
computes the ratio over non-constant features only and counts the constant ones
separately, so the statistic no longer flags a perfectly preserved feature as
the worst drift in the dataset.

What remains after that fix is real, and is a different finding:

| | CIC | EMBER |
|---|---|---|
| constant in the population | 3 | 83 |
| constant in the subsample | 4 | 217 |
| **features flattened by subsampling** | **1** | **134** |

CIC's one is `callbacks.ngeneric`; 29,291 of 29,298 rows share one value and 7
hold another, so a 1000-row draw misses all 7. EMBER's 134 are the same shape —
`F701`, for instance, has a single outlier row in 14,325. EMBER's 2.783 comes
from `F1101`, whose SD moves from 0.0084 to 0.0316: real, but on a feature whose
absolute spread is negligible.

**This changes nothing for any model**, and the reason is specific rather than
hand-waving: `build_feature_pipeline` opens with
`VarianceThreshold(threshold=0.0)`, fit inside the train fold. Any feature that
is constant in the training data is dropped before the correlation filter, the
scaler, or PCA ever see it. The features the subsample flattens are precisely
the features the pipeline discards.

---

# Section B — Why CIC 15-class fails and EMBER 15-class does not

Week 4 established the contrast and closed with *"The actual cause is
unidentified."* Restated with sweep-aware numbers (latest clean sweep per cell,
grouped by `parent_run_id` — see `w5_day29_ember_four_model.md` §7):

| CIC multiclass | nc=1 | nc=3 | nc=6 |
|---|---|---|---|
| QSVM (best encoding) | 0.0559 | 0.0707 | 0.0794 |
| classical SVM | 0.1075 | 0.1309 | 0.1647 |
| random baseline (15 balanced classes) | 0.0667 | 0.0667 | 0.0667 |

QSVM straddles the random baseline at every `n_components`. Classical SVM is
1.6–2.5× random — poor, but unambiguously above chance. On EMBER 15-class the
same QSVM reaches 0.42–0.52 (`w5_day29_ember_four_model.md` §5).

## B.1 Family composition

| | CIC | EMBER |
|---|---|---|
| families in the subsample | 15 | 15 |
| smallest / largest family | 48 / 82 | 66 / 67 |
| imbalance ratio | 1.71× | 1.02× |

Week 4 already ruled out imbalance as the sole cause, and this confirms why the
question stayed open: 1.71× is mild, and QSVM's CIC collapse is total.

## B.2 Geometry of the projected features

Means over the 5 outer folds, pipeline fit on train-fold rows only — the same
leakage discipline the models use, so this is the geometry they actually saw.

| nc | dataset | mean Fisher | min Fisher | silhouette | mean centroid dist | cum. explained var |
|---|---|---|---|---|---|---|
| 1 | CIC | 0.0696 | 0.0003 | −0.2228 | 1.23 | 0.328 |
| 1 | EMBER | **0.5685** | **0.0159** | −0.2065 | 10.58 | 0.077 |
| 3 | CIC | 0.0572 | 0.0018 | −0.1848 | 1.52 | 0.572 |
| 3 | EMBER | **0.4525** | **0.0156** | −0.0595 | 16.95 | 0.174 |
| 6 | CIC | 0.0512 | 0.0035 | −0.1473 | 1.69 | 0.738 |
| 6 | EMBER | **0.3987** | **0.0351** | −0.0018 | 19.02 | 0.257 |
| 8 | CIC | 0.0491 | 0.0042 | −0.1454 | 1.72 | 0.807 |
| 8 | EMBER | **0.3995** | **0.0340** | **+0.0691** | 21.34 | 0.300 |

**Read the centroid-distance column with care — it is not comparable across
datasets.** `StandardScaler` gives every feature unit variance before PCA, so
total variance equals the feature count: 2,381 for EMBER against 55 for CIC.
EMBER's PCA coordinates therefore live on a much larger scale by construction,
and its 10–20× larger centroid distances are substantially a dimensionality
artifact. Fisher ratio and silhouette are scale-invariant (between- and
within-class scatter both scale quadratically), so they are the columns that
carry a real cross-dataset comparison. The table keeps centroid distance because
it is informative *within* a dataset — both grow with `nc`, confirming the
projections are not degenerate.

On the comparable metrics:

- **EMBER's families are ~8× more separable.** Mean Fisher ratio 0.57 vs 0.070
  at nc=1, 0.40 vs 0.051 at nc=6.
- **The worst-case family is where the gap is widest.** Min Fisher ratio: EMBER
  0.016–0.035, CIC 0.0003–0.0042 — up to 50×. CIC has families that are, in the
  projected space, very nearly not distinguished at all.
- **CIC's silhouette is negative at every `nc`.** The average CIC point sits
  closer to some other family's centroid than to its own. EMBER climbs through
  zero and turns positive by nc=8.
- **The variance columns invert the story.** CIC's PCA captures 33–81% of
  variance, EMBER's only 8–30%. EMBER separates families far better while
  explaining far less variance — meaning CIC's variance is concentrated in
  directions that do not discriminate families. A 55-feature memory-forensics
  vector has strong shared structure (process counts, handle counts) that varies
  with workload rather than with family.

## B.3 What this explains, and what it does not

It explains why *everything* does badly on CIC 15-class. Negative silhouette and
a near-zero minimum Fisher ratio are properties of the features, and both the
classical and the quantum pipeline consume the same projected features.

**It does not explain the QSVM-specific failure, and it cannot.** Classical SVM
extracts 0.108–0.165 from exactly these features while QSVM extracts 0.056–0.079.
Whatever separability CIC carries is demonstrably present — the classical kernel
finds it, and the fidelity kernel does not. Since the input geometry is
identical by construction, the difference has to lie in the kernel.

That is precisely the question **Day 33** takes up: whether the fidelity kernel
concentrates on CIC — off-diagonal Gram entries collapsing toward a constant, so
every pair of points looks equally similar and the SVM has nothing to separate —
and whether kernel-target alignment on CIC is materially lower than on EMBER.
`gram_offdiag_std` is already logged for every QSVM fold across both datasets, so
that comparison needs no new Gram builds.

---

## Artifacts

- `src/common/geometry.py` — `fisher_ratio`, `centroid_distances`,
  `class_proportion_drift`, `feature_drift`, `ks_reject_count`.
- `tests/test_geometry.py` — 9 tests, including two pinning the constant-feature
  behaviour that A.3 turns on.
- `scripts/compare_class_geometry.py` — the driver.
- `docs/reports/logs/w5_day31/geometry.json` — full numeric output.

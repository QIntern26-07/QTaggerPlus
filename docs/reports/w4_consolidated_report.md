# Week 4 Consolidated Report — QSVM Track (Days 22-28)

**Author:** Anh
**Date:** 2026-07-28
**Scope.** My own items in `temp/week-4-plan.md`, Team C / "Anh — QSVM Track" table only.
The Elizabeth/Ge table (generalized quantum-feature-extraction, hybrid-readout confidence
intervals, etc.) is not covered here and is not my scope.

**Format and honesty standard.** This follows `w3_jul-21_anh_qsvm_progress_audit.md`: every
number below is traceable to a committed report under `docs/reports/` or to
`results/mlflow_runs.csv`, nothing here is estimated, and where I fell short of the plan I
say so plainly rather than rounding up. This week's headline is largely negative — I have
tried not to soften it.

**Reproducing the numbers.** `scripts/export_mlflow_runs.py` was re-run before writing this
report; `results/mlflow_runs.csv` now holds 868 runs. `cic-malmem` (225 binary + 532
multiclass rows) and `ember-2018` (57 binary + 54 multiclass rows) are both present.
`sorel-20m` rows are **absent** — expected, since no SOREL sweep was run this week (see Day
25/26 below). `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest` was run before writing this
report: 116 passed (the system ROS2 `launch_testing` plugin breaks a bare `uv run pytest` on
this machine, unrelated to this repo).

---

## Status at a glance

| Day | Item (as written in `temp/week-4-plan.md`) | Status |
|---|---|---|
| 22 | EMBER sweeps (n≈1000, angle+iqp, nc=1/3/6, quantum then classical on shared splits) | **DONE** |
| 23 | Close the subsample-sizing decision | **DONE** |
| 24 | Extend the 15-class angle-vs-iqp sweep past nc=6; switch to held-out macro-F1 | **DONE** (reached nc=8) |
| 25 | SOREL-20M phase 1: binary (`is_malware`) subset + sweep | **NOT STARTED** |
| 26 | SOREL-20M phase 2: dominant-tag multiclass labels + sweep | **NOT STARTED** — labelling *design* is CLOSED |
| 27 | Buffer / begin drafting the Week 4 report | absorbed into execution (Days 22-24 ran to their planned scope; no overrun to buffer) |
| 28 | Consolidate Week 4 findings | this document |

Every Day 22-28 row that has a status of DONE has a corresponding artifact under
`docs/reports/`:

- Day 22 → `w4_jul-25_ember_binary.md`, `w4_jul-26_ember_multiclass.md`
- Day 23 → `w4_subsample_sizing_decision.md`
- Day 24 → `w4_jul-27_encoding_verdict.md`
- Day 25/26 → `w4_sorel_labelling_decision.md` (labelling design only — no sweep artifact
  exists because no sweep ran)
- Day 28 → this document

## SOREL, stated plainly

**Day 25 (SOREL binary sweep) and Day 26 (SOREL dominant-tag sweep) did not happen.** No
SOREL QSVM run exists in `results/mlflow_runs.csv`, no SOREL feature subset was built, and no
SOREL numbers appear anywhere in this report. Do not read "labelling closed" as partial
credit for the sweeps — it is not. The two are independent pieces of work and only one of
them is done.

**Why:** SOREL's feature store, `s3://sorel-20m/09-DEC-2020/processed-data/ember_features/data.mdb`,
is a single 71.6 GiB LMDB file. LMDB has no key-level remote-read API — the entire file must
be resident locally before any single key can be opened, so there is no way to stream a
stratified subset of feature vectors the way the EMBER subset builder streams columns from a
parquet row group. Downloading 71.6 GiB and sweeping it was out of scope for the time
available this week.

**What was done instead:** only `meta.db` (3.5 GiB — labels and the 11 behavior-tag columns,
no features) was fetched, via public unsigned S3 access, and used to settle the multiclass
labelling question: **dominant tag**, i.e. argmax over the 11 raw per-tag detector-vote
counts, ties broken by a fixed column order, all-zero-tag rows dropped. This was chosen over
binary-only (duplicates the existing binary task, wastes the point of a third dataset) and
iterative multi-label stratification (produces no single class per sample, breaks the
project's shared-metric comparison protocol). Full reasoning, schema verification, and the
19.7M-row label statistics (48.5% labelled, 62:1 largest:smallest class ratio in the raw
population, cleared at subsample scale) are in `w4_sorel_labelling_decision.md`; nothing here
adds to or supersedes those numbers.

`docs/quantum_todo.md`'s SOREL entry already reflects this split (labelling decided,
features/sweeps still open) as of the labelling-decision commit — verified below, not
duplicated.

---

## 1. EMBER sweeps (Day 22)

First EMBER 2018 results in the project, both tasks, both encodings run as **separate
single-encoding invocations** (not jointly tuned — see §2 for why), quantum first with
classical SVM replayed on the identical rows/folds via `--load-quantum-splits`. n=1000,
5-fold outer CV, nc ∈ {1, 3, 6}. Source: `w4_jul-25_ember_binary.md`,
`w4_jul-26_ember_multiclass.md`.

### Binary (malicious vs. benign, ~8:1 imbalanced)

| n_components | QSVM (angle) | QSVM (iqp) | SVM |
|---|---|---|---|
| 1 | 0.4532 | 0.4587 | **0.5314** |
| 3 | 0.4924 | 0.5084 | **0.6306** |
| 6 | 0.5138 | 0.5137 | **0.6724** |

SVM wins at every nc, by a widening margin (+0.078 at nc=1 → +0.159 at nc=6). At nc=1, QSVM
partially collapses to an always-malicious predictor: class-0 (benign) F1 is exactly 0.000 in
3 of 5 outer folds, both encodings, masked by a superficially reasonable raw accuracy
(0.73-0.74) — the reason this project reports macro-F1, not accuracy, as the headline metric.

**This is the finding that overturns last week's binary reading.** Week 3's CIC binary result
was QSVM at 0.990-0.992 macro-F1, statistically tied with all four classical models — "binary
is saturated and tied" was the audit's own conclusion. EMBER binary QSVM tops out at 0.5138
(nc=6) against SVM's 0.6724, a 0.159 gap, and never comes close to tying. Same estimator,
same fairness protocol, a different dataset — Week 3's "quantum is competitive on binary"
does not survive a second dataset. It was a property of CIC's binary task being
near-ceiling for everyone, not a property of the QSVM.

### 15-class family attribution (avclass families, exactly balanced: 955 rows/family)

| n_components | QSVM (angle) | QSVM (iqp) | SVM |
|---|---|---|---|
| 1 | 0.1688 | 0.1541 | **0.5294** |
| 3 | 0.3908 | **0.4797** | **0.7389** |
| 6 | 0.4232 | **0.5195** | **0.7930** |

SVM wins at every nc, and this is the largest quantum-classical gap measured anywhere in the
project to date: **0.27 at nc=6** (0.5195 vs 0.7930). Unlike CIC 15-class, QSVM does not
collapse: it recognizes 8/15 (iqp) to 9/15 (angle) families with nonzero F1 even at the worst
setting (nc=1), and 14-15/15 from nc=3 upward.

That non-collapse is the important comparison. CIC 15-class QSVM (Week 3 report) scored
0.0577-0.0708 across nc=1/3/6 — barely above the ~0.067 random baseline, consistent with
recognizing essentially one family and nothing else. EMBER's families are exactly balanced
(955/class); CIC's are ~1.7x imbalanced. If imbalance had been the cause of CIC's collapse,
removing it on EMBER should have fixed the problem outright. It didn't — EMBER 15-class QSVM
still loses to SVM by 0.27, its largest margin in the project. What it *did* fix is the
collapse pattern specifically: EMBER QSVM is a genuinely multi-family (if weak) classifier,
CIC QSVM was not. **Class imbalance is ruled out as the sole explanation for CIC's collapse.
The actual cause is unidentified — I have not determined what in CIC's memory-dump feature
geometry differs from EMBER's static-PE features to produce it, and I am not going to guess
at it here.**

---

## 2. The closed 15-class encoding verdict (Day 24)

Source: `w4_jul-27_encoding_verdict.md`. This closes the open item carried from the Week 3
audit (§2, "iqp-vs-angle on the 15-class task"), which had rested on **inner-CV win-counting**
— tallying which encoding an Optuna trial *picked* per fold when both were live options under
one joint tuning budget, not measuring how much better either one actually scored. That
methodology conflated selection with performance and is **retired for encoding comparisons on
this project as of this report.** Every number below (and every EMBER number above) instead
comes from a dedicated, single-encoding invocation reporting its own held-out macro-F1.

CIC 15-class, held-out macro-F1, mean ± std over 5 outer folds, extended for the first time
past nc=6 to **nc=8** (probe-gated: projected ~464s/Gram at nc=8 iqp, comfortably under the
15-minute stop-rule; nc=10 was optional and not attempted — not worth the runtime for a data
point outside this report's required table):

| nc | angle | iqp | delta (iqp - angle) | classical svm |
|---|---|---|---|---|
| 1 | 0.0550 ± 0.0095 | 0.0559 ± 0.0077 | +0.0009 | 0.1075 ± 0.0164 |
| 3 | 0.0585 ± 0.0057 | 0.0707 ± 0.0176 | +0.0122 | 0.1309 ± 0.0265 |
| 6 | 0.0681 ± 0.0147 | 0.0794 ± 0.0171 | +0.0113 | 0.1647 ± 0.0233 |
| 8 | 0.0693 ± 0.0062 | 0.0739 ± 0.0103 | +0.0046 | 0.1772 ± 0.0355 |

**Verdict: iqp beats angle at every nc tested on CIC 15-class, but every single margin is
smaller than either encoding's own fold-to-fold std** (deltas 0.0009-0.0122 against stds of
0.006-0.018). This does not sharpen or grow monotonically with qubit count — it peaks at
nc=3, not at nc=8. Contrast this with EMBER 15-class (§1), where iqp's edge over angle was
large and clean at every nc tested (+0.089 at nc=3, +0.096 at nc=6, both well outside either
encoding's own std). **The angle-vs-iqp difference is dataset-dependent, not a fixed property
of the encoding itself** — the same pair of circuits produces a decisive, trustworthy
separation on one dataset and a directionally-consistent-but-statistically-thin one on
another. nc=8 did not rescue CIC 15-class either: 0.0740 (iqp) sits barely above the ~0.067
random baseline, essentially the same place nc=6 (0.0794) and nc=1 (0.0559) sit. **More
qubits is not the missing ingredient for CIC 15-class.**

`--load-quantum-splits` determinism check (incidental, worth recording): the nc=1/3/6
classical SVM numbers re-run in this sweep reproduced the Jul-15 committed values **exactly,
to full float precision** (0.1075, 0.1309, 0.1647), three-plus weeks and an independent
invocation apart — the shared-split protocol this whole project's fairness claim depends on
is confirmed fully deterministic end to end.

---

## 3. SOREL binary and dominant-tag results (Days 25-26)

Covered in full above ("SOREL, stated plainly"). No results to report — restated here only
because the plan names this as its own line item and I don't want it silently missing from a
section a reader might scan for.

---

## 4. CTGAN run status

**Still blocked on Team B**, unchanged since the Week 3 audit. Verified for this report: no
`ctgan`, `augmented`, or `smote` file or directory exists anywhere under `data/` (checked by
name search). No augmented dataset has been handed over as of this report. My side remains
ready — the pipeline accepts any frame through `task_xy` + `build_feature_pipeline`, and the
n=200 → n=1000 ramp protocol is already established — but there is nothing to run against.

---

## 5. Memory constraint: what it forced

The development machine had roughly **4 GB free RAM** for most of this week's EMBER and SOREL
work. Two modules exist specifically because of that ceiling, and both would look like
unmotivated scope creep without this context:

- **`src/common/ember_subset.py` (column-batched EMBER subset builder).** The published EMBER
  test parquet is a single row group, 200,000 x 2,381 float32 — loading it directly risks
  ~1.9 GB resident (more with the pyarrow intermediate), an OOM risk on a ~4 GB-free machine
  on every load, quantum or classical. A single row group cannot be streamed row-wise, so
  this streams **column-wise** instead: read the three label columns cheaply, decide which
  rows are needed, then read feature columns `col_batch` (default 200) at a time and keep
  only the needed rows. Peak memory scales as `n_rows * col_batch * 4 bytes` rather than the
  full matrix — the recorded peak for this approach was **1.36 GB**, against ~1.9 GB for the
  naive full-frame load. `scripts/make_ember_subset.py` runs this once, offline, producing
  the 50 MB `data/ember/ember2018_quantum_subset.parquet` that every EMBER sweep in this
  report actually reads.
- **Chunked read of SOREL's `meta.db` (19,724,997 rows).** A full-table `pd.read_sql_query`
  read of all 13 columns — including a 64-character hex `sha256` string per row as a Python
  object — was estimated at roughly 3.2-3.3 GiB (sha256 column alone ~1.3-1.4 GiB, ~12 int64
  columns ~1.9 GiB, before `read_sql_query`'s own row-buffer overhead), against a machine
  reporting only ~1.2 GiB free at the time (`free -h`) and one OOM-risk incident already that
  week — judged unsafe. `scripts/sorel_label_stats.py::chunked_label_stats` instead reads via
  `pd.read_sql_query(..., chunksize=1_000_000)`, calling the same tested `label_stats`
  function per 1M-row chunk and merging the accumulators (`total_rows`, `dropped_all_zero`,
  `labelled_rows`, `tied_rows`, `class_counts`). The full 19.7M-row statistic set in
  `w4_sorel_labelling_decision.md` was produced this way, in 63 seconds across 20 chunks, with
  no observed swap growth. The chunking is purely an accumulation strategy layered on top of
  the already-unit-tested `label_stats` — the tested function's interface (one DataFrame in)
  is unchanged.

---

## 6. Anomaly carried forward: orphaned MLflow fold rows

`results/mlflow_runs.csv` holds **two orphaned child (fold-level) run rows** for
`(dataset=ember-2018, task=binary, n_components=1, encoding=angle)` — 7 child rows logged
where the raw stdout log shows exactly 5 "outer fold" iterations. Root cause (confirmed, not
a code bug): the nc=1 angle sweep's first invocation was interrupted mid-run and relaunched;
the two extra rows are orphaned folds from that first, interrupted launch. In the CSV they sit
at UTC `03:33:13` (f1_macro=0.470899) and `03:33:59` (f1_macro=0.399101), and both **predate
the parent run's own `start_time` of 03:34:36**. Every other sweep in this week's work (18 of
19 quantum invocations) has exactly the expected child-row count with no such orphans.

The parent aggregate metrics used throughout §1 (`f1_macro_mean`, `_std`, etc.) match a
hand-computed mean of the 5 real "fold done" lines in the raw stdout log exactly, to 6 decimal
places — **the reported aggregates are unaffected.** For anyone re-deriving fold-level
statistics directly from the CSV rather than trusting the parent aggregates: **the filtering
rule is to drop any child row whose `start_time` precedes its own parent run's `start_time`**
— that discards exactly the two orphaned rows from the interrupted-and-relaunched sweep and
nothing else.

---

## 7. `docs/quantum_todo.md` consistency check

Verified against the current file rather than assumed:

- **EMBER experiments** — was listed under `## Open` ("EMBER experiments not yet run") going
  into this report; this is now stale (§1 above closes it) and has been moved to
  `## Decided` as part of this report's commit.
- **Subsample sizing** — already under `## Decided` (closed 2026-07-25, matches
  `w4_subsample_sizing_decision.md`'s n=900 recommendation). No change needed.
- **SOREL-20M** — already rewritten (by an earlier task this week) to read "labelling decided,
  features not fetched," correctly narrowed to feature acquisition + sweeps as the remaining
  open scope, and correctly citing `w4_sorel_labelling_decision.md`. Verified accurate against
  §"SOREL, stated plainly" above; not duplicated.
- **VQC** — remains under `## Open`, correctly marked "not implemented and not planned to be
  implemented by this contributor." Left unchanged — out of scope for the QSVM track.

---

## 8. Methodology constraints (must be read before comparing across weeks)

1. **Classical comparison is SVM only this week, not the four-model batch** (RF/XGBoost/
   LightGBM) used in earlier weeks' CIC reports. RF/XGBoost/LightGBM are another team member's
   scope this week. Every "classical" claim in this report — every SVM number in §1 and §2 —
   means SVM specifically. A reader comparing against Week 3's four-model CIC numbers (e.g.
   Random Forest as the classical winner on CIC 15-class at nc=1/3) must not read this
   report's SVM-only numbers as directly comparable to that four-model set; they are a
   narrower baseline.
2. **~4 GB free RAM shaped two modules** — see §5. This is recorded so the column-batched
   EMBER builder and the chunked SOREL reader don't read as unexplained scope additions to
   someone reviewing the diff.

---

## 9. Summary

Everything that depended only on me and was scoped for this week is done except SOREL's
sweeps, which needed a 71.6 GiB file this week's time budget did not allow for. The EMBER
sweeps overturn last week's most quotable finding: **QSVM does not tie classical on binary
classification in general — it tied on CIC specifically, because CIC's binary task is
near-ceiling for every model tried. On a second dataset (EMBER), classical SVM beats QSVM
clearly on both binary (+0.159 at nc=6) and 15-class (+0.27 at nc=6, the project's largest
gap to date).** The positive-sounding half of this week's work is narrower than it looks:
EMBER's exact class balance rules out imbalance as the explanation for CIC 15-class's
near-total collapse, but it does not identify the real cause, and it does not close the
classical-quantum gap — it only changes collapse into a wide, stable loss. The encoding
question is closed with a weaker verdict than Week 3 implied: iqp's CIC 15-class edge is
real in direction but statistically thin at every qubit count tried, up to and including the
newly-reached nc=8, in contrast to a large, clean iqp advantage on EMBER 15-class. SOREL's
labelling question is settled; its features are not fetched and no sweep has run. CTGAN
remains blocked on Team B with nothing new to report.

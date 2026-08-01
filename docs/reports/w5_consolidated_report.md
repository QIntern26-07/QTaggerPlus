# Week 5 consolidated report — QSVM track

**Author:** Anh (Team C, quantum / QSVM)
**Date:** 2026-08-01
**Scope:** Days 29–35, plus every still-open item from Weeks 1–4 that falls in
the QSVM track.

## Format and honesty standard

Every number in this report traces to a committed per-day report or to
`results/mlflow_runs.csv`. Where something was not done, it says so and gives the
reason. Where a result is weaker than the headline, the weaker reading is stated
in the same paragraph rather than in a footnote. Two claims made earlier in the
week were wrong and are retracted in place — §7 lists them.

**Reproduction state at close:**

- Test suite: **143 passed** (`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest`).
  The plugin-autoload flag is required because a system-wide ROS 2
  `launch_testing` plugin breaks bare `pytest`.
- `results/mlflow_runs.csv`: **976 runs × 73 columns**.
- Datasets present: CIC-MalMem-2022, EMBER 2018. Absent: SOREL-20M (§4).

## 1. Status at a glance

| Day | Item | Status | Artifact |
|---|---|---|---|
| 29 | EMBER four-model classical baseline | Done | `w5_day29_ember_four_model.md` |
| 30 | SOREL-20M close-out | Done | `w5_day30_sorel_closeout.md` |
| 31 | CIC vs EMBER separability + subsample fidelity | Done | `w5_day31_cic_vs_ember_separability.md` |
| 32 | Quantum feature extraction / C0-V0-H2 | Not this contributor's scope | — |
| 33 | Kernel concentration and target alignment | Done | `w5_day33_kernel_diagnostics.md` |
| 34 | Master tables + significance tests | Done | `w5_day34_master_tables.md` |
| 35 | Consolidated report + paper sections | Done | this file, `docs/paper/qsvm_sections_draft.md` |

## 2. Day 29 — EMBER four-model classical baseline

Week 4 ran only `svm` on EMBER's classical side, so every EMBER
quantum-classical margin on record was a QSVM-vs-SVM margin. Adding
`random_forest`, `xgboost` and `lightgbm` at `nc ∈ {1,3,6}` for both tasks
brings EMBER to CIC's coverage. All 24 (task × model × nc) cells now hold
exactly 5 folds.

**`random_forest` beats `svm` on binary at every `nc`**, so the Week 4 binary
margins were understated by 0.011–0.027 macro-F1. On multiclass `svm` genuinely
is the best baseline at `nc ∈ {3,6}`, so those margins stand. Four of six cells
move, all in the same direction: **Week 4 flattered QSVM on EMBER.**

Cost: 32 minutes total across six foreground invocations, no cell over 10
minutes, no scope reduction needed.

## 3. Day 30 — SOREL-20M closed

Promoted from *deferred* to *closed*. The labelling design is delivered
(`w4_sorel_labelling_decision.md`); the feature store was never fetched and **no
SOREL sweep was ever run** — zero `sorel-20m` rows exist in
`results/mlflow_runs.csv`. The two are independent pieces of work and only one
was done; the labelling report is not partial credit for the sweeps.

Measured at close-out: 227 GiB disk free against 71.6 GiB needed, so disk is not
the blocker. RAM is not either — the store is a memory-mapped LMDB whose reads
page through the OS cache. The blocker is the fixed 71.6 GiB download,
unavoidable because LMDB offers no key-level remote access, so there is no way to
fetch only the 1000 rows a sweep would need. Resume conditions and the source URI
are in the close-out; nothing needs re-investigating.

## 4. Days 31 and 33 — the CIC 15-class collapse

`w4_consolidated_report.md` closed with *"The actual cause is unidentified."*
Two days of measurement narrow it to two distinct effects.

**Day 31 — feature geometry.** EMBER's families are ~8× more separable than
CIC's by Fisher ratio and up to 50× on the worst-case family. CIC's silhouette is
negative at every `nc`: the average CIC point sits closer to another family's
centroid than to its own. Counter-intuitively CIC's PCA explains far *more*
variance (33–81%) than EMBER's (8–30%), so CIC's variance lives in directions
that do not discriminate families — plausible for a 55-feature memory-forensics
vector whose process and handle counts track workload rather than family.

This explains why *everything* does badly on CIC 15-class. It cannot explain the
QSVM-specific failure, because classical SVM extracts 0.108–0.165 macro-F1 from
exactly the same projected features where QSVM extracts 0.056–0.079.

**Day 33 — kernel geometry.** With an RBF control built on the identical matrix:

- **Alignment separates the datasets, not the kernels.** Kernel-target alignment
  must be read against its floor — a constant kernel scores 1/√15 ≈ 0.258 on 15
  balanced classes. On CIC the fidelity kernel lands **0.000069 below** that
  baseline, indistinguishable from an all-ones matrix, and RBF is below it too.
  On EMBER both clear it by ~0.03.
- **Concentration separates the kernels, specifically on CIC.** RBF produces
  near-identical spread on both datasets (0.2212 vs 0.2184), so CIC's features do
  not make a classical kernel concentrate. The fidelity kernel tracks RBF on
  EMBER (0.92×) but collapses to **0.40×** on CIC — every off-diagonal similarity
  squeezed into a band 2.5× narrower from the same inputs. The Gram approaches a
  scaled identity, which is what a macro-F1 pinned to the random baseline looks
  like.

The QSVM-specific mechanism is therefore **kernel concentration**, and
`bandwidth` is the named lever. That lever has not been tuned on CIC beyond its
default, so this is a hypothesis with a mechanism, not a measured fix.

**The tension this leaves, stated rather than hidden:** alignment says neither
kernel's Gram carries label information on CIC, yet SVM reaches 0.16 there.
Alignment is a global unweighted average over all pairs and an SVM needs no such
thing — it needs support vectors, `C` and class weights, none of which alignment
models. Alignment is useful for ranking dataset difficulty and unreliable for
predicting achievable SVM performance. The concentration result does not have
this weakness.

## 5. Day 34 — significance tests

Week 1 Day 6 assigned significance tests. The infrastructure was built and unit
tested in Week 1 and **had never been run** — four weeks of "QSVM loses to
classical" rested on mean differences alone.

**All 84 testable quantum-vs-classical pairs favour classical, and all 84 reach
p < 0.05 on the paired t-test.** Worst p across every pair: 0.0241.

**Wilcoxon returned exactly 0.0625 on all 84, and that is not a null result.**
With 5 paired samples, 0.0625 is the *smallest attainable* two-sided p-value, so
the test cannot reach 0.05 at this fold count regardless of effect size. All 84
hitting the floor means every pair achieved the most extreme rank configuration
available. Reporting "Wilcoxon found nothing significant" would be a serious
misreading. Six folds would put 0.03125 in reach.

**What could not be tested.** All CIC binary comparisons — 9 cells × 4 models.
Every CIC binary run, classical *and* quantum, dates from 2026-07-09…07-12,
before nested MLflow logging existed, and carries no `parent_run_id` or
`tags.sweep`. The raw rows show QSVM at 0.92–1.00 macro-F1, consistent with Week
3's "quantum ties classical on binary", but the 45 rows per model resolve into
two distinct sweeps per `nc` with no identifier separating them. Grouping by
time-gap clustering would invent a boundary the data does not record; no attempt
was made. The limitation is symmetric across frameworks, not a quantum-specific
gap.

McNemar ran classical-vs-classical only, since QSVM predictions were never
persisted. On EMBER 15-class `random_forest` beats `xgboost` at p = 0.0022; on
CIC 15-class **no pair separates at all**, consistent with Day 31.

## 6. What did not happen, and why

| Item | Reason |
|---|---|
| Day 32 quantum feature extraction / C0-V0-H2 | Elizabeth / Ge scope, not Team C's |
| VQC | Not implemented and not planned by this contributor. Week 2 Day 2's "extend the existing binary QSVM/VQC architecture" is QSVM-only unless someone else picks it up |
| CTGAN-augmented quantum run | Blocked on Team B. Ready on this side — the CLI takes any parquet |
| Evaluation-protocol finalization | Blocked on Team A. Skeleton delivered in Week 1 |
| SOREL-20M sweeps | Closed, not deferred — §3 |

None of these is a QSVM-track failure; each is either another team's scope or
another team's dependency. They are listed so a reader does not have to infer
their absence.

## 7. Two claims made this week that were wrong

Recorded because the correction changes what a reader should believe, and
because a consolidation report that only lists successes is not a record.

1. **"The Day 29 crash was caused by a missing `--csv` flag, and the completed
   run scored the wrong rows."** Wrong on both counts. The corrected re-run
   reproduced the crashed run's `f1_macro_mean = 0.558183651105556` to every
   recorded digit across all five folds, which is only possible if both runs read
   the same rows. The flag had been passed. The actual cause was `--n-jobs` left
   at its default `-1`, spawning 20 loky workers on a 20-core machine; the
   machine died when a second such pool started. Capped at `--n-jobs 4`: peak RSS
   879 MB, zero swaps.
2. **"Week 4's stated reason for not fetching SOREL had become invalid."**
   Wrong. `w4_consolidated_report.md` already gave the reason as *"out of scope
   for the time available this week"* — a time argument, which remains correct.
   Day 30 promotes *deferred* to *closed*; it does not correct a false statement.

## 8. Backlog reconciliation — whole project

**This table was built by auditing the original plan documents (`temp/qi26_7.md`
and the Weeks 1–5 execution plans), not the self-authored progress reports.** The
distinction matters: a progress report can only list gaps its author already
noticed. Four of the items below — significance tests, subsample representativeness
validation, weighted F1, and basis encoding — appear in **no** prior progress
report at all. That is why this list is trustworthy and why it is longer than
expected.

| Item | Origin | Disposition |
|---|---|---|
| EMBER four-model classical baseline | W5 D29 | **Closed this week** |
| SOREL-20M features + sweeps | W2 D2 / W4 D25-26 | **Closed as *not done***, with reasons and resume conditions |
| Subsample sizing decision | W2 D2 | Closed in Week 4 (`w4_subsample_sizing_decision.md`) |
| iqp-vs-angle 15-class verdict | W3 D1-3 | Closed in Week 4 (`w4_jul-27_encoding_verdict.md`) |
| CIC 15-class collapse cause | W4 open question | **Addressed this week** (Days 31, 33) — mechanism named, fix not yet measured |
| Statistical significance tests | W1 D6 | **Never run before; closed this week** |
| Subsample representativeness validation | W2 D2 | **Never run before; closed this week** |
| Weighted F1 | W3 D6 | **Never implemented; added this week.** No sweep was re-run, so no existing result carries it |
| Basis encoding | W1 D2 | **Closed as a deliberate omission** — it maps discrete values to computational basis states, but the pipeline's inputs are continuous PCA components, so it would need a quantisation step that discards exactly what PCA was used to retain |
| UMAP as a second reduction | W2 D2 | Closed as a deliberate deviation — PCA only, because PCA is the classical/quantum alignment point. Rationale in `w3_jul-21_anh_qsvm_progress_audit.md`; restated, not re-argued |
| VQC | W2 D2 | Not implemented, not planned by this contributor |
| CTGAN-augmented quantum run | W3 D4 | Blocked on Team B, ready on this side |
| Evaluation-protocol finalization | W2 D3 | Blocked on Team A, skeleton delivered |
| Quantum-feature-extraction / C0-V0-H2 | W5 D32 | Elizabeth / Ge scope |
| Research-paper draft (QSVM sections) | `qi26_7.md` W5 | **Drafted this week** (`docs/paper/qsvm_sections_draft.md`) |

## 9. Engineering defects found and fixed this week

Three were live bugs, found while doing the analysis rather than by looking for
them:

1. **`split_paths('cic', 'binary')` pointed at a file that did not exist.** The
   repo carried the CIC binary quantum sample index under its pre-per-task name
   `quantum_sample_idx.json`; the multiclass counterpart had been renamed and the
   binary one was missed. `classical --dataset cic --tasks binary
   --load-quantum-splits` died with `FileNotFoundError` — the shared-split
   contract broken on its most common path. Renamed, content verified (n = 1000,
   matching `cic_binary_quantum_folds.json`'s 0..999 coverage).
2. **`scripts/export_mlflow_runs.py` dropped `run_id` and every `tags.mlflow.*`
   column**, including `mlflow.parentRunId` — precisely the columns needed to
   group folds into sweeps. Fixed to emit `run_id`, `parent_run_id` and `status`.
3. **`common.geometry.feature_drift` reported perfectly preserved features as
   maximal drift.** Zero-variance features were given a denominator of 1.0, so a
   feature constant in the population (and therefore in any subsample of it)
   scored a deviation of exactly 1.0. Now computed over non-constant features
   only, with the constant ones counted separately.

Two known traps were worked around rather than fixed, deliberately:

- `src/classical/__main__.py:119` names prediction files
  `{model}_{task}_predictions.npz` with **no `n_components`**, so each new `nc`
  overwrites the last. Day 29 worked around it with per-`nc` `--predictions-dir`.
  Fixing the template invalidates how the existing legacy files should be read,
  so it belongs with a re-run, not with this week's analysis.
- `src/classical/__main__.py:85`/`:124` rewrites the metrics CSV wholesale on
  every invocation. Worked around by giving every cell its own `--out`.

## 10. Methodology constraints a cross-week reader must know

- **Every quantum result is computed on a 1000-row subsample**, forced by the
  kernel's O(n²) cost. Day 31 verified this is faithful — class proportions,
  per-feature means and per-feature distributions all preserved, KS rejections
  far below chance on both datasets. This had been assumed for four weeks.
- **PCA `n_components` is the single alignment point.** It fixes the classical
  feature count and the quantum qubit budget simultaneously, and is fit inside
  the training fold only.
- **Quantum runs first, classical replays** via `--load-quantum-splits`. A
  full-dataset classical run must never be compared against a subsampled quantum
  run.
- **EMBER's reference frame is the 18,014-row quantum subset**, not the
  200,000-row test parquet. `ember_family_xy` re-downsamples families at load
  time, so feeding a different input frame produces a different pool and the
  persisted sample indices select the wrong rows. The full parquet is also a
  single unstreamable row group at ~1.9 GB resident.
- **Macro-F1, not accuracy.** The worked example is EMBER binary at `nc = 1`:
  QSVM reaches **0.890 accuracy in 3 of 5 folds while the benign class's F1 is
  exactly 0.000** in those same folds. The model labels everything malware and is
  right 89% of the time because 89% of rows are malware. Accuracy is not merely
  optimistic here — it is actively misleading.
- **Simulator only.** All results are PennyLane `lightning.qubit`; no hardware,
  no noise model.

## 11. Summary

The QSVM track closes Week 5 with its own backlog empty. Of the six Week 5 days
in scope, all six are delivered. Of the nine carried items the original plans
assigned, four were closed this week, four were already closed in Week 4, and one
(the CIC collapse) is now explained at the mechanism level with a named,
untested lever. Four further items belong to other teams or other contributors
and are listed rather than silently omitted.

The project's central claim is stronger than it was on Monday and points the same
way. **Classical beats quantum on every testable configuration, and now with
p < 0.05 on all 84 of them rather than on mean differences alone.** The Week 4
EMBER margins turned out to be understated, not overstated. The one place the
quantum pipeline was ever competitive — CIC binary — cannot be significance-tested
from what was logged, and this report says so rather than quietly dropping it.

What is genuinely new is the *why*: the fidelity kernel does not fail on CIC
because CIC's features are hard, since a classical RBF kernel on the identical
features does not concentrate. It fails because the fidelity kernel itself
collapses to 0.40× the classical kernel's spread on that data. That is a
statement about the method, not about the dataset, and it is the most useful
thing this track has produced for the paper.

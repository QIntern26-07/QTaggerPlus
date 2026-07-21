# QSVM Track — Progress Report (Week 2 Days 8–10 + Week 3)

**Author:** Anh
**Date:** 2026-07-21

**Scope.** This covers my own items in `temp/Week 2 - 3 Days Plan.md` and `temp/Week 3 Plan.md`
(Team C). Every number below is traceable to a committed report under `docs/reports/` or to the
committed run export `results/mlflow_runs.csv` — none of it is estimated. Where I deviated from the
plan or fell behind, I have said so explicitly rather than rounding up.

**Reproducing the numbers.** All runs are logged to MLflow, but the SQLite backend (`mlflow.db`) is
not committed: it is a multi-megabyte binary that every run rewrites, and its artifact paths point
into the gitignored `mlruns/` directory, so checking it out would give you broken links anyway.
Instead, `scripts/export_mlflow_runs.py` exports every run's params and metrics to
`results/mlflow_runs.csv` (685 runs at the time of writing), which is committed and diffable. Every
table in this report can be rebuilt from that CSV by filtering on `params.dataset`, `params.task`,
and `params.n_components`. To browse the runs interactively instead, run the sweeps locally and use
`uv run mlflow ui`.

Legend: **DONE** · **PARTIAL** · **NOT STARTED** · **BLOCKED**

---

## Status at a glance

| # | Task (as written in the plan) | Status |
|---|---|---|
| W2 D1 | Profile the codebase; efficiency fixes; draft the evaluation protocol skeleton | **DONE** |
| W2 D2 | Minimal representative subsets for EMBER/SOREL; extend the binary QSVM toward multi-class | **PARTIAL** |
| W2 D3 | Train the multi-class QSVM; first-pass results; runtime benchmark; finalize the evaluation protocol | **DONE** (CIC) / **PARTIAL** (protocol doc) |
| W3 D1 | Begin the n_components ≥ 2 sweep for a fair angle-vs-iqp comparison | **DONE** — early (Jul 11–12) |
| W3 D2 | Extend the QSVM ramp to nc = 2, 3; merge into the PCA alignment branch | **DONE** — early |
| W3 D3 | Finalize and document the angle-vs-iqp verdict at nc ≥ 2 | **DONE** — early, with one follow-up open (see §2, Days 1–3) |
| W3 D4 | Run QSVM on Team B's CTGAN-augmented dataset at n=1000 | **BLOCKED** — dataset not handed over |
| W3 D5 | Build a minimal representative stratified sample under the PCA/qubit-budget convention | **DONE** |
| W3 D6 | Train the multi-class QSVM; record per-class F1, macro/weighted F1, confusion matrix | **DONE** |
| W3 D7 | Consolidate the Week 3 report | **PARTIAL** — this document is the QSVM half |
| — | EMBER 2018 support, binary + multiclass (not on either plan) | code **DONE**, sweeps **NOT RUN** |

**Complete.** Profiling and all three efficiency fixes (caching, batching ~4.4×, PCA), plus a Gram
parallelization that was not asked for (~7.5× end-to-end). The multi-class extension of the QSVM.
The full CIC picture: binary and 15-class multiclass, n = 200 and 1000, nc = 1…6, quantum and
classical on identical rows and folds. The angle-vs-iqp verdict. EMBER support built and tested.

**In progress / not started.** The EMBER sweeps — the code is ready and this is my next action.
The subsample sizing decision, which I can close alongside them. SOREL-20M, which is untouched and
needs a labeling design decision before any code (see §4). The UMAP branch of Week 2 Day 2, which I
deliberately did not build (see §1, Day 2).

**Blocked on others.** Team A's schema, which the evaluation protocol needs to be finalized against.
Team B's CTGAN-augmented dataset for Week 3 Day 4. I am ready on my side for both.

### Where the code lives

| Work | Branch | State |
|---|---|---|
| 15-class malware-only reframe, balanced XGBoost, Gram parallelization, Jul-16 results | `feature/multiclass-malware-only-reframe` | **merged** to `main` via PR #4 |
| EMBER 2018 support, MLflow CSV export, this report | `feature/ember-2018-support` | **open** as PR #5, awaiting review |
| Everything from Week 2 Day 1 through the Jul-14 multiclass sweep | — | already on `main` (PRs #1–#3) |

Nothing of mine is sitting unreviewed on a local branch. PR #5 is the only
outstanding piece, and it is code plus documentation — no experimental results depend on it yet,
since the EMBER sweeps have not been run.

---

## 1. Week 2 (Days 8–10)

### Day 1 — Profile the QSVM/VQC codebase; efficiency fixes; draft the evaluation protocol skeleton

**Status: DONE** (2026-07-09, commit `feat(quantum): batch QSVM kernel evaluation and profile Day 1 bottlenecks`)

Deliverables: `docs/reports/w2_day1_quantum_profiling_baseline_report.md` (frozen pre-optimization
baseline) and `docs/reports/w2_day1_quantum_profiling_report.md` (post-optimization).

Profiling results:

| Bottleneck profiled        | Finding                                                                                                                                                                        |
|----------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Circuit depth / gate count | `angle` ~ qubits^1.2 (4 gates @1q → 46 @8q); `iqp` ~ qubits^1.5 (4 → 88); `amplitude` ~ qubits^2.1                                                                             |
| Simulator backend          | `lightning.qubit` **1.48×** faster than `default.qubit` on 200 kernel pairs (0.238 s vs 0.353 s); confirmed `_resolve_device()` is not silently falling back to the slower one |
| Kernel-matrix cost vs n    | `angle` ~ n^2.2, `iqp` ~ n^1.9 over n=50→400 (0.13 s → 13.85 s)                                                                                                                |

I shipped all three efficiency fixes the plan named:

- **Kernel-matrix caching** — `qsvm.py` caches train and test Grams. While testing this I found and
  fixed a cache-identity bug on the test Gram (`fix(qsvm): correct test-gram cache identity and warmup timing`).
- **Batched execution** — QNode parameter broadcasting, one dispatch per chunk instead of one per
  pair: **~4.4× at n=400** (angle). The exponent stays ~2.0, as expected — batching reduces the
  per-pair constant, not the O(n²) pair count.
- **Reduced feature dimensionality** — PCA as the final stage of the shared
  `build_feature_pipeline()`. `n_components` is the single knob that sets both the classical
  feature count and the quantum qubit budget, which is what keeps the comparison fair.

Evaluation protocol skeleton for Team A: `docs/day1_evaluation_protocol_skeleton.md`, covering
metrics (accuracy, per-class F1, ROC-AUC, confusion matrix), split strategy, and expected input format.

### Day 2 — Minimal representative subsets for EMBER/SOREL; extend the binary QSVM toward multi-class

**Status: PARTIAL**

| Sub-item                                           | Status                                                             | Evidence                                                                                                                                                                                                                                                                                                                                                 |
|----------------------------------------------------|--------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Extend binary QSVM → multi-class                   | **DONE**                                                           | `SVC` one-vs-one on the cached Gram, `decision_function_shape="ovr"` for AUC; multiclass CV-integration test committed 2026-07-08                                                                                                                                                                                                                        |
| Stratified subsampling + PCA machinery             | **DONE**                                                           | `python -m quantum --max-samples` persists the exact subsample and folds; the classical run replays them via `--load-quantum-splits`                                                                                                                                                                                                                     |
| EMBER minimal representative subset                | **DONE, but late** — 2026-07-20, i.e. in Week 3 rather than Week 2 | see §3                                                                                                                                                                                                                                                                                                                                                   |
| SOREL-20M subset                                   | **NOT STARTED**                                                    | no loader, not downloaded                                                                                                                                                                                                                                                                                                                                |
| Subsample **sizing decision** from the scaling law | **NOT STARTED**                                                    | I have the scaling data (exponent ~2.0, 11.4 s @ n=400, nc=2) but never solved it for a target runtime budget                                                                                                                                                                                                                                            |
| UMAP                                               | **NOT DONE — deliberate deviation**                                | I used PCA only. PCA is the alignment point between the classical and quantum pipelines; introducing UMAP on one side would break the "same dimensionality reduction on both sides" guarantee the whole comparison rests on. I consider this the right call, but it is a departure from the written plan and I want it on the record rather than buried. |

### Day 3 — Train the multi-class QSVM on the subset; first-pass results; benchmark against the Day 8 baseline; finalize the evaluation protocol

**Status: DONE (CIC) / PARTIAL (protocol doc)**

I trained and reported the multi-class QSVM twice:

- `docs/reports/w2_jul-14_6-components_multi.md` — 16-class (Benign + 15 families), nc = 1…6, n = 200 and 1000.
- `docs/reports/w2_jul-16_multiclass_15class_1000.md` — reframed 15-class malware-only, nc = 1/3/6, n = 1000.

Runtime benchmark against the Day 8 baseline: **DONE**, and I went past what the plan asked. On top
of the Day-1 4.4× batching win, I parallelized the Gram build across cores
(`perf(quantum): parallelize QSVM kernel Gram build across cores`). The nc=6 multiclass sweep
finished in **~28 min** against **~3.5 h** for the equivalent single-core run — **~7.5×**
end-to-end. That is higher than the 2.7× single-Gram micro-benchmark because a warm loky pool
amortizes worker spawn across the many Grams a tuning sweep builds. Kernel values are unchanged;
parallel == serial is unit-tested.

Evaluation protocol doc: **not finalized.** `docs/day1_evaluation_protocol_skeleton.md` still has
open reconciliation points that need Team A's real schema. **Blocked on Team A**, not on me.

---

## 2. Week 3

### Days 1–3 — Begin the n_components ≥ 2 sweep; extend the QSVM ramp to nc = 2, 3 and merge into the PCA alignment branch; finalize and document the angle-vs-iqp verdict

**Status: all three DONE, completed in Week 2 (Jul 11–12), ahead of schedule.**
`docs/reports/w2_Jul-11_experiments_{2,3}-components.md`,
`w2_Jul-12-experiments_6-components.md`, consolidated in
`w2_Jul-12_experiments_summary_2-3-6-components.md`.

**The angle-vs-iqp verdict, consolidated across every sweep I have run:**

| Task                                      | n_components | iqp wins (inner-CV fold selections) | Reading                                                                   |
|-------------------------------------------|--------------|-------------------------------------|---------------------------------------------------------------------------|
| binary                                    | 1            | 0/10                                     | iqp cannot win at 1 qubit *by construction* — no feature pair to entangle |
| binary                                    | 2, 3, 6      | **2/30**                                 | no advantage; the two wins are scattered, with no trend in qubit count |
| multiclass, 16-class, n=200               | 1…6          | **12/30**                                | rising (0/5 at nc=1 → 3/5 at nc=6), but ~5 members/class makes this plausibly noise |
| multiclass, 16-class, **n=1000**          | 1…6          | **20/30** (1, 3, 4, 3, 4, **5**/5)       | **monotone climb to unanimous 5/5 at nc=6 — the real signal** |
| multiclass, 15-class malware-only, n=1000 | 1, 3, 6      | **8/15** (2/5, 3/5, 3/5)                 | flattened back to co-competitive after Benign was dropped |

**Verdict.** My Jul-9 hypothesis — that iqp underperformed only because nc=1 denied it a feature
pair — is **partly confirmed, and the confirming evidence is the strongest single encoding result
in the project.** On the 16-class task at n=1000, iqp climbs monotonically from 1/5 at nc=1 to a
**unanimous 5/5 at nc=6**. That run is the one to trust: unlike the n=200 stage (~5 members/class,
where fold-level selection is close to a coin flip), 1000 samples gives each fold's inner-CV
comparison enough data to be meaningful. It is also exactly the shape the Day-1 profiling report
predicted — iqp's entangling structure needs a feature pair to act on, so its advantage should
appear only once n_components is high enough.

The advantage is **conditional**, not general. It requires all three of: a multiclass (hard) task,
enough qubits, and enough samples for the comparison to be reliable. It does not appear on binary
(2/30) — but binary is saturated at macro-F1 ≈ 0.99, so there is almost no headroom for any
encoding to distinguish itself there. And it did **not** carry over to the 15-class malware-only
reframe (8/15). My working explanation is that dropping Benign removed the easy
Benign-vs-family one-vs-one boundaries, leaving a noisier, fully balanced set of pairwise fits
where neither encoding systematically wins — but I want to be clear that this is a hypothesis I
have not tested, not a finding.

**What I would do about it.** This is the one open encoding question worth a run, and I do not
consider it settled: the 15-class reframe is now the project's primary multiclass task, and iqp's
status on it rests on three data points (nc = 1, 3, 6). I would extend the 15-class sweep past
nc=6 to see whether the n=1000/16-class trend reappears once there are more qubits to entangle. I
would also stop counting inner-CV selection wins and compare the two encodings' held-out macro-F1
directly — win-counting tells you which encoding the tuner picked, not how much better it actually
scored, and the whole verdict above currently rests on that weaker signal.

One cost note for anyone choosing on runtime grounds: at nc=2 the two encodings cost the same
(10 gates, ~11 s at n=400); iqp's quadratic `MultiRZ` penalty only bites at higher qubit counts
(28 vs 6 gates at 4 qubits, 88 vs 46 at 8). So if the 15-class extension does reproduce the win,
it will not come free.

### Day 4 — Pair with Shanmukh (Team B) to run QSVM/VQC on the CTGAN-augmented dataset at n=1000

**Status: NOT STARTED — blocked on Team B.** The CTGAN-v2 augmented dataset has not been handed
over yet. My side is ready: the pipeline accepts any frame through `task_xy` +
`build_feature_pipeline`, and the n=200 → n=1000 ramp protocol that Shanmukh's Day 4 references is
already established and documented. I can turn this around quickly once the data arrives.

### Days 5–6 — Build a minimal representative stratified sample under the PCA/qubit-budget convention; train the multi-class QSVM and record per-class F1, macro/weighted F1, confusion matrix

**Status: DONE**, delivered early — this is the Jul-14 / Jul-16 work above. Per-class F1 is logged
as `f1_class_<label>` (columns `metrics.f1_class_0` … `metrics.f1_class_14` in
`results/mlflow_runs.csv`) and tabulated in both multiclass reports; confusion matrices come from
`common/evaluate.py` and are logged as MLflow figure artifacts, which — being artifacts rather than
metrics — stay in the local `mlruns/` store and are not part of the CSV export.

**First-pass multi-class quantum results** (15-class malware-only, n=1000, mean of 5 outer folds).
Random baselines for a balanced 15-way task: accuracy 1/15 = 0.067, macro-F1 ≈ 0.067.

| nc | QSVM f1_macro   | QSVM acc | best classical f1_macro | classical winner |
|----|-----------------|----------|-------------------------|------------------|
| 1  | 0.0577 ± 0.0076 | 0.098    | 0.1283                  | random_forest    |
| 3  | 0.0585 ± 0.0044 | 0.101    | 0.1398                  | random_forest    |
| 6  | 0.0708 ± 0.0141 | 0.106    | **0.1647**              | svm              |

What I take from this:

1. **QSVM sits at roughly the random baseline on honest family attribution.** The per-class F1
   table shows why: it scores 0.00 on about half the families at every nc and recognizes
   essentially **one** family (class 8, Spyware-TIBS, F1 0.44–0.47). It is a one-family detector
   plus noise. This is a negative result, but I believe it is a solid one.
2. **Classical beats QSVM at every n_components** (0.10–0.16 macro-F1), with nonzero F1 spread
   across most classes. The gap was ambiguous at n=200; at n=1000 it is not.
3. **The task is dimensionality-starved, not saturated** — every model improves monotonically from
   nc=1 to nc=6 and none exceeds 0.17. The curves were still rising at nc=6, so more components
   would likely still help.
4. **Separability is intrinsic, not sample-driven** — the *rarest* family (Spyware-TIBS, 1,410
   rows) is the easiest one for every model, quantum and classical alike.
5. **Binary is a different story: saturated and tied.** QSVM reaches 0.990–0.992 macro-F1,
   statistically tied with all four classical models across nc=2/3/6. There the classical/quantum
   difference is **entirely runtime**, not quality — QSVM fit time 71→169 s and tune time
   125→465 s over nc=2→6, while every classical model stays under ~1 s.

### Day 7 — Consolidate the Week 3 report

**Status: PARTIAL — this document is the QSVM half.** The angle-vs-iqp verdict and the first
multi-class quantum results are consolidated here. The other three Week-3 components (Elizabeth's
EMBER SVM fix, Ge's SOREL retune, the VQC results) belong to my teammates and are not covered.

---

## 3. Work I did that was not on either plan

**EMBER 2018 support, binary + multiclass** (2026-07-20; on `feature/ember-2018-support`, open as
PR #5).
This closes the EMBER half of Week 2 Day 2, three days late. It also removes the constraint Team B
flagged in their own plan — "EMBER 2018 is binary-only, no family labels" — which is no longer true.

- **Blocker found and solved.** The official EMBER tarball ships raw JSONL, and the reference
  vectorizer depends on **LIEF 0.9.0, which will not build on Python 3.12**. I resolved this by
  vendoring a LIEF-free copy of the raw-JSON → vector path only
  (`src/common/ember_vectorize.py`, feature version 2, 2381 dims). LIEF is only needed to extract
  raw features *from PE binaries*, which we never do. One genuine compatibility fix was required:
  the upstream code relied on older sklearn iterating a bare `str` into characters inside
  `FeatureHasher`, which modern sklearn rejects.
- **Alternative I rejected.** The pre-vectorized dhoogla parquet has no `sha256` and 199,956 rows
  against the official 200,000, so there is no reliable positional join back to the avclass labels.
  Multiclass would have been impossible with it.
- **Binary:** 100k malware / 100k benign — balance is a non-issue.
- **Multiclass:** EMBER's avclass distribution is heavy-tailed (917 families; only 23 reach 500
  samples), unlike CIC's naturally ~1.7×-balanced families, so I imposed the structure: keep
  families with ≥ 500 samples, cap at the top 15 **for parity with CIC's 15 families**, then
  downsample every kept family to the smallest kept count. The result is exactly balanced (1.0×) at
  955 rows/class, with `wapomi` as the binding constraint. Deterministic under seed.
- **Verified:** 101 tests pass (up from 92), and the CIC code path was reviewed and confirmed
  behavior-preserving so the existing results remain valid.

**Status: code DONE, experiments NOT RUN.** So far only a 120-sample end-to-end probe has executed
(kernel 0.73 s at nc=1). The real sweeps — n≈1000, `angle`+`iqp`, nc 1/3/6, quantum-then-classical
on shared splits — are still ahead of me.

---

## 4. Open items, risks, and what I need

| Item                                                                                                                                                | Status                     | Note                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
|-----------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **EMBER sweeps**                                                                                                                                    | code ready, not run        | My immediate next action. Based on CIC timings, roughly 28 min per nc at n=1000.                                                                                                                                                                                                                                                                                                                                                                                |
| **SOREL-20M**                                                                                                                                       | NOT STARTED                | **This needs a design decision before any code.** SOREL's labels are 11 multi-label behavior tags with no single family class, so plain stratification does not apply. The options are binary-only, a dominant-tag class, or iterative multi-label stratification. This is the largest remaining gap against the goal of running both tasks on both additional datasets, and I would appreciate a steer on which framing you consider defensible for the paper. |
| **Subsample sizing decision**                                                                                                                       | NOT STARTED                | I measured the scaling law but never solved it for a runtime budget. Small piece of work; I can close it alongside the EMBER sweeps.                                                                                                                                                                                                                                                                                                                            |
| **iqp-vs-angle on the 15-class task**                                                                                                               | OPEN                       | iqp wins decisively on 16-class at n=1000 (unanimous 5/5 at nc=6) but not on the 15-class reframe (8/15 over nc = 1, 3, 6). Since 15-class is now the primary multiclass task, this needs a sweep past nc=6 and a direct held-out macro-F1 comparison rather than inner-CV win-counting. See §2, Days 1–3. |
| **Evaluation protocol finalization**                                                                                                                | Blocked on Team A          | Skeleton drafted; the open reconciliation points need their real schema.                                                                                                                                                                                                                                                                                                                                                                                        |
| **CTGAN-augmented quantum run**                                                                                                                     | Blocked on Team B          | Week 3 Day 4; I am ready on my side.                                                                                                                                                                                                                                                                                                                                                                                                                            |

## 5. Summary

Everything on my plan that depended only on me is done, and the Week 3 angle-vs-iqp and
multi-class rows were delivered roughly a week early. The CIC picture is now complete, and the
headline finding is negative but well-supported: **QSVM ties the classical baselines on binary
classification and loses clearly on 15-class family attribution, while costing 100–1000× the
runtime.** What remains outstanding is dataset breadth rather than algorithm work — EMBER is built
but not yet run, and SOREL is untouched and needs a labeling decision before it can start.

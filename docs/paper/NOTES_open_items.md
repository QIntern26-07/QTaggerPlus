# Open items and scope boundaries — working notes

Not part of any report. The reports state limitations *of the results they
present*; this file tracks work that was never carried out, so that it stays
recorded without cluttering the scientific narrative.

Keep the distinction when editing either document:

- **belongs in the report** — "bandwidth was never tuned on CIC, so the
  mechanism is identified but the fix is unmeasured". That bounds a claim the
  report actually makes.
- **belongs here** — "SOREL sweeps were never run". That is absent scope, and a
  reader of the results does not need it to judge them.

---

## SOREL-20M — closed, not deferred

**Delivered:** the multiclass labelling scheme. Dominant-tag argmax over the 11
raw behaviour-tag counts, ties broken by declared column order, all-zero-tag rows
dropped. Validated across 19.7M rows of `meta.db`. Of those, 9.57M were labelled
across 11 classes, 51.5% were dropped as all-zero-tag, and 13.5% involved
tie-breaking.

**Never started:** feature-store download, subset build, binary sweep,
dominant-tag multiclass sweep, any classical baseline on SOREL rows.
`results/mlflow_runs.csv` contains **zero** rows with `params.dataset =
sorel-20m`.

These are independent pieces of work. The labelling design is not partial credit
for the sweeps.

**What is not the blocker** (measured 2026-08-01):

- disk — 227 GiB free against a 71.6 GiB requirement
- memory — `ember_features/data.mdb` is a memory-mapped LMDB; random key reads
  page through the OS cache and the file never needs to be resident

**What is the blocker:** the fixed 71.6 GiB transfer. LMDB offers no key-level
remote access, so the entire file must be local before the first row is readable.
There is no way to fetch only the 1000 rows a sweep would need, which is why the
cost is the same regardless of how small the intended sweep is.

**To resume:**

1. Fetch `s3://sorel-20m/09-DEC-2020/processed-data/ember_features/data.mdb` over
   public unsigned S3 — the path `scripts/fetch_sorel_meta.py` already uses for
   `meta.db`.
2. Build a stratified subset mirroring `src/common/ember_subset.py`, keyed on the
   `dominant_tag` labelling.
3. Run quantum first, then classical with `--load-quantum-splits`.

The CLI surface is already wired: `classical/__main__.py` and
`quantum/__main__.py` both raise an explanatory `SystemExit` for
`--dataset sorel`, and `data.load_sorel` / `task_xy(..., dataset="sorel")`
implement the task. Only the bytes are missing.

---

## Variational quantum classifier

Not implemented, and not planned by this contributor. The Week 2 plan's
"extend the existing binary QSVM/VQC architecture toward multi-class" is
QSVM-only unless someone else picks it up.

Consequence for the reports: they compare *one* quantum method against classical
baselines, never "quantum methods" in general. That framing is already in the
limitations sections and should stay there.

---

## Basis encoding

Deliberately omitted rather than overlooked. It maps discrete values onto
computational basis states, but the pipeline's inputs are continuous PCA
components, so using it would require a quantisation step that discards exactly
what PCA was used to retain.

---

## UMAP as a second dimensionality reduction

Deliberately omitted. PCA is the single classical/quantum alignment point; a
second reduction with no classical counterpart would break it. Rationale first
recorded in `w3_jul-21_anh_qsvm_progress_audit.md`.

---

## CTGAN-augmented run

Blocked on the augmented dataset from Team B, which does not exist in the shared
data directory. Nothing is outstanding on this side — the CLI accepts any
parquet.

---

## Weighted F1

`f1_weighted` was added to `common.evaluate.compute_metrics`, but no sweep was
re-run afterwards, so no recorded result carries it. Adding a metric cannot
produce values for runs that finished before it existed.

It matters most on CIC 15-class, where families run 48–82 rows in the subsample
(1.71× imbalance). EMBER's are balanced at 66–67, where macro and weighted F1
nearly coincide. Any future CIC 15-class sweep should report both.

---

## McNemar between frameworks

Requires per-sample predictions from both models on a shared test set. QSVM
predictions were not persisted during the reported sweeps. Persistence has since
been added (`quantum/__main__.py`), but no sweep was re-run, so nothing is
recoverable retroactively.

This one *does* also appear in the reports, as a limitation on the statistical
evidence they present. That is correct — it bounds a claim being made.

---

## Day 32 / quantum feature extraction (C0-V0-H2)

Elizabeth and Ge's scope, not Team C's.

---

## Evaluation-protocol finalisation

Blocked on Team A. The Week 1 skeleton was delivered.

---

## Known CLI traps, worked around rather than fixed

Both are live and both were handled per-invocation rather than repaired, because
changing them alters how existing output files should be interpreted and so
belongs with a re-run.

- `src/classical/__main__.py:119` names prediction files
  `{model}_{task}_predictions.npz` with **no `n_components`**, so a run at a new
  `nc` silently overwrites the previous one. Worked around with a per-`nc`
  `--predictions-dir`.
- `src/classical/__main__.py:85`/`:124` rewrites the metrics CSV wholesale on
  every invocation. Worked around by giving every cell its own `--out`.

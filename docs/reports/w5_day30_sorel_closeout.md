# Week 5 / Day 30 — SOREL-20M close-out

**Date:** 2026-08-01
**Decision:** SOREL-20M is **CLOSED** for this project's QSVM track — not deferred,
not blocked, not in progress.

## 1. What this closes, and what it does not

Two independent pieces of SOREL work existed. One was delivered, one was never
started, and they should not be conflated:

| Piece | Status |
|---|---|
| Multiclass **labelling design** | **Delivered** — `docs/reports/w4_sorel_labelling_decision.md` |
| Feature acquisition + any QSVM **sweep** | **Never started** |

The labelling design is real, finished work: dominant-tag argmax over the 11 raw
behaviour-tag counts, ties broken by declared column order, all-zero-tag rows
dropped, validated across 19.7M rows from `meta.db` (3.5 GiB, fetched and
analysed). It answers "what does a SOREL family label even mean", which was a
genuinely open design question.

It is **not partial credit for the sweeps.** No SOREL model was ever trained,
classical or quantum. `results/mlflow_runs.csv` contains **zero** rows with
`params.dataset = sorel-20m`, at any `n_components`, for any model. Anyone reading
the labelling report should not infer that results exist.

## 2. What was never done

- Download of the feature store `ember_features/data.mdb`.
- Build of a stratified quantum subset (the SOREL equivalent of
  `data/ember/ember2018_quantum_subset.parquet`).
- Binary sweep.
- Dominant-tag multiclass sweep.
- Any classical baseline on SOREL rows.

## 3. The stated reason, restated precisely

`w4_consolidated_report.md` gives the reason as *"out of scope for the time
available this week"*. That was and remains a **time** argument. This document
does not correct a false statement — it promotes *deferred* to *closed* and pins
down the resource facts so that no future contributor has to re-investigate them.

Measured at close-out (2026-08-01):

```
$ df -h /home/al | tail -1
/dev/nvme0n1p2  468G  218G  227G  49% /

$ free -h | head -2
               total        used        free      shared  buff/cache   available
Mem:            15Gi       7.4Gi       3.0Gi       1.2Gi       6.7Gi       7.9Gi
```

**Disk is not the blocker.** 227 GiB free against a 71.6 GiB requirement — roughly
3× headroom, and that is before considering that only a subset needs to be
retained once extracted.

**RAM is not the blocker.** `ember_features/data.mdb` is a memory-mapped LMDB.
Random key reads page through the OS page cache; the file does not need to be
resident in RAM, and 15 GiB total / 7.9 GiB available is ample for reading an
arbitrary number of individual keys. This is worth stating explicitly because the
*Week 4* wording — "a single 71.6 GiB memory-mapped LMDB with no key-level remote
access, so the whole file must be local before any row can be read" — is easy to
misread as a memory constraint. It is a **locality** constraint: the whole file
must be *on disk*, not in memory.

**The blocker is download bandwidth against the time available.** 71.6 GiB must
transfer in full before the first row is readable, because LMDB offers no
key-level remote access. There is no way to fetch "just the 1000 rows we need" —
that is the entire difficulty, and it is why the cost is fixed regardless of how
small the intended sweep is.

## 4. Resume conditions

A future contributor needs no further investigation. To resume:

1. Fetch `s3://sorel-20m/09-DEC-2020/processed-data/ember_features/data.mdb` via
   public unsigned S3 access — the same access path `scripts/fetch_sorel_meta.py`
   already uses for `meta.db`. Budget 71.6 GiB of transfer.
2. Build a stratified quantum subset mirroring `src/common/ember_subset.py`,
   keyed on the `dominant_tag` labelling already specified in
   `w4_sorel_labelling_decision.md`.
3. Run quantum first, then classical with `--load-quantum-splits`, per the
   project's comparison protocol. `data.split_paths("sorel", task)` already
   returns the correct persistence paths.

The CLI surface is fully wired and waiting for the data — nothing else is
missing. `src/classical/__main__.py:70-78` and `src/quantum/__main__.py:49-57`
both raise an explanatory `SystemExit` for `--dataset sorel` when the parquet is
absent, and `common.data.load_sorel` / `task_xy(..., dataset="sorel")` already
implement the dominant-tag task. Only the bytes are absent.

## 5. Why closing rather than deferring is the honest call

The project's remaining scope is QSVM analysis on CIC-MalMem and EMBER. A third
dataset would strengthen generalisation claims, but the Week 5 deliverables do
not depend on it, and carrying it as "open" across the rest of the project would
misrepresent an item nobody intends to pick up as one that is merely waiting.
Closing it with exact resume conditions preserves every bit of the option value
while stating the true status.

The claim this project can make about generalisation is therefore bounded to two
datasets, and the Week 5 consolidated report should say so plainly rather than
implying broader coverage.

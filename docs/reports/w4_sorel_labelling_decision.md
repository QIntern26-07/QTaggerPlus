# Week 4 — SOREL-20M labelling decision

Closes the SOREL open item in `docs/quantum_todo.md`: "decide binary-only vs.
dominant-tag class vs. iterative multi-label stratification before building
anything." That decision, not the feature download, was the actual blocker —
SOREL has sat untouched for weeks because nobody had settled what a
"multiclass" label even means for a dataset that ships no single family
class. This report settles it from `meta.db` alone.

## Why this task changed scope

The original plan assumed a subset of SOREL features could be streamed from
S3 by fetching selected `sha256` keys out of the feature store. That is not
possible. Direct bucket listing shows:

```
processed-data/meta.db                      3.5 GiB   labels + 11 tag columns
processed-data/ember_features/data.mdb     71.6 GiB   LMDB, single file
lightGBM-features/validation-features.npz  22.2 GiB
lightGBM-features/test-features.npz        37.2 GiB
lightGBM-features/train-features.npz      112.7 GiB
```

`ember_features/data.mdb` is a single memory-mapped LMDB B-tree. LMDB has no
key-level remote-read API — the entire 71.6 GiB file has to be local before
any single key can be opened. There is no way to pull a stratified subsample
of feature vectors without first downloading the whole store. That download,
and any QSVM sweep against it, is out of scope for this task and remains
open. **No SOREL features were downloaded and no SOREL QSVM sweep was run.**
This report only fetches and analyzes `meta.db` (3.5 GiB), which is
sufficient to answer the labelling question on its own.

## Three framings considered

SOREL's 11 tags (`adware`, `flooder`, `ransomware`, `dropper`, `spyware`,
`packed`, `crypto_miner`, `file_infector`, `installer`, `worm`,
`downloader`) are **counts of independent detector votes per sample**, not a
one-hot family label — a single sample can (and very often does) have
nonzero counts in several tags at once. Three ways to turn that into a label
were considered:

1. **Binary-only** (`is_malware`). Trivially available, and it is what CIC
   and EMBER's binary task already exercise. It throws away the entire point
   of adding a third dataset — SOREL exists in this project specifically to
   test the multiclass QSVM comparison at a different data source and scale,
   not to duplicate the binary task with different numbers.

2. **Iterative multi-label stratification** (treat all nonzero tags per
   sample as a label set, use e.g. `iterstrat`'s
   `MultilabelStratifiedKFold` to fold-split while preserving per-tag
   balance). This is the "correct" treatment of multi-label data in the
   literature sense, but it doesn't produce a single class per sample —
   there is no multiclass target to hand to `SVC(kernel="precomputed")` or
   to any of the classical baselines' `fit(X, y)` calls without a further
   reduction step, and it would require a new metric (per-label F1 /
   Jaccard) that CIC and EMBER's multiclass reports don't use, breaking the
   apples-to-apples comparison protocol this project is built around
   (CLAUDE.md: "same CV folds, same PCA dimensionality, same metrics").

3. **Dominant tag** — argmax over the raw per-tag counts, ties broken by a
   fixed column order, all-zero-tag rows dropped. This produces exactly one
   label per sample, so it slots into the same `task_xy`-shaped multiclass
   pipeline CIC (`Category` family) and EMBER (`avclass` family) already
   use, with no new metric and no new CV machinery. It also uses the
   **counts**, not a threshold or a binarized "yes/no" per tag, which keeps
   the label decision principled (the detector engines most strongly agree
   on the tag with the most votes) instead of picking an arbitrary
   binarization cutoff that would need its own justification.

**Decision: dominant tag.** It is the only framing of the three that
produces a single class per sample compatible with the existing
classical/quantum comparison pipeline, without inventing new metrics or
reduction logic that would make SOREL's results incomparable to CIC's and
EMBER's.

## The rule as implemented

`src/common/sorel_labels.py`:

- `TAG_COLS` — the 11 tag columns, in a fixed order that **is** the
  tie-break: `adware, flooder, ransomware, dropper, spyware, packed,
  crypto_miner, file_infector, installer, worm, downloader`.
- `dominant_tag_labels(meta, tag_cols)` — per row, `argmax` over the 11 raw
  tag counts. `np.argmax` returns the first maximal index, so ties resolve
  to whichever tied tag appears earliest in `TAG_COLS`. Rows where all 11
  tags are zero are dropped (no detector flagged any category — there is
  nothing to label).
- `label_stats(meta, tag_cols)` — `total_rows`, `dropped_all_zero`,
  `labelled_rows`, `tied_rows`, and `class_counts` (dominant-tag value
  counts).
- `read_meta(meta_db, table="meta")` — full-table read of
  `sha256, is_malware` + the 11 tag columns via `pd.read_sql_query`. Kept as
  the simple, directly-tested interface for small reads (unit tests, or a
  future subsample query with a `WHERE`/`LIMIT`); not what was used for the
  real 19.7M-row statistics below (see next section).

Five unit tests in `tests/test_sorel_labels.py` cover: plain argmax,
all-zero-row dropping, tie-break-by-column-order, tie-break stability under
row reordering, and `label_stats`' drop/tie/count bookkeeping. All five pass.

## Schema verification

The brief's assumed column list was checked against the real database rather
than trusted:

```
tables: [('meta',)]
columns: ['sha256', 'is_malware', 'rl_fs_t', 'rl_ls_const_positives',
          'adware', 'flooder', 'ransomware', 'dropper', 'spyware', 'packed',
          'crypto_miner', 'file_infector', 'installer', 'worm', 'downloader']
row count: 19,724,997
```

The table is named `meta` as assumed, and all 11 `TAG_COLS` plus `sha256`
and `is_malware` exist with exactly those names. Two extra columns exist
that the brief didn't mention (`rl_fs_t`, `rl_ls_const_positives` —
reputation/detection-count metadata unrelated to the 11 behavior tags);
`read_meta` does not select them, so they don't affect the labelling logic.

## Download

```
$ uv run --with awscli python scripts/fetch_sorel_meta.py
```

`aws s3 cp --no-sign-request s3://sorel-20m/09-DEC-2020/processed-data/meta.db data/sorel/meta.db`
— bucket is public, no credentials needed. Completed in ~8m41s at an average
~7.4 MiB/s, landing a 3.6 GiB file (`data/sorel/meta.db`, gitignored under
`data/*`, never staged).

## How the statistics were computed: chunked, not full-table

Before running the real read, machine memory was checked:

```
$ free -h
               total        used        free      shared  buff/cache   available
Mem:            15Gi       8.4Gi       1.2Gi       1.0Gi       7.1Gi       6.9Gi
```

Only ~1.2 GiB reported free (6.9 GiB "available" counting reclaimable
page cache, but this machine already had one OOM-risk incident this week and
the reclaim isn't guaranteed under memory pressure). A full
`read_meta`-style read materializes 19,724,997 rows across 13 columns,
including a 64-character hex `sha256` string per row as a Python object —
back-of-envelope, roughly 1.3-1.4 GiB just for the sha256 column as pandas
objects, plus ~12 int64 columns at 19.7M x 8 bytes each (~1.9 GiB), before
accounting for `pd.read_sql_query`'s own row-buffer overhead during
construction. That's in the same range as the free memory, with no margin —
judged unsafe to risk.

Instead, `scripts/sorel_label_stats.py::chunked_label_stats` reads via
`pd.read_sql_query(..., chunksize=1_000_000)`, calling the tested
`label_stats` function on each 1M-row chunk and summing
`total_rows`/`dropped_all_zero`/`labelled_rows`/`tied_rows` and merging
`class_counts` with a `Counter`. This keeps `label_stats`'s tested interface
completely unchanged (it still takes one DataFrame, as the brief requires
for the unit tests) — the chunking is purely an accumulation strategy
layered on top in the script, not a change to the tested function. Peak
resident memory per chunk is roughly 1/20th of the full-table estimate
above. The run completed in 63 seconds across 20 chunks with no swap
pressure (`free -h` before/after showed free memory essentially unchanged,
~780 MiB-1.2 GiB, no growth in swap usage).

A `--full` flag exists on the script for a from-scratch full-table read if a
machine with more headroom is available later, but it was not used to
produce the numbers below.

## Statistics

```json
{
  "total_rows": 19724997,
  "dropped_all_zero": 10158278,
  "labelled_rows": 9566719,
  "tied_rows": 1287189,
  "class_counts": {
    "spyware": 1722713,
    "dropper": 1414233,
    "adware": 1008500,
    "worm": 1265735,
    "packed": 964299,
    "ransomware": 721508,
    "downloader": 808987,
    "file_infector": 1063928,
    "installer": 300247,
    "flooder": 27896,
    "crypto_miner": 268673
  }
}
```

- **Total rows**: 19,724,997.
- **All-zero-tag rows dropped**: 10,158,278 (51.5% of the table) — over half
  of SOREL's samples have no positive count on any of the 11 behavior tags.
  Combined with `is_malware`, most of this is presumably benign/unflagged
  software rather than malware missing a family assignment, but this report
  does not cross-tabulate against `is_malware` to confirm that split
  quantitatively — noted as an open question below.
- **Labelled rows**: 9,566,719 (48.5%) — still a very large usable pool, two
  orders of magnitude above anything a QSVM subsample sweep would need.
- **Tied rows**: 1,287,189 (13.5% of labelled rows) — a substantial minority
  of samples have two or more tags tied for the top count and get
  deterministically assigned to whichever tied tag sits earliest in
  `TAG_COLS`. This is a real source of label noise inherent to the
  dominant-tag framing (see Open Questions).
- **Per-class distribution** (11 classes, sorted descending):

  | class | count | % of labelled |
  |---|---:|---:|
  | spyware | 1,722,713 | 18.0% |
  | dropper | 1,414,233 | 14.8% |
  | worm | 1,265,735 | 13.2% |
  | file_infector | 1,063,928 | 11.1% |
  | adware | 1,008,500 | 10.5% |
  | packed | 964,299 | 10.1% |
  | downloader | 808,987 | 8.5% |
  | ransomware | 721,508 | 7.5% |
  | installer | 300,247 | 3.1% |
  | crypto_miner | 268,673 | 2.8% |
  | flooder | 27,896 | 0.3% |

## Balance assessment for a stratified QSVM subsample

The ratio between the largest class (spyware, 1,722,713) and the smallest
(flooder, 27,896) is about **62:1** — this is a severely imbalanced
population by any standard, consistent with the general expectation for
multi-label malware tags stated in the task brief.

That said, "imbalanced in the full 19.7M-row population" is a different
question from "unusable for a stratified subsample," and the latter is what
actually matters here. `docs/reports/w4_subsample_sizing_decision.md`
derived a per-class test-fold floor for a 15-class task
(`n/5 >= 15*10 => n >= 750`) and recommended **n=900** at `n_components=2`
for future EMBER/SOREL sweeps. Re-deriving the same arithmetic for SOREL's
11 classes: `n/5 >= 11*10 => n >= 550`. At the recommended n=900 with roughly
equal per-class allocation (~82 rows/class before the 5-fold split, ~16 per
test fold), every one of the 11 classes needs at least ~82 *available* rows
to draw a balanced subsample from. The smallest class, flooder, has 27,896
available — over 300x the ~82 rows a balanced n=900 subsample would need
from it. **Every class clears the floor by a wide margin**; the imbalance is
in the full population's proportions, not in absolute availability at
subsample scale.

**Conclusion: usable, with an explicit caveat.** A *balanced* stratified
subsample (equal draw per class, the same "downsample-to-balance" approach
EMBER's multiclass task already uses per `docs/quantum_todo.md`'s Decided
section) is straightforward to build at n=900 — flooder and crypto_miner
being far smaller than spyware in the full population doesn't block it,
because even the smallest class has orders of magnitude more rows than a
balanced subsample would draw. What the imbalance *does* imply for a future
SOREL sweep: a **non-stratified** (proportional/simple random) subsample at
n=900 would draw roughly 3 flooder rows and 5 crypto_miner rows out of 900 —
nowhere near enough for those classes' per-fold F1 to mean anything. The
sweep must explicitly balance the subsample per class (as EMBER's multiclass
task already does), not draw it proportionally to SOREL's natural label
frequencies, or the two rarest classes (flooder, crypto_miner) will produce
noise-level metrics despite the 750-row-floor style arithmetic technically
being satisfied on average.

## What was and wasn't done

- **Fetched**: `meta.db` only (3.6 GiB), via public `--no-sign-request` S3
  access.
- **Not fetched**: `ember_features/data.mdb` (71.6 GiB LMDB feature store) —
  LMDB requires the entire file local before any key-level read is possible,
  so no partial/subsampled feature download exists. This remains open.
- **Not run**: any SOREL QSVM sweep, timing probe, or classical baseline —
  there is no feature data to run one against yet. This task settles the
  labelling design only.

## Open questions

- The 51.5% all-zero-tag drop rate should be cross-checked against
  `is_malware` before the labelling pipeline is trusted end-to-end — if a
  meaningful fraction of `is_malware=1` rows are being dropped as
  all-zero-tag (rather than the drop being almost entirely benign samples,
  as assumed above), that changes how much of the malware population this
  labelling scheme actually covers. Not computed here; flagged for whoever
  picks up the feature-acquisition half of this work.
- 13.5% of labelled rows are tied at the top count. The fixed tie-break
  (declared `TAG_COLS` order) is deterministic and reproducible but
  arbitrary in which tag "wins" a tie — worth a sensitivity check (e.g. does
  reordering `TAG_COLS` materially shift `class_counts`?) before the label
  set is treated as final, since 1.29M rows is not a negligible slice.
- No cross-tabulation was done between dominant tag and `is_malware`/benign
  status; it's assumed (not verified) that dominant-tag labels are
  meaningful mostly within the malware subset.

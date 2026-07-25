# Week 4 Jul 27 — Closing the 15-class angle-vs-iqp verdict on CIC-MalMem, past nc=6

## What this closes

The open item in `docs/reports/w3_jul-21_anh_qsvm_progress_audit.md` §4
("iqp-vs-angle on the 15-class task") rested on **inner-CV win-counting**:
across the nc=1/3/6 CIC 15-class sweeps, the Optuna inner loop was tuned
jointly over both `angle` and `iqp` per fold, and the verdict was "iqp won
8/15 folds" — a count of which encoding the tuner *picked*, not a measurement
of how much better either one *scored*. That methodology is retired as of
this report.

This report re-runs nc=1, 3, 6 one encoding at a time (so each cell is a
dedicated invocation with a single fixed encoding, not a joint tune), and
extends the sweep to nc=8 for the first time on CIC 15-class. Every number
below is a held-out macro-F1 from its own invocation.

## Step 1: cost probe at nc=8

```sh
uv run python -m quantum --probe --tasks multiclass --n-components 8 \
  --max-samples 200 --encodings angle iqp --n-jobs 8
```

Output:

```
[probe] angle nc=8 kernel_train=7.354s fit=7.360s infer=3.835s
[probe] iqp   nc=8 kernel_train=18.565s fit=18.569s infer=10.335s
```

Projected to n=1000 (x25, per the brief's O(n^2) scaling rule): angle
~184s/Gram, iqp ~464s/Gram (~7.7 min). Both comfortably under the 15-minute
gate, so nc=8 proceeded. nc=10 was **not attempted** — it's out of this
report's required table (nc in {1,3,6,8}), and per the task instructions it
was optional and only worth attempting opportunistically; with iqp nc=8
already costing ~41 minutes wall-clock for the full 5-fold sweep (see below),
extrapolated nc=10 cost was judged not worth the extra unplanned runtime for
a data point the deliverable doesn't need.

## Commands run

All runs: CIC-MalMem (`--dataset` defaults to `cic`), `--max-samples 1000
--folds 5 --n-jobs 8 --mlflow`, one encoding per invocation, in this order
(never overlapping, each polled to completion by PID before the next
launched):

```sh
mkdir -p docs/reports/logs/w4_cic_15class_encoding

# Step 2 — nc=8, the expensive cell, run first per the brief
uv run python -m quantum --tasks multiclass --n-components 8 --encodings angle ...
uv run python -m quantum --tasks multiclass --n-components 8 --encodings iqp ...

# Step 3 — backfill nc=1/3/6, one encoding at a time
for nc in 1 3 6; do
  for enc in angle iqp; do
    uv run python -m quantum --tasks multiclass --n-components $nc --encodings $enc ...
  done
done

# Step 4 — classical SVM reference at nc=8, replayed on the quantum splits
uv run python -m classical --models svm --tasks multiclass --n-components 8 \
  --load-quantum-splits --folds 5 --mlflow

# classical SVM reference at nc=1/3/6 too (needed for the full table; the
# existing committed SVM numbers at these nc turned out to be reproducible —
# see anomaly note below — but were re-run today for an exact same-day match)
for nc in 1 3 6; do
  uv run python -m classical --models svm --tasks multiclass --n-components $nc \
    --load-quantum-splits --folds 5 --mlflow
done

uv run python scripts/export_mlflow_runs.py
```

Logs: `docs/reports/logs/w4_cic_15class_encoding/nc{1,3,6,8}_{angle,iqp}_quantum.txt`,
`nc{1,3,6,8}_classical.txt`, `probe_nc8.txt`.

## Wall-clock per invocation (log timestamps, first fold start -> last fold done)

| run | wall clock |
|---|---|
| nc=1 angle quantum | 3m 41s |
| nc=1 iqp quantum | 2m 26s |
| nc=3 angle quantum | 7m 27s |
| nc=3 iqp quantum | 7m 45s |
| nc=6 angle quantum | 12m 49s |
| nc=6 iqp quantum | 24m 11s |
| nc=8 angle quantum | 16m 22s |
| nc=8 iqp quantum | 41m 00s |
| nc=1/3/6/8 classical (svm) | 14-20s each |

Total quantum sweep: ~115 minutes across 8 invocations, dominated by nc=8 iqp
(41 min) as predicted by the Day 1 profiling report's gate-count projection
(88 vs 46 `MultiRZ` gates at 8 qubits). nc=6 iqp (24m 11s) essentially matches
the EMBER 15-class nc=6 iqp wall-clock (24m 26s, same n=1000/5-fold shape),
consistent with kernel cost being dataset-size-driven, not dataset-identity-
driven.

## Held-out macro-F1 per encoding — the required table

Mean +/- std over 5 outer folds, MLflow parent-run aggregate. `delta` =
iqp - angle.

| nc | angle | iqp | delta | classical svm |
|---|---|---|---|---|
| 1 | 0.0550 ± 0.0095 | 0.0559 ± 0.0077 | +0.0009 | 0.1075 ± 0.0164 |
| 3 | 0.0585 ± 0.0057 | 0.0707 ± 0.0176 | +0.0122 | 0.1309 ± 0.0265 |
| 6 | 0.0681 ± 0.0147 | 0.0794 ± 0.0171 | +0.0113 | 0.1647 ± 0.0233 |
| 8 | 0.0693 ± 0.0062 | 0.0739 ± 0.0103 | +0.0046 | 0.1772 ± 0.0355 |

(Random baseline for 15 balanced classes ≈ 0.067, marked here for reference —
CIC's 15 malware families are ~1.7x imbalanced so the true chance level is
somewhat below this, but it's the same reference figure used throughout this
project's CIC 15-class reports.)

## Verdict

**iqp beats angle at every n_components tested (1, 3, 6, 8) on CIC 15-class**,
but the margin is small and inconsistent in size — +0.0009 (nc=1, essentially
a wash), +0.0122 (nc=3), +0.0113 (nc=6), +0.0046 (nc=8) — and at every nc the
delta is smaller than either encoding's own fold-to-fold std (0.006-0.018).
This is **not** the same pattern as EMBER 15-class, where iqp's advantage was
large, monotone-ish, and clearly separated from angle (+0.089 at nc=3, +0.096
at nc=6, both deltas well outside the ~0.03-0.06 per-encoding std observed
there). On CIC 15-class the "16-class n=1000 monotone iqp climb" **does not
reappear**: there is a consistent directional edge for iqp, but it is thin,
noisy relative to its own variance, and does not grow monotonically with
qubit count (it peaks at nc=3, not at nc=8, the highest qubit count tested).

Given the fold-to-fold std at every nc is 1.5-3x the observed delta, this
sweep does not support a strong claim that iqp is meaningfully better than
angle on CIC 15-class — only that it has not been worse, in any of these
four single-encoding, non-jointly-tuned runs. That is already a materially
different and more honest conclusion than the retired win-counting method
implied (which under-stated iqp, counting only 8/15 raw fold wins with no
magnitude information at all).

Classical SVM (nc=8: 0.1772 ± 0.0355) beats both QSVM encodings by a wide
margin at every n_components, consistent with every CIC 15-class and EMBER
result recorded so far in this project — QSVM has never closed this gap on
any dataset/task combination tried to date.

**Methodology retirement.** Inner-CV win-counting (counting which encoding an
Optuna trial selected per fold, when both encodings are tunable candidates in
the same search) is retired for encoding comparisons on this project as of
this report. It conflates "which one got picked when both were live options
under one joint budget" with "which one performs better," and — as this
report demonstrates — can produce an anemic or even misleading signal (8/15
"iqp wins" reads as ambiguous; the direct macro-F1 deltas show iqp actually
ahead in every single nc tested, just not decisively). All future
angle-vs-iqp comparisons in this project should use dedicated,
single-encoding invocations and report macro-F1 deltas with std, as done
here and in the EMBER reports.

## Anomaly note

The nc=1/3/6 classical SVM numbers re-run today (0.1075, 0.1309, 0.1647)
reproduced **exactly** the SVM numbers from the Jul 15 committed history
(same three values to 6 decimal places) despite being independent
invocations three-plus weeks apart. This confirms the `--load-quantum-splits`
protocol is fully deterministic end to end (seed=42 stratified subsample +
StratifiedKFold folds, independent of `n_components` since folds are built
from labels only) — a reassuring consistency check, not a bug. `data/splits/`
did not change as a result of today's re-runs (`git status` on that directory
was clean throughout), for the same reason.

## Open questions

- Whether iqp's thin-but-consistent edge on CIC 15-class would sharpen or
  vanish with more outer folds / larger n (this sweep is still n=1000,
  5-fold, same as every other report this project has produced) is untested.
- Why CIC 15-class QSVM plateaus so far below EMBER 15-class QSVM (best
  CIC macro-F1 0.0794 at nc=6 vs EMBER's 0.5195) is analyzed in the Jul 26
  EMBER report as likely a combination of CIC's class imbalance and a harder
  underlying separability problem (memory-dump features vs static PE
  features) — this report does not re-derive that, only confirms CIC 15-class
  QSVM is still barely above random baseline at every nc tested, including 8.

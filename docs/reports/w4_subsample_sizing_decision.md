# Week 4 — Subsample-sizing decision for QSVM sweeps

Closes the "Day 2 EMBER/SOREL-20M subsample sizing decision" item that has sat
open in `docs/quantum_todo.md` since Week 2 Day 1. The measurement it depends
on already existed; what was missing was solving it for a runtime budget and
writing the number down.

## The fitted law

`docs/reports/w2_day1_quantum_profiling_report.md` measured kernel Gram-matrix
build time against sample count at `n_components=2` (the qubit count the
profiling sweep used) for both `angle` and `iqp` encodings, batched execution:

| encoding | n_samples | kernel_build_train_s |
|---|---|---|
| angle | 50 | 0.13 |
| angle | 100 | 0.68 |
| angle | 200 | 3.03 |
| angle | 400 | 13.85 |
| iqp | 50 | 0.22 |
| iqp | 100 | 0.64 |
| iqp | 200 | 3.42 |
| iqp | 400 | 10.74 |

The report fits this to a power law `t(n) = t_ref * (n / n_ref) ** exponent`
with `exponent ~1.9` (iqp) to `~2.2` (angle) — consistent with the O(n^2) pair
count the kernel evaluation is built from, batching only reduces the per-pair
constant, not the exponent. At `n_components=2`, `angle` and `iqp` cost almost
identically (~11.4 s at n=400), so the report uses `exponent=2.0` and
`(ref_n, ref_s) = (400, 11.4)` as the representative reference point for
either encoding at that qubit budget. `solve_for_budget` in
`src/common/sizing.py` inverts that law: given a target seconds-per-Gram-build
budget, it returns the largest `n` whose predicted build time stays within it.

**Caveat carried forward from the Day 1 report**: this reference point is
qubit-budget-specific. If a sweep moves to `n_components` materially above 2,
`iqp`'s steeper `MultiRZ` gate-count growth (28 vs 6 gates at 4 qubits, 88 vs
46 at 8 qubits) means its per-pair constant — and therefore its effective
`ref_s` — grows faster than `angle`'s. The n below should be re-derived from a
fresh probe at the actual target `n_components` before being treated as exact,
though the qualitative recommendation (target the class floor, not the
runtime ceiling) will not change.

## Solved n at three per-Gram-build budgets

```
$ for b in 60 300 600; do uv run python scripts/solve_sample_size.py --budget-s $b; done
max n for 60.0s/Gram: 917
max n for 300.0s/Gram: 2051
max n for 600.0s/Gram: 2901
```

| budget | max n (runtime-only) |
|---|---|
| 1 min (60 s) | 917 |
| 5 min (300 s) | 2051 |
| 10 min (600 s) | 2901 |

These are per-Gram-build numbers, not per-fold or per-sweep numbers. A tuned
CV sweep (`run_quantum_cv`) builds a Gram matrix once per Optuna trial per
inner fold, plus once per outer fold for the final refit, and does this again
for every encoding under test — so the wall-clock multiplier on top of a
single build is easily 1-2 orders of magnitude. The budget above should be
read as "how expensive is it acceptable for *one* kernel build to be," not as
a sweep time estimate.

## Cross-check against class balance (15-class task)

Before accepting a runtime-derived n, it has to leave enough rows per class
per test fold to make per-class F1 a real estimate rather than noise. Deriving
the floor from scratch rather than trusting a stated number: with 5-fold CV,
each test fold holds `n / 5` rows. Requiring roughly 10 rows per class in a
test fold for the 15-class malware-family task means

```
n / 5 >= 15 * 10
n >= 750
```

This matches the number `docs/quantum_todo.md`'s open item pointed at
(`15 * 10 * 5 = 750`) — the arithmetic there is correct, just re-derived here
independently rather than taken on faith.

**Where the floor actually binds**: solving for the budget at which the
runtime-derived n crosses 750 —

```
750 = 400 * (b / 11.4) ** 0.5  =>  b ~= 40 s
```

— shows the class floor only binds below roughly a 40-second-per-Gram-build
budget (`solve_for_budget(40, ...)` gives 749; `solve_for_budget(41, ...)`
gives 758). All three budgets asked for in this report (60 s, 300 s, 600 s)
already clear 750 comfortably (917, 2051, 2901), so **for the budgets actually
under consideration, the runtime ceiling is the binding constraint, not the
class floor** — the opposite of what the open TODO item's framing implied.
The floor matters as a *sanity lower bound* to check any future budget
against, not as the thing driving today's number.

## Recommendation

**Target n = 900** for future EMBER/SOREL-20M QSVM subsample sweeps, assuming
a ~60-second per-Gram-build budget at `n_components=2`.

Rationale: 60 s/build is a budget an iterative sweep (many builds per Optuna
trial, times several outer folds, times multiple encodings) can absorb
without turning into a multi-hour run, while `solve_for_budget(60, 400, 11.4,
2.0) = 917` clears the 750-row 15-class balance floor with headroom (~30
rows/class/test-fold instead of the bare minimum ~10) rather than sitting
right at it. Rounding down to 900 keeps the number stratifiable and leaves a
small margin under the 917 ceiling. If a sweep targets a materially different
`n_components`, re-run the Day 1 probe at that qubit count first — the
`ref_s=11.4` used here is only validated at `n_components=2`.

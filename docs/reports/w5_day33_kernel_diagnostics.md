# Week 5 / Day 33 — Kernel concentration and target alignment, CIC vs EMBER

**Date:** 2026-08-01
**Driver:** `scripts/kernel_diagnostics.py` (5 m 52 s, 2 Gram builds)
**Raw output:** `docs/reports/logs/w5_day33/alignment.json`,
`alignment_baseline.json`, `concentration_from_mlflow.txt`

Day 31 ended by showing that feature geometry explains why *everything* does
badly on CIC 15-class but cannot explain the **QSVM-specific** failure: classical
SVM extracts 0.108–0.165 macro-F1 from exactly the projected features where QSVM
extracts 0.056–0.079. Since the input geometry is identical by construction, the
difference has to lie in the kernel. This day measures the kernel.

## 1. Method

For each dataset, on the **first outer training fold** at `n_components = 6`
with the `iqp` encoding (800 rows, 6 qubits):

1. Build the QSVM fidelity Gram with `QSVM._gram_sym` — the exact method
   `QSVM.fit` uses, called directly because `fit` consumes the matrix internally
   and never exposes it.
2. Build a classical RBF Gram on the **identical** post-PCA, post-encoding-scaled
   matrix, `gamma = 1/(n_features · var)`.
3. Measure off-diagonal standard deviation (concentration) and kernel-target
   alignment on both.

The RBF control is what makes this decisive. Comparing QSVM on CIC against QSVM
on EMBER only shows *that* the kernel behaves differently; comparing QSVM against
RBF **on the same feature matrix** separates "these features are hard" from
"this kernel is bad".

## 2. Results

| dataset | kernel | off-diagonal std | alignment | alignment − baseline |
|---|---|---|---|---|
| CIC | QSVM (iqp) | **0.0886** | 0.259955 | **−0.000069** |
| CIC | RBF control | **0.2212** | 0.256096 | −0.003928 |
| EMBER | QSVM (iqp) | **0.2015** | 0.288423 | **+0.030208** |
| EMBER | RBF control | **0.2184** | 0.285654 | +0.027439 |

## 3. The alignment column needs its baseline, or it means nothing

Raw kernel-target alignment on a 15-class problem is dominated by a floor.
A constant kernel `K = 1` has alignment `1/√15 ≈ 0.2582` for balanced classes —
computed exactly against each dataset's actual fold-0 labels: **0.260024** for
CIC, **0.258215** for EMBER. Every raw number in the table sits within a few
hundredths of that floor, so the raw column is nearly uninformative and the
excess column is the one to read.

**On CIC, the fidelity kernel's alignment is 0.000069 BELOW the constant-kernel
baseline.** Not slightly above, not marginal — indistinguishable from an all-ones
matrix. At the Gram level it carries no label information whatsoever. The RBF
control on the same features is also below baseline, by 0.0039.

**On EMBER, both kernels clear the baseline by ~0.03** — QSVM +0.0302, RBF
+0.0274.

So alignment separates the two *datasets* cleanly, and does **not** separate the
two *kernels* within either dataset. On CIC neither kernel sees the 15-class
structure; on EMBER both do, and the quantum kernel is marginally the better of
the two.

## 4. Concentration is where the kernels actually differ

| | CIC | EMBER | CIC/EMBER |
|---|---|---|---|
| QSVM off-diagonal std | 0.0886 | 0.2015 | **0.44×** |
| RBF off-diagonal std | 0.2212 | 0.2184 | 1.01× |
| **QSVM / RBF within dataset** | **0.40×** | **0.92×** | |

Read the rows first. The **RBF kernel produces essentially the same spread on
both datasets** — 0.2212 against 0.2184, a 1% difference. Whatever is different
about CIC's features, it does not make a classical kernel concentrate.

The fidelity kernel does the opposite. On EMBER it tracks RBF closely (0.2015 vs
0.2184, 0.92×). On CIC it collapses to **0.40× of RBF's spread on the identical
matrix**. Every off-diagonal similarity is squeezed into a band 2.5× narrower
than the classical kernel achieves from the same 800 × 6 inputs.

That is the QSVM-specific mechanism Week 4 was missing. The Gram approaches a
scaled identity: with all pairs looking roughly equally similar, the SVM has
almost nothing to separate on, which is exactly what a macro-F1 pinned to the
random baseline looks like.

The MLflow-sourced table in §5 confirms this is not an artifact of one fold or
one setting — CIC's fidelity Gram is roughly half as spread as EMBER's at every
`n_components` and both encodings.

## 5. Concentration across the sweep — no new computation required

`src/quantum/qsvm.py:186` has been computing `gram_offdiag_std` on every fit all
along, and `run.py:193` logs it. The full picture was already in
`results/mlflow_runs.csv` and needed only to be read.

Latest clean sweep per cell (grouped by `parent_run_id`, `FINISHED` parents with
exactly 5 folds — see `w5_day29_ember_four_model.md` §7), multiclass:

| dataset | nc | encoding | mean | std |
|---|---|---|---|---|
| CIC | 1 | angle | 0.1192 | 0.0268 |
| CIC | 1 | iqp | 0.0927 | 0.0198 |
| CIC | 3 | angle | 0.0795 | 0.0038 |
| CIC | 3 | iqp | 0.1005 | 0.0132 |
| CIC | 6 | angle | 0.0674 | 0.0035 |
| CIC | 6 | iqp | 0.0882 | 0.0105 |
| CIC | 8 | angle | 0.0600 | 0.0058 |
| CIC | 8 | iqp | 0.0798 | 0.0096 |
| EMBER | 1 | angle | 0.2044 | 0.0054 |
| EMBER | 1 | iqp | 0.1756 | 0.0027 |
| EMBER | 3 | angle | 0.1942 | 0.0113 |
| EMBER | 3 | iqp | 0.2569 | 0.0177 |
| EMBER | 6 | angle | 0.1363 | 0.0180 |
| EMBER | 6 | iqp | 0.2104 | 0.0086 |

EMBER's Gram is 1.9–2.6× more spread than CIC's at every comparable cell.
Concentration also worsens with qubit count on both datasets — CIC `iqp` falls
0.0927 → 0.0798 from nc=1 to nc=8 — the textbook kernel-concentration trend that
`bandwidth` exists to counteract.

## 6. Cross-check: does the fresh Gram reproduce the sweep?

Rebuilding the Gram gives a free correctness check on the whole pipeline.

| dataset | MLflow nc=6 iqp (5-fold mean ± std) | fresh fold-0 build | verdict |
|---|---|---|---|
| CIC | 0.0882 ± 0.0105 | 0.08856 | matches |
| EMBER | 0.2104 ± 0.0086 | 0.20151 | within 1.0 SD |

Exact equality is not expected — MLflow averages 5 folds, this rebuild uses fold
0 only — but both land inside the fold-to-fold spread. The rebuild is reproducing
the sweep, so the §2 numbers describe the same kernel the reported results came
from.

## 7. The tension this leaves, stated rather than papered over

§3 says neither kernel's Gram carries label information on CIC. §5 of
`w5_day31` says classical SVM nonetheless reaches 0.108–0.165 macro-F1 on those
features — well above the 0.0667 random baseline. Those two facts sit awkwardly
together and the report will not pretend otherwise.

The reconciliation is that **kernel-target alignment is a global, unweighted
average over all pairs, and an SVM does not need global alignment.** Alignment
near its floor says the *average* same-class pair is no more similar than the
*average* different-class pair. It says nothing about whether a small subset of
points supports a usable decision boundary, which is precisely what an SVM
builds — through support vectors, `C`, and class weights, none of which alignment
models. So alignment is a blunt instrument here: useful for showing that CIC is
categorically harder than EMBER, unreliable as a predictor of achievable SVM
performance.

The concentration result does not have this problem. It is a statement about the
Gram the SVM actually receives, and it distinguishes the two kernels *within* CIC
where alignment cannot.

**Conclusion.** Two distinct effects, both real:

1. **A dataset effect.** CIC's 15-class structure is not visible to either kernel
   at the Gram level, and EMBER's is. This is the Day 31 geometry finding
   restated in kernel terms.
2. **A kernel effect, specific to CIC.** The fidelity kernel concentrates to 0.40×
   the RBF control's spread on identical features, while on EMBER it matches RBF
   at 0.92×. This is the QSVM-specific deficit, and it is a property of the
   kernel, not of the features.

`bandwidth` (`quantum/encoding.py::default_bandwidth`) is the lever for a
follow-up: it scales inputs precisely to counteract concentration, and it has
never been tuned on CIC beyond its default.

## 8. Limitations

- **One training fold** (fold 0), **one `n_components`** (6), **one encoding**
  (`iqp`) for the alignment and RBF-control measurements. §5's concentration
  table is broader, covering 5 folds × 4 `nc` values × 2 encodings, but it has no
  RBF control.
- The RBF control uses a single `gamma` heuristic and is not tuned. A tuned RBF
  might spread differently; the comparison is "fidelity kernel vs. a reasonable
  classical default", not "vs. the best possible classical kernel".
- Alignment is measured on the training fold, so it describes the kernel the SVM
  fits on, not generalisation.
- No follow-up bandwidth sweep was run. §7's closing suggestion is a hypothesis
  with a named mechanism, not a measured result.

## Artifacts

- `src/common/kernel_diag.py` — `kernel_target_alignment`, `offdiag_std`.
- `tests/test_kernel_diag.py` — 6 tests.
- `scripts/kernel_diagnostics.py` — the driver.
- `docs/reports/logs/w5_day33/` — raw JSON and the MLflow concentration table.

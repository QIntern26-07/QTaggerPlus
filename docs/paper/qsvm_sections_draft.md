# QSVM sections — research paper draft

**Author:** Anh (Team C, quantum / QSVM)
**Date:** 2026-08-01
**Status:** first draft, QSVM portion only.

## Scope note

`temp/qi26_7.md` (the mentors' six-week plan) lists Week 5 as *"Draft initial
version of the research paper and implementation documentation"*, while
`temp/Week 5 Plan.md` does not assign a paper draft to Team C. The two documents
disagree. This draft resolves the disagreement by covering **the QSVM portion
only** — methodology, results, encoding comparison, the collapse investigation,
and limitations. Classical-baseline framing, dataset description, related work
and the introduction are not attempted here.

**No new analysis appears below.** Every number is assembled from the committed
per-day reports; where a claim is thin, it is reported as thin.

---

## 1. Methodology

### 1.1 The comparison protocol

The point of the study is an apples-to-apples comparison, which requires
controlling three things that are easy to let drift.

**Identical rows and folds.** The quantum kernel costs O(n²) to evaluate, which
makes full-dataset quantum runs infeasible: every quantum result is computed on a
stratified 1000-row subsample. The subsample and the outer folds computed over it
are persisted to disk by the quantum CLI; the classical CLI then replays them via
`--load-quantum-splits`. A full-dataset classical run is never compared against a
subsampled quantum run.

**Identical dimensionality.** Both pipelines consume the output of one shared
feature pipeline: variance filter → correlation filter → standardisation →
PCA. PCA's `n_components` is the single alignment point — it fixes the classical
feature count and the quantum qubit budget simultaneously. The pipeline is fit
**inside the training fold only**, so no test-fold information reaches the
projection.

**Identical evaluation.** Nested cross-validation: an Optuna inner loop tunes
hyperparameters within each outer training fold, followed by one timed refit on
the full training fold. The quantum side uses a two-tier variant that exploits a
property of the fidelity kernel — the kernel depends only on the encoding and
bandwidth, not on the SVM's `C` — so the inner Gram is built once per
(encoding, bandwidth) and `C`/`class_weight` are swept cheaply on top of it.

### 1.2 Why macro-F1 and not accuracy

Macro-F1 is the headline metric throughout. The reason is concrete rather than
conventional. On EMBER's binary task at `n_components = 1`, QSVM reaches
**0.890 accuracy in three of five outer folds while the benign class's F1 is
exactly 0.000 in those same folds**. The model labels every sample malware and is
right 89% of the time because 89% of the rows are malware. Accuracy here is not
merely optimistic — it reports near-success for a classifier that never once
identifies the class the task exists to find.

### 1.3 Subsample validity

Because every quantum result rests on 1000 rows, the subsample's faithfulness is
a precondition for the whole study rather than a detail. It was measured
directly: class proportions, per-feature means, per-feature standard deviations
and per-feature distributions were compared against the full population for both
datasets.

The decisive statistic is the two-sample Kolmogorov–Smirnov rejection count read
against its own null expectation, since a perfectly faithful subsample still
produces about `α · n_features` rejections by chance. **CIC rejects 0 of 55
features where ~2.75 were expected; EMBER rejects 1 of 2,381 where ~119 were
expected.** Maximum class-proportion drift is below 0.001 on both. The
subsamples are faithful.

---

## 2. Results

macro-F1, mean ± std over 5 outer folds. Best classical model bolded.

### 2.1 EMBER 2018, binary

| nc | random_forest | xgboost | lightgbm | svm | QSVM angle | QSVM iqp |
|---|---|---|---|---|---|---|
| 1 | **0.5582 ± 0.0242** | 0.5451 ± 0.0196 | 0.5093 ± 0.0335 | 0.5314 ± 0.0277 | 0.4532 ± 0.0277 | 0.4587 ± 0.0229 |
| 3 | **0.6575 ± 0.0544** | 0.6465 ± 0.0614 | 0.6434 ± 0.0160 | 0.6306 ± 0.0192 | 0.4924 ± 0.0548 | 0.5084 ± 0.0347 |
| 6 | **0.6835 ± 0.0741** | 0.6680 ± 0.0489 | 0.6593 ± 0.0708 | 0.6724 ± 0.0376 | 0.5138 ± 0.0084 | 0.5137 ± 0.0372 |

### 2.2 EMBER 2018, 15-class

| nc | random_forest | xgboost | lightgbm | svm | QSVM angle | QSVM iqp |
|---|---|---|---|---|---|---|
| 1 | **0.5562 ± 0.0268** | 0.5313 ± 0.0315 | 0.5410 ± 0.0414 | 0.5294 ± 0.0328 | 0.1688 ± 0.0313 | 0.1541 ± 0.0262 |
| 3 | 0.7194 ± 0.0102 | 0.7089 ± 0.0162 | 0.7035 ± 0.0216 | **0.7389 ± 0.0061** | 0.3908 ± 0.0590 | 0.4797 ± 0.0277 |
| 6 | 0.7903 ± 0.0238 | 0.7659 ± 0.0214 | 0.7709 ± 0.0166 | **0.7930 ± 0.0208** | 0.4232 ± 0.0751 | 0.5195 ± 0.0582 |

### 2.3 CIC-MalMem, 15-class

Random baseline for 15 balanced classes ≈ 0.0667.

| nc | random_forest | xgboost | lightgbm | svm | QSVM angle | QSVM iqp |
|---|---|---|---|---|---|---|
| 1 | **0.1283 ± 0.0359** | 0.1010 ± 0.0162 | 0.1094 ± 0.0226 | 0.1075 ± 0.0164 | 0.0550 ± 0.0095 | 0.0559 ± 0.0077 |
| 3 | **0.1398 ± 0.0130** | 0.1270 ± 0.0087 | 0.1185 ± 0.0164 | 0.1309 ± 0.0265 | 0.0585 ± 0.0057 | 0.0707 ± 0.0176 |
| 6 | 0.1510 ± 0.0262 | 0.1494 ± 0.0249 | 0.1571 ± 0.0221 | **0.1647 ± 0.0233** | 0.0681 ± 0.0147 | 0.0794 ± 0.0171 |

Every QSVM cell straddles the random baseline; every classical cell is 1.5–2.5×
above it.

### 2.4 Significance

Paired t-test over the 5 outer-fold macro-F1 values, for every quantum-classical
pair with a complete fold group.

**All 84 testable pairs favour classical, and all 84 reach p < 0.05.** The worst
p-value across every pair is 0.0241; the best is 4 × 10⁻⁶.

The Wilcoxon signed-rank test returned exactly 0.0625 for all 84 pairs. This is
**not** a null result: with 5 paired samples, 0.0625 is the smallest attainable
two-sided p-value, so the test cannot reach 0.05 at this fold count regardless of
effect size. All 84 hitting that floor means every pair achieved the most extreme
rank configuration available. Any future work wanting a non-parametric test
should raise the fold count to 6, which puts 0.03125 in reach.

**CIC binary is absent from this analysis and from §2.** Every CIC binary run —
classical and quantum alike — predates nested experiment-tracking and carries no
recoverable sweep boundary. Earlier work reported QSVM at 0.92–1.00 macro-F1
there, essentially tied with classical, because CIC's binary task is near-ceiling
for every model. That directional reading stands on the earlier report; it could
not be significance-tested from the logged data, and no grouping was reconstructed
after the fact.

### 2.5 Capacity scaling

Both frameworks improve monotonically with `n_components` on every
dataset/task combination. The rates differ, and the difference is the point.

On EMBER binary, QSVM gains +0.055 macro-F1 going from 1 to 6 components
(0.4587 → 0.5138) while `random_forest` gains +0.125 (0.5582 → 0.6835). **The gap
widens with qubit count rather than closing.** Additional qubits are not buying
the quantum model the representational headroom the classical model extracts from
the same additional components.

---

## 3. Encoding comparison

Three feature maps were implemented: `angle` (low entanglement), `iqp`
(ZZ-interaction), and `amplitude`. The comparison that matters is `angle` vs
`iqp` on the 15-class task, and its verdict is **dataset-dependent** — which is
itself the finding.

**On EMBER 15-class, `iqp` wins clearly.** +0.0889 at `nc=3` (0.3908 → 0.4797)
and +0.0963 at `nc=6` (0.4232 → 0.5195). Both deltas sit well outside the
per-encoding fold-to-fold standard deviation (~0.03–0.06), so the advantage is
real and large.

**On CIC 15-class, `iqp` is ahead everywhere but thinly.** Measured at `nc ∈
{1,3,6,8}`: +0.0009, +0.0122, +0.0113, +0.0046. At every `nc` the delta is
**smaller than either encoding's own fold-to-fold standard deviation**
(0.006–0.018), and the advantage does not grow with qubit count — it peaks at
`nc=3`, not at the highest `nc` tested. What the data supports is that `iqp` has
never been worse on CIC, not that it is meaningfully better.

A methodological note worth carrying into the paper: comparing encodings by
counting which one an inner-CV tuner selected per fold was tried and **retired**.
It conflates "which was picked when both were live options under one joint
budget" with "which performs better", and produced an ambiguous 8/15 win count on
data where the direct macro-F1 deltas showed `iqp` ahead at every single `nc`.
Encoding comparisons use dedicated single-encoding runs and report macro-F1
deltas with their standard deviations.

`amplitude` packs 2ⁿ features into n qubits and yields a near-classical kernel;
it was not carried into the headline comparisons. Basis encoding was deliberately
not implemented: it maps discrete values onto computational basis states, but the
pipeline's inputs are continuous PCA components, so it would require a
quantisation step that discards exactly what PCA was used to retain.

---

## 4. The collapse investigation

QSVM's failure on CIC 15-class is total — it sits at the random baseline — while
on EMBER 15-class the same method reaches 0.42–0.52. Two measurements narrow the
cause, and they point at two distinct effects.

### 4.1 Feature geometry

Measured on the identical projected features both frameworks consume, fit on
training folds only:

- EMBER's families are ~8× more separable by one-vs-rest Fisher ratio (0.57 vs
  0.070 at `nc=1`), and up to 50× on the worst-case family.
- CIC's silhouette coefficient is **negative at every `n_components`**: the
  average CIC point lies closer to some other family's centroid than to its own.
  EMBER's climbs through zero and turns positive.
- CIC's PCA explains substantially *more* variance (33–81%) than EMBER's (8–30%)
  while separating families far less, so CIC's variance is concentrated in
  directions that do not discriminate families — consistent with a
  memory-forensics feature vector whose process and handle counts track workload
  rather than malware family.

This explains why every method does badly on CIC 15-class. **It cannot explain
the quantum-specific failure**, because a classical SVM extracts 0.108–0.165
macro-F1 from exactly these features where QSVM extracts 0.056–0.079.

### 4.2 Kernel geometry

The decisive experiment builds a classical RBF Gram on the **identical** feature
matrix, which separates "these features are hard" from "this kernel is bad".

**Kernel-target alignment separates the datasets, not the kernels.** Alignment
must be read against its floor: a constant kernel scores 1/√15 ≈ 0.258 on 15
balanced classes. On CIC the fidelity kernel lands 0.000069 *below* that
baseline — indistinguishable from an all-ones matrix — and the RBF control is
below it too. On EMBER both clear it by ~0.03.

**Kernel concentration separates the kernels, specifically on CIC.** Off-diagonal
standard deviation of the Gram:

| | CIC | EMBER |
|---|---|---|
| Fidelity kernel | 0.0886 | 0.2015 |
| RBF control | 0.2212 | 0.2184 |
| ratio | **0.40×** | 0.92× |

The RBF kernel produces near-identical spread on both datasets, so CIC's features
do not make a classical kernel concentrate. The fidelity kernel tracks RBF on
EMBER but collapses to 0.40× of its spread on CIC — every off-diagonal similarity
squeezed into a band 2.5× narrower from the same inputs. The Gram approaches a
scaled identity, and a support-vector machine given a near-constant Gram has
almost nothing to separate on. This is what a macro-F1 pinned to the random
baseline looks like from the inside.

The pattern is not an artifact of one setting: across the full sweep, the CIC
fidelity Gram is 1.9–2.6× less spread than EMBER's at every `n_components` and
both encodings, and concentration worsens with qubit count on both — the standard
kernel-concentration trend that the `bandwidth` parameter exists to counteract.

### 4.3 What this supports, and at what confidence

Supported: **the fidelity kernel's failure on CIC is a property of the kernel,
not of the dataset.** A classical kernel on the same features does not
concentrate; the quantum one does.

Not supported: that tuning `bandwidth` would fix it. `bandwidth` is the named
mechanism and it has never been tuned on CIC beyond its default, but no sweep was
run. This is a hypothesis with an identified lever, not a measured result.

One tension is stated rather than resolved. §4.2 finds that neither kernel's Gram
carries label information on CIC by the alignment measure, yet a classical SVM
nonetheless reaches 0.16 there. Kernel-target alignment is a global unweighted
average over all pairs, and an SVM needs no such thing — it needs support
vectors, a regularisation constant and class weights, none of which alignment
models. Alignment is a useful measure of relative dataset difficulty and an
unreliable predictor of achievable SVM performance. The concentration result does
not share this weakness, because it describes the Gram the SVM actually receives.

---

## 5. Limitations

**Subsample scale.** Every quantum result comes from 1000 rows, forced by the
kernel's quadratic cost, against source datasets of ~29K (CIC, malware-only) and
200K (EMBER) rows. The subsamples were verified faithful, but a faithful small
sample is still a small sample: all reported standard deviations are over 5 folds
of 200 test rows each.

**Two datasets, not three.** SOREL-20M was closed without any sweep. Its
multiclass labelling scheme was designed and validated, but its feature store —
a single 71.6 GiB memory-mapped LMDB with no key-level remote access, so the
entire file must be transferred before one row is readable — was never fetched.
Generalisation claims are bounded to CIC-MalMem and EMBER.

**Simulator only.** All results use PennyLane's `lightning.qubit` state-vector
simulator. No hardware execution, no noise model. Concentration behaviour on
real devices, where sampling noise adds its own floor, is untested.

**QSVM only.** No variational quantum classifier was implemented. The comparison
is fidelity-kernel QSVM against classical baselines, not "quantum methods"
against classical ones.

**No sample-level test between frameworks.** McNemar's test requires per-sample
predictions from both models on a shared test set. QSVM predictions were not
persisted during the sweeps reported here, so quantum-vs-classical comparisons
rest on 5 paired fold-level scores rather than 1000 paired sample-level outcomes.
Persistence has since been added, but no sweep was re-run, so nothing is
recoverable retroactively for these results.

**Fold count.** Five outer folds is enough for the paired t-test and too few for
Wilcoxon, whose minimum attainable p-value at n=5 is 0.0625. Six folds would
remove that ceiling.

**One CIC configuration untested for significance.** CIC binary — the only
configuration where the quantum model was ever competitive — could not be
significance-tested, because those runs predate nested experiment tracking and
their sweep boundaries are unrecoverable.

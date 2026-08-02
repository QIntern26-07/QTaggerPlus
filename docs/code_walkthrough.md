# QTagger+ code walkthrough

A guided tour of the codebase for someone who has not worked in it before.
Written to be read top to bottom once, then used as a reference.

Line references are `file.py:NN` and were accurate at the time of writing; if
one drifts, search for the function name instead.

---

## 0. What this project is, in one paragraph

QTagger+ compares **classical machine-learning baselines** against a **quantum
support-vector machine (QSVM)** on malware classification. The whole point is
that the comparison be *fair* — same rows, same cross-validation folds, same
number of input features, same metrics. Almost every design decision in this
repo exists to protect that fairness. If you understand nothing else, understand
that: most of the code is not "run a model", it is "make sure the two sides are
measured identically".

Two datasets, two tasks:

| | binary task | 15-class task |
|---|---|---|
| **CIC-MalMem-2022** | benign vs malware, ~58K rows, 55 features | malware family, malware-only rows |
| **EMBER 2018** | benign vs malware, 200K rows, 2,381 features | `avclass` family, balanced |

---

## 1. The mental model: one experiment, end to end

Before looking at any file, hold this picture in your head. This is what happens
when you run one quantum experiment and its classical counterpart.

```
                    raw dataset (CSV or parquet)
                              |
                     [1] task_xy(df, task)
                          pick the rows this task uses
                              |
                     [2] stratified subsample -> 1000 rows
                          (quantum only; kernel cost is O(n^2))
                          SAVED TO DISK: data/splits/*.json
                              |
                     [3] make_outer_folds -> 5 stratified folds
                          SAVED TO DISK: data/splits/*.json
                              |
              +---------------+---------------+
              |                               |
       for each fold:                  for each fold:
              |                               |
   [4] build_feature_pipeline        [4] build_feature_pipeline
       variance -> corr -> scale         (identical, same n_components)
       -> PCA(n_components)
              |                               |
   [5] EncodingScaler                 [5] (nothing — classical uses
       map into [0, pi]                    the PCA output directly)
              |                               |
   [6] QSVM: build Gram matrix        [6] Optuna tunes RF/XGB/LGBM/SVM
       via quantum circuit                 inside the training fold
       -> SVC(kernel="precomputed")        -> refit on full training fold
              |                               |
              +---------------+---------------+
                              |
                     [7] compute_metrics on the test fold
                              |
                     [8] log to MLflow (one parent run per sweep,
                          one child run per fold)
```

**Step 2 and 3 are why this project works.** The quantum side runs first, writes
its subsample indices and fold indices to `data/splits/`, and the classical side
then *replays* them with `--load-quantum-splits`. Without that, the two sides
would be scored on different rows and every comparison would be meaningless.

---

## 2. Repository layout

```
src/
  common/      framework-agnostic. Both pipelines depend on this.
  classical/   RF / XGBoost / LightGBM / SVM baselines
  quantum/     fidelity-kernel QSVM built on PennyLane
scripts/       one-off drivers: data prep, analysis, exports
tests/         144 tests, mirroring src/ file by file
docs/reports/  week-by-week findings (read these for RESULTS, not code)
data/splits/   committed JSON — the contract between the two pipelines
```

`pyproject.toml` sets `pythonpath = ["src"]`, so `import common`, `import
quantum`, `import classical` work without installing anything.

**Everything runs through `uv run`.** Never a bare `python` or `pytest` — the
project pins its own environment.

---

## 3. Suggested reading order

If you read the files in this order, each one only depends on things you have
already seen:

1. `src/common/data.py` — what a "task", a "fold", a "subsample" mean here
2. `src/common/preprocess.py` — the shared feature pipeline (short, important)
3. `src/common/evaluate.py` — metrics
4. `src/quantum/encoding.py` — how classical numbers become a quantum circuit
5. `src/quantum/qsvm.py` — the kernel and the estimator
6. `src/quantum/run.py` — cross-validation for the quantum side
7. `src/classical/run.py` — the classical mirror image of the above
8. `src/common/tracking.py` — MLflow logging
9. `src/common/significance.py` — how results are compared afterwards

Sections 4–9 below follow that order.

---

## 4. `src/common/` — the shared core

### 4.1 `data.py` (278 lines) — the single source of truth for rows and folds

This is the most important file to understand, and the one where a subtle
mistake does the most damage.

**`task_xy(df, task, dataset)` — `data.py:140`.** Returns the `(X, y)` that a
given task operates on. Read the docstring; the rules differ per dataset:

- `cic` binary: all rows, benign-vs-malware label.
- `cic` multiclass: **malware rows only**, 15-class family label. Benign is
  dropped — detecting benign is the binary task's job, and keeping it makes the
  label space pathologically imbalanced.
- `ember` binary: all rows, `label` column.
- `ember` multiclass: `ember_family_xy` — keeps families with ≥ 500 samples, caps
  at the top 15, then **downsamples every family to the smallest kept count** so
  the pool is exactly balanced.

That last one has a consequence worth internalising: **`ember_family_xy`
re-derives its row set every time it is called.** Feed it a different input
frame and you get a different pool, in a different order. Persisted row indices
computed against one frame are meaningless against another. This is a real trap
— see §10.

**Folds and subsamples — `data.py:187-243`.**

- `split_paths(dataset, task)` returns the two JSON paths for a
  `(dataset, task)` pair. Always ask this function; never hardcode a filename.
- `make_outer_folds(y, n_splits, seed)` → list of `(train_idx, test_idx)`.
- `save_folds` / `load_folds`, `save_sample_idx` / `load_sample_idx`.

The JSON files under `data/splits/` **are committed to git** even though they are
generated. They are the contract between the two pipelines, so they must be
identical for everyone.

**Predictions — `data.py:245-277`.** `save_predictions(records, path)` takes the
per-fold record dicts and flattens them into one `.npz` with four parallel
arrays: `fold_ids`, `test_idx`, `y_true`, `y_pred`. `load_predictions` reads it
back. This is what makes sample-level statistical tests (McNemar) possible.

### 4.2 `preprocess.py` (64 lines) — the alignment point

Short file, outsized importance.

```python
steps = [
    ("variance",    VarianceThreshold(threshold=variance_threshold)),
    ("decorrelate", DropCorrelated(threshold=corr_threshold)),   # custom, :16
    ("scale",       StandardScaler()),
]
if n_components is not None:
    steps.append(("pca", PCA(n_components=n_components, random_state=seed)))
```

`build_feature_pipeline()` at `preprocess.py:42`. Four stages: drop
zero-variance columns, drop one of each highly correlated pair (>0.95),
standardise, then optionally compress to `n_components` principal components.

**`n_components` is the single alignment point between classical and quantum.**
It simultaneously sets:

- how many features the classical models see, and
- how many **qubits** the quantum circuit uses.

So "nc=6" means both sides get 6 numbers per sample. That is what makes the
comparison apples-to-apples, and it is why you will see `--n-components` on
every CLI invocation in the docs.

**Leakage rule:** the pipeline is always `fit` on training-fold rows only, then
`transform` applied to the test fold. Never fit on the whole dataset. If you add
a new experiment, copy this discipline.

### 4.3 `evaluate.py` (106 lines) — metrics

`compute_metrics(y_true, y_pred, y_proba, task)` at `:23` returns accuracy,
precision, recall, `f1_macro`, `f1_weighted`, `mcc`, `roc_auc`.

**Why macro-F1 is the headline and not accuracy.** A concrete example from this
project: on EMBER binary at nc=1, the QSVM reached **0.890 accuracy in three of
five folds while the benign class's F1 was exactly 0.000** in those same folds.
The model labelled everything malware and was right 89% of the time because 89%
of rows are malware. Accuracy reported near-success for a classifier that never
once found the class the task exists to find. Macro-F1 averages per-class F1, so
a class you never predict drags it down.

Other helpers: `per_class_f1` (`:42`), `aggregate_metrics` (`:57`, computes
mean/std across folds), `timed` (`:76`, wraps a call and returns
`(result, seconds)`), `auc_scores` (`:91`, handles models with `predict_proba`
vs. only `decision_function`).

### 4.4 `tracking.py` (47 lines) — MLflow

One context manager, `tracking.run(...)` at `:27`:

```python
with tracking.run("qtaggerplus", "my-run", params, tags={...}) as log:
    log.log_metrics({"f1_macro": 0.5})
    log.log_figure(fig, "confusion_matrix.png")
```

Local file store, no login, no network. `nested=True` makes the run a **child**
of the currently active run. That parent/child structure matters enormously for
reading results later — see §9.

---

## 5. `src/quantum/` — the QSVM

### 5.1 `encoding.py` (62 lines) — classical numbers → quantum circuit

Three feature maps behind one entry point, `feature_map(x, wires, encoding,
bandwidth)` at `:49`:

- **`angle`** — one feature per qubit as a Y-rotation, a ring of CNOTs, then
  Z-rotations. Low entanglement; the simple baseline.
- **`iqp`** — `qml.IQPEmbedding`, a ZZ-interaction map. Entangled, conjectured
  classically hard, empirically the best of the three on malware data.
- **`amplitude`** — packs 2ⁿ features into n qubits. The resulting kernel is
  near-classical.

`n_qubits_for(encoding, n_components)` at `:37`: for `angle` and `iqp` it is
just `n_components` (one feature per qubit); for `amplitude` it is
`ceil(log2(n_components))`.

**`bandwidth`** at `:44`, default `n_qubits ** -0.5`. This scales the inputs
before rotation. It exists because fidelity kernels **concentrate** as qubit
count grows — every pair of points starts looking equally similar, the Gram
matrix approaches a scaled identity, and the SVM has nothing to separate on.
Bandwidth counteracts that. This is not a tuning knob someone added for fun; it
is load-bearing, and Week 5's Day 33 report found concentration to be the
mechanism behind the QSVM's failure on CIC 15-class.

Basis encoding is deliberately **not** implemented: it maps discrete values to
computational basis states, but the pipeline's inputs are continuous PCA
components, so it would need a quantisation step that throws away exactly what
PCA was used to keep.

### 5.2 `preprocess.py` (41 lines) — `EncodingScaler`

Runs *after* PCA, *before* the circuit. For `angle` and `iqp` it min-max scales
each feature into `[0, π]` (rotation angles); for `amplitude` it pads or
truncates to 2ⁿ and L2-normalises (amplitudes must form a unit vector).

Kept separate from `common/preprocess.py` so the QSVM never re-runs PCA. Fit on
the training fold only, same as everything else.

### 5.3 `qsvm.py` (205 lines) — the kernel and the estimator

**The quantum part is 5 lines.** Everything else is bookkeeping and speed.

```python
@qml.qnode(self._dev)                       # qsvm.py:103
def kernel(x1, x2):
    feature_map(x1, self._wires, encoding, self.bandwidth)
    qml.adjoint(feature_map)(x2, self._wires, encoding, self.bandwidth)
    return qml.probs(wires=self._wires)
```

Encode `x1`, then apply the *inverse* of encoding `x2`. If the two states are
identical the circuit returns to |0…0⟩ and the probability of measuring all
zeros is 1. If they are orthogonal it is 0. So `probs[0]` **is** the fidelity
|⟨φ(x₂)|φ(x₁)⟩|², which is the kernel value. That single number is the entire
quantum contribution to this project.

The rest of the class:

| Method | Line | What it does |
|---|---|---|
| `_kernel_pairs(X1, X2)` | `:117` | Evaluate the kernel for paired rows. Splits large jobs across processes — `lightning.qubit` is single-threaded on circuits this small, so the O(n²) pair count parallelises cleanly. Output is identical to the serial path. |
| `gram(A, B)` | `:153` | Full rectangular Gram matrix (used for test-vs-train) |
| `_gram_sym(A)` | `:165` | Symmetric Gram (training). Computes only the upper triangle and mirrors it — half the work |
| `fit(X, y)` | `:177` | Build the training Gram, then `SVC(kernel="precomputed")` |
| `predict(X)` | `:201` | Build the test Gram against stored training rows, then `SVC.predict` |

**`gram_offdiag_std` at `:186`** — computed on every `fit` and logged to MLflow.
It is the standard deviation of the off-diagonal Gram entries, i.e. the
concentration health check described above. Because it has been logged since the
beginning, the entire Week 5 concentration analysis needed **zero** new
computation.

**Cost model, so you can predict runtimes:** a training Gram on n rows needs
n(n−1)/2 circuit evaluations. At n=800 that is ~320,000. This is why every
quantum run uses a 1000-row subsample and not the full dataset — the cost is
quadratic, not linear.

### 5.4 `run.py` (224 lines) — quantum cross-validation

**`tune_and_fit_qsvm` at `:49` — the two-tier trick.** Naively, tuning would
rebuild the Gram for every hyperparameter combination. But the quantum kernel
depends only on `(encoding, bandwidth)` — **not** on the SVM's `C` or
`class_weight`. So:

```
for encoding:
    for bandwidth:
        build Gram ONCE                 <- expensive, quantum
        for C:
            for class_weight:
                fit SVC on that Gram    <- cheap, classical
```

That turns a 12-combination search into 2 Gram builds instead of 12.

**`evaluate_fold_quantum` at `:92`** runs one outer fold and returns a record
dict containing metrics, timings, chosen hyperparameters, `gram_offdiag_std`,
and `test_idx` / `y_true` / `y_pred`.

**`run_quantum_cv` at `:131`** loops the outer folds and handles MLflow: one
**parent** run for the sweep holding mean/std aggregates, one nested **child**
run per fold. Remember this shape — §9 depends on it.

**`timing_probe` at `:200`** — a single untuned fit, no CV, to measure
wall-clock before committing to a full sweep. Use it. A sweep that turns out to
cost six hours is much better discovered in ninety seconds.

### 5.5 `__main__.py` (136 lines) — the `python -m quantum` CLI

Order of operations: load dataset → `task_xy` → stratified subsample to
`--max-samples` → **save sample indices** → build folds → **save folds** → run
CV → **save predictions**.

`predictions_path` at `:42` names the output
`results/<dataset>/qsvm_<task>_nc<N>_<encoding>_predictions.npz`. Note
`n_components` is in the filename: a sweep at a different qubit budget is a
different result, not an overwrite of this one.

---

## 6. `src/classical/` — the baselines

Deliberately the mirror image of `src/quantum/`, so the two are comparable by
inspection.

**`models.py` (102 lines).** `make_model(name, params, task, seed, n_jobs)` at
`:31` builds one of `random_forest`, `xgboost`, `lightgbm`, `svm`.
`suggest_params(name, trial)` at `:66` defines each model's Optuna search space.
Add a model in these two functions and the rest of the pipeline picks it up.

`BalancedXGBClassifier` at `:14` exists because XGBoost has no `class_weight`
parameter; this subclass computes sample weights at `fit` time instead.

**`run.py` (182 lines).** Same shape as the quantum runner:

- `tune_and_fit` (`:38`) — Optuna inner loop *inside* the training fold
- `evaluate_fold` (`:74`) — fit pipeline on train, transform test, score
- `run_nested_cv` (`:116`) — loop outer folds, log parent + child runs

"Nested CV" means: an inner loop tunes hyperparameters using only training-fold
data, then the tuned model is refit and scored once on the held-out test fold.
The test fold never influences tuning. That is the whole reason for the nesting.

**`compare.py` (37 lines).** Three statistical tests:
`paired_ttest(a, b)` and `wilcoxon(a, b)` compare two models' per-fold scores;
`mcnemar(y_true, pred_a, pred_b)` compares two models' per-sample predictions on
a shared test set.

---

## 7. `scripts/` — drivers

Scripts are thin: they load data, call functions from `src/`, and print. **All
the real logic lives in `src/` where it is unit-tested.** If you find yourself
writing an algorithm inside a script, it probably belongs in `src/common/`.

| Script | Purpose |
|---|---|
| `download_cic.py` | fetch the CIC dataset (needs a Kaggle token) |
| `prepare_ember.py`, `make_ember_subset.py` | build the EMBER parquets |
| `export_mlflow_runs.py` | MLflow → `results/mlflow_runs.csv` |
| `compare_class_geometry.py` | Day 31: separability + subsample fidelity |
| `kernel_diagnostics.py` | Day 33: concentration and alignment vs. an RBF control |
| `run_significance_tests.py` | Day 34: paired tests over all model pairs |
| `export_week5_csv.py` | derived results → seven CSVs |
| `profile_quantum_day1.py` | early circuit profiling |

---

## 8. `tests/` — 144 tests

One test file per source module. The convention is: **pure functions in `src/`
get unit tests; scripts do not.** That split is intentional — scripts are I/O
shells around tested logic.

Run them:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_qsvm.py -v
```

**You need `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.** A system-wide ROS 2
`launch_testing` plugin hijacks bare `pytest` on this machine. Without the flag
you get a confusing collection error that has nothing to do with this project.

The tests are also the fastest way to learn an unfamiliar module — each one is a
worked example with the expected answer written next to it.

---

## 9. Reading results: MLflow's parent/child structure

This trips up everyone, so it gets its own section.

Every sweep produces:

- **one parent run** — carries `f1_macro_mean`, `f1_macro_std` (aggregates)
- **five child runs** — one per fold, each carrying `f1_macro` (no `_mean`)

`scripts/export_mlflow_runs.py` flattens all of it into
`results/mlflow_runs.csv`, keeping `run_id`, `parent_run_id` and `status`.

**Now the important part.** You might reasonably think you can find a result by
filtering on `(dataset, task, n_components, model, encoding)`. **You cannot**, for
two reasons:

1. **The same cell has been swept several times** over the project's weeks. A
   param-based filter returns 10, 15 or 20 rows for a 5-fold sweep and averages
   distinct hyperparameter searches together. 57 param-groups in the export have
   a size other than 5.
2. **`encoding` is a per-fold *outcome* for QSVM, not a cell key.** The two-tier
   tuner picks the winning encoding *inside each fold*, so one joint sweep can
   appear as "4 angle folds + 1 iqp fold". Grouping by encoding splits a single
   sweep in two.

The correct rule lives in **`src/common/significance.py`**:

- `clean_sweep_folds(df)` at `:45` — keep only folds whose **parent is
  `FINISHED` with exactly 5 children**
- `fold_scores(df, dataset, task, nc, model, encoding=None)` at `:76` — return
  one sweep's 5 per-fold scores, most recent clean sweep wins

Use these. Do not re-implement the rule — a second copy that drifts from this
one would make the CSVs and the significance tests silently disagree.

Under this rule the store holds **125 nested sweeps, 123 of them clean**. The
two exceptions are an interrupted sweep whose parent stayed `RUNNING`, and a
truncated 3-fold sweep from a past out-of-memory incident. Both are excluded
automatically rather than deleted, so history stays intact.

**230 of 850 fold rows have no `parent_run_id` at all** — they predate nested
logging (early July). They cannot be grouped into sweeps. That is why no CIC
binary comparison appears in the Week 5 significance tables.

---

## 10. Traps — read this before running anything

These have each cost real time. They are not hypothetical.

### 10.1 Always pass `--csv` for EMBER

The default is the full `ember2018_test.parquet`: 200,000 × 2,384 in a **single
row group**, which cannot be streamed and loads at ~1.9 GB resident.

Worse, it is a *correctness* problem. The persisted sample indices were computed
against the 18,014-row subset, and `ember_family_xy` re-derives its pool from
whatever frame it is given. Apply subset indices to the full frame and you
select rows with different `sha256` values and different labels.

**Always:** `--csv data/ember/ember2018_quantum_subset.parquet`

### 10.2 Always cap `--n-jobs`

The default `-1` takes every core. On a 20-core machine that spawns 20 worker
processes, each holding its own copy of the fold data. This exhausted the
machine mid-sweep during Week 5. `--n-jobs 4` measured 879 MB peak RSS and zero
swapping.

### 10.3 The classical CLI overwrites its own outputs

- `classical/__main__.py:119` names predictions
  `{model}_{task}_predictions.npz` — **no `n_components`**. Run nc=1 then nc=3
  and nc=1's file is gone. Work around it with `--predictions-dir results/<ds>/nc<N>`.
- `classical/__main__.py:85` / `:124` — the metrics CSV is rewritten wholesale
  every invocation. Give each run its own `--out`.

Both are known and documented rather than fixed, because changing the path
template changes how existing files should be interpreted.

### 10.4 Running the quantum CLI overwrites `data/splits/`

Those JSON files are committed and shared. A quick smoke test with
`--max-samples 40` will silently replace the real 1000-row indices. If you run a
throwaway experiment, restore them afterwards:

```bash
git checkout -- data/splits/
```

### 10.5 Run quantum first, then classical

```bash
# 1. quantum writes the subsample + folds
uv run python -m quantum --tasks binary --n-components 3 --folds 5 --mlflow

# 2. classical replays exactly those rows and folds
uv run python -m classical --tasks binary --n-components 3 \
    --load-quantum-splits --mlflow
```

Never compare a full-dataset classical run against a subsampled quantum run.

---

## 11. Common commands

```bash
uv sync                                   # install dependencies

# time one quantum fit before committing to a sweep
uv run python -m quantum --probe --n-components 1 --max-samples 120 \
    --encodings angle iqp

# a fast classical smoke test
uv run python -m classical --models random_forest --tasks binary \
    --folds 3 --trials 3

uv run mlflow ui                          # browse runs at 127.0.0.1:5000
uv run python scripts/export_mlflow_runs.py

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest
```

---

## 12. How to make common changes

**Add a classical model.** Add a branch to `make_model` (`classical/models.py:31`)
and a search space to `suggest_params` (`:66`). Add a case to
`tests/test_models.py`. Nothing else changes.

**Add a quantum encoding.** Add a branch to `feature_map`
(`quantum/encoding.py:49`), update `n_qubits_for` if the qubit count differs from
`n_components`, and add scaling to `EncodingScaler`
(`quantum/preprocess.py:25`) if the input range differs. Add it to `ENCODINGS`
at `:34` and test it in `tests/test_encoding.py`.

**Add a metric.** Add it to the dict in `compute_metrics`
(`common/evaluate.py:23`). Note that `tests/test_evaluate.py` asserts an *exact*
key set, so that test must be updated in the same change — this is expected, not
a failure.

**Add an analysis.** Put the computation as pure functions in `src/common/`,
unit-test them, then write a thin driver in `scripts/`. Follow
`geometry.py` + `compare_class_geometry.py` as the template.

**Add a dataset.** Add a loader and a `task_xy` branch in `common/data.py`, a
`split_paths` branch at `:187`, and a CLI branch in both `__main__.py` files.
`sorel` is a half-finished worked example of exactly this.

---

## 13. Where to look next

- **Results, not code:** `docs/reports/` — the week-by-week findings.
  `w5_consolidated_report.md` is the best single overview.
- **Machine-readable results:** `docs/reports/logs/w5_csv/` — seven CSVs.
- **Project conventions:** `CLAUDE.md` at the repo root.
- **Methodology rationale:** `w5_day31_*` and `w5_day33_*` explain why the
  geometry and kernel diagnostics exist and what they found.

If something in this document contradicts the code, the code is right — please
fix the document.

"""Select comparable per-fold scores out of the flat MLflow export.

The hard part is not the statistics — `classical.compare` already provides
those — it is deciding which rows constitute ONE cross-validation run.

Filtering by `(dataset, task, n_components, model, encoding)` does not work, and
this was measured rather than assumed (see `w5_day29_ember_four_model.md` §7):

  * The same cell has legitimately been swept several times over the project's
    weeks. A param-based groupby returns 10, 15 or 20 rows for a 5-fold sweep and
    averages distinct hyperparameter searches together. 57 param-groups in the
    export have a size other than 5, most of them exact multiples of 5.
  * `encoding` is a per-fold OUTCOME for QSVM, not a cell key. The two-tier tuner
    in `quantum/run.py` picks the winning encoding inside each fold and
    `_log_quantum_fold` logs that choice, so one joint sweep can appear as
    "4 angle folds + 1 iqp fold". The sweep-level intent lives on the PARENT row.

So a sweep is identified by `parent_run_id`, and only parents that are FINISHED
with exactly `expected_folds` children are trusted. That rule also excludes folds
orphaned by an interrupted sweep — the contamination
`w4_consolidated_report.md` §6 had to clean by hand.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _sweep_encoding(parent_row, child_rows) -> str | None:
    """The encoding a sweep was configured with.

    Prefers the parent's `params.encoding` ("angle", "iqp" or "joint"). Older
    sweeps predate that param, so fall back to the children: if every fold chose
    the same encoding the sweep was single-encoding, otherwise it was joint.
    """
    tag = parent_row.get("params.encoding")
    if isinstance(tag, str) and tag:
        return tag
    encodings = sorted(set(child_rows["params.encoding"].dropna()))
    if not encodings:
        return None
    return encodings[0] if len(encodings) == 1 else "joint"


def fold_scores(df: pd.DataFrame, dataset: str, task: str, n_components,
                model: str, encoding: str | None = None,
                expected_folds: int = 5,
                metric: str = "metrics.f1_macro") -> np.ndarray:
    """Per-fold `metric` values for ONE sweep, in fold order.

    When several clean sweeps match the cell, the most recently started one wins,
    deterministically. Raises `ValueError` when no clean sweep exists, so the
    exception means "nothing trustworthy here" rather than "this cell was run
    more than once".
    """
    required = {"run_id", "parent_run_id", "status", "start_time", metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"export is missing {sorted(missing)}; re-run "
            "scripts/export_mlflow_runs.py (it used to drop run_id and "
            "tags.mlflow.parentRunId, which makes per-sweep grouping impossible)"
        )

    parents = df.set_index("run_id")
    children = df[
        df["parent_run_id"].notna()
        & df[metric].notna()
        & (df["params.dataset"] == dataset)
        & (df["params.task"] == task)
        & (df["params.n_components"].astype(str) == str(n_components))
        & (df["params.model"] == model)
    ]

    candidates = []
    for parent_id, group in children.groupby("parent_run_id"):
        if parent_id not in parents.index:
            continue
        parent = parents.loc[parent_id]
        if parent["status"] != "FINISHED" or len(group) != expected_folds:
            continue
        if encoding is not None and _sweep_encoding(parent, group) != encoding:
            continue
        candidates.append((parent["start_time"], parent_id, group))

    if not candidates:
        raise ValueError(
            f"{dataset}/{task}/nc={n_components}/{model}"
            f"{'/' + encoding if encoding else ''}: no FINISHED sweep with "
            f"exactly {expected_folds} fold rows"
        )
    _, _, group = max(candidates, key=lambda c: c[0])
    return group.sort_values("start_time")[metric].to_numpy(dtype=float)

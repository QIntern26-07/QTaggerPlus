import numpy as np
import pandas as pd
import pytest

from common import significance


def _parent(run_id, start, status="FINISHED", encoding=None):
    return {
        "run_id": run_id, "parent_run_id": np.nan, "status": status,
        "start_time": start, "params.dataset": "cic-malmem",
        "params.task": "binary", "params.n_components": 3,
        "params.model": "svm", "params.encoding": encoding,
        "metrics.f1_macro": np.nan,
    }


def _child(run_id, parent_id, f1, start, model="svm", encoding=None):
    return {
        "run_id": run_id, "parent_run_id": parent_id, "status": "FINISHED",
        "start_time": start, "params.dataset": "cic-malmem",
        "params.task": "binary", "params.n_components": 3,
        "params.model": model, "params.encoding": encoding,
        "metrics.f1_macro": f1,
    }


def _sweep(parent_id, base, scores, status="FINISHED", model="svm",
           encoding=None, child_encodings=None):
    rows = [_parent(parent_id, f"2026-07-01T00:0{base}:00", status, encoding)]
    rows[0]["params.model"] = model
    for i, f1 in enumerate(scores):
        enc = child_encodings[i] if child_encodings else encoding
        rows.append(_child(f"{parent_id}c{i}", parent_id, f1,
                           f"2026-07-01T00:0{base}:0{i + 1}", model, enc))
    return rows


def test_returns_one_sweeps_folds_in_start_time_order():
    df = pd.DataFrame(_sweep("p1", 1, [0.3, 0.1, 0.2], ))
    scores = significance.fold_scores(df, "cic-malmem", "binary", 3, "svm",
                                      expected_folds=3)
    # ordered by the child's own start_time, not by the order rows appear
    assert np.allclose(scores, [0.3, 0.1, 0.2])


def test_two_clean_sweeps_of_one_cell_return_the_later_sweep_only():
    # The same cell has legitimately been swept more than once across weeks.
    # A param-based filter would return 6 values and silently average two
    # distinct hyperparameter searches together.
    df = pd.DataFrame(_sweep("old", 1, [0.1, 0.1, 0.1])
                      + _sweep("new", 5, [0.9, 0.9, 0.9]))
    scores = significance.fold_scores(df, "cic-malmem", "binary", 3, "svm",
                                      expected_folds=3)
    assert len(scores) == 3
    assert np.allclose(scores, [0.9, 0.9, 0.9])


def test_children_of_an_unfinished_parent_are_excluded():
    # An interrupted sweep relaunched under a new parent leaves orphan folds:
    # exactly the contamination w4_consolidated_report.md section 6 cleaned by
    # hand. The parent's status excludes them without deleting history.
    df = pd.DataFrame(_sweep("killed", 1, [0.5, 0.5], status="RUNNING")
                      + _sweep("relaunch", 5, [0.2, 0.4, 0.6]))
    scores = significance.fold_scores(df, "cic-malmem", "binary", 3, "svm",
                                      expected_folds=3)
    assert np.allclose(scores, [0.2, 0.4, 0.6])


def test_a_joint_quantum_sweep_with_mixed_fold_encodings_stays_one_group():
    # The two-tier tuner picks the winning encoding INSIDE each fold, so a joint
    # sweep legitimately logs different per-fold encodings. Grouping on the
    # child's encoding would split one sweep into two incomplete ones.
    df = pd.DataFrame(_sweep("j", 1, [0.1, 0.2, 0.3], model="qsvm",
                             encoding="joint",
                             child_encodings=["angle", "iqp", "angle"]))
    scores = significance.fold_scores(df, "cic-malmem", "binary", 3, "qsvm",
                                      encoding="joint", expected_folds=3)
    assert np.allclose(scores, [0.1, 0.2, 0.3])


def test_encoding_falls_back_to_the_children_when_the_parent_lacks_the_param():
    # Sweeps logged before params.encoding existed on the parent row.
    df = pd.DataFrame(_sweep("old", 1, [0.4, 0.5, 0.6], model="qsvm",
                             encoding=None,
                             child_encodings=["iqp", "iqp", "iqp"]))
    scores = significance.fold_scores(df, "cic-malmem", "binary", 3, "qsvm",
                                      encoding="iqp", expected_folds=3)
    assert np.allclose(scores, [0.4, 0.5, 0.6])


def test_raises_when_no_sweep_has_the_expected_fold_count():
    df = pd.DataFrame(_sweep("short", 1, [0.1, 0.2]))
    with pytest.raises(ValueError, match="no FINISHED sweep"):
        significance.fold_scores(df, "cic-malmem", "binary", 3, "svm",
                                 expected_folds=3)


def test_raises_when_the_export_predates_the_parent_run_id_column():
    df = pd.DataFrame([{"params.dataset": "cic-malmem", "metrics.f1_macro": 0.1}])
    with pytest.raises(ValueError, match="re-run"):
        significance.fold_scores(df, "cic-malmem", "binary", 3, "svm")


def test_clean_sweep_folds_is_the_shared_selection_rule():
    # scripts/export_week5_csv.py builds the master-table CSVs from this same
    # function. Two implementations of the rule would let the CSVs and the
    # significance tests drift apart silently.
    df = pd.DataFrame(_sweep("ok", 1, [0.1, 0.2, 0.3])
                      + _sweep("killed", 5, [0.9, 0.9], status="RUNNING")
                      + _sweep("short", 7, [0.4, 0.5]))
    clean = significance.clean_sweep_folds(df, expected_folds=3)
    assert set(clean["parent_run_id"]) == {"ok"}
    assert len(clean) == 3

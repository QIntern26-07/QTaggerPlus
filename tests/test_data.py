import pandas as pd
import pytest
from common import data


def _toy_df():
    # Category format in CIC-MalMem is like "Ransomware-Conti-<hash>" or "Benign"
    return pd.DataFrame(
        {
            "feat_a": [0.1, 0.2, 0.3, 0.4, 0.5],
            "feat_b": [10, 20, 30, 40, 50],
            "Category": [
                "Benign",
                "Ransomware-Conti-abc",
                "Ransomware-Ako-differenthash",
                "Spyware-180-xyz",
                "Benign",
            ],
            "Class": ["Benign", "Malware", "Malware", "Malware", "Benign"],
        }
    )


def test_build_xy_drops_label_columns():
    X, y_bin, y_multi = data.build_xy(_toy_df())
    assert list(X.columns) == ["feat_a", "feat_b"]
    assert "Category" not in X.columns and "Class" not in X.columns


def test_build_xy_binary_encoding():
    _, y_bin, _ = data.build_xy(_toy_df())
    assert y_bin.tolist() == [0, 1, 1, 1, 0]


def test_build_xy_multiclass_family_prefix():
    _, _, y_multi = data.build_xy(_toy_df())
    # Four distinct families: Benign, Ransomware-Conti, Ransomware-Ako, Spyware-180
    assert y_multi.nunique() == 4
    # Benign rows (0 and 4) share the same class
    assert y_multi.iloc[0] == y_multi.iloc[4]
    # Rows 1 and 2 are both Ransomware but different subfamilies (Conti vs Ako),
    # so they must have different classes (proves two-token extraction, not just first-token)
    assert y_multi.iloc[1] != y_multi.iloc[2]
    # Rows 1 and 3 are from different top-level families
    assert y_multi.iloc[1] != y_multi.iloc[3]


def test_malware_family_xy_drops_benign_rows():
    # Toy df has 2 Benign rows (0, 4) and 3 malware rows (1, 2, 3).
    X_m, y_fam = data.malware_family_xy(_toy_df())
    assert len(X_m) == 3
    assert len(y_fam) == 3
    # feature columns preserved, label columns dropped
    assert list(X_m.columns) == ["feat_a", "feat_b"]
    # the surviving rows are exactly the malware rows (feat_a 0.2, 0.3, 0.4)
    assert X_m["feat_a"].tolist() == [0.2, 0.3, 0.4]


def test_malware_family_xy_reencodes_families_contiguously_from_zero():
    # 3 malware families (Ransomware-Conti, Ransomware-Ako, Spyware-180) must be
    # re-encoded to a contiguous 0..2 with no gap left by the removed Benign class.
    _, y_fam = data.malware_family_xy(_toy_df())
    assert sorted(y_fam.unique().tolist()) == [0, 1, 2]
    # the two distinct Ransomware subfamilies stay distinct
    assert y_fam.iloc[0] != y_fam.iloc[1]


def test_malware_family_xy_resets_index():
    # Downstream code indexes the returned frame positionally (sample_idx, folds),
    # so the index must be 0..n-1, not the original sparse malware-row indices.
    X_m, y_fam = data.malware_family_xy(_toy_df())
    assert X_m.index.tolist() == [0, 1, 2]
    assert y_fam.index.tolist() == [0, 1, 2]


def test_task_xy_binary_keeps_all_rows():
    # Both CLIs route through task_xy so they can't drift on what a task means.
    X, y = data.task_xy(_toy_df(), "binary")
    assert len(X) == 5  # all rows, Benign included
    assert y.tolist() == [0, 1, 1, 1, 0]


def test_task_xy_multiclass_is_malware_only():
    # multiclass must mean the 15-class malware-family task (Benign dropped),
    # identical to malware_family_xy — not the Benign-inclusive build_xy label.
    X, y = data.task_xy(_toy_df(), "multiclass")
    assert len(X) == 3  # Benign rows dropped
    assert sorted(y.unique().tolist()) == [0, 1, 2]


def test_task_xy_rejects_unknown_task():
    with pytest.raises(ValueError):
        data.task_xy(_toy_df(), "regression")

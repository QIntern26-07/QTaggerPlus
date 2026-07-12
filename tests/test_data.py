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

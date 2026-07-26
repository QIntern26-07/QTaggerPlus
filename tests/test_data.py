import numpy as np
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


def _toy_ember_df():
    # Cached EMBER 2018 test parquet layout: F1..Fn feature columns plus
    # lowercase int `label` (0 benign / 1 malicious), `avclass` family string
    # (empty if none), and `sha256`.
    return pd.DataFrame(
        {
            "F1": [0.1, 0.2, 0.3, 0.4],
            "F2": [1.0, 2.0, 3.0, 4.0],
            "label": [0, 1, 1, 0],
            "avclass": ["", "zbot", "ramnit", ""],
            "sha256": ["a", "b", "c", "d"],
        }
    )


def test_load_ember_reads_parquet(tmp_path):
    path = tmp_path / "ember_test.parquet"
    _toy_ember_df().to_parquet(path)
    df = data.load_ember(str(path))
    assert list(df.columns) == ["F1", "F2", "label", "avclass", "sha256"]
    assert len(df) == 4


def test_task_xy_ember_binary():
    X, y = data.task_xy(_toy_ember_df(), "binary", dataset="ember")
    assert list(X.columns) == ["F1", "F2"]
    assert y.tolist() == [0, 1, 1, 0]
    assert y.dtype.kind == "i"


def _toy_ember_family_df():
    # 3 families with counts 6/4/2 and some benign + empty-avclass malware.
    rows = []

    def add(n, label, av):
        for _ in range(n):
            rows.append({"F1": 0.1, "F2": 0.2, "label": label, "avclass": av,
                         "sha256": "x"})
    add(6, 1, "zbot")
    add(4, 1, "ramnit")
    add(2, 1, "sality")     # below a min_per_class=3 threshold -> dropped
    add(3, 1, "")           # malware, empty avclass -> dropped
    add(5, 0, "")           # benign -> dropped from multiclass
    return pd.DataFrame(rows)


def test_ember_family_xy_drops_benign_and_empty_avclass():
    X, y = data.ember_family_xy(_toy_ember_family_df(), min_per_class=3,
                                max_families=15)
    # only zbot(6) and ramnit(4) qualify (>=3); each downsampled to min kept (4)
    assert len(y) == 8
    assert list(X.columns) == ["F1", "F2"]


def test_ember_family_xy_balanced_and_contiguous():
    _, y = data.ember_family_xy(_toy_ember_family_df(), min_per_class=3,
                                max_families=15)
    counts = y.value_counts()
    assert counts.min() == counts.max() == 4          # exactly balanced
    assert sorted(y.unique().tolist()) == [0, 1]       # contiguous from 0


def test_ember_family_xy_caps_family_count():
    # 3 families all >=3, but max_families=2 keeps the two largest (zbot, ramnit).
    _, y = data.ember_family_xy(_toy_ember_family_df(), min_per_class=1,
                                max_families=2)
    assert y.nunique() == 2


def test_ember_family_xy_is_deterministic():
    a = data.ember_family_xy(_toy_ember_family_df(), min_per_class=3, seed=42)[1]
    b = data.ember_family_xy(_toy_ember_family_df(), min_per_class=3, seed=42)[1]
    assert a.tolist() == b.tolist()


def test_task_xy_ember_multiclass_now_returns():
    # task_xy routes to ember_family_xy with its real defaults (min_per_class=
    # 500), so unlike the other family tests above (which override the
    # threshold to exercise small toy data), this needs a fixture where a
    # family actually clears that bar.
    rows = [{"F1": 0.1, "F2": 0.2, "label": 1, "avclass": "zbot", "sha256": "x"}
            for _ in range(500)]
    rows += [{"F1": 0.1, "F2": 0.2, "label": 1, "avclass": "ramnit", "sha256": "x"}
             for _ in range(500)]
    df = pd.DataFrame(rows)
    X, y = data.task_xy(df, "multiclass", dataset="ember")
    assert len(y) > 0
    assert list(X.columns) == ["F1", "F2"]


def test_task_xy_default_dataset_is_cic():
    # Existing callers pass no dataset arg; behavior must be unchanged.
    X, y = data.task_xy(_toy_df(), "binary")
    assert y.tolist() == [0, 1, 1, 1, 0]


def test_split_paths_cic_keeps_legacy_filenames():
    # data/splits/*.json for CIC are committed and shared between the classical
    # and quantum CLIs — renaming them would break the existing contract.
    sample, folds = data.split_paths("cic", "binary")
    assert sample == "data/splits/quantum_sample_idx_binary.json"
    assert folds == "data/splits/cic_binary_quantum_folds.json"


def test_split_paths_ember_is_dataset_prefixed():
    sample, folds = data.split_paths("ember", "binary")
    assert sample == "data/splits/ember_quantum_sample_idx_binary.json"
    assert folds == "data/splits/ember_binary_quantum_folds.json"


def _sorel_frame(n=40, n_features=6):
    rng = np.random.RandomState(0)
    df = pd.DataFrame(rng.rand(n, n_features).astype(np.float32),
                      columns=[f"F{i + 1}" for i in range(n_features)])
    df["sha256"] = [f"{i:064x}" for i in range(n)]
    df["label"] = ([0, 1] * (n // 2))[:n]
    df["dominant_tag"] = (["ransomware", "worm", "adware", "dropper"] * (n // 4))[:n]
    return df


def test_sorel_binary_task_xy_drops_non_features():
    df = _sorel_frame()
    X, y = data.task_xy(df, "binary", dataset="sorel")
    assert not {"sha256", "label", "dominant_tag"} & set(X.columns)
    assert set(y.unique()) == {0, 1}
    assert len(X) == len(df)


def test_sorel_multiclass_encodes_dominant_tag():
    df = _sorel_frame()
    X, y = data.task_xy(df, "multiclass", dataset="sorel")
    assert y.nunique() == 4
    assert y.dtype.kind in "iu"
    assert len(X) == len(y)


def test_sorel_split_paths_are_dataset_prefixed():
    s, f = data.split_paths("sorel", "binary")
    assert s == "data/splits/sorel_quantum_sample_idx_binary.json"
    assert f == "data/splits/sorel_binary_quantum_folds.json"

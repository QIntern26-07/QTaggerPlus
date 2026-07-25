import numpy as np
import pandas as pd
import pytest

from common.ember_subset import build_ember_subset


def _toy_parquet(path, n_per_family=30, n_benign=50, n_features=8, seed=0):
    """A miniature stand-in for the real EMBER parquet, same schema."""
    rng = np.random.RandomState(seed)
    families = ["wapomi", "emotet", "zusy"]
    rows, labels, avclass = [], [], []
    for fam in families:
        for _ in range(n_per_family):
            rows.append(rng.rand(n_features))
            labels.append(1)
            avclass.append(fam)
    for _ in range(n_benign):
        rows.append(rng.rand(n_features))
        labels.append(0)
        avclass.append("")
    df = pd.DataFrame(np.asarray(rows, dtype=np.float32),
                      columns=[f"F{i + 1}" for i in range(n_features)])
    df["label"] = labels
    df["avclass"] = avclass
    df["sha256"] = [f"{i:064x}" for i in range(len(df))]
    df.to_parquet(path, index=False)
    return path


def test_subset_preserves_schema_and_shrinks_rows(tmp_path):
    src = _toy_parquet(tmp_path / "src.parquet")
    dst = tmp_path / "subset.parquet"
    info = build_ember_subset(str(src), str(dst), binary_n=20,
                              min_per_class=10, max_families=3, col_batch=3)
    out = pd.read_parquet(dst)
    assert list(out.columns) == list(pd.read_parquet(src).columns)
    assert len(out) < 140
    assert info["rows"] == len(out)
    assert set(info["families"]) == {"wapomi", "emotet", "zusy"}


def test_subset_multiclass_pool_is_balanced(tmp_path):
    src = _toy_parquet(tmp_path / "src.parquet")
    dst = tmp_path / "subset.parquet"
    build_ember_subset(str(src), str(dst), binary_n=20,
                       min_per_class=10, max_families=3, col_batch=3)
    out = pd.read_parquet(dst)
    counts = out[out["label"] == 1]["avclass"].value_counts()
    assert counts.nunique() == 1, f"families not balanced: {counts.to_dict()}"


def test_subset_is_deterministic(tmp_path):
    src = _toy_parquet(tmp_path / "src.parquet")
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    build_ember_subset(str(src), str(a), binary_n=20, min_per_class=10,
                       max_families=3, col_batch=3, seed=7)
    build_ember_subset(str(src), str(b), binary_n=20, min_per_class=10,
                       max_families=3, col_batch=3, seed=7)
    assert pd.read_parquet(a)["sha256"].tolist() == pd.read_parquet(b)["sha256"].tolist()


def test_column_batching_does_not_change_result(tmp_path):
    """The batch size is a memory knob only — it must not affect output."""
    src = _toy_parquet(tmp_path / "src.parquet")
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    build_ember_subset(str(src), str(a), binary_n=20, min_per_class=10,
                       max_families=3, col_batch=2, seed=7)
    build_ember_subset(str(src), str(b), binary_n=20, min_per_class=10,
                       max_families=3, col_batch=8, seed=7)
    pd.testing.assert_frame_equal(pd.read_parquet(a), pd.read_parquet(b))

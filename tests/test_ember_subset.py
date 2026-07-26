import numpy as np
import pandas as pd

from common import data
from common.ember_subset import _select_rows, build_ember_subset


def _toy_parquet(path, n_per_family=None, n_benign=50, n_features=8, seed=0):
    """A miniature stand-in for the real EMBER parquet, same schema.

    Family sizes are UNEQUAL by default (mirrors the real EMBER distribution,
    where the malware pool `_select_rows` draws is not balanced going in —
    see `ember_family_xy`, which restores balance later at load time by
    downsampling every kept family to the smallest kept count)."""
    rng = np.random.RandomState(seed)
    families = ["wapomi", "emotet", "zusy"]
    if n_per_family is None:
        n_per_family = {"wapomi": 30, "emotet": 45, "zusy": 60}
    elif isinstance(n_per_family, int):
        n_per_family = {fam: n_per_family for fam in families}
    rows, labels, avclass = [], [], []
    for fam in families:
        for _ in range(n_per_family[fam]):
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


def test_subset_multiclass_pool_is_unbalanced_but_load_time_rebalances(tmp_path):
    """`_select_rows`'s malware pool is NOT balanced on its own — the binary
    pool draws positives independently and injects extra rows into families
    already selected for the family pool (see `ember_subset.py`'s docstring).
    With unequal family sizes (30/45/60) that imbalance actually shows up in
    the written subset. Balance is restored later, at load time, by
    `data.ember_family_xy`'s downsample-to-smallest-kept-count step — which is
    the real end-to-end guarantee this module exists to support. Assert that
    guarantee via `data.task_xy(..., dataset="ember")`, not a false invariant
    about the subset file itself."""
    # `data.task_xy(..., dataset="ember")` calls `ember_family_xy` with ITS
    # OWN defaults (min_per_class=500, max_families=15) — it doesn't forward
    # any params — so the fixture's families must clear 500 rows each for this
    # to exercise the real production path rather than a stand-in threshold.
    src = _toy_parquet(
        tmp_path / "src.parquet",
        n_per_family={"wapomi": 600, "emotet": 750, "zusy": 900},
        n_benign=50,
    )
    dst = tmp_path / "subset.parquet"
    build_ember_subset(str(src), str(dst), binary_n=20,
                       min_per_class=500, max_families=3, col_batch=3)
    out = pd.read_parquet(dst)
    counts = out[out["label"] == 1]["avclass"].value_counts()
    assert counts.nunique() > 1, (
        "expected the raw subset's malware pool to be unbalanced with unequal "
        f"family sizes, got balanced counts: {counts.to_dict()}"
    )

    _, y = data.task_xy(out, "multiclass", dataset="ember")
    assert y.value_counts().nunique() == 1


def test_select_rows_and_ember_family_xy_agree_on_family_list(tmp_path):
    """Both paths must at least agree on the family LABEL SPACE (which family
    names qualify), even though (per the finding above) they do not select the
    same rows for those families. This pins the one guarantee the
    `_select_rows` docstring actually claims — see its "mirrors ... exactly"
    note, which is about family selection only, not row selection."""
    src = _toy_parquet(
        tmp_path / "src.parquet",
        n_per_family={"wapomi": 30, "emotet": 45, "zusy": 60},
        n_benign=50,
    )
    df = pd.read_parquet(src)
    labels = df[["label", "avclass", "sha256"]]

    _, select_rows_families, _ = _select_rows(
        labels, binary_n=20, min_per_class=10, max_families=3, seed=42
    )

    # Independently derive which families `ember_family_xy` keeps, using the
    # same eligibility rule it documents (>= min_per_class, top max_families
    # by count), without reaching into its internals.
    m = df[(df["label"] == 1) & df["avclass"].notna() & (df["avclass"].astype(str) != "")]
    counts = m["avclass"].value_counts()
    counts = counts.sort_index().sort_values(ascending=False, kind="stable")
    expected_families = list(counts[counts >= 10].head(3).index)

    assert set(select_rows_families) == set(expected_families)

    # And confirm ember_family_xy actually returns that many distinct classes.
    _, y = data.ember_family_xy(df, min_per_class=10, max_families=3, seed=42)
    assert y.nunique() == len(expected_families)


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

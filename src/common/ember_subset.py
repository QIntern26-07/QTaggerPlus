"""Build a small, quantum-ready EMBER parquet without loading the full frame.

The published EMBER parquet is a SINGLE row group of 200,000 x 2381 float32
(~1.9 GB resident, more with the pyarrow intermediate). On a laptop with ~4 GB
free that is an out-of-memory risk on every load, and both the quantum and the
classical CLI would pay it.

A single row group cannot be streamed row-wise, so this streams COLUMN-wise
instead: read the three label columns (cheap), decide which rows are needed,
then read feature columns `col_batch` at a time and keep only those rows. Peak
memory is ~ n_rows * col_batch * 4 bytes rather than the whole matrix.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from loguru import logger

_LABEL_COLS = ("label", "avclass", "sha256")


def _select_rows(labels: pd.DataFrame, binary_n: int, min_per_class: int,
                 max_families: int, seed: int) -> tuple[np.ndarray, list[str], int]:
    """Row positions to keep: a balanced family pool plus a stratified binary pool.

    Family selection mirrors `common.data.ember_family_xy` exactly (>= min_per_class,
    top max_families by count, downsample to the smallest kept count) so the subset
    reproduces the same label space the full-frame path would produce.
    """
    rng = np.random.RandomState(seed)
    mal = labels[(labels["label"] == 1)
                 & labels["avclass"].notna()
                 & (labels["avclass"].astype(str) != "")]
    counts = mal["avclass"].value_counts()
    counts = counts.sort_index().sort_values(ascending=False, kind="stable")
    kept = counts[counts >= min_per_class].head(max_families)
    if len(kept) == 0:
        raise ValueError(f"no avclass family reaches min_per_class={min_per_class}")
    per_class = int(kept.min())
    family_names = list(kept.index)

    parts = []
    for name in family_names:
        idx = mal.index[mal["avclass"] == name].to_numpy()
        parts.append(rng.choice(idx, size=per_class, replace=False))
    multiclass_idx = np.concatenate(parts)

    # Binary pool: stratified 50/50, drawn independently of the family pool.
    pos = labels.index[labels["label"] == 1].to_numpy()
    neg = labels.index[labels["label"] == 0].to_numpy()
    half = min(binary_n // 2, len(pos), len(neg))
    binary_idx = np.concatenate([
        rng.choice(pos, size=half, replace=False),
        rng.choice(neg, size=half, replace=False),
    ])

    keep = np.unique(np.concatenate([multiclass_idx, binary_idx]))
    keep.sort()
    return keep, family_names, len(binary_idx)


def build_ember_subset(src_parquet: str, dst_parquet: str, binary_n: int = 4000,
                       seed: int = 42, col_batch: int = 200,
                       min_per_class: int = 500, max_families: int = 15) -> dict:
    """Write a subset parquet containing only the rows the sweeps will use."""
    pf = pq.ParquetFile(src_parquet)
    all_cols = [f.name for f in pf.schema_arrow]
    feat_cols = [c for c in all_cols if c not in _LABEL_COLS]

    labels = pf.read(columns=list(_LABEL_COLS)).to_pandas()
    keep, families, actual_binary_n = _select_rows(labels, binary_n, min_per_class,
                                                    max_families, seed)
    logger.info(f"[ember-subset] keeping {len(keep)} of {len(labels)} rows, "
                f"{len(families)} families")

    pieces = []
    for start in range(0, len(feat_cols), col_batch):
        chunk_cols = feat_cols[start:start + col_batch]
        chunk = pf.read(columns=chunk_cols).to_pandas()
        pieces.append(chunk.iloc[keep].reset_index(drop=True))
        del chunk
    out = pd.concat(pieces, axis=1)
    del pieces

    kept_labels = labels.iloc[keep].reset_index(drop=True)
    for c in _LABEL_COLS:
        out[c] = kept_labels[c]
    out = out[all_cols]
    out.to_parquet(dst_parquet, index=False)

    n_multi = int(((out["label"] == 1) & out["avclass"].isin(families)).sum())
    return {"rows": len(out), "binary_rows": int(actual_binary_n),
            "multiclass_rows": n_multi, "families": families}

"""SOREL-20M multiclass label construction.

SOREL has no single family class: it ships 11 multi-label behavior tags, so plain
stratification does not apply and a labelling decision had to be made before any
SOREL experiment could run. The framing adopted here is DOMINANT TAG — argmax over
the raw per-tag counts, taken BEFORE any binarization, with all-zero-tag rows
dropped. Ties are broken by the declared `TAG_COLS` order, which is fixed and
documented so the label set is reproducible run to run.

Features are deliberately not handled here. SOREL's feature store is a single
71.6 GiB LMDB with no key-level remote access, so feature work is deferred; the
labelling question this module answers is the part that was actually blocking.
"""
from __future__ import annotations

import sqlite3

import pandas as pd

TAG_COLS = ["adware", "flooder", "ransomware", "dropper", "spyware", "packed",
            "crypto_miner", "file_infector", "installer", "worm", "downloader"]


def dominant_tag_labels(meta: pd.DataFrame, tag_cols: list[str]) -> pd.Series:
    """Argmax over raw tag counts. All-zero rows dropped; ties -> first column."""
    counts = meta[tag_cols].to_numpy(dtype=float)
    keep = counts.sum(axis=1) > 0
    idx = meta.index[keep]
    # np.argmax returns the FIRST maximal index, so tag_cols order IS the tie-break.
    winners = counts[keep].argmax(axis=1)
    return pd.Series([tag_cols[w] for w in winners], index=idx, name="dominant_tag")


def label_stats(meta: pd.DataFrame, tag_cols: list[str]) -> dict:
    """The numbers a reviewer will ask about before trusting this label set."""
    counts = meta[tag_cols].to_numpy(dtype=float)
    nonzero = counts.sum(axis=1) > 0
    kept = counts[nonzero]
    row_max = kept.max(axis=1, keepdims=True) if kept.size else kept
    tied = int((kept == row_max).sum(axis=1).__gt__(1).sum()) if kept.size else 0
    labels = dominant_tag_labels(meta, tag_cols)
    return {
        "total_rows": int(len(meta)),
        "dropped_all_zero": int((~nonzero).sum()),
        "labelled_rows": int(nonzero.sum()),
        "tied_rows": tied,
        "class_counts": labels.value_counts().to_dict(),
    }


def read_meta(meta_db: str, table: str = "meta") -> pd.DataFrame:
    """Read sha256 + is_malware + the 11 tag columns out of meta.db."""
    cols = ", ".join(["sha256", "is_malware"] + TAG_COLS)
    with sqlite3.connect(meta_db) as con:
        return pd.read_sql_query(f"SELECT {cols} FROM {table}", con)

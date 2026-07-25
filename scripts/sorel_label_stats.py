"""Print SOREL dominant-tag label statistics from a local meta.db.

meta.db holds ~19.7M rows. Materializing sha256 + 11 tag columns for all of
them as a single pandas DataFrame risks exhausting RAM on a memory-constrained
machine, so this script accumulates `label_stats` chunk-by-chunk via
`pd.read_sql_query(..., chunksize=...)` instead of calling `read_meta` (which
loads everything at once and is kept only for small/test-sized reads).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter

import pandas as pd

from common.sorel_labels import TAG_COLS, label_stats


def chunked_label_stats(meta_db: str, tag_cols: list[str], table: str = "meta",
                         chunksize: int = 1_000_000) -> dict:
    """Same statistics as `label_stats`, computed over chunked reads of meta_db.

    Accumulates by calling the tested `label_stats` function on each chunk and
    summing the results, so the tested aggregation logic is reused unchanged
    rather than reimplemented against a running total.
    """
    cols = ", ".join(["sha256", "is_malware"] + tag_cols)
    total = {"total_rows": 0, "dropped_all_zero": 0, "labelled_rows": 0, "tied_rows": 0}
    class_counts: Counter[str] = Counter()
    n_chunks = 0
    with sqlite3.connect(meta_db) as con:
        for chunk in pd.read_sql_query(f"SELECT {cols} FROM {table}", con,
                                        chunksize=chunksize):
            n_chunks += 1
            s = label_stats(chunk, tag_cols)
            total["total_rows"] += s["total_rows"]
            total["dropped_all_zero"] += s["dropped_all_zero"]
            total["labelled_rows"] += s["labelled_rows"]
            total["tied_rows"] += s["tied_rows"]
            class_counts.update(s["class_counts"])
    total["class_counts"] = dict(class_counts)
    total["n_chunks"] = n_chunks
    total["chunksize"] = chunksize
    return total


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="SOREL label stats")
    p.add_argument("--meta-db", default="data/sorel/meta.db")
    p.add_argument("--chunksize", type=int, default=1_000_000)
    p.add_argument("--full", action="store_true",
                    help="Read the whole table into one DataFrame instead of "
                         "chunking (unsafe on memory-constrained machines).")
    args = p.parse_args(argv)
    if args.full:
        from common.sorel_labels import read_meta
        meta = read_meta(args.meta_db)
        stats = label_stats(meta, TAG_COLS)
    else:
        stats = chunked_label_stats(args.meta_db, TAG_COLS, chunksize=args.chunksize)
    print(json.dumps(stats, indent=2, default=int))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

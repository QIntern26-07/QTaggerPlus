# scripts/prepare_ember.py
"""One-time: vectorize EMBER 2018 test_features.jsonl into a cached parquet.

Reads the official raw JSONL (each row carries sha256/label/avclass + raw
feature groups), vectorizes via the vendored LIEF-free extractor, and writes
F1..F2381 + label + avclass + sha256. Drops label==-1 (unlabeled).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from loguru import logger

from common.ember_vectorize import PEFeatureExtractor


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", default="data/ember/ember2018/test_features.jsonl")
    p.add_argument("--out", default="data/ember/ember2018_test.parquet")
    args = p.parse_args(argv)

    ex = PEFeatureExtractor(feature_version=2, print_feature_warning=False)
    vecs, labels, avclasses, shas = [], [], [], []
    with open(args.jsonl) as fh:
        for i, line in enumerate(fh):
            r = json.loads(line)
            if r.get("label") == -1:
                continue
            vecs.append(ex.process_raw_features(r))
            labels.append(int(r["label"]))
            avclasses.append(r.get("avclass") or "")
            shas.append(r["sha256"])
            if (i + 1) % 20000 == 0:
                logger.info(f"vectorized {i + 1} rows")

    X = np.vstack(vecs).astype(np.float32)
    df = pd.DataFrame(X, columns=[f"F{i + 1}" for i in range(X.shape[1])])
    df["label"] = labels
    df["avclass"] = avclasses
    df["sha256"] = shas
    df.to_parquet(args.out)
    logger.info(f"wrote {len(df)} rows x {X.shape[1]} features to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

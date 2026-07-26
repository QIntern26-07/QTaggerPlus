"""Build the quantum-ready EMBER subset once, offline.

Run this before any EMBER sweep. Loading the full published parquet inside a
sweep risks OOM on a ~4 GB-free laptop; this produces a file small enough to
load repeatedly without thinking about it.
"""
from __future__ import annotations

import argparse

from loguru import logger

from common.ember_subset import build_ember_subset


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build the EMBER quantum subset")
    p.add_argument("--src", default="data/ember/ember2018_test.parquet")
    p.add_argument("--dst", default="data/ember/ember2018_quantum_subset.parquet")
    p.add_argument("--binary-n", type=int, default=4000)
    p.add_argument("--col-batch", type=int, default=200,
                   help="feature columns read per pass; lower it if memory is tight")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    info = build_ember_subset(args.src, args.dst, binary_n=args.binary_n,
                              seed=args.seed, col_batch=args.col_batch)
    logger.info(f"[ember-subset] wrote {args.dst}: {info}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

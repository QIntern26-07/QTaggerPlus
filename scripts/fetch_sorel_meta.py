"""Download SOREL-20M's meta.db (3.5 GiB) — labels only, no feature store.

The feature store (ember_features/data.mdb) is a single 71.6 GiB LMDB with no
key-level remote access, so it is deliberately not fetched here.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from loguru import logger

SRC = "s3://sorel-20m/09-DEC-2020/processed-data/meta.db"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Fetch SOREL meta.db")
    p.add_argument("--dst", default="data/sorel/meta.db")
    args = p.parse_args(argv)
    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    cmd = ["aws", "s3", "cp", "--no-sign-request", SRC, args.dst]
    logger.info(f"[sorel] {' '.join(cmd)}")
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())

"""CLI: quantum QSVM baselines over CIC-MalMem-2022 (mirrors classical CLI)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.model_selection import train_test_split

from common import data
from quantum.run import run_quantum_cv, timing_probe


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CIC-MalMem QSVM baselines")
    p.add_argument("--csv", default="data/cic_malmem/Obfuscated-MalMem2022.csv")
    p.add_argument("--tasks", nargs="+", default=["binary"])
    p.add_argument("--n-components", type=int, default=1)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--max-samples", type=int, default=400,
                   help="stratified subsample size (QSVM kernel is O(n^2)).")
    p.add_argument("--encodings", nargs="+", default=["angle", "iqp"])
    p.add_argument("--probe", action="store_true",
                   help="single untuned fit to measure wall-clock, then exit.")
    p.add_argument("--mlflow", action="store_true")
    p.add_argument("--tracking-uri", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logger.add("run.log", rotation="10 MB")
    df = data.load_cic_malmem(args.csv)
    X_df, y_bin, y_multi = data.build_xy(df)
    X_full = X_df.to_numpy()
    y_bin_full = y_bin.to_numpy()
    y_multi_full = y_multi.to_numpy()

    # Subsample once by INDEX (kernel cost is quadratic); stratify on the
    # multiclass label when multiclass is requested (its smallest families are
    # a fraction of the binary classes' size, so binary-stratification can't
    # guarantee every family survives with enough members for StratifiedKFold),
    # otherwise stratify on the binary label. Indexing X, y_bin, and y_multi
    # with the same sample_idx keeps all three row-aligned regardless of which
    # task is requested.
    if len(X_full) > args.max_samples:
        all_idx = np.arange(len(X_full))
        strat_labels = y_multi_full if "multiclass" in args.tasks else y_bin_full
        sample_idx, _ = train_test_split(
            all_idx, train_size=args.max_samples, stratify=strat_labels,
            random_state=args.seed,
        )
    else:
        sample_idx = np.arange(len(X_full))

    X = X_full[sample_idx]
    targets = {"binary": y_bin_full[sample_idx], "multiclass": y_multi_full[sample_idx]}

    # Persist the subsample so `classical --load-quantum-splits` can score the
    # identical rows for a fair classical-vs-quantum comparison. One file since
    # the row subsample itself is task-independent (only the folds below are
    # task-specific, since they're stratified on that task's own labels).
    Path("data/splits").mkdir(parents=True, exist_ok=True)
    data.save_sample_idx(sample_idx, "data/splits/quantum_sample_idx.json")

    for task in args.tasks:
        y = targets[task]
        if args.probe:
            for enc in args.encodings:
                rec = timing_probe(X, y, task, args.n_components, enc, args.seed)
                logger.info(
                    f"[probe] {enc} nc={args.n_components} "
                    f"kernel_train={rec['kernel_build_train_s']:.3f}s "
                    f"fit={rec['fit_time_sec']:.3f}s infer={rec['inference_time_sec']:.3f}s"
                )
            continue
        # These folds index INTO THE SUBSAMPLE (0..len(sample_idx)-1, NOT into
        # the full ~58K-row dataset) since y here is already row-aligned to
        # sample_idx above.
        folds = data.make_outer_folds(y, n_splits=args.folds, seed=args.seed)
        data.save_folds(folds, f"data/splits/cic_{task}_quantum_folds.json")
        grid = {"encoding": args.encodings, "bandwidth": [None],
                "C": [0.1, 1.0, 10.0], "class_weight": [None, "balanced"]}
        run_quantum_cv(X, y, task, folds, args.n_components, grid=grid,
                       seed=args.seed, use_mlflow=args.mlflow,
                       tracking_uri=args.tracking_uri)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

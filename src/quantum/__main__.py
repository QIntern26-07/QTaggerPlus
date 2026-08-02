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
    p = argparse.ArgumentParser(description="QSVM baselines (CIC-MalMem / EMBER)")
    p.add_argument("--dataset", choices=["cic", "ember", "sorel"], default="cic")
    p.add_argument("--csv", default=None,
                   help="dataset file; defaults per --dataset (CIC csv / EMBER parquet).")
    p.add_argument("--tasks", nargs="+", default=["binary"])
    p.add_argument("--n-components", type=int, default=1)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--max-samples", type=int, default=400,
                   help="stratified subsample size (QSVM kernel is O(n^2)).")
    p.add_argument("--encodings", nargs="+", default=["angle", "iqp"])
    p.add_argument("--probe", action="store_true",
                   help="single untuned fit to measure wall-clock, then exit.")
    p.add_argument(
        "--n-jobs", type=int, default=-1,
        help="worker processes for the QSVM kernel Gram build (each ~1 core). "
             "lightning.qubit on these small circuits is single-threaded, so the "
             "O(n^2) pair count is split across processes. -1 = all cores; cap it "
             "(e.g. 8) to leave headroom for the desktop/IDE and avoid "
             "oversubscription.",
    )
    p.add_argument("--mlflow", action="store_true")
    p.add_argument("--tracking-uri", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p


def predictions_path(dataset: str, task: str, n_components: int, encodings) -> str:
    """Per-fold prediction file for one quantum sweep.

    Mirrors the classical CLI's persistence so paired significance tests can
    reach QSVM predictions at sample level. The encoding tag follows
    run.run_quantum_cv's rule: the single encoding name, or "joint" when one
    sweep tunes over several. n_components is in the name because a sweep at a
    different qubit budget is a different result, not an overwrite of this one.
    """
    tag = encodings[0] if len(encodings) == 1 else "joint"
    return f"results/{dataset}/qsvm_{task}_nc{n_components}_{tag}_predictions.npz"


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logger.add("run.log", rotation="10 MB")
    if args.dataset == "ember":
        df = data.load_ember(args.csv or "data/ember/ember2018_test.parquet")
    elif args.dataset == "sorel":
        sorel_path = args.csv or "data/sorel/sorel_quantum_subset.parquet"
        if not Path(sorel_path).exists():
            raise SystemExit(
                f"--dataset sorel: {sorel_path} does not exist. SOREL-20M features "
                "require downloading its 71.6 GiB LMDB feature store "
                "(s3://sorel-20m/09-DEC-2020/processed-data/ember_features/data.mdb), "
                "which has not been fetched — this is a deliberate, documented decision, "
                "not a bug. See docs/reports/w4_sorel_labelling_decision.md for the "
                "labelling design and what remains (feature-subset acquisition + sweep)."
            )
        df = data.load_sorel(sorel_path)
    else:
        df = data.load_cic_malmem(args.csv or "data/cic_malmem/Obfuscated-MalMem2022.csv")
    Path("data/splits").mkdir(parents=True, exist_ok=True)

    # Each task now operates on its OWN row set — binary on all rows, multiclass
    # on malware-only rows (Benign dropped, 15-class family task). So the
    # subsample and its persisted index are per-task, not shared: a multiclass
    # sample_idx points into the malware-only frame, a binary one into the full
    # frame. classical --load-quantum-splits must apply the matching task_xy
    # before indexing.
    for task in args.tasks:
        X_df, y_ser = data.task_xy(df, task, dataset=args.dataset)
        X_full = X_df.to_numpy()
        y_full = y_ser.to_numpy()
        sample_path, folds_path = data.split_paths(args.dataset, task)

        # Subsample by INDEX (kernel cost is quadratic), stratified on this
        # task's own labels so every class survives for StratifiedKFold.
        if len(X_full) > args.max_samples:
            all_idx = np.arange(len(X_full))
            sample_idx, _ = train_test_split(
                all_idx, train_size=args.max_samples, stratify=y_full,
                random_state=args.seed,
            )
        else:
            sample_idx = np.arange(len(X_full))

        X = X_full[sample_idx]
        y = y_full[sample_idx]
        # Persist the per-task subsample so `classical --load-quantum-splits`
        # can score the identical rows for a fair comparison.
        data.save_sample_idx(sample_idx, sample_path)

        if args.probe:
            for enc in args.encodings:
                rec = timing_probe(X, y, task, args.n_components, enc, args.seed,
                                   n_jobs=args.n_jobs)
                logger.info(
                    f"[probe] {enc} nc={args.n_components} "
                    f"kernel_train={rec['kernel_build_train_s']:.3f}s "
                    f"fit={rec['fit_time_sec']:.3f}s infer={rec['inference_time_sec']:.3f}s"
                )
            continue
        # These folds index INTO THE SUBSAMPLE (0..len(sample_idx)-1), since y
        # is already row-aligned to sample_idx above.
        folds = data.make_outer_folds(y, n_splits=args.folds, seed=args.seed)
        data.save_folds(folds, folds_path)
        grid = {"encoding": args.encodings, "bandwidth": [None],
                "C": [0.1, 1.0, 10.0], "class_weight": [None, "balanced"]}
        dataset_name = {"cic": "cic-malmem", "ember": "ember-2018",
                        "sorel": "sorel-20m"}[args.dataset]
        records = run_quantum_cv(X, y, task, folds, args.n_components, grid=grid,
                                 seed=args.seed, use_mlflow=args.mlflow,
                                 tracking_uri=args.tracking_uri, n_jobs=args.n_jobs,
                                 dataset_name=dataset_name)
        pred_path = predictions_path(args.dataset, task, args.n_components,
                                     args.encodings)
        Path(pred_path).parent.mkdir(parents=True, exist_ok=True)
        data.save_predictions(records, pred_path)
        logger.info(f"wrote per-fold predictions to {pred_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

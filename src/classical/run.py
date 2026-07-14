"""Nested-CV orchestration: Optuna tuning inside outer stratified folds."""
from __future__ import annotations

import contextlib
import warnings

import numpy as np
import optuna
from loguru import logger
from sklearn.model_selection import StratifiedKFold, cross_val_score

from common.evaluate import (
    aggregate_metrics,
    auc_scores,
    compute_metrics,
    confusion_matrix_figure,
    per_class_f1,
    timed,
)
from common.preprocess import build_feature_pipeline
from common.tracking import run as mlflow_run
from classical.models import make_model, suggest_params

optuna.logging.set_verbosity(optuna.logging.WARNING)

# LightGBM's sklearn wrapper retains feature-name metadata from one CV fit and
# warns when a later predict() call (inside the same Optuna inner-CV loop)
# gets a plain numpy array — reproduces even with pure-numpy input with no
# reference to this pipeline, so it's an upstream LightGBM/sklearn quirk, not
# something this code is doing wrong. Cosmetic-only suppression.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)


def tune_and_fit(name, task, X_train, y_train, n_trials, inner_splits, seed, n_jobs=-1):
    """Optuna inner-CV tuning, then a timed final refit on the full train fold.

    Returns (model, best_params, fit_time_sec) where fit_time_sec is the wall
    time of the single final refit only — tuning time is measured by the caller.

    `n_jobs` controls parallelism across the inner CV folds (via cross_val_score).
    The model itself is always built with n_jobs=1 *inside* the inner-CV objective,
    since cross_val_score already parallelizes across `inner_splits` folds — giving
    each fold's model its own n_jobs=-1 as well would oversubscribe CPU/RAM by a
    factor of inner_splits. The single final refit (one model, not nested) uses the
    full `n_jobs` value.
    """
    inner_cv = StratifiedKFold(
        n_splits=inner_splits, shuffle=True, random_state=seed
    )

    def objective(trial):
        params = suggest_params(name, trial)
        model = make_model(name, params, task, seed=seed, n_jobs=1)
        scores = cross_val_score(
            model, X_train, y_train, cv=inner_cv, scoring="f1_macro", n_jobs=n_jobs
        )
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = make_model(name, study.best_params, task, seed=seed, n_jobs=n_jobs)
    _, fit_time = timed(best.fit, X_train, y_train)
    return best, study.best_params, fit_time


def evaluate_fold(
    name, task, X, y, train_idx, test_idx, n_trials, seed, inner_splits=3, n_jobs=-1,
    n_components=None,
):
    """Tune+fit on train fold, score on test fold; return a per-fold record."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    pipeline = build_feature_pipeline(n_components=n_components, seed=seed)
    X_tr_t = pipeline.fit_transform(X_tr)  # fit on TRAIN fold only (no leakage)
    X_te_t = pipeline.transform(X_te)

    (model, best_params, fit_time), tune_plus_fit = timed(
        tune_and_fit, name, task, X_tr_t, y_tr, n_trials, inner_splits, seed,
        n_jobs=n_jobs,
    )
    tune_time = tune_plus_fit - fit_time
    y_pred, infer_time = timed(model.predict, X_te_t)
    y_score = auc_scores(model, X_te_t, task)

    metrics = compute_metrics(y_te, y_pred, y_score, task)
    logger.info(f"[{name}/{task}] fold done: f1_macro={metrics['f1_macro']:.4f}")
    return {
        "model": name,
        "task": task,
        "metrics": metrics,
        "f1_per_class": per_class_f1(y_te, y_pred),
        "best_params": best_params,
        "fit_time_sec": fit_time,
        "tune_time_sec": tune_time,
        "inference_time_sec": infer_time,
        "test_idx": np.asarray(test_idx),
        "y_true": y_te,
        "y_pred": np.asarray(y_pred),
    }


_TIMING_KEYS = ("fit_time_sec", "tune_time_sec", "inference_time_sec")


def run_nested_cv(
    X, y, task, name, folds, n_trials=25, seed=42, use_mlflow=False,
    inner_splits=3, dataset_name="cic-malmem", n_jobs=-1, tracking_uri=None,
    extra_params=None, n_components=None,
):
    """Run every outer fold for one model x task; optionally log each to MLflow.

    When `use_mlflow`, every fold is logged as a nested child run under one
    sweep-level parent run, and the parent gets the mean/std aggregate across
    folds — so the sweep has a single summary row instead of only per-fold rows.
    """
    records = []
    sweep_tag = f"{name}-{task}-nc{n_components}"
    parent_cm = (
        mlflow_run(
            "qtaggerplus", f"{name}-{task}-sweep",
            {"model": name, "task": task, "dataset": dataset_name,
             "n_jobs": n_jobs, **(extra_params or {})},
            tracking_uri=tracking_uri, tags={"sweep": sweep_tag},
        )
        if use_mlflow else contextlib.nullcontext()
    )
    with parent_cm as parent_log:
        for i, (train_idx, test_idx) in enumerate(folds):
            logger.info(f"[{name}/{task}] outer fold {i + 1}/{len(folds)}")
            rec = evaluate_fold(
                name, task, X, y, train_idx, test_idx, n_trials, seed,
                inner_splits=inner_splits, n_jobs=n_jobs, n_components=n_components,
            )
            rec["fold"] = i
            records.append(rec)
            if use_mlflow:
                _log_fold_to_mlflow(
                    rec, dataset_name=dataset_name, n_jobs=n_jobs,
                    tracking_uri=tracking_uri, extra_params=extra_params or {},
                    sweep_tag=sweep_tag,
                )
        if use_mlflow:
            combined = [
                {**r["metrics"], **{k: r[k] for k in _TIMING_KEYS}} for r in records
            ]
            parent_log.log_metrics(aggregate_metrics(combined))
    return records


def _log_fold_to_mlflow(rec, dataset_name, n_jobs, tracking_uri, extra_params, sweep_tag):
    """One nested MLflow run per model x task x fold, under the sweep's parent run."""
    import matplotlib.pyplot as plt

    params = {
        "model": rec["model"], "task": rec["task"], "dataset": dataset_name,
        "n_jobs": n_jobs, **extra_params, **rec["best_params"],
    }
    with mlflow_run(
        "qtaggerplus", f"{rec['model']}-{rec['task']}-fold{rec['fold']}",
        params, tracking_uri=tracking_uri, tags={"sweep": sweep_tag}, nested=True,
    ) as log:
        log.log_metrics({
            **rec["metrics"],
            **{f"f1_class_{k}": v for k, v in rec["f1_per_class"].items()},
            "fit_time_sec": rec["fit_time_sec"],
            "tune_time_sec": rec["tune_time_sec"],
            "inference_time_sec": rec["inference_time_sec"],
        })
        fig = confusion_matrix_figure(rec["y_true"], rec["y_pred"])
        log.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

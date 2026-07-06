"""Nested-CV orchestration: Optuna tuning inside outer stratified folds."""
from __future__ import annotations

import numpy as np
import optuna
from loguru import logger
from sklearn.model_selection import StratifiedKFold, cross_val_score

from classical.evaluate import compute_metrics, confusion_matrix_figure, timed
from classical.features import build_feature_pipeline
from classical.models import make_model, suggest_params

optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_and_fit(name, task, X_train, y_train, n_trials, inner_splits, seed):
    """Optuna inner-CV tuning on the training fold; refit best on full train fold."""
    inner_cv = StratifiedKFold(
        n_splits=inner_splits, shuffle=True, random_state=seed
    )

    def objective(trial):
        params = suggest_params(name, trial)
        model = make_model(name, params, task, seed=seed)
        scores = cross_val_score(
            model, X_train, y_train, cv=inner_cv, scoring="f1_macro", n_jobs=-1
        )
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = make_model(name, study.best_params, task, seed=seed)
    best.fit(X_train, y_train)
    return best, study.best_params


def _proba_for(model, X, task):
    proba = model.predict_proba(X)
    return proba[:, 1] if task == "binary" else proba


def evaluate_fold(name, task, X, y, train_idx, test_idx, n_trials, seed, inner_splits=3):
    """Tune+fit on train fold, score on test fold; return a per-fold record."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    pipeline = build_feature_pipeline()
    X_tr_t = pipeline.fit_transform(X_tr)  # fit on TRAIN fold only (no leakage)
    X_te_t = pipeline.transform(X_te)

    (model, best_params), train_time = timed(
        tune_and_fit, name, task, X_tr_t, y_tr, n_trials, inner_splits, seed
    )
    y_pred, infer_time = timed(model.predict, X_te_t)
    y_proba = _proba_for(model, X_te_t, task)

    metrics = compute_metrics(y_te, y_pred, y_proba, task)
    logger.info(f"[{name}/{task}] fold done: f1_macro={metrics['f1_macro']:.4f}")
    return {
        "model": name,
        "task": task,
        "metrics": metrics,
        "best_params": best_params,
        "train_time_sec": train_time,
        "inference_time_sec": infer_time,
        "test_idx": np.asarray(test_idx),
        "y_true": y_te,
        "y_pred": np.asarray(y_pred),
    }


def run_nested_cv(
    X, y, task, name, folds, n_trials=25, seed=42, use_wandb=False,
    inner_splits=3, dataset_name="cic-malmem",
):
    """Run every outer fold for one model x task; optionally log each to W&B."""
    records = []
    for i, (train_idx, test_idx) in enumerate(folds):
        logger.info(f"[{name}/{task}] outer fold {i + 1}/{len(folds)}")
        rec = evaluate_fold(
            name, task, X, y, train_idx, test_idx, n_trials, seed,
            inner_splits=inner_splits,
        )
        rec["fold"] = i
        records.append(rec)
        if use_wandb:
            _log_fold_to_wandb(rec, dataset_name=dataset_name)
    return records


def _log_fold_to_wandb(rec, dataset_name="cic-malmem"):
    """One W&B run per model x task x fold; imported lazily so tests stay offline."""
    import wandb

    run = wandb.init(
        project="qtaggerplus-classical",
        group=f"{dataset_name}-{rec['task']}",
        name=f"{rec['model']}-fold{rec['fold']}",
        config={"model": rec["model"], "task": rec["task"],
                "dataset": dataset_name, **rec["best_params"]},
        reinit=True,
    )
    wandb.log({
        **rec["metrics"],
        "train_time_sec": rec["train_time_sec"],
        "inference_time_sec": rec["inference_time_sec"],
    })

    fig = confusion_matrix_figure(rec["y_true"], rec["y_pred"])
    wandb.log({"confusion_matrix": wandb.Image(fig)})
    import matplotlib.pyplot as plt
    plt.close(fig)

    table = wandb.Table(columns=["test_idx", "y_true", "y_pred"])
    for idx, yt, yp in zip(rec["test_idx"], rec["y_true"], rec["y_pred"]):
        table.add_data(int(idx), int(yt), int(yp))
    wandb.log({"predictions": table})

    run.finish()

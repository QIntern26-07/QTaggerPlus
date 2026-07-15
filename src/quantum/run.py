"""Quantum (QSVM) runner: shared folds, two-tier tuning, MLflow, timing probe.

Two-tier tuning (R3 motivation): the quantum kernel depends only on the
encoding/bandwidth, not on C. So we build the inner Gram once per
(encoding, bandwidth) and sweep C/class_weight cheaply on it, instead of
rebuilding the kernel for every hyperparameter combination.
"""
from __future__ import annotations

import contextlib

import numpy as np
from loguru import logger
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

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
from quantum.encoding import default_bandwidth, n_qubits_for
from quantum.preprocess import EncodingScaler
from quantum.qsvm import QSVM

DEFAULT_GRID = {
    "encoding": ["angle", "iqp"],
    "bandwidth": [None],            # None -> encoding's default_bandwidth
    "C": [0.1, 1.0, 10.0],
    "class_weight": [None, "balanced"],
}


def _prep(X_tr, X_te, n_components, encoding, n_qubits, seed):
    """Shared PCA pipeline (fit on train) then encoding-specific scaling."""
    pipe = build_feature_pipeline(n_components=n_components, seed=seed)
    Ztr = pipe.fit_transform(X_tr)
    Zte = pipe.transform(X_te)
    scaler = EncodingScaler(encoding, n_qubits).fit(Ztr)
    return scaler.transform(Ztr), scaler.transform(Zte)


def tune_and_fit_qsvm(X_tr, y_tr, task, n_components, grid, seed, n_jobs=-1):
    """Two-tier tuning on the train fold; refit best QSVM on the full fold.

    Returns (model, best_params, fit_time_sec) with fit_time_sec the final refit
    only. Encoding-specific scaling is folded into each candidate via _prep at
    the caller; here X_tr is already scaled for a *single* encoding is NOT the
    case — we scale per encoding inside the loop using the raw train fold.
    """
    Xa, Xb, ya, yb = train_test_split(
        X_tr, y_tr, test_size=0.25, stratify=y_tr, random_state=seed
    )
    best = None  # (score, params)
    for encoding in grid["encoding"]:
        n_qubits = n_qubits_for(encoding, n_components)
        Za, Zb = _prep(Xa, Xb, n_components, encoding, n_qubits, seed)
        for bw in grid["bandwidth"]:
            bandwidth = default_bandwidth(n_qubits) if bw is None else bw
            probe = QSVM(encoding=encoding, n_components=n_components,
                         bandwidth=bandwidth, task=task, seed=seed, n_jobs=n_jobs)
            Kaa = probe._gram_sym(Za)          # inner-train Gram, built ONCE
            Kba = probe.gram(Zb, Za)           # inner-val Gram, built ONCE
            for C in grid["C"]:
                for cw in grid["class_weight"]:
                    svc = SVC(kernel="precomputed", C=C, class_weight=cw,
                              decision_function_shape="ovr", random_state=seed)
                    svc.fit(Kaa, ya)
                    pred = svc.predict(Kba)
                    score = f1_score(yb, pred, average="macro")
                    params = {"encoding": encoding, "bandwidth": bandwidth,
                              "C": C, "class_weight": cw}
                    if best is None or score > best[0]:
                        best = (score, params)
    params = best[1]
    n_qubits = n_qubits_for(params["encoding"], n_components)
    Ztr, _ = _prep(X_tr, X_tr[:1], n_components, params["encoding"], n_qubits, seed)
    model = QSVM(encoding=params["encoding"], n_components=n_components,
                 bandwidth=params["bandwidth"], C=params["C"],
                 class_weight=params["class_weight"], task=task, seed=seed,
                 n_jobs=n_jobs)
    _, fit_time = timed(model.fit, Ztr, y_tr)
    return model, params, fit_time


def evaluate_fold_quantum(X, y, task, train_idx, test_idx, n_components, grid,
                          seed, n_jobs=-1):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    (model, best_params, fit_time), tune_plus_fit = timed(
        tune_and_fit_qsvm, X_tr, y_tr, task, n_components, grid, seed, n_jobs=n_jobs
    )
    tune_time = tune_plus_fit - fit_time

    # scale the test fold with the winning encoding, fit on the train fold
    n_qubits = n_qubits_for(best_params["encoding"], n_components)
    _, Zte = _prep(X_tr, X_te, n_components, best_params["encoding"], n_qubits, seed)

    y_pred, infer_time = timed(model.predict, Zte)
    y_score = auc_scores(model, Zte, task)
    metrics = compute_metrics(y_te, y_pred, y_score, task)
    logger.info(f"[qsvm/{task}] fold done: f1_macro={metrics['f1_macro']:.4f}")
    return {
        "model": "qsvm", "task": task, "metrics": metrics,
        "f1_per_class": per_class_f1(y_te, y_pred),
        "best_params": best_params,
        "n_qubits": n_qubits,
        "fit_time_sec": fit_time, "tune_time_sec": tune_time,
        "inference_time_sec": infer_time,
        "kernel_build_train_s": model.kernel_build_train_s,
        "kernel_build_test_s": model.kernel_build_test_s,
        "kernel_evals": model.kernel_evals,
        "gram_offdiag_std": model.gram_offdiag_std,
        "test_idx": np.asarray(test_idx), "y_true": y_te,
        "y_pred": np.asarray(y_pred),
    }


_TIMING_KEYS = ("fit_time_sec", "tune_time_sec", "inference_time_sec")


def run_quantum_cv(X, y, task, folds, n_components, grid=None, seed=42,
                   use_mlflow=False, tracking_uri=None, dataset_name="cic-malmem",
                   n_jobs=-1):
    """Run every outer fold; optionally log each as a nested child run under one
    sweep-level parent run that gets the mean/std aggregate across folds."""
    grid = grid or DEFAULT_GRID
    records = []
    sweep_tag = f"qsvm-{task}-nc{n_components}"
    parent_cm = (
        mlflow_run(
            "qtaggerplus", f"qsvm-{task}-nc{n_components}-sweep",
            {"framework": "quantum", "model": "qsvm", "task": task,
             "dataset": dataset_name, "n_components": n_components},
            tracking_uri=tracking_uri, tags={"sweep": sweep_tag},
        )
        if use_mlflow else contextlib.nullcontext()
    )
    with parent_cm as parent_log:
        for i, (train_idx, test_idx) in enumerate(folds):
            logger.info(f"[qsvm/{task}] outer fold {i + 1}/{len(folds)}")
            rec = evaluate_fold_quantum(X, y, task, train_idx, test_idx,
                                        n_components, grid, seed, n_jobs=n_jobs)
            rec["fold"] = i
            records.append(rec)
            if use_mlflow:
                _log_quantum_fold(rec, dataset_name, n_components, tracking_uri, sweep_tag)
        if use_mlflow:
            combined = [
                {**r["metrics"], **{k: r[k] for k in _TIMING_KEYS}} for r in records
            ]
            parent_log.log_metrics(aggregate_metrics(combined))
    return records


def _log_quantum_fold(rec, dataset_name, n_components, tracking_uri, sweep_tag):
    import matplotlib.pyplot as plt
    params = {
        "framework": "quantum", "model": "qsvm", "task": rec["task"],
        "dataset": dataset_name, "n_components": n_components,
        "n_qubits": rec["n_qubits"],
        "encoding": rec["best_params"]["encoding"],
        "bandwidth": rec["best_params"]["bandwidth"],
        "C": rec["best_params"]["C"],
        "class_weight": rec["best_params"]["class_weight"],
    }
    with mlflow_run("qtaggerplus",
                    f"qsvm-{rec['task']}-nc{n_components}-fold{rec['fold']}",
                    params, tracking_uri=tracking_uri, tags={"sweep": sweep_tag},
                    nested=True) as log:
        log.log_metrics({
            **rec["metrics"],
            **{f"f1_class_{k}": v for k, v in rec["f1_per_class"].items()},
            "fit_time_sec": rec["fit_time_sec"],
            "tune_time_sec": rec["tune_time_sec"],
            "inference_time_sec": rec["inference_time_sec"],
            "kernel_build_train_s": rec["kernel_build_train_s"],
            "kernel_build_test_s": rec["kernel_build_test_s"],
            "kernel_evals": rec["kernel_evals"],
            "gram_offdiag_std": rec["gram_offdiag_std"],
        })
        fig = confusion_matrix_figure(rec["y_true"], rec["y_pred"])
        log.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)


def timing_probe(X, y, task, n_components, encoding, seed=42, n_jobs=-1):
    """Single untuned QSVM fit/predict to measure wall-clock before a full sweep.

    Deliberately no CV and no tuning: the point is to learn the per-Gram cost at
    a given (encoding, n_components) so grid and sample sizes can be chosen
    sanely. Uses an 80/20 split of the provided data.
    """
    from sklearn.model_selection import train_test_split as _tts
    X = np.asarray(X, dtype=float)
    Xtr, Xte, ytr, yte = _tts(X, y, test_size=0.2, stratify=y, random_state=seed)
    n_qubits = n_qubits_for(encoding, n_components)
    Ztr, Zte = _prep(Xtr, Xte, n_components, encoding, n_qubits, seed)
    model = QSVM(encoding=encoding, n_components=n_components, task=task,
                 seed=seed, n_jobs=n_jobs)
    _, fit_time = timed(model.fit, Ztr, ytr)
    y_pred, infer_time = timed(model.predict, Zte)
    y_score = auc_scores(model, Zte, task)
    metrics = compute_metrics(yte, y_pred, y_score, task)
    return {
        "encoding": encoding, "n_components": n_components, "n_qubits": n_qubits,
        "fit_time_sec": fit_time, "inference_time_sec": infer_time,
        "kernel_build_train_s": model.kernel_build_train_s,
        "kernel_build_test_s": model.kernel_build_test_s,
        "kernel_evals": model.kernel_evals, "metrics": metrics,
    }

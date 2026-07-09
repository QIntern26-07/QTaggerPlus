"""Model factories and Optuna hyperparameter search spaces."""
from __future__ import annotations

import optuna
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

MODEL_NAMES = ("random_forest", "xgboost", "lightgbm", "svm")


def make_model(name: str, params: dict, task: str, seed: int = 42, n_jobs: int = -1):
    """Construct an estimator with task-appropriate defaults merged with `params`.

    `n_jobs` controls the estimator's own internal parallelism (ignored for SVM,
    which has no such parameter). Callers doing nested parallelism (e.g. cross
    validating several of these concurrently) should pass n_jobs=1 here to avoid
    oversubscribing CPU/RAM.
    """
    if name == "random_forest":
        return RandomForestClassifier(
            random_state=seed, n_jobs=n_jobs, class_weight="balanced", **params
        )
    if name == "xgboost":
        objective = "binary:logistic" if task == "binary" else "multi:softprob"
        return XGBClassifier(
            random_state=seed, n_jobs=n_jobs, objective=objective,
            eval_metric="logloss", tree_method="hist", **params,
        )
    if name == "lightgbm":
        return LGBMClassifier(
            random_state=seed, n_jobs=n_jobs, class_weight="balanced",
            verbose=-1, **params,
        )
    if name == "svm":
        # probability estimation is off by default in sklearn; leaving the
        # kwarg unset (rather than passing probability=False) avoids the
        # FutureWarning sklearn now emits whenever `probability` is passed
        # explicitly at all, regardless of value.
        return SVC(
            random_state=seed,
            decision_function_shape="ovr", **params,
        )
    raise ValueError(f"unknown model name: {name}")


def suggest_params(name: str, trial: optuna.Trial) -> dict:
    """Sample a hyperparameter dict for `name` from the trial's search space."""
    if name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 4, 32),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            ),
        }
    if name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
    if name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
    if name == "svm":
        return {
            "C": trial.suggest_float("C", 1e-2, 1e2, log=True),
            "gamma": trial.suggest_float("gamma", 1e-4, 1e1, log=True),
            "class_weight": trial.suggest_categorical(
                "class_weight", [None, "balanced"]
            ),
            "kernel": "rbf",
        }
    raise ValueError(f"unknown model name: {name}")

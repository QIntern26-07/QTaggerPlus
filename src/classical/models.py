"""Model factories and Optuna hyperparameter search spaces."""
from __future__ import annotations

import optuna
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

MODEL_NAMES = ("random_forest", "xgboost", "lightgbm", "svm")


def make_model(name: str, params: dict, task: str, seed: int = 42):
    """Construct an estimator with task-appropriate defaults merged with `params`."""
    if name == "random_forest":
        return RandomForestClassifier(
            random_state=seed, n_jobs=-1, class_weight="balanced", **params
        )
    if name == "xgboost":
        objective = "binary:logistic" if task == "binary" else "multi:softprob"
        return XGBClassifier(
            random_state=seed, n_jobs=-1, objective=objective,
            eval_metric="logloss", tree_method="hist", **params,
        )
    if name == "lightgbm":
        return LGBMClassifier(
            random_state=seed, n_jobs=-1, class_weight="balanced",
            verbose=-1, **params,
        )
    if name == "svm":
        return SVC(
            random_state=seed, probability=True, class_weight="balanced", **params
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
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
            "kernel": trial.suggest_categorical("kernel", ["rbf"]),
        }
    raise ValueError(f"unknown model name: {name}")

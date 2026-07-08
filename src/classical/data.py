"""Load CIC-MalMem-2022 and construct binary and multiclass labels."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

FEATURE_LABEL_COLS = ("Category", "Class")


def load_cic_malmem(csv_path: str) -> pd.DataFrame:
    """Read the raw CIC-MalMem-2022 CSV into a DataFrame."""
    return pd.read_csv(csv_path)


def build_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Split a raw frame into features X, binary y, and multiclass (family) y.

    Binary: Class == "Malware" -> 1, else 0.
    Multiclass: family = first two tokens of Category (e.g. "Ransomware-Ako", "Benign").
    """
    X = df.drop(columns=list(FEATURE_LABEL_COLS))
    y_binary = (df["Class"].str.strip().str.lower() == "malware").astype(int)
    y_binary = pd.Series(y_binary, name="y_binary")

    family = df["Category"].str.split("-").str[:2].str.join("-").str.strip()
    y_multiclass = pd.Series(
        LabelEncoder().fit_transform(family), name="y_multiclass"
    )
    return X, y_binary, y_multiclass


def make_outer_folds(y, n_splits: int = 5, seed: int = 42):
    """Return a list of (train_idx, test_idx) numpy arrays via stratified k-fold."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X_dummy = np.zeros((len(y), 1))
    return [
        (train_idx, test_idx) for train_idx, test_idx in skf.split(X_dummy, y)
    ]


def save_folds(folds, path: str) -> None:
    """Persist folds as JSON lists so any framework (incl. quantum) can reload them."""
    payload = [
        {"train": train.tolist(), "test": test.tolist()} for train, test in folds
    ]
    with open(path, "w") as fh:
        json.dump(payload, fh)


def load_folds(path: str):
    """Load folds saved by save_folds back into (train_idx, test_idx) numpy arrays."""
    with open(path) as fh:
        payload = json.load(fh)
    return [
        (np.array(item["train"]), np.array(item["test"])) for item in payload
    ]


def save_predictions(records, path: str) -> None:
    """Persist per-fold test_idx/y_true/y_pred for one model x task run.

    `records` is the list of per-fold dicts returned by run.run_nested_cv (each
    must contain "fold", "test_idx", "y_true", "y_pred"). Stored as four
    parallel flat arrays (one row per test sample, repeated fold id) inside a
    single .npz so paired significance tests in compare.py can later reload
    real classical-model predictions without depending on aggregated metrics.
    """
    fold_ids, test_idx, y_true, y_pred = [], [], [], []
    for rec in records:
        n = len(rec["test_idx"])
        fold_ids.append(np.full(n, rec["fold"]))
        test_idx.append(np.asarray(rec["test_idx"]))
        y_true.append(np.asarray(rec["y_true"]))
        y_pred.append(np.asarray(rec["y_pred"]))
    np.savez(
        path,
        fold_ids=np.concatenate(fold_ids),
        test_idx=np.concatenate(test_idx),
        y_true=np.concatenate(y_true),
        y_pred=np.concatenate(y_pred),
    )


def load_predictions(path: str) -> dict:
    """Load predictions saved by save_predictions back into a dict of flat arrays."""
    with np.load(path) as npz:
        return {
            "fold_ids": npz["fold_ids"],
            "test_idx": npz["test_idx"],
            "y_true": npz["y_true"],
            "y_pred": npz["y_pred"],
        }

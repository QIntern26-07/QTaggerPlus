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


def load_ember(parquet_path: str = "data/ember/ember2018_test.parquet") -> pd.DataFrame:
    """Read the cached EMBER 2018 test parquet (see scripts/prepare_ember.py).

    Columns: F1..F2381 vectorized features (feature version 2) + int `label`
    (0 benign / 1 malicious) + `avclass` family string + `sha256`.
    """
    return pd.read_parquet(parquet_path)


_EMBER_NON_FEATURE = ("label", "avclass", "sha256")


def _ember_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop non-feature EMBER columns, tolerating their absence (e.g. in toy
    test frames that only carry `label`)."""
    cols = [c for c in _EMBER_NON_FEATURE if c in df.columns]
    return df.drop(columns=cols)


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


def malware_family_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Features and 15-class family labels for the malware-only multiclass task.

    Drops Benign rows entirely (Benign detection is already the binary task, and
    keeping it makes the multiclass label space pathologically imbalanced —
    Benign ~21x any single family). The remaining malware families are internally
    near-balanced (~1.7x) and re-encoded contiguously from 0. Index is reset so
    downstream positional indexing (subsample idx, CV folds) is valid.
    """
    is_malware = df["Class"].str.strip().str.lower() == "malware"
    dfm = df[is_malware].reset_index(drop=True)
    X = dfm.drop(columns=list(FEATURE_LABEL_COLS))
    family = dfm["Category"].str.split("-").str[:2].str.join("-").str.strip()
    y_family = pd.Series(
        LabelEncoder().fit_transform(family), name="y_family"
    )
    return X, y_family


def ember_family_xy(
    df: pd.DataFrame,
    min_per_class: int = 500,
    max_families: int = 15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Features and balanced K-class avclass-family labels for EMBER multiclass.

    EMBER's avclass distribution is heavy-tailed (unlike CIC's naturally
    ~1.7x-balanced families), so structure is imposed: keep malware rows with a
    non-empty avclass, keep families with count >= min_per_class, cap at the
    top `max_families` by count, then downsample every kept family to the
    smallest kept count so the returned pool is exactly balanced (1.0x).
    Deterministic under `seed`. Benign is excluded (binary task's job).
    """
    df = df.reset_index(drop=True)
    m = df[(df["label"] == 1) & df["avclass"].notna() & (df["avclass"].astype(str) != "")]
    counts = m["avclass"].value_counts()
    counts = counts.sort_index().sort_values(ascending=False, kind="stable")
    eligible = counts[counts >= min_per_class]
    kept = eligible.head(max_families)
    if len(kept) == 0:
        raise ValueError(
            f"no avclass family reaches min_per_class={min_per_class}"
        )
    per_class = int(kept.min())
    keep_names = list(kept.index)
    rng = np.random.RandomState(seed)
    parts = []
    for name in keep_names:
        idx = m.index[m["avclass"] == name].to_numpy()
        chosen = rng.choice(idx, size=per_class, replace=False)
        parts.append(chosen)
    sel = np.concatenate(parts)
    sel.sort()
    dfm = df.loc[sel].reset_index(drop=True)
    X = _ember_features(dfm)
    y = pd.Series(
        LabelEncoder().fit_transform(dfm["avclass"]), name="y_family"
    )
    return X, y


def task_xy(
    df: pd.DataFrame, task: str, dataset: str = "cic"
) -> tuple[pd.DataFrame, pd.Series]:
    """Return the (X, y) row set a task operates on. Single source of truth so
    the classical and quantum CLIs can never drift on what a task means (a drift
    would silently misalign their shared subsample/fold contract).

    cic binary: all rows, Benign-vs-Malware label.
    cic multiclass: malware-only rows, 15-class family label (see malware_family_xy).
    ember binary: all rows, `label` column as int 0/1.
    ember multiclass: balanced top-15 avclass family label (see ember_family_xy).
    """
    if dataset == "cic":
        if task == "binary":
            X, y_binary, _ = build_xy(df)
            return X, y_binary
        if task == "multiclass":
            return malware_family_xy(df)
        raise ValueError(f"unknown task: {task}")
    if dataset == "ember":
        if task == "binary":
            X = _ember_features(df)
            y = df["label"].astype(int).rename("y_binary")
            return X, y
        if task == "multiclass":
            return ember_family_xy(df)
        raise ValueError(f"unknown task: {task}")
    raise ValueError(f"unknown dataset: {dataset}")


def split_paths(dataset: str, task: str) -> tuple[str, str]:
    """Paths for the persisted quantum subsample index and outer folds.

    CIC keeps its original un-prefixed filenames — those JSONs are committed and
    are the existing contract between the classical and quantum CLIs. Other
    datasets get dataset-prefixed names alongside them.
    """
    if dataset == "cic":
        return (
            f"data/splits/quantum_sample_idx_{task}.json",
            f"data/splits/cic_{task}_quantum_folds.json",
        )
    return (
        f"data/splits/{dataset}_quantum_sample_idx_{task}.json",
        f"data/splits/{dataset}_{task}_quantum_folds.json",
    )


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


def save_sample_idx(sample_idx, path: str) -> None:
    """Persist a row-index subsample as JSON so another framework can reload it."""
    with open(path, "w") as fh:
        json.dump(np.asarray(sample_idx).tolist(), fh)


def load_sample_idx(path: str):
    """Load a subsample saved by save_sample_idx back into a numpy index array."""
    with open(path) as fh:
        payload = json.load(fh)
    return np.array(payload)


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

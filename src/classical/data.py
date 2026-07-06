"""Load CIC-MalMem-2022 and construct binary and multiclass labels."""
from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import LabelEncoder

FEATURE_LABEL_COLS = ("Category", "Class")


def load_cic_malmem(csv_path: str) -> pd.DataFrame:
    """Read the raw CIC-MalMem-2022 CSV into a DataFrame."""
    return pd.read_csv(csv_path)


def build_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Split a raw frame into features X, binary y, and multiclass (family) y.

    Binary: Class == "Malware" -> 1, else 0.
    Multiclass: family = first token of Category (e.g. "Ransomware", "Benign").
    """
    X = df.drop(columns=list(FEATURE_LABEL_COLS))
    y_binary = (df["Class"].str.strip().str.lower() == "malware").astype(int)
    y_binary = pd.Series(y_binary, name="y_binary")

    family = df["Category"].str.split("-").str[0].str.strip()
    y_multiclass = pd.Series(
        LabelEncoder().fit_transform(family), name="y_multiclass"
    )
    return X, y_binary, y_multiclass

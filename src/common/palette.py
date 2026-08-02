"""One colour vocabulary shared by every figure in the project.

Figures were previously coloured per-script, which let the same hue mean three
different things across a single document: `#D55E00` was "random forest" in the
results chart, "RBF control" in the kernel chart, and "CIC-MalMem" in the
geometry chart. A reader who learns a colour in one figure should not be misled
by it in the next.

The rule is that hue *family* encodes the semantic family:

    warm (orange / vermillion / pink / brown)  ->  classical
    cool (blue / green)                        ->  quantum
    purple and olive                           ->  datasets
    grey                                       ->  reference / null / control

Datasets deliberately sit outside both framework families, so a dataset-keyed
chart can never borrow a model's colour. Within the classical and quantum
families a specific hue identifies the specific model or encoding.

Base hues are Okabe-Ito, which stays legible in the common forms of colour
blindness and in greyscale print.
"""
from __future__ import annotations

# --- classical: warm -------------------------------------------------------
CLASSICAL = {
    "random_forest": "#D55E00",   # vermillion
    "xgboost":       "#E69F00",   # orange
    "lightgbm":      "#CC79A7",   # pink
    "svm":           "#8C510A",   # brown
}

# --- quantum: cool ---------------------------------------------------------
QUANTUM = {
    "qsvm":       "#0072B2",      # blue -- the QSVM as a single series
    "qsvm-angle": "#0072B2",      # blue -- angle encoding
    "qsvm-iqp":   "#009E73",      # green -- IQP encoding
}

MODEL = {**CLASSICAL, **QUANTUM}

# --- framework-level, for charts that compare the two sides directly -------
FRAMEWORK = {
    "classical": "#D55E00",
    "quantum":   "#0072B2",
}

# --- kernels: a fidelity kernel is quantum, an RBF kernel is classical, so
#     these intentionally reuse the framework hues rather than inventing new
#     ones. The families still read correctly.
KERNEL = {
    "qsvm_fidelity": QUANTUM["qsvm"],
    "rbf_control":   FRAMEWORK["classical"],
}

# --- datasets: outside both framework families -----------------------------
DATASET = {
    "cic-malmem": "#6A3D9A",      # purple
    "ember-2018": "#767B33",      # olive
    "cic":        "#6A3D9A",      # short aliases used by some drivers
    "ember":      "#767B33",
}

# A dataset with two tasks keeps the dataset hue and varies lightness, so the
# dataset stays identifiable and the task is a secondary distinction.
DATASET_TASK = {
    ("cic-malmem", "binary"):     "#9D77C1",
    ("cic-malmem", "multiclass"): "#6A3D9A",
    ("ember-2018", "binary"):     "#A8AE6B",
    ("ember-2018", "multiclass"): "#767B33",
}

# --- reference / null / control --------------------------------------------
NEUTRAL = "#BBBBBB"
RULE = "0.35"                     # chance lines, zero lines, annotations

# --- markers, kept alongside the colours so the two never drift apart ------
MARKER = {
    "random_forest": "o", "xgboost": "s", "lightgbm": "^", "svm": "D",
    "qsvm": "v", "qsvm-angle": "v", "qsvm-iqp": "P",
}

LABEL = {
    "random_forest": "Random forest", "xgboost": "XGBoost",
    "lightgbm": "LightGBM", "svm": "SVM",
    "qsvm": "QSVM", "qsvm-angle": "QSVM (angle)", "qsvm-iqp": "QSVM (IQP)",
    "cic-malmem": "CIC-MalMem", "ember-2018": "EMBER 2018",
    "binary": "binary", "multiclass": "15-class",
}

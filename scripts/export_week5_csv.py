"""Flatten Week 5's derived results into committable CSVs.

`results/mlflow_runs.csv` carries the RAW per-run metrics, but every Week 5
conclusion is a DERIVED quantity — a sweep-level aggregate, a paired test, a
geometry statistic — and those lived only in JSON under `docs/reports/logs/`.
JSON is fine for an audit trail and poor for a spreadsheet or a paper table.

`results/` is gitignored except for `mlflow_runs.csv` (see .gitignore:290), so
these land under `docs/reports/logs/w5_csv/`, which is tracked.

Regenerates from the committed JSON plus the MLflow export, so it is
reproducible rather than hand-assembled.

Run: uv run python scripts/export_week5_csv.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from loguru import logger

OUT_DIR = Path("docs/reports/logs/w5_csv")
MLFLOW_CSV = "results/mlflow_runs.csv"
DAY31 = "docs/reports/logs/w5_day31/geometry.json"
DAY33 = "docs/reports/logs/w5_day33/alignment.json"
DAY33_BASE = "docs/reports/logs/w5_day33/alignment_baseline.json"
DAY34_SIG = "docs/reports/logs/w5_day34/significance.json"
DAY34_MCN = "docs/reports/logs/w5_day34/mcnemar_classical.json"


def clean_sweeps(df: pd.DataFrame, metric: str = "metrics.f1_macro") -> pd.DataFrame:
    """Fold rows belonging to a FINISHED parent with exactly 5 children.

    Same rule as `common.significance.fold_scores` — see that module for why
    cell params do not identify a CV run.
    """
    parents = df.set_index("run_id")
    folds = df[df["parent_run_id"].notna() & df[metric].notna()].copy()
    folds["parent_status"] = folds["parent_run_id"].map(parents["status"])
    sizes = folds.groupby("parent_run_id").size()
    keep = set(sizes[sizes == 5].index) & set(
        folds.loc[folds["parent_status"] == "FINISHED", "parent_run_id"]
    )
    return folds[folds["parent_run_id"].isin(keep)]


def master_tables(df: pd.DataFrame) -> pd.DataFrame:
    """One row per clean sweep: the aggregate every report table is built from."""
    parents = df.set_index("run_id")
    rows = []
    for parent_id, group in clean_sweeps(df).groupby("parent_run_id"):
        parent = parents.loc[parent_id]
        child_encodings = sorted(set(group["params.encoding"].dropna()))
        tag = parent.get("params.encoding")
        if not isinstance(tag, str) or not tag:
            tag = (child_encodings[0] if len(child_encodings) == 1
                   else "joint") if child_encodings else ""
        rows.append({
            "dataset": group["params.dataset"].iloc[0],
            "task": group["params.task"].iloc[0],
            "n_components": int(group["params.n_components"].iloc[0]),
            "framework": group["params.framework"].iloc[0],
            "model": group["params.model"].iloc[0],
            "encoding": tag,
            "f1_macro_mean": group["metrics.f1_macro"].mean(),
            "f1_macro_std": group["metrics.f1_macro"].std(ddof=0),
            "accuracy_mean": group["metrics.accuracy"].mean(),
            "gram_offdiag_std_mean": group["metrics.gram_offdiag_std"].mean(),
            "n_folds": len(group),
            "sweep_start": parent["start_time"],
            "parent_run_id": parent_id,
        })
    out = pd.DataFrame(rows)
    # `is_latest` marks the sweep each report table actually quoted, without
    # discarding the superseded ones — a reader can see what was replaced.
    key = ["dataset", "task", "n_components", "model", "encoding"]
    out = out.sort_values(key + ["sweep_start"])
    out["is_latest"] = ~out.duplicated(key, keep="last")
    return out


def significance() -> pd.DataFrame:
    raw = json.load(open(DAY34_SIG))
    rows = [{
        "dataset": c["dataset"], "task": c["task"],
        "n_components": c["n_components"],
        "quantum": c["quantum"], "classical": c["classical"],
        "qsvm_f1_macro_mean": c["qsvm_mean"],
        "classical_f1_macro_mean": c["classical_mean"],
        "delta_classical_minus_qsvm": c["delta"],
        "ttest_statistic": c["ttest"]["statistic"],
        "ttest_pvalue": c["ttest"]["pvalue"],
        "wilcoxon_statistic": c["wilcoxon"]["statistic"],
        "wilcoxon_pvalue": c["wilcoxon"]["pvalue"],
        "ttest_significant_at_05": c["ttest"]["pvalue"] < 0.05,
    } for c in raw["comparisons"]]
    return pd.DataFrame(rows)


def skipped() -> pd.DataFrame:
    """Cells that could not be tested. Absence is a result; keep it in the data."""
    raw = json.load(open(DAY34_SIG))
    rows = []
    for line in raw["skipped"]:
        cell, reason = line.split(": ", 1)
        parts = cell.split("/")
        rows.append({
            "dataset": parts[0], "task": parts[1],
            "n_components": int(parts[2].removeprefix("nc=")),
            "model": parts[3], "encoding": parts[4] if len(parts) > 4 else "",
            "reason": reason,
        })
    return pd.DataFrame(rows)


def mcnemar() -> pd.DataFrame:
    raw = json.load(open(DAY34_MCN))
    return pd.DataFrame([{
        "group": r["group"], "model_a": r["pair"][0], "model_b": r["pair"][1],
        "shared_test_set": r["shared_test_set"], "n_samples": r["n"],
        "n01_a_wrong_b_right": r.get("n01"), "n10_a_right_b_wrong": r.get("n10"),
        "statistic": r.get("statistic"), "pvalue": r.get("pvalue"),
        "significant_at_05": (r["pvalue"] < 0.05) if "pvalue" in r else None,
        "skipped": r.get("skipped", ""),
    } for r in raw])


def geometry() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = json.load(open(DAY31))
    rep, geo = [], []
    for dataset, d in raw.items():
        rep.append({"dataset": dataset, "source": d["source"],
                    "n_full_rows": d["n_full_rows"],
                    "n_subsample_rows": d["n_subsample_rows"],
                    **d["representativeness"]})
        for nc, stats in d["geometry"].items():
            row = {"dataset": dataset, "n_components": int(nc)}
            for name, ms in stats.items():
                row[f"{name}_mean"] = ms["mean"]
                row[f"{name}_std"] = ms["std"]
            geo.append(row)
    return pd.DataFrame(rep), pd.DataFrame(geo)


def kernel_diagnostics() -> pd.DataFrame:
    raw = json.load(open(DAY33))
    base = json.load(open(DAY33_BASE))
    rows = []
    for dataset, d in raw.items():
        b = base[dataset]["constant_kernel_alignment_baseline"]
        for kernel, key in (("qsvm_fidelity", "quantum"), ("rbf_control", "rbf_control")):
            rows.append({
                "dataset": dataset, "kernel": kernel,
                "n_train_rows": d["n_train_rows"], "n_qubits": d["n_qubits"],
                "offdiag_std": d[key]["offdiag_std"],
                "alignment": d[key]["alignment"],
                "constant_kernel_baseline": b,
                # Raw alignment is dominated by the 1/sqrt(n_classes) floor, so
                # the excess is the column that carries information.
                "alignment_excess_over_baseline": d[key]["alignment"] - b,
                "rbf_gamma": d[key].get("gamma"),
            })
    return pd.DataFrame(rows)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(MLFLOW_CSV)
    rep, geo = geometry()
    tables = {
        "w5_master_tables.csv": master_tables(df),
        "w5_significance_ttest_wilcoxon.csv": significance(),
        "w5_significance_skipped.csv": skipped(),
        "w5_mcnemar_classical.csv": mcnemar(),
        "w5_subsample_representativeness.csv": rep,
        "w5_feature_geometry.csv": geo,
        "w5_kernel_diagnostics.csv": kernel_diagnostics(),
    }
    for name, table in tables.items():
        path = OUT_DIR / name
        table.to_csv(path, index=False)
        logger.info(f"wrote {len(table):4d} rows x {len(table.columns):2d} cols -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

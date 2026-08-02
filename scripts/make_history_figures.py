"""Figures for the cumulative Week 1 -> Week 5 report.

Separate from `make_report_figures.py`, which covers Week 5 only. These cover
work from earlier weeks that was written up in prose and never plotted.

Sources, and why they differ:
  * Weeks 2-5 read `results/mlflow_runs.csv` -- every run of those weeks is in
    the experiment log.
  * Week 1 is hardcoded below. Its runs predate MLflow tracking in this project
    (every logged run carries an `n_components`, which Week 1's full-dataset
    baselines did not use), so the numbers come from the Week 1 report and are
    marked at their point of use.

Run: uv run python scripts/make_history_figures.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from loguru import logger  # noqa: E402

from common import palette  # noqa: E402
from common.significance import clean_sweep_folds  # noqa: E402

MLFLOW_CSV = "results/mlflow_runs.csv"
TARGETS = {
    "paper":  {"dir": Path("docs/paper/figures_history"),        "base": 9.0,  "h": 2.5},
    "slides": {"dir": Path("docs/paper/figures_history_slides"), "base": 11.5, "h": 3.0},
}
FULL_W = 6.3
OUT_DIR = TARGETS["paper"]["dir"]
HALF_H = 2.5
BAR_FS = 6.5

CLASSICAL = ["random_forest", "xgboost", "lightgbm", "svm"]
# Shared vocabulary -- see common.palette for what each hue family means.
NICE = palette.LABEL
COLOR = palette.MODEL
MARKER = palette.MARKER

# --- Week 1, from week1_cic_malmem_classical_baseline_report.md -------------
# Full CIC-MalMem, no PCA, no subsampling. These predate experiment tracking in
# this project, so they are transcribed rather than read from the log.
W1_BINARY = {"random_forest": 0.99995, "lightgbm": 0.99990,
             "svm": 0.99986, "xgboost": 0.99985}
# Multiclass BEFORE the malware-only reframe: benign still included as a class.
W1_MULTI = {"lightgbm": 0.5738, "random_forest": 0.5726,
            "xgboost": 0.5676, "svm": 0.3853}
W1_MULTI_STD = {"lightgbm": 0.0030, "random_forest": 0.0060,
                "xgboost": 0.0044, "svm": 0.0038}
W1_TRAIN_S = {"random_forest": 971, "xgboost": 1095,
              "lightgbm": 1871, "svm": 4302}


def setup_style(base: float) -> None:
    have_lm = any("Latin Modern Roman" in f.name
                  for f in matplotlib.font_manager.fontManager.ttflist)
    matplotlib.rcParams.update({
        "text.usetex": False,      # usetex drops minus signs on this TeX Live
        "font.family": "serif",
        "font.serif": (["Latin Modern Roman"] if have_lm else []) + ["DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "font.size": base, "axes.labelsize": base, "axes.titlesize": base,
        "legend.fontsize": base - 1.5,
        "xtick.labelsize": base - 1.0, "ytick.labelsize": base - 1.0,
        "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "grid.linewidth": 0.4, "grid.alpha": 0.35,
        "lines.linewidth": 1.2 * base / 9.0, "lines.markersize": 4 * base / 9.0,
        "legend.frameon": False, "pdf.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "axes.unicode_minus": True,
    })


def _nc():
    return r"$n_{\mathrm{components}}$"


def sweeps(df: pd.DataFrame) -> pd.DataFrame:
    """One row per clean sweep, latest per cell."""
    parents = df.set_index("run_id")
    rows = []
    for pid, g in clean_sweep_folds(df).groupby("parent_run_id"):
        p = parents.loc[pid]
        encs = sorted(set(g["params.encoding"].dropna()))
        tag = p.get("params.encoding")
        if not isinstance(tag, str) or not tag:
            tag = (encs[0] if len(encs) == 1 else "joint") if encs else ""
        rows.append({
            "dataset": g["params.dataset"].iloc[0],
            "task": g["params.task"].iloc[0],
            "nc": int(g["params.n_components"].iloc[0]),
            "model": g["params.model"].iloc[0],
            "encoding": tag,
            "mean": g["metrics.f1_macro"].mean(),
            "std": g["metrics.f1_macro"].std(ddof=0),
            "start": p["start_time"],
        })
    out = pd.DataFrame(rows).sort_values("start")
    key = ["dataset", "task", "nc", "model", "encoding"]
    return out[~out.duplicated(key, keep="last")]


# ---------------------------------------------------------------- week 1
def fig_week1_baseline() -> None:
    """The result that set the project's direction: binary is saturated."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_W, HALF_H))
    order = ["random_forest", "xgboost", "lightgbm", "svm"]
    x = np.arange(len(order))
    cols = [COLOR[m] for m in order]

    ax1.bar(x, [W1_BINARY[m] for m in order], 0.62, color=cols)
    ax1.set_ylim(0.995, 1.0005)
    ax1.set_yticks([0.995, 0.9975, 1.0])
    ax1.set_title("(a) Binary \u2014 saturated")
    ax1.set_ylabel("macro-F1")
    for i, m in enumerate(order):
        ax1.text(i, W1_BINARY[m] - 0.0004, f"{W1_BINARY[m]:.5f}", ha="center",
                 va="top", fontsize=BAR_FS, color="white", rotation=90)

    ax2.bar(x, [W1_MULTI[m] for m in order], 0.62, color=cols,
            yerr=[W1_MULTI_STD[m] for m in order], capsize=2.5,
            error_kw={"elinewidth": 0.7})
    ax2.set_ylim(0, 0.68)
    ax2.set_title("(b) Multiclass, benign included")
    ax2.set_ylabel("macro-F1")
    for i, m in enumerate(order):
        ax2.text(i, W1_MULTI[m] + 0.02, f"{W1_MULTI[m]:.3f}", ha="center",
                 fontsize=BAR_FS)

    for ax in (ax1, ax2):
        ax.set_xticks(x, [NICE[m] for m in order], rotation=18, ha="right")
        ax.grid(True, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_w1_baseline.pdf")
    plt.close(fig)


def fig_week1_cost() -> None:
    """Why the project could not simply scale up: classical tuning cost."""
    fig, ax = plt.subplots(figsize=(FULL_W * 0.54, HALF_H))
    order = sorted(W1_TRAIN_S, key=W1_TRAIN_S.get)
    y = np.arange(len(order))
    bars = ax.barh(y, [W1_TRAIN_S[m] / 60 for m in order], 0.6,
                   color=[COLOR[m] for m in order])
    ax.bar_label(bars, fmt="%.0f min", fontsize=BAR_FS, padding=2)
    ax.set_yticks(y, [NICE[m] for m in order])
    ax.set_xlabel("minutes per fold, incl. tuning")
    ax.set_xlim(0, 92)
    ax.grid(True, axis="x")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_w1_cost.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- week 2
def fig_binary_ceiling(df: pd.DataFrame) -> None:
    """CIC binary: every model within noise of every other.

    Takes the RAW fold rows, not clean sweeps. Every CIC binary run predates
    parent/child run tracking, so no sweep boundary is recoverable and
    `clean_sweep_folds` returns nothing for this cell. The individual fold
    scores are still valid measurements; they simply cannot be attributed to a
    named sweep, so this aggregates over every recorded fold and the caption
    says so.
    """
    b = df[(df["params.dataset"] == "cic-malmem")
           & (df["params.task"] == "binary")
           & df["metrics.f1_macro"].notna()]
    fig, ax = plt.subplots(figsize=(FULL_W * 0.60, HALF_H))
    for model in CLASSICAL + ["qsvm"]:
        g = b[b["params.model"] == model].groupby("params.n_components")
        if not len(g):
            continue
        m, sd = g["metrics.f1_macro"].mean(), g["metrics.f1_macro"].std(ddof=0)
        ax.errorbar(m.index, m.values, yerr=sd.values, color=COLOR[model],
                    marker=MARKER[model], capsize=1.8, elinewidth=0.6,
                    linestyle="--" if model == "qsvm" else "-",
                    label=NICE[model])
    ax.set_xlabel(_nc())
    ax.set_ylabel("macro-F1")
    ax.set_ylim(0.965, 1.005)
    ax.set_xticks([1, 2, 3, 4, 6])
    ax.grid(True, axis="y")
    # Five entries do not fit beside the data once the slide font is applied;
    # putting the legend under the axes keeps the curves unobstructed at both
    # type sizes.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3,
              columnspacing=1.2, handlelength=1.7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_cic_binary_ceiling.pdf")
    plt.close(fig)


def fig_quantum_cost(df: pd.DataFrame) -> None:
    """What the O(n^2) kernel actually costs as the qubit budget grows."""
    # One session only. Worker counts and the parallel Gram path changed across
    # the project, and per-pair wall clock is sensitive to both -- pooling
    # sessions produced a spurious dip at the largest qubit count, because that
    # point existed only in the later, more parallel session. 2026-07-25 is the
    # one session covering n_components 1/3/6/8 under a single configuration.
    SESSION = "2026-07-25"
    q = df[(df["params.model"] == "qsvm")
           & (df["params.dataset"] == "cic-malmem")
           & df["start_time"].str.startswith(SESSION)
           & df["metrics.kernel_build_train_s"].notna()
           & (df["metrics.kernel_evals"] > 0)].copy()
    # Subsample size changed across the project (200 -> 400 -> 1000 rows), so a
    # raw wall-clock comparison across n_components would mostly measure how
    # many pairs each run happened to evaluate. Dividing by the evaluation
    # count isolates the per-circuit cost, which is what grows with qubits.
    q["us_per_eval"] = (q["metrics.kernel_build_train_s"]
                        / q["metrics.kernel_evals"] * 1e6)
    g = q.groupby("params.n_components")["us_per_eval"]
    m, s = g.mean(), g.std(ddof=0)
    fig, ax = plt.subplots(figsize=(FULL_W * 0.52, HALF_H))
    ax.errorbar(m.index, m.values, yerr=s.values,
                color=palette.FRAMEWORK["quantum"], marker="o",
                capsize=2, elinewidth=0.7)
    ax.set_xlabel(_nc() + "  (= qubits)")
    ax.set_ylabel(r"cost per circuit pair ($\mu$s)")
    ax.set_xticks(sorted(m.index))
    ax.set_ylim(0, max(m.values + s.values) * 1.18)
    ax.grid(True, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_quantum_cost.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- week 4
def fig_encoding_verdict(sw: pd.DataFrame) -> None:
    """angle vs IQP: a clean win on EMBER, inside the noise on CIC."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_W, HALF_H))
    panels = [(ax1, "cic-malmem", "CIC-MalMem 15-class", [1, 3, 6, 8]),
              (ax2, "ember-2018", "EMBER 15-class", [1, 3, 6])]
    for ax, dataset, title, ncs in panels:
        s = sw[(sw.dataset == dataset) & (sw.task == "multiclass")
               & sw.encoding.isin(["angle", "iqp"])]
        a = s[s.encoding == "angle"].set_index("nc").reindex(ncs)
        i = s[s.encoding == "iqp"].set_index("nc").reindex(ncs)
        delta = i["mean"] - a["mean"]
        # The comparison that matters is the delta against the fold-to-fold
        # spread of the encodings themselves: a delta inside that band is not
        # a result. Band = larger of the two per-encoding stds.
        band = np.maximum(a["std"].values, i["std"].values)
        x = np.arange(len(ncs))
        ax.fill_between(x, -band, band, color="0.85",
                        label="fold-to-fold spread")
        ax.bar(x, delta.values, 0.45, color=palette.QUANTUM["qsvm-iqp"],
               label="IQP $-$ angle")
        ax.axhline(0, color="0.2", lw=0.7)
        ax.set_xticks(x, [str(n) for n in ncs])
        ax.set_xlabel(_nc())
        ax.set_title(title)
        ax.grid(True, axis="y")
    ax1.set_ylabel("macro-F1 difference")
    ax1.set_ylim(-0.09, 0.12)
    ax2.set_ylim(-0.09, 0.12)
    ax2.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_encoding_verdict.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- overall
RAW: pd.DataFrame


def fig_project_arc(sw: pd.DataFrame) -> None:
    """Best classical against best quantum in every cell that can be compared."""
    cells = [("cic-malmem", "binary", "CIC\nbinary"),
             ("cic-malmem", "multiclass", "CIC\n15-class"),
             ("ember-2018", "binary", "EMBER\nbinary"),
             ("ember-2018", "multiclass", "EMBER\n15-class")]
    # Restrict to the nc grid the master tables report. Without this the max
    # would pick up an nc=2/4/5 sweep from an earlier week and silently mix
    # configurations across weeks.
    GRID = [1, 3, 6]
    best_c, best_q, labels, from_raw = [], [], [], []
    for dataset, task, label in cells:
        s = sw[(sw.dataset == dataset) & (sw.task == task) & sw.nc.isin(GRID)]
        c, q = s[s.model.isin(CLASSICAL)], s[s.model == "qsvm"]
        if c.empty or q.empty:
            # CIC binary: no sweep is recoverable, so fall back to raw folds.
            r = RAW[(RAW["params.dataset"] == dataset)
                    & (RAW["params.task"] == task)
                    & RAW["params.n_components"].isin(GRID)]
            if r.empty:
                continue
            rc = r[r["params.model"].isin(CLASSICAL)]
            rq = r[r["params.model"] == "qsvm"]
            if rc.empty or rq.empty:
                continue
            best_c.append(rc.groupby("params.model")["metrics.f1_macro"].mean().max())
            best_q.append(rq["metrics.f1_macro"].mean())
            from_raw.append(True)
        else:
            best_c.append(c["mean"].max())
            best_q.append(q["mean"].max())
            from_raw.append(False)
        labels.append(label)
    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(FULL_W * 0.56, HALF_H))
    hatch = ["//" if r else "" for r in from_raw]
    b1 = ax.bar(x - w / 2, best_c, w, label="best classical",
                color=palette.FRAMEWORK["classical"],
                hatch=hatch, edgecolor="white", linewidth=0)
    b2 = ax.bar(x + w / 2, best_q, w, label="best QSVM",
                color=palette.FRAMEWORK["quantum"],
                hatch=hatch, edgecolor="white", linewidth=0)
    ax.bar_label(b1, fmt="%.3f", fontsize=BAR_FS - 0.5, padding=6, rotation=90)
    ax.bar_label(b2, fmt="%.3f", fontsize=BAR_FS - 0.5, padding=6, rotation=90)
    ax.set_xticks(x, labels)
    ax.set_ylabel("best macro-F1 attained")
    ax.set_ylim(0, 1.30)
    ax.grid(True, axis="y")
    ax.legend(loc="upper right", ncol=1)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_project_arc.pdf")
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", choices=sorted(TARGETS) + ["all"], default="paper")
    args = ap.parse_args(argv)
    names = sorted(TARGETS) if args.target == "all" else [args.target]

    global OUT_DIR, BAR_FS, HALF_H
    global RAW
    df = pd.read_csv(MLFLOW_CSV)
    RAW = df[df["metrics.f1_macro"].notna()]
    sw = sweeps(df)
    for name in names:
        spec = TARGETS[name]
        OUT_DIR, HALF_H = spec["dir"], spec["h"]
        BAR_FS = 6.5 * spec["base"] / 9.0
        setup_style(spec["base"])
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        fig_week1_baseline()
        fig_week1_cost()
        fig_binary_ceiling(df)
        fig_quantum_cost(df)
        fig_encoding_verdict(sw)
        fig_project_arc(sw)
        logger.info(f"[{name}] base={spec['base']}pt")
        for p in sorted(OUT_DIR.glob("*.pdf")):
            logger.info(f"    {p.name}  ({p.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

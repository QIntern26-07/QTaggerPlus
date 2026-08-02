"""Render the Week 5 report figures as vector PDFs for LaTeX inclusion.

Reads only the committed CSVs under `docs/reports/logs/w5_csv/`, so a figure can
never disagree with the numbers in the reports. Nothing is recomputed here.

Font matching without `text.usetex`. The document is typeset in Latin Modern,
so the figures set `font.serif = ["Latin Modern Roman"]` and render maths with
matplotlib's `cm` fontset — the same typeface, reached without invoking LaTeX.

`text.usetex = True` was tried first and rejected: on this TeX Live it silently
drops the minus sign from negative tick labels, so a silhouette of -0.22 prints
as "0.22" and the sign of the result inverts. Setting `axes.unicode_minus =
False` does not fix it, and neither does forcing tick labels through an explicit
math-mode formatter. A figure that misstates a sign is worse than one whose
font is a hair off, and in this case there is no trade-off — Latin Modern is
the same font LaTeX would have used.

Run: uv run python scripts/make_report_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from loguru import logger  # noqa: E402

CSV_DIR = Path("docs/reports/logs/w5_csv")
OUT_DIR = Path("docs/paper/figures")

# Column width of a standard article at 10pt is ~4.8in; these are sized to be
# placed at \textwidth without rescaling, so fonts stay at their true size.
FULL_W = 6.3
HALF_H = 2.5

CLASSICAL = ["random_forest", "xgboost", "lightgbm", "svm"]
NICE = {
    "random_forest": "Random forest", "xgboost": "XGBoost",
    "lightgbm": "LightGBM", "svm": "SVM",
    "qsvm-angle": "QSVM (angle)", "qsvm-iqp": "QSVM (IQP)",
    "cic-malmem": "CIC-MalMem", "ember-2018": "EMBER 2018",
    "binary": "binary", "multiclass": "15-class",
}
# Colour-blind-safe (Okabe-Ito). Classical models share a warm family, quantum a
# cool one, so the two frameworks separate at a glance even in greyscale print.
COLOR = {
    "random_forest": "#D55E00", "xgboost": "#E69F00",
    "lightgbm": "#CC79A7", "svm": "#8C510A",
    "qsvm-angle": "#0072B2", "qsvm-iqp": "#009E73",
}
MARKER = {
    "random_forest": "o", "xgboost": "s", "lightgbm": "^", "svm": "D",
    "qsvm-angle": "v", "qsvm-iqp": "P",
}
RANDOM_15 = 1.0 / 15.0


def setup_style() -> bool:
    have_lm = any("Latin Modern Roman" in f.name
                  for f in matplotlib.font_manager.fontManager.ttflist)
    matplotlib.rcParams.update({
        "text.usetex": False,          # see module docstring — drops minus signs
        "font.family": "serif",
        "font.serif": (["Latin Modern Roman"] if have_lm else []) + ["DejaVu Serif"],
        "mathtext.fontset": "cm",      # Computer Modern maths, incl. the minus
        "axes.formatter.use_mathtext": True,
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.unicode_minus": True,
    })
    return have_lm


def _nc_label() -> str:
    return r"$n_{\mathrm{components}}$"


def load_master() -> pd.DataFrame:
    df = pd.read_csv(CSV_DIR / "w5_master_tables.csv")
    df = df[df["is_latest"] & df["n_components"].isin([1, 3, 6])].copy()
    df["series"] = np.where(
        df["model"] == "qsvm", "qsvm-" + df["encoding"].fillna(""), df["model"]
    )
    return df


def fig_capacity_scaling(master: pd.DataFrame) -> None:
    """Macro-F1 against qubit/feature budget, per dataset and task."""
    panels = [("cic-malmem", "multiclass"), ("ember-2018", "binary"),
              ("ember-2018", "multiclass")]
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.35), sharex=True)
    for ax, (dataset, task) in zip(axes, panels):
        sub = master[(master["dataset"] == dataset) & (master["task"] == task)]
        for series in CLASSICAL + ["qsvm-angle", "qsvm-iqp"]:
            s = sub[sub["series"] == series].sort_values("n_components")
            if s.empty:
                continue
            ax.errorbar(s["n_components"], s["f1_macro_mean"],
                        yerr=s["f1_macro_std"], label=NICE.get(series, series),
                        color=COLOR[series], marker=MARKER[series],
                        capsize=1.8, elinewidth=0.6,
                        linestyle="--" if series.startswith("qsvm") else "-")
        if task == "multiclass":
            ax.axhline(RANDOM_15, color="0.35", lw=0.7, ls=":", zorder=0)
            # Right edge, below the line: the only region no series occupies in
            # either multiclass panel.
            ax.text(6.0, RANDOM_15 - 0.018, "chance", fontsize=6.5,
                    va="top", ha="right", color="0.35")
        ax.set_title(f"{NICE[dataset]}, {NICE[task]}")
        ax.set_xticks([1, 3, 6])
        ax.set_xlabel(_nc_label())
        ax.grid(True, axis="y")
        ax.set_ylim(0, 0.87)
    axes[0].set_ylabel("macro-F1")
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(bottom=0.34)
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.015), columnspacing=1.6,
               handlelength=1.8, labelspacing=0.35)
    fig.savefig(OUT_DIR / "fig_capacity_scaling.pdf", bbox_inches=None)
    plt.close(fig)


def fig_kernel_diagnostics() -> None:
    """The decisive experiment: fidelity kernel against an RBF control."""
    k = pd.read_csv(CSV_DIR / "w5_kernel_diagnostics.csv")
    order = ["cic", "ember"]
    names = {"cic": "CIC-MalMem", "ember": "EMBER 2018"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_W, HALF_H))
    x = np.arange(len(order))
    w = 0.34
    for i, (kern, label, col) in enumerate([
        ("qsvm_fidelity", "Fidelity kernel (QSVM)", "#0072B2"),
        ("rbf_control", "RBF control", "#D55E00"),
    ]):
        vals = [k[(k.dataset == d) & (k.kernel == kern)]["offdiag_std"].iloc[0]
                for d in order]
        bars = ax1.bar(x + (i - 0.5) * w, vals, w, label=label, color=col)
        ax1.bar_label(bars, fmt="%.3f", fontsize=6.5, padding=1.5)
    ax1.set_xticks(x, [names[d] for d in order])
    ax1.set_ylabel("Gram off-diagonal std")
    ax1.set_title("(a) Kernel concentration")
    ax1.set_ylim(0, 0.34)
    ax1.grid(True, axis="y")
    ax1.legend(loc="upper center", ncol=1)

    for i, (kern, label, col) in enumerate([
        ("qsvm_fidelity", "Fidelity kernel (QSVM)", "#0072B2"),
        ("rbf_control", "RBF control", "#D55E00"),
    ]):
        vals = [k[(k.dataset == d) & (k.kernel == kern)]
                ["alignment_excess_over_baseline"].iloc[0] for d in order]
        bars = ax2.bar(x + (i - 0.5) * w, vals, w, label=label, color=col)
        ax2.bar_label(bars, fmt="%+.4f", fontsize=6.5, padding=1.5)
    ax2.axhline(0, color="0.2", lw=0.7)
    ax2.set_xticks(x, [names[d] for d in order])
    ax2.set_ylabel("alignment above chance floor")
    ax2.set_title("(b) Kernel-target alignment")
    ax2.set_ylim(-0.016, 0.042)
    ax2.grid(True, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_kernel_diagnostics.pdf")
    plt.close(fig)


def fig_concentration_sweep(master: pd.DataFrame) -> None:
    """Concentration across the whole sweep, from values logged since Week 2."""
    df = pd.read_csv(CSV_DIR / "w5_master_tables.csv")
    df = df[df["is_latest"] & (df["model"] == "qsvm")
            & (df["task"] == "multiclass")
            & df["gram_offdiag_std_mean"].notna()]
    fig, ax = plt.subplots(figsize=(FULL_W * 0.52, HALF_H))
    for dataset, ls in [("cic-malmem", "-"), ("ember-2018", "--")]:
        for enc, col in [("angle", "#0072B2"), ("iqp", "#009E73")]:
            s = df[(df.dataset == dataset) & (df.encoding == enc)] \
                .sort_values("n_components")
            if s.empty:
                continue
            ax.plot(s["n_components"], s["gram_offdiag_std_mean"], ls=ls,
                    color=col, marker=MARKER[f"qsvm-{enc}"],
                    label=f"{NICE[dataset]}, {enc}")
    ax.set_xlabel(_nc_label())
    ax.set_ylabel("Gram off-diagonal std")
    ax.set_xticks([1, 3, 6, 8])
    ax.set_ylim(0, 0.38)
    ax.grid(True, axis="y")
    ax.legend(loc="upper right", ncol=1)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_concentration_sweep.pdf")
    plt.close(fig)


def fig_significance() -> None:
    """Effect size against p-value for all 84 tested pairs."""
    s = pd.read_csv(CSV_DIR / "w5_significance_ttest_wilcoxon.csv")
    fig, ax = plt.subplots(figsize=(FULL_W * 0.56, HALF_H + 0.2))
    groups = [("cic-malmem", "multiclass", "#D55E00", "o"),
              ("ember-2018", "binary", "#0072B2", "s"),
              ("ember-2018", "multiclass", "#009E73", "^")]
    for dataset, task, col, mk in groups:
        g = s[(s.dataset == dataset) & (s.task == task)]
        ax.scatter(g["ttest_pvalue"], g["delta_classical_minus_qsvm"],
                   s=14, color=col, marker=mk, alpha=0.85, linewidths=0,
                   label=f"{NICE[dataset]}, {NICE[task]}")
    ax.axvline(0.05, color="0.2", lw=0.8, ls="--")
    ax.text(0.043, 0.015, r"$p=0.05$", fontsize=7, color="0.2", ha="right")
    ax.set_xscale("log")
    ax.set_xlim(2e-6, 0.13)
    ax.set_xticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    ax.set_xlabel("paired $t$-test $p$-value")
    ax.set_ylabel("classical advantage (macro-F1)")
    ax.set_ylim(0, 0.55)
    ax.grid(True)
    ax.legend(loc="upper left", ncol=1)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_significance.pdf")
    plt.close(fig)


def fig_geometry() -> None:
    """Class separability of the projected features, CIC against EMBER."""
    g = pd.read_csv(CSV_DIR / "w5_feature_geometry.csv").sort_values("n_components")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_W, HALF_H))
    for dataset, col, mk in [("cic", "#D55E00", "o"), ("ember", "#0072B2", "s")]:
        s = g[g.dataset == dataset]
        label = NICE["cic-malmem"] if dataset == "cic" else NICE["ember-2018"]
        ax1.errorbar(s["n_components"], s["mean_fisher_ratio_mean"],
                     yerr=s["mean_fisher_ratio_std"], color=col, marker=mk,
                     capsize=1.8, elinewidth=0.6, label=label)
        ax2.errorbar(s["n_components"], s["silhouette_mean"],
                     yerr=s["silhouette_std"], color=col, marker=mk,
                     capsize=1.8, elinewidth=0.6, label=label)
    ax1.set_ylabel("mean Fisher ratio")
    ax1.set_title("(a) Class separability")
    ax2.axhline(0, color="0.2", lw=0.7, ls=":")
    ax2.set_ylabel("silhouette")
    ax2.set_title("(b) Cluster quality")
    for ax in (ax1, ax2):
        ax.set_xlabel(_nc_label())
        ax.set_xticks([1, 3, 6, 8])
        ax.grid(True, axis="y")
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_geometry.pdf")
    plt.close(fig)


def fig_representativeness() -> None:
    """KS rejections against what chance alone produces."""
    r = pd.read_csv(CSV_DIR / "w5_subsample_representativeness.csv")
    fig, ax = plt.subplots(figsize=(FULL_W * 0.52, HALF_H))
    names = {"cic": "CIC-MalMem", "ember": "EMBER 2018"}
    x = np.arange(len(r))
    w = 0.34
    b1 = ax.bar(x - w / 2, r["expected_rejects_under_null"], w,
                label="expected under the null", color="#BBBBBB")
    b2 = ax.bar(x + w / 2, r["n_reject"], w, label="observed",
                color="#0072B2")
    ax.bar_label(b1, fmt="%.2f", fontsize=6.5, padding=1.5)
    ax.bar_label(b2, fmt="%.0f", fontsize=6.5, padding=1.5)
    ax.set_xticks(x, [f"{names[d]}\n({int(n)} features)"
                      for d, n in zip(r["dataset"], r["n_features"])])
    ax.set_ylabel("KS rejections at $\\alpha=0.05$")
    ax.set_ylim(0, 168)
    ax.grid(True, axis="y")
    ax.legend(loc="upper left", ncol=1)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_representativeness.pdf")
    plt.close(fig)


def main() -> int:
    have_lm = setup_style()
    logger.info(f"Latin Modern available: {have_lm}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    master = load_master()
    fig_capacity_scaling(master)
    fig_kernel_diagnostics()
    fig_concentration_sweep(master)
    fig_significance()
    fig_geometry()
    fig_representativeness()
    for p in sorted(OUT_DIR.glob("*.pdf")):
        logger.info(f"  {p}  ({p.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

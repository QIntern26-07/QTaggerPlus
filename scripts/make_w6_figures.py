"""Render the Week 6 bandwidth figure as a vector PDF for LaTeX inclusion.

Reads only the committed CSVs under `docs/reports/logs/w6_csv/`, so the figure
can never disagree with the numbers in the report. Nothing is recomputed here.
Style and colour vocabulary come from `make_report_figures` and `common.palette`
so this figure sits beside the Week 5 set without a second visual language.

Two panels, one measure each. Panel (a) is Gram spread against bandwidth; panel
(b) is macro-F1 by model. Alignment is deliberately NOT plotted: it is flat
across the whole bracket, and putting it on panel (a) would require a second
y-scale.

Run: uv run python scripts/make_w6_figures.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from loguru import logger  # noqa: E402

from common import palette  # noqa: E402
from make_report_figures import FULL_W, TARGETS, setup_style  # noqa: E402

CSV_DIR = Path("docs/reports/logs/w6_csv")

# 15 balanced-ish classes; the chance level a macro-F1 must clear to mean
# anything. Stated in the report as approximately 0.067.
CHANCE = 1.0 / 15.0


def fig_bandwidth(out_dir: Path, half_h: float) -> None:
    bw = pd.read_csv(CSV_DIR / "w6_bandwidth_diagnostics.csv")
    cmp_ = pd.read_csv(CSV_DIR / "w6_sixfold_comparison.csv")

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(FULL_W, half_h))

    # --- (a) concentration against bandwidth -------------------------------
    ax_a.axhline(1.0, color=palette.KERNEL["rbf_control"], lw=1.0, ls="--",
                 zorder=1)
    ax_a.annotate("RBF control", xy=(0.052, 1.0), xytext=(0, 3),
                  textcoords="offset points",
                  color=palette.KERNEL["rbf_control"],
                  fontsize=matplotlib.rcParams["legend.fontsize"])
    ax_a.plot(bw["bandwidth"], bw["ratio_to_rbf"], marker="o",
              color=palette.KERNEL["qsvm_fidelity"], zorder=3)

    default = bw[bw["is_default"]].iloc[0]
    ax_a.plot([default["bandwidth"]], [default["ratio_to_rbf"]], marker="o",
              markersize=matplotlib.rcParams["lines.markersize"] * 2.0,
              markerfacecolor="none", markeredgewidth=1.2,
              color=palette.KERNEL["qsvm_fidelity"], zorder=4)
    ax_a.annotate("default", xy=(default["bandwidth"], default["ratio_to_rbf"]),
                  xytext=(-4, -13), textcoords="offset points", ha="right",
                  fontsize=matplotlib.rcParams["legend.fontsize"])

    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlabel("bandwidth")
    ax_a.set_ylabel("Gram spread / RBF")
    ax_a.set_title("(a) concentration is set by bandwidth", loc="left",
                   fontsize=matplotlib.rcParams["axes.titlesize"] - 1.5)
    ax_a.grid(True, which="major", axis="both")

    # --- (b) macro-F1 by model --------------------------------------------
    order = ["svm", "random_forest", "lightgbm", "xgboost"]
    classical = cmp_[cmp_["framework"] == "classical"].set_index("model")
    q_tuned = cmp_[(cmp_["framework"] == "quantum")
                   & (cmp_["bandwidth"].str.startswith("tuned"))].iloc[0]
    q_default = cmp_[(cmp_["framework"] == "quantum")
                     & (cmp_["bandwidth"].str.startswith("default"))].iloc[0]

    labels, values, errors, colors, hatches = [], [], [], [], []
    for m in order:
        row = classical.loc[m]
        labels.append(palette.LABEL[m])
        values.append(row["f1_macro_mean"])
        errors.append(row["f1_macro_std"])
        colors.append(palette.CLASSICAL[m])
        hatches.append("")
    for row, label, hatch in ((q_tuned, "QSVM (tuned)", ""),
                              (q_default, "QSVM (default)", "///")):
        labels.append(label)
        values.append(row["f1_macro_mean"])
        errors.append(row["f1_macro_std"])
        colors.append(palette.QUANTUM["qsvm-iqp"])
        hatches.append(hatch)

    bars = ax_b.bar(range(len(values)), values, yerr=errors, capsize=2,
                    color=colors, edgecolor="white", linewidth=0.8, zorder=3,
                    error_kw={"lw": 0.8, "zorder": 4})
    for bar, hatch in zip(bars, hatches):
        if hatch:
            bar.set_hatch(hatch)
            bar.set_edgecolor(palette.QUANTUM["qsvm-iqp"])
            bar.set_facecolor("white")

    # Blank gutter on the left so the chance label has somewhere to sit: the
    # line passes through every bar, so any label placed over the plot area
    # lands on a filled mark.
    ax_b.set_xlim(-1.85, len(values) - 0.4)
    ax_b.axhline(CHANCE, color=palette.RULE, lw=1.0, ls=":", zorder=2)
    ax_b.annotate("chance", xy=(-1.8, CHANCE), xytext=(0, 3),
                  textcoords="offset points", ha="left", color=palette.RULE,
                  fontsize=matplotlib.rcParams["legend.fontsize"])

    ax_b.set_xticks(range(len(labels)))
    ax_b.set_xticklabels(labels, rotation=30, ha="right")
    ax_b.set_ylabel("macro-F1")
    ax_b.set_title("(b) CIC 15-class, shared folds", loc="left",
                   fontsize=matplotlib.rcParams["axes.titlesize"] - 1.5)
    ax_b.grid(True, axis="y")
    ax_b.set_axisbelow(True)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig_bandwidth.pdf")
    plt.close(fig)
    logger.info(f"wrote {out_dir / 'fig_bandwidth.pdf'}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", choices=sorted(TARGETS), default="paper")
    args = p.parse_args(argv)
    spec = TARGETS[args.target]
    have_lm = setup_style(spec["base"])
    if not have_lm:
        logger.warning("Latin Modern Roman not found; falling back to DejaVu Serif")
    fig_bandwidth(spec["dir"], spec["half_h"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

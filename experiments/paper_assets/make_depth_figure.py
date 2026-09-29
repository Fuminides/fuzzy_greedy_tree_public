"""Matched-budget figure for the FERL paper (depth sweep of FERL-deep).

Reads results/ignorance_depth.csv (experiments/reliability/ds_ignorance_semantics.py
--study depth) and writes paper/figures/depth_matched.pdf plus
paper/generated/depth_summary.tex, which defines \\DepthSummary (the prose
summary quoted in the results section). Run from the repository root:

    python experiments/paper_assets/make_depth_figure.py
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

GEN = Path(os.environ.get("FERL_GEN_DIR", "paper/generated"))
FIG = Path(os.environ.get("FERL_FIG_DIR", "paper/figures"))
# Colours shared with the other paper figures (Okabe-Ito; FERL in vermilion).
C_FERL, C_CART, C_FIGS, C_GREY = "#D55E00", "#0072B2", "#009E73", "#767676"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "pdf.fonttype": 42,
})


def main():
    r = pd.read_csv("results/ignorance_depth.csv")
    per = r.groupby(["dataset", "max_depth"]).mean(numeric_only=True).reset_index()
    g = per.groupby("max_depth").mean(numeric_only=True)
    n_figs = per.groupby("max_depth").figs_leafmatch_accuracy.count()
    depths = g.index.values

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    ax = axes[0]
    ax.plot(depths, 100 * g.accuracy, "-o", c=C_FERL, lw=2, ms=4, label="FERL-deep")
    ax.plot(depths, 100 * g.cart_depth_accuracy, "-s", c=C_CART, lw=2, ms=4, label="CART")
    ax.set_xlabel("maximum depth")
    ax.set_ylabel("accuracy (%)")
    ax.set_xticks(depths)
    ax.legend(loc="lower right")
    ax.set_title("(a) same depth", loc="left", fontsize=9)

    ax = axes[1]
    ax.plot(g.n_leaves, 100 * g.accuracy, "-o", c=C_FERL, lw=2, ms=4, label="FERL-deep")
    ax.plot(g.n_leaves, 100 * g.cart_leafmatch_accuracy, "-s", c=C_CART, lw=2, ms=4,
            label="CART")
    full = n_figs == per.dataset.nunique()
    ax.plot(g.n_leaves[full], 100 * g.figs_leafmatch_accuracy[full], "-^", c=C_FIGS, lw=2,
            ms=4, label="FIGS")
    ax.set_xscale("log")
    ax.set_xlabel("leaves (log scale)")
    ax.set_ylabel("accuracy (%)")
    ax.legend(loc="lower right")
    ax.set_title("(b) same number of leaves", loc="left", fontsize=9)

    ax = axes[2]
    ax.plot(depths, 100 * g.coverage, "-o", c=C_FERL, lw=2, ms=4, label="set coverage")
    ax.plot(depths, 100 * g.ignorance, "--o", c=C_GREY, lw=2, ms=4, label="ignorance")
    ax.set_xlabel("maximum depth")
    ax.set_ylabel("%")
    ax.set_xticks(depths)
    ax.set_ylim(0, 100)
    ax.legend(loc="center right")
    ax.set_title("(c) FERL-deep read-out", loc="left", fontsize=9)

    fig.tight_layout(pad=1)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "depth_matched.pdf", dpi=300)
    plt.close(fig)

    d_lo, d_hi = depths.min(), depths.max()
    gap = 100 * (g.accuracy - g.cart_depth_accuracy)
    small = g.loc[4]
    big = g.loc[8]
    text = (
        f"FERL-deep is more accurate than CART grown to the same depth at every depth, "
        f"by {gap[d_lo]:.1f} points at depth {d_lo} and {gap[d_hi]:.1f} at depth {d_hi} "
        f"(Figure~\\ref{{fig:depth}}a). With the same number of leaves the comparison depends "
        f"on the budget (Figure~\\ref{{fig:depth}}b). At about {small.n_leaves:.0f} leaves, "
        f"best-first CART and FIGS use their leaves better ({100*small.cart_leafmatch_accuracy:.1f}\\% "
        f"and {100*small.figs_leafmatch_accuracy:.1f}\\% against {100*small.accuracy:.1f}\\%). "
        f"From about {big.n_leaves:.0f} leaves on, FERL-deep leads "
        f"({100*big.accuracy:.1f}\\% against {100*big.cart_leafmatch_accuracy:.1f}\\% for CART). "
        f"Depth-first growth spends leaves evenly across the tree, so at small budgets the "
        f"best-first variants FERL-compact and FERL-medium are the relevant ones. FIGS is "
        f"compared only up to 128 rules, which covers all datasets up to depth "
        f"{int(depths[full].max())}. As the tree deepens, ignorance grows from "
        f"{100*g.ignorance[d_lo]:.0f}\\% to {100*g.ignorance[d_hi]:.0f}\\% and set coverage from "
        f"{100*g.coverage[d_lo]:.0f}\\% to {100*g.coverage[d_hi]:.0f}\\% "
        f"(Figure~\\ref{{fig:depth}}c): more leaves mean more transition bands, as "
        f"Proposition~\\ref{{prop:representation}} anticipates."
    )
    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / "depth_summary.tex").write_text("\\newcommand{\\DepthSummary}{" + text + "}\n")
    print("wrote depth_matched.pdf and depth_summary.tex")


if __name__ == "__main__":
    main()

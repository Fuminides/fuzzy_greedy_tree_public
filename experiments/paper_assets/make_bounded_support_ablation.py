"""Generate the bounded-support inference ablation table.

Reads results/bounded_support_id.csv and results/bounded_support_ood.csv from
experiments/reliability/ds_bounded_support_ablation.py (main tabular protocol:
30 datasets, 5 folds) and writes tab_bounded_support_ablation.tex to
paper/generated/ (FERL_GEN_DIR overrides), plus prose macros in
bounded_macros.tex.
"""
import os
from pathlib import Path

import pandas as pd

ID_SOURCE = Path("results/bounded_support_id.csv")
OOD_SOURCE = Path("results/bounded_support_ood.csv")
GEN = Path(os.environ.get("FERL_GEN_DIR", "paper/generated"))
DEFAULT_MARGIN = "1.0"
MARGINS = [("unbounded", "Gate off"), ("0.5", "Margin 0.5"),
           (DEFAULT_MARGIN, "Margin 1 (used)"), ("2.0", "Margin 2")]


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}"


def main() -> None:
    ids = pd.read_csv(ID_SOURCE)
    ood = pd.read_csv(OOD_SOURCE)
    ood["margin"] = ood["margin"].astype(str)
    id_mean = ids.groupby("ds").mean(numeric_only=True).mean()
    ood_mean = (ood.groupby(["ds", "shift", "margin"]).mean(numeric_only=True)
                .groupby(["shift", "margin"]).mean())

    rows = []
    for margin, label in MARGINS:
        a, o = ood_mean.loc[("all", margin)], ood_mean.loc[("one", margin)]
        if margin == "unbounded":
            idc = f"{_pct(id_mean.acc_unb)} & {_pct(id_mean.cov_unb)} & {id_mean.size_unb:.2f}"
        elif margin == DEFAULT_MARGIN:
            idc = f"{_pct(id_mean.acc_bnd)} & {_pct(id_mean.cov_bnd)} & {id_mean.size_bnd:.2f}"
        else:
            idc = r"\multicolumn{3}{c}{--}"
        rows.append(f"{label} & {idc} & {_pct(a.auroc_firing)} & {_pct(a.auroc_ignorance)} & "
                    f"{_pct(a.phi_ood)} & {_pct(o.auroc_firing)} & {_pct(o.auroc_ignorance)}\\\\")

    n_ds, n_folds = ids.ds.nunique(), ids.fold.nunique()
    lines = [
        r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
        r"\caption{\textbf{Inference-only bounded-support ablation for FERL-deep} "
        f"({n_ds} datasets, {n_folds} folds; 0--100 scale except set size). The fitted "
        r"tree is unchanged; the support gate is switched off, or its margin changed, "
        r"only at inference. In-distribution columns use the soft-vote label and the "
        r"native leaves-only Dempster set. Geometric-OOD inputs move either every "
        r"feature (\emph{all}) or one randomly chosen feature (\emph{one}) of a test "
        r"input outside the training range; AUROC separates them from the unmodified "
        r"test inputs by the lost routing mass ($1-\sum_\ell\phi_\ell$) or by the "
        r"ignorance, and $\bar\Phi_{\mathrm{OOD}}$ is their mean total firing.}",
        r"\label{tab:bounded-support-ablation}",
        r"\begin{tabular}{@{}lccccccccc@{}}", r"\toprule",
        r"& \multicolumn{3}{c}{In distribution} & \multicolumn{3}{c}{OOD, all features} "
        r"& \multicolumn{2}{c}{OOD, one feature}\\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(l){8-9}",
        r"Support gate & Acc. & Cov. & Size & Firing & Ign. & $\bar\Phi_{\mathrm{OOD}}$ "
        r"& Firing & Ign.\\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / "tab_bounded_support_ablation.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    a_on, a_off = ood_mean.loc[("all", DEFAULT_MARGIN)], ood_mean.loc[("all", "unbounded")]
    o_on, o_off = ood_mean.loc[("one", DEFAULT_MARGIN)], ood_mean.loc[("one", "unbounded")]
    macros = {
        "BsAccOff": _pct(id_mean.acc_unb), "BsAccOn": _pct(id_mean.acc_bnd),
        "BsCovOff": _pct(id_mean.cov_unb), "BsCovOn": _pct(id_mean.cov_bnd),
        "BsSizeOff": f"{id_mean.size_unb:.2f}", "BsSizeOn": f"{id_mean.size_bnd:.2f}",
        "BsAllFireOff": _pct(a_off.auroc_firing), "BsAllFireOn": _pct(a_on.auroc_firing),
        "BsAllIgnOff": _pct(a_off.auroc_ignorance), "BsAllIgnOn": _pct(a_on.auroc_ignorance),
        "BsAllPhiOn": _pct(a_on.phi_ood), "BsPhiId": _pct(a_on.phi_id),
        "BsOneFireOff": _pct(o_off.auroc_firing), "BsOneFireOn": _pct(o_on.auroc_firing),
        "BsOneIgnOn": _pct(o_on.auroc_ignorance), "BsOnePhiOn": _pct(o_on.phi_ood),
        "BsNds": str(n_ds), "BsNfolds": str(n_folds),
    }
    (GEN / "bounded_macros.tex").write_text(
        "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()) + "\n")
    print(f"wrote {GEN / 'tab_bounded_support_ablation.tex'} and bounded_macros.tex")


if __name__ == "__main__":
    main()

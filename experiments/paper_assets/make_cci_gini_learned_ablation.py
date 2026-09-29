"""Ablation: split criterion for FERL-deep (the standalone LearnedFuzzyTree).

FERL-deep scores splits by weighted Gini, whereas the rest of the FERL family
uses the Complete Classification Index (CCI). This measures the switch on the
thirty tabular benchmarks with the complete benchmark protocol held fixed:
five seeded stratified outer folds, with 25% of each outer-training fold
reserved for calibration and therefore excluded from model fitting. The
held-out calibration data are unused by these native point predictors. The
script reports accuracy under each criterion and the class-count breakdown. It
writes tab_cci_gini_learned.tex (summary) and
tab_cci_gini_learned_full.tex (per-dataset) under paper/generated/ (FERL_GEN_DIR overrides).

Run from the repo root:
    python experiments/paper_assets/make_cci_gini_learned_ablation.py
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score

from ferl.pipeline.run_configs import SELECTED_30, load_filtered
from ferl.core.learned_tree import LearnedFuzzyTree

GEN = Path(os.environ.get("FERL_GEN_DIR", "paper/generated"))
GEN.mkdir(parents=True, exist_ok=True)
DEPTH, N_FOLDS, CAL_FRAC = 12, 5, 0.25


def evaluate(datasets=SELECTED_30):
    rows = []
    for name in datasets:
        try:
            X, y = load_filtered(name)
        except Exception as e:
            print(f"  {name}: SKIP ({e})", flush=True); continue
        C = len(np.unique(y))
        skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=33)
        acc = {"gini": [], "cci": []}
        for fold, (outer_train, te) in enumerate(skf.split(X, y)):
            try:
                train, _calibration = train_test_split(
                    outer_train,
                    test_size=CAL_FRAC,
                    random_state=fold,
                    stratify=y[outer_train],
                )
            except ValueError:
                train, _calibration = train_test_split(
                    outer_train,
                    test_size=CAL_FRAC,
                    random_state=fold,
                )
            for crit in ("gini", "cci"):
                m = LearnedFuzzyTree(max_depth=DEPTH, criterion=crit,
                                     random_state=0).fit(X[train], y[train])
                acc[crit].append(accuracy_score(y[te], m.predict(X[te])))
        g, c = float(np.mean(acc["gini"])), float(np.mean(acc["cci"]))
        rows.append(dict(Dataset=name, C=C, gini=g, cci=c, delta=c - g))
        print(f"  {name:14s} C={C:2d} gini={g*100:.2f} cci={c*100:.2f} "
              f"delta={(c-g)*100:+.2f}", flush=True)
    return pd.DataFrame(rows)


def summary_table(m):
    b, mu = m[m.C == 2], m[m.C > 2]
    wins = int((m.delta > 1e-9).sum()); ties = int((m.delta.abs() <= 1e-9).sum())
    loss = int((m.delta < -1e-9).sum()); rho = m.delta.corr(m.C)

    def r(lbl, s):
        return (f"{lbl} & {len(s)} & {s.gini.mean()*100:.2f} & "
                f"{s.cci.mean()*100:.2f} & {s.delta.mean()*100:+.2f}\\\\")

    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Split-criterion ablation for FERL-deep (the standalone learned "
        r"tree) on the thirty tabular benchmarks: weighted Gini vs.\ the Complete "
        r"Classification Index, with the outer folds, held-out calibration split, "
        r"and depth fixed. Accuracy is on a 0--100 scale (mean over datasets). "
        r"For the learned-threshold deep tree, CCI "
        f"wins on only {wins}" + r" of $30$ datasets and is "
        + f"{abs(m.delta.mean()*100):.2f}" + r" points lower on average, on both "
        r"the binary and multiclass subsets (Pearson $r{=}"
        + f"{rho:.2f}" + r"$ between $\Delta$ and class count $C$).}",
        r"\label{tab:cci-gini-learned}", r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
        r"Subset & \# & Gini & CCI & $\Delta$\\", r"\midrule",
        r(r"Binary ($C{=}2$)", b), r(r"Multiclass ($C{>}2$)", mu), r"\midrule",
        r(r"\emph{All}", m),
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    (GEN / "tab_cci_gini_learned.tex").write_text("\n".join(lines) + "\n")
    print("wrote tab_cci_gini_learned.tex")


def perdataset_table(m):
    m = m.sort_values("delta", ascending=False)

    def row(x):
        return (f"{x.Dataset} & {x.C} & {x.gini*100:.2f} & {x.cci*100:.2f} & "
                f"{x.delta*100:+.2f}\\\\")

    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Per-dataset accuracy (0--100) for FERL-deep under the Gini and "
        r"CCI split criteria, sorted by gain $\Delta=\mathrm{CCI}-\mathrm{Gini}$.}",
        r"\label{tab:cci-gini-learned-full}", r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule", r"Dataset & $C$ & Gini & CCI & $\Delta$\\", r"\midrule",
        *[row(x) for x in m.itertuples()],
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    (GEN / "tab_cci_gini_learned_full.tex").write_text("\n".join(lines) + "\n")
    print("wrote tab_cci_gini_learned_full.tex")


if __name__ == "__main__":
    import sys
    CSV = "results/cci_gini_learned_ablation.csv"
    if "--from-csv" in sys.argv:           # rebuild the tables without refitting
        m = pd.read_csv(CSV)
    else:
        m = evaluate()
        m.to_csv(CSV, index=False)
    summary_table(m)
    perdataset_table(m)
    print(f"\nN={len(m)}  gini={m.gini.mean()*100:.2f}  cci={m.cci.mean()*100:.2f}  "
          f"delta={m.delta.mean()*100:+.2f}  wins={(m.delta>1e-9).sum()}/"
          f"{len(m)}  binary_delta={m[m.C==2].delta.mean()*100:+.2f}  "
          f"multi_delta={m[m.C>2].delta.mean()*100:+.2f}")

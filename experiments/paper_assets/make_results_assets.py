"""Generate publication tables and figures for the FERL paper from the result
CSVs. Tables are emitted as standalone LaTeX snippets under paper/generated/
(\\input into the manuscript sources); figures as PDF under paper/figures/.
Set FERL_GEN_DIR / FERL_FIG_DIR to redirect.

All accuracies/AURC are reported on the 0-100 scale with two decimals and a
standard deviation (over CV folds for tabular, over detector x model seeds for
CBM). Run from the repo root:  python3 experiments/paper_assets/make_results_assets.py
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GEN = Path(os.environ.get("FERL_GEN_DIR", "paper/generated"))
FIG = Path(os.environ.get("FERL_FIG_DIR", "paper/figures"))
GEN.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# colour-blind-safe (Okabe-Ito); one hue per model family
C_INTERP = "#0072B2"   # blue   -- interpretable baselines
C_MODERN = "#009E73"   # green  -- modern rule learners
C_FERL = "#D55E00"     # vermil -- FERL family (ours)
C_CEIL = "#999999"     # grey   -- non-interpretable ceilings

FERL = ["FERL-compact", "FERL-medium", "FERL-deep", "FERL-deep-tuned"]
INTERP = ["CART", "C45", "FIGS", "RuleFit", "FURIA"]
MODERN = ["FuzzyUCS-DS", "SampledRuleList", "RRL", "RL-Net", "NeuRules"]
CEIL = ["RF", "GBDT", "NGBoost", "EDL", "LogReg"]
PRETTY = {"FERL-compact": "FERL-compact", "FERL-medium": "FERL-medium",
          "FERL-deep": "FERL-deep", "C45": "C4.5", "LogReg": "Logistic reg.",
          "FuzzyUCS-DS": "FUCS (DS)", "SampledRuleList": "SamRuLe", "ICDT": "CDT",
          "FERL-deep-tuned": "FERL-deep (tuned)"}

# Ordered frontier-table spec mirroring the paper's Baselines section. Every
# promised baseline appears as a row; methods with no benchmark results render
# as ``--`` (see B6). Each entry is (model_key_in_csv, pretty_name).
TABLE_GROUPS = [
    ("Trees \\& rule ensembles",
     [("CART", "CART"), ("C45", "C4.5"), ("FIGS", "FIGS"), ("RuleFit", "RuleFit")]),
    ("Fuzzy rule-based",
     [("FURIA", "FURIA"), ("FuzzyUCS-DS", "FUCS (DS)")]),
    ("Neural \\& sampled rule learners",
     [("RRL", "RRL"), ("RL-Net", "RL-Net"), ("NeuRules", "NeuRules"),
      ("SampledRuleList", "SamRuLe")]),
    ("Credal \\& evidential",
     [("NCC", "NCC"), ("ICDT", "CDT$^{\\dagger}$")]),
    ("FERL family (ours)",
     [("FERL-compact", "FERL-compact"), ("FERL-medium", "FERL-medium"),
      ("FERL-deep", "FERL-deep"),
      ("FERL-deep-tuned", "FERL-deep, tuned width$^{\\ddagger}$")]),
    ("Linear model",
     [("LogReg", "Logistic reg.")]),
    ("Non-interpretable",
     [("RF", "RF"), ("GBDT", "GB"), ("EDL", "EDL")]),
]
NON_INTERP = {"RF", "GBDT", "EDL"}


def _pm(mean, std, scale=100.0, dp=2):
    return f"{mean * scale:.{dp}f}\\,$\\pm$\\,{std * scale:.{dp}f}"


# --- tabular frontier table + hero figure ------------------------------------

def _benchmark_rows():
    """Per-fold benchmark rows restricted to the 30 paper datasets (the CSV also
    holds an extra dataset that only some methods were run on)."""
    from ferl.pipeline.run_configs import SELECTED_30
    raw = pd.read_csv("results/benchmark2_per_fold.csv")
    return raw[raw.dataset.isin(SELECTED_30)]


def tabular_assets():
    raw = _benchmark_rows()
    d = raw[raw.status == "ok"] if "status" in raw else raw
    n_datasets = raw.dataset.nunique()

    n_folds = raw.fold.nunique()
    # models whose row aggregates fewer than the full dataset set (DNF folds).
    # Report both the fold completion rate and the number of datasets with all
    # folds finished, so the reader can gauge the incompleteness directly.
    table_models = {m: name for _, members in TABLE_GROUPS for m, name in members}
    incomplete = []
    for m, name in table_models.items():
        sub = d[d.model == m]
        done_folds = len(sub)
        full_ds = int(sub.groupby("dataset").fold.nunique().eq(n_folds).sum())
        if 0 < done_folds < n_datasets * n_folds:
            incomplete.append((name, done_folds, full_ds))
    dnf_note = ""
    if incomplete:
        parts = [f"{name} completed {folds}/{n_datasets * n_folds} folds "
                 f"({full} of {n_datasets} datasets fully)"
                 for name, folds, full in incomplete]
        dnf_note = (" " + "; ".join(parts)
                    + ". Rows for these methods aggregate only their completed "
                    "folds; the supplementary significance analysis uses "
                    "matched complete-case pairs.")

    # Collapse folds to one score per dataset *before* aggregating, so the
    # reported std is over datasets (as the caption states), not over the
    # pooled dataset x fold rows.
    per_ds = d.groupby(["model", "dataset"]).mean(numeric_only=True).reset_index()
    g = per_ds.groupby("model").agg(
        acc_m=("acc", "mean"), acc_s=("acc", "std"),
        nodes_m=("complexity", "mean"),
        aurc_m=("aurc", "mean"), aurc_s=("aurc", "std")).reset_index()
    gi = g.set_index("model")

    present = [m for m in table_models if m in gi.index]
    interp = [m for m in present if m not in NON_INTERP]
    nonint = [m for m in present if m in NON_INTERP]
    best = {("acc_m", True): gi.loc[interp].acc_m.idxmax(),
            ("aurc_m", True): gi.loc[interp].aurc_m.idxmin(),
            ("acc_m", False): gi.loc[nonint].acc_m.idxmax() if nonint else None,
            ("aurc_m", False): gi.loc[nonint].aurc_m.idxmin() if nonint else None}

    def mark(m, col, text):
        if best.get((col, m not in NON_INTERP)) == m:
            return f"\\textbf{{{text}}}" if m not in NON_INTERP else f"\\underline{{{text}}}"
        return text

    def row(m, name):
        if m not in gi.index:
            return f"{name} & -- & -- & --\\\\"
        r = gi.loc[m]
        size = "--" if m in ("LogReg", "RF", "GBDT", "EDL") else f"{r.nodes_m:.0f}"
        return (f"{name} & {mark(m, 'acc_m', _pm(r.acc_m, r.acc_s))} & {size} & "
                f"{mark(m, 'aurc_m', _pm(r.aurc_m, r.aurc_s))}\\\\")

    body = []
    for i, (title, members) in enumerate(TABLE_GROUPS):
        if i:
            body.append(r"\midrule")
        body.append(f"\\multicolumn{{4}}{{@{{}}l}}{{\\emph{{{title}}}}}\\\\")
        body += [row(m, name) for m, name in members]

    fucs = per_ds[per_ds.model == "FuzzyUCS-DS"].set_index("dataset").acc
    fucs_opt, fucs_wo = 100 * fucs.get("optdigits", np.nan), 100 * fucs.drop("optdigits", errors="ignore").mean()
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{\textbf{Accuracy and selective-risk AURC on the thirty "
        r"tabular benchmarks} (mean\,$\pm$\,std over datasets, 0--100 scale; "
        r"lower AURC is better; AURC ranks test points by the maximum predicted "
        r"probability, for FERL the normalised soft vote). ``Size'' is the number "
        r"of leaves for trees and of rules for rule sets and rule ensembles; it is "
        r"comparable within a family and only an order-of-magnitude signal across "
        r"families. Bold: best interpretable model; underlined: best "
        r"non-interpretable model. $^{\dagger}$The credal C4.5 tree and its "
        r"set-valued variant ICDT share one tree and one point prediction; they "
        r"differ only in the set output (Table~\ref{tab:credal}). $^{\ddagger}$Band-width "
        r"rule chosen in each training fold by inner 3-fold cross-validation (Brier score) "
        r"among the bootstrap rule and 0.25, 0.5 and 1 node-weighted standard deviations. "
        r"FUCS is at "
        f"chance level on \\emph{{optdigits}} ({fucs_opt:.1f}\\%), which lowers its mean; "
        f"without that dataset it averages {fucs_wo:.1f}." + dnf_note + r"}",
        r"\label{tab:frontier}", r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
        r"Method & Acc.\ $\uparrow$ & Size & AURC $\downarrow$\\", r"\midrule",
        *body,
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    (GEN / "tab_frontier.tex").write_text("\n".join(lines) + "\n")

    # hero figure: accuracy vs complexity (log-x), coloured by family
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for fam, colour, label in ((INTERP, C_INTERP, "Classical rule learners"),
                               (MODERN, C_MODERN, "Modern rule learners"),
                               (CEIL, C_CEIL, "Non-interp. ceilings"),
                               (FERL, C_FERL, "FERL (ours)")):
        present = [m for m in fam if m in gi.index]
        if not present:
            continue
        pts = gi.loc[present]
        ax.scatter(pts.nodes_m, pts.acc_m * 100, s=70, c=colour, label=label,
                   zorder=3, edgecolor="white", linewidth=0.7)
    # FERL frontier line (sorted by complexity)
    fp = gi.loc[[m for m in FERL if m in gi.index]].sort_values("nodes_m")
    if len(fp) > 1:
        ax.plot(fp.nodes_m, fp.acc_m * 100, "-", c=C_FERL, lw=1.6, zorder=2, alpha=0.8)
    for m in [m for m in FERL + MODERN + ["CART", "RF", "GBDT"]
              if m in gi.index]:
        r = gi.loc[m]
        ax.annotate(PRETTY.get(m, m), (r.nodes_m, r.acc_m * 100),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("model size (leaves or rules, log scale)")
    ax.set_ylabel("accuracy (\\%)".replace("\\%", "%"))
    ax.set_title("Accuracy--compactness frontier (30 tabular datasets)")
    ax.grid(True, which="both", axis="both", ls=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIG / "frontier.pdf")
    plt.close(fig)
    print("wrote tab_frontier.tex and frontier.pdf")


# --- CBM accuracy table (CUB + AwA2), mean +/- std over seeds ----------------

def _cbm_accuracy(perf_dir: Path, subsets):
    """mean/std predicted accuracy per (subset, method) over detector x model seeds."""
    p = perf_dir / "e7_calibration.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    out = {}
    for sub in subsets:
        s = d[d.subset.astype(str) == str(sub)]
        if s.empty:
            continue
        agg = s.groupby(["method", "variant"]).accuracy.agg(["mean", "std"]).fillna(0.0)
        out[sub] = agg
    return out


def cbm_assets():
    cub = _cbm_accuracy(Path("results/cub_cbm_perf"), ["20", "full"])
    awa = _cbm_accuracy(Path("results/awa2_cbm_perf"), ["awa2"])
    if cub is None and awa is None:
        print("no CBM e7 data; skipping tab_cbm"); return
    methods = [("ferl-deep", "FERL-deep"), ("decision_tree", "CART"),
               ("logistic_l1", "Sparse LR (concept-matched)"),
               ("logistic_regression", "Logistic reg.\\ (ref.)")]
    # map display keys back to the raw method names stored in e7_calibration.csv
    method_key = {"ferl-deep": "ferl-deep"}
    cols = []  # (header, agg-frame)
    if cub:
        cols += [("CUB-20", cub.get("20")), ("CUB-200", cub.get("full"))]
    if awa:
        cols += [("AwA2", awa.get("awa2"))]
    cols = [(h, a) for h, a in cols if a is not None]

    def cell(agg, method, variant):
        try:
            r = agg.loc[(method, variant)]
            return _pm(r["mean"], r["std"], dp=1)
        except KeyError:
            return "--"

    header = " & ".join(f"\\multicolumn{{2}}{{c}}{{{h}}}" for h, _ in cols)
    cmid = "".join(f"\\cmidrule(lr){{{2+2*i}-{3+2*i}}}" for i in range(len(cols)))
    subhdr = " & ".join("raw & calib." for _ in cols)
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Concept-bottleneck accuracy with detector-predicted concepts "
        r"(0--100 scale, mean\,$\pm$\,std over seeds), before (raw) and after "
        r"per-concept isotonic calibration. Calibration lifts FERL to the CART "
        r"level where the detector is noisy (CUB) and is a no-op where it is "
        r"already good (AwA2). ``Sparse LR'' is an $\ell_1$ head restricted to the "
        r"FERL tree's concept budget---the matched-capacity interpretable "
        r"comparator---while the dense logistic head is an unconstrained "
        r"reference ceiling.}",
        r"\label{tab:cbm}", r"\footnotesize", r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{@{}l" + "cc" * len(cols) + r"@{}}", r"\toprule",
        f"& {header}\\\\", cmid, f"Method & {subhdr}\\\\", r"\midrule",
    ]
    for method, pretty in methods:
        key = method_key.get(method, method)
        cells = " & ".join(cell(a, key, v) for _, a in cols for v in ("raw", "calibrated"))
        lines.append(f"{pretty} & {cells}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (GEN / "tab_cbm.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote tab_cbm.tex (columns: {[h for h,_ in cols]})")


def adaptive_cbm_assets(perf_dir="results/cub_cbm_perf"):
    """Adaptive intervention and shared-vocabulary CBM paper figures/tables."""
    perf = Path(perf_dir)
    e10_path = perf / "e10_matched_adaptive_lr.csv"
    e11_path = perf / "e11_concept_budget_frontier.csv"
    if not e10_path.exists() or not e11_path.exists():
        print("no E10/E11 CBM results; skipping adaptive CBM assets")
        return

    e10 = pd.read_csv(e10_path)
    e10 = e10[
        e10.subset.astype(str).eq("full")
        & e10.concepts.eq("calibrated")
        & e10.ferl_max_depth.eq(60)
        & e10.detector_seed.isin(range(5))
        & e10.k.isin([0, 1, 2, 4, 8])
    ].drop_duplicates(
        ["detector_seed", "method", "policy", "k"], keep="last",
    )
    curves = [
        (("ferl-deep", "adaptive_support"),
         "FERL structural", C_FERL, "--"),
        (("ferl-deep", "adaptive_reliability_purity"),
         "FERL utility-aware", "#CC79A7", "-"),
        (("logistic_regression", "adaptive_eig"),
         "LR adaptive EIG", C_INTERP, "-."),
    ]
    panels = [
        ("net_correction_rate", "net correction among rejected (%)"),
        ("wrong_singleton_rate", "wrong accepted decision among rejected (%)"),
        ("accuracy", "test accuracy (%)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.0))
    for (method, policy), label, colour, linestyle in curves:
        selected = e10[(e10.method == method) & (e10.policy == policy)]
        summary = selected.groupby("k").agg(
            queries=("mean_queries", "mean"),
            **{
                f"{metric}_mean": (metric, "mean")
                for metric, _ in panels
            },
            **{
                f"{metric}_std": (metric, "std")
                for metric, _ in panels
            },
        ).reset_index()
        for axis, (metric, _) in zip(axes, panels):
            axis.errorbar(
                summary.queries,
                100.0 * summary[f"{metric}_mean"],
                yerr=100.0 * summary[f"{metric}_std"].fillna(0.0),
                marker="o", ms=3.5, lw=1.5, capsize=2,
                color=colour, linestyle=linestyle, label=label,
            )
    for axis, (_, ylabel) in zip(axes, panels):
        axis.set_xlabel("mean verified concepts / image")
        axis.set_ylabel(ylabel)
        axis.grid(True, ls=":", alpha=0.4)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "adaptive_intervention_cub.pdf")
    plt.close(fig)

    # Compact k=8 table. All percentages use the initially-rejected set as the
    # denominator except point accuracy; query cost is averaged over all images.
    k8 = e10[e10.k == 8]
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\caption{Adaptive intervention on calibrated CUB-200 at query budget "
        r"$k=8$ (mean$\,\pm\,$std over five detector seeds). Correct and wrong "
        r"acceptance rates use the initially rejected set as denominator; FERL "
        r"acceptance is a singleton credal set. Net "
        r"correction is wrong-to-correct minus correct-to-wrong.}",
        r"\label{tab:adaptive-cbm}",
        r"\begin{tabular}{@{}llccccc@{}}", r"\toprule",
        r"Head & policy & queries/image & correct accepted & wrong accepted & "
        r"net correction & accuracy\\", r"\midrule",
    ]
    table_rows = [
        ("ferl-deep", "adaptive_support", "FERL-deep", "structural"),
        ("ferl-deep", "adaptive_reliability_purity", "FERL-deep", "utility-aware"),
        ("logistic_regression", "adaptive_eig", "Logistic reg.", "adaptive EIG"),
    ]
    for method, policy, head, pretty_policy in table_rows:
        group = k8[(k8.method == method) & (k8.policy == policy)]
        cells = []
        for column, scale in (
            ("mean_queries", 1.0),
            ("correct_recovery_rate", 100.0),
            ("wrong_singleton_rate", 100.0),
            ("net_correction_rate", 100.0),
            ("accuracy", 100.0),
        ):
            cells.append(
                f"{group[column].mean() * scale:.2f}" r"$\,\pm\,$"
                f"{group[column].std() * scale:.2f}"
            )
        lines.append(f"{head} & {pretty_policy} & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (GEN / "tab_adaptive_intervention.tex").write_text("\n".join(lines) + "\n")

    e11 = pd.read_csv(e11_path)
    e11 = e11[
        e11.concepts.eq("calibrated")
        & e11.concept_budget.isin([8, 16, 32, 64, 112])
    ].drop_duplicates(
        ["subset", "detector_seed", "method", "ferl_max_depth", "concept_budget"],
        keep="last",
    )
    methods = [
        ("ferl-deep", "FERL-deep", C_FERL, "o"),
        ("decision_tree", "CART", C_MODERN, "s"),
        ("logistic_regression", "Logistic reg.", C_INTERP, "^"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
    for axis, subset, depth, title in (
        (axes[0], "20", 12, "CUB-20"),
        (axes[1], "full", 60, "CUB-200"),
    ):
        selected = e11[
            e11.subset.astype(str).eq(subset) & e11.ferl_max_depth.eq(depth)
        ]
        for method, label, colour, marker in methods:
            summary = selected[selected.method == method].groupby("concept_budget").accuracy.agg(
                ["mean", "std"],
            )
            axis.errorbar(
                summary.index, 100.0 * summary["mean"],
                yerr=100.0 * summary["std"].fillna(0.0), marker=marker,
                ms=3.5, lw=1.5, capsize=2, color=colour, label=label,
            )
        axis.set_title(title)
        axis.set_xlabel("available train-ranked concepts")
        axis.grid(True, ls=":", alpha=0.4)
    axes[0].set_ylabel("test accuracy (%)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "concept_budget_cub.pdf")
    plt.close(fig)
    print("wrote adaptive_intervention_cub.pdf, concept_budget_cub.pdf, "
          "and tab_adaptive_intervention.tex")


# --- FERL component ablation (same 30 tabular datasets) ----------------------

def ablation_table():
    d = pd.read_csv("results/benchmark2_per_fold.csv")
    if "status" in d:
        d = d[d.status == "ok"]
    rows_spec = [
        ("FERL-compact", "FERL-compact (quantile, point)"),
        ("FERL-credal", "FERL-compact, DS read-out"),
        ("FERL-medium", "FERL-medium (learned thresholds)"),
        ("FERL-deep", "FERL-deep (learned+depth, DS)"),
    ]
    # per-dataset means first, so std is over datasets not pooled folds
    per_ds = d.groupby(["model", "dataset"]).mean(numeric_only=True).reset_index()
    g = per_ds.groupby("model")

    def stat(m, col):
        return g.get_group(m)[col].mean(), g.get_group(m)[col].std()

    def row(model, label):
        acc = stat(model, "acc"); aurc = stat(model, "aurc"); ece = stat(model, "ece_raw")
        u65 = stat(model, "u65"); nodes = g.get_group(model)["complexity"].mean()
        return (f"{label} & {_pm(*acc)} & {nodes:.0f} & {_pm(*aurc)} & "
                f"{_pm(*ece)} & {_pm(*u65)}\\\\")

    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Component ablation of FERL on the thirty tabular benchmarks "
        r"(0--100, mean\,$\pm$\,std) along two axes. Adding the Dempster--Shafer "
        r"read-out (row~1$\to$2) sharpens calibration (ECE) and set-valued utility "
        r"($u_{65}$) at fixed accuracy and size; learned-threshold splits and "
        r"depth (rows~1,3,4) raise accuracy and lower selective risk; FERL-deep "
        r"combines both. Lower AURC/ECE better, higher acc/$u_{65}$ better.}",
        r"\label{tab:ablation}", r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
        r"Variant & Acc.\ $\uparrow$ & \#Nodes & AURC $\downarrow$ & "
        r"ECE $\downarrow$ & $u_{65}$ $\uparrow$\\", r"\midrule",
        *[row(m, lab) for m, lab in rows_spec],
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    (GEN / "tab_ablation.tex").write_text("\n".join(lines) + "\n")
    print("wrote tab_ablation.tex")


# --- residual near-OOD combination-rule ablation (supplementary) ------------

def residual_ood_table():
    """AUROC of the residual near-OOD score under alternative per-node
    combination rules, per FERL variant, vs the dedicated detectors (which are
    model-independent). Reads results/ds_ood_residual_<cfg>.csv (per-dataset,
    per-seed) from experiments/reliability/ds_ood_residual_variants.py."""
    cfgs = [("ferl-compact", "FERL-compact"), ("ferl-medium", "FERL-medium"),
            ("ferl-deep", "FERL-deep")]
    rules = [("residual", r"Firing-weighted avg.\ (used)"),
             ("residual_leaf", "Leaves only"),
             ("residual_chi2", r"$\chi^2$-CDF"),
             ("residual_chi2log", r"$\chi^2$ log-survival")]
    dets = [("maha", "Mahalanobis"), ("knn", "$k$NN distance"),
            ("isoforest", "Isolation Forest")]
    stats = {}
    for cfg, _ in cfgs:
        df = pd.read_csv(f"results/ds_ood_residual_{cfg}.csv")
        sm = df.groupby("seed").mean(numeric_only=True)   # dataset-mean per seed
        stats[cfg] = {c: (sm[c].mean(), sm[c].std()) for c in sm.columns}
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Combination-rule ablation for the residual near-OOD score: "
        r"leave-one-class-out AUROC (0--100, mean\,$\pm$\,std over 3 seeds, "
        r"dataset-averaged over the 12 multiclass benchmarks). The plain "
        r"firing-weighted average used in the main paper is best or tied for "
        r"every variant; the dedicated detectors (bottom block) are fit on the "
        r"input features and do not depend on the FERL variant.}",
        r"\label{tab:residual-comb}", r"\setlength{\tabcolsep}{3.5pt}",
        r"\footnotesize", r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
        r"Combination rule & Compact & Medium & Deep\\",
        r"\midrule",
        *[f"{lab} & " + " & ".join(_pm(*stats[cfg][col]) for cfg, _ in cfgs) + r"\\"
          for col, lab in rules],
        r"\midrule",
        *[f"{lab} & \\multicolumn{{3}}{{c}}{{{_pm(*stats['ferl-compact'][col])}}}\\\\"
          for col, lab in dets],
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    (GEN / "tab_residual_ood.tex").write_text("\n".join(lines) + "\n")
    print("wrote tab_residual_ood.tex")


# --- statistical significance: ranks, Friedman, Nemenyi CD diagram ----------

# Rule-learner cohort compared on equal footing (interpretable + rule learners
# with per-dataset accuracy on the shared benchmark). Ceilings (RF/GB/EDL/
# NGBoost) are excluded: they are not interpretable and only frame the frontier.
SIG_METHODS = ["CART", "C45", "FIGS", "RuleFit", "LogReg", "FURIA",
               "FuzzyUCS-DS", "RRL", "RL-Net", "NeuRules", "SampledRuleList",
               "NCC", "ICDT", "FERL-compact", "FERL-medium", "FERL-deep"]


def _pval(p):
    """p-value for math mode: two significant digits, powers of ten when tiny."""
    if p >= 0.001:
        return f"{p:.2g}"
    mant, exp = f"{p:.1e}".split("e")
    return f"{mant}\\times10^{{{int(exp)}}}"


def _holm(pvalues):
    """Holm step-down adjusted p-values, returned in the input order."""
    order = np.argsort(pvalues)
    adjusted, running, k = {}, 0.0, len(pvalues)
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, pvalues[idx] * (k - rank)))
        adjusted[idx] = running
    return adjusted


def significance_assets():
    import scikit_posthocs as sp
    from scipy.stats import friedmanchisquare, wilcoxon

    d = _benchmark_rows()
    if "status" in d:
        d = d[d.status == "ok"]
    g = d.groupby(["dataset", "model"]).mean(numeric_only=True).reset_index()

    def analyse(metric, ascending):
        """Return (avg-rank series, Friedman p, per-dataset score matrix) on the
        datasets where every cohort method has a score. rank 1 = best."""
        piv = g.pivot(index="dataset", columns="model", values=metric)
        piv = piv[[m for m in SIG_METHODS if m in piv.columns]].dropna()
        ranks = piv.rank(axis=1, ascending=ascending)
        p = friedmanchisquare(*[piv[c] for c in piv.columns]).pvalue
        return ranks.mean().sort_values(), p, piv

    acc_rank, acc_p, acc_piv = analyse("acc", ascending=False)
    aurc_rank, aurc_p, _ = analyse("aurc", ascending=True)

    # --- CD diagram on accuracy (Nemenyi post-hoc over Friedman) -------------
    nemenyi = sp.posthoc_nemenyi_friedman(acc_piv)
    nemenyi.index = [PRETTY.get(m, m) for m in nemenyi.index]
    nemenyi.columns = [PRETTY.get(m, m) for m in nemenyi.columns]
    ranks_pretty = acc_rank.copy()
    ranks_pretty.index = [PRETTY.get(m, m) for m in ranks_pretty.index]

    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    colours = {}
    for m, r in ranks_pretty.items():
        key = {v: k for k, v in PRETTY.items()}.get(m, m)
        fam = (C_FERL if key in FERL else C_MODERN if key in MODERN
               else C_CEIL if key in CEIL else C_INTERP)
        colours[m] = fam
    sp.critical_difference_diagram(
        ranks_pretty, nemenyi, ax=ax,
        label_props={"fontsize": 9},
        color_palette=colours,
        crossbar_props={"linewidth": 1.4},
    )
    ax.set_title("Accuracy ranks on the tabular benchmark "
                 f"(Nemenyi CD, Friedman $p<10^{{{int(np.floor(np.log10(acc_p))) + 1}}}$)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "cd_accuracy.pdf", bbox_inches="tight")
    plt.close(fig)

    # --- pairwise Wilcoxon of FERL-deep vs every cohort method (Holm) --------
    piv_acc = g.pivot(index="dataset", columns="model", values="acc")
    piv_aur = g.pivot(index="dataset", columns="model", values="aurc")
    others = [m for m in SIG_METHODS if m != "FERL-deep"]

    def wilcox(piv, a, b, better_low):
        sub = piv[[a, b]].dropna()
        try:
            p = wilcoxon(sub[a], sub[b]).pvalue
        except ValueError:  # all-zero differences
            p = 1.0
        delta = (sub[a] - sub[b]).mean() * 100
        return delta, p, len(sub)

    rows = []
    for m in others:
        da, pa, na = wilcox(piv_acc, "FERL-deep", m, better_low=False)
        du, pu, _ = wilcox(piv_aur, "FERL-deep", m, better_low=True)
        rows.append((m, da, pa, du, pu, na))
    holm = _holm([r[2] for r in rows])          # accuracy family
    holm_aurc = _holm([r[4] for r in rows])     # AURC family

    def stars(p):
        return ("$^{***}$" if p < 0.001 else "$^{**}$" if p < 0.01
                else "$^{*}$" if p < 0.05 else "")

    body = []
    for i, (m, da, pa, du, pu, na) in enumerate(rows):
        name = PRETTY.get(m, m)
        acc_cell = f"{da:+.2f}{stars(holm[i])}"
        aur_cell = f"{du:+.2f}{stars(holm_aurc[i])}"
        body.append(f"{name} & {acc_cell} & {aur_cell}\\\\")

    lines = [
        r"\begin{table}[h!]", r"\centering", r"\small",
        r"\caption{\textbf{Wilcoxon signed-rank comparisons on the tabular "
        r"benchmark.} Pairwise comparison of \emph{FERL-deep} against every "
        r"interpretable and rule-learning baseline across 30 datasets. Entries "
        r"are mean paired differences: $\Delta$Acc.\ $>0$ and "
        r"$\Delta$AURC $<0$ favour FERL. $p$-values are Holm-corrected "
        r"within each metric's comparison family. $^{*}p<0.05$, "
        r"$^{**}p<0.01$, $^{***}p<0.001$.}",
        r"\label{tab:sig}", r"\begin{tabular}{@{}lcc@{}}", r"\toprule",
        r"vs.\ baseline & $\Delta$Acc.\ $\uparrow$ & $\Delta$AURC $\downarrow$\\",
        r"\midrule", *body, r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    (GEN / "tab_significance.tex").write_text("\n".join(lines) + "\n")

    # --- machine-readable macros for the prose ------------------------------
    macros = [
        r"\newcommand{\SigNdatasets}{%d}" % len(acc_piv),
        r"\newcommand{\SigFriedmanAccP}{%s}" % _pval(acc_p),
        r"\newcommand{\SigFriedmanAurcP}{%s}" % _pval(aurc_p),
        r"\newcommand{\SigFerlAccRank}{%.2f}" % acc_rank["FERL-deep"],
        r"\newcommand{\SigFerlAurcRank}{%.2f}" % aurc_rank["FERL-deep"],
        r"\newcommand{\SigRuleFitAccRank}{%.2f}" % acc_rank["RuleFit"],
        r"\newcommand{\SigRuleFitAurcRank}{%.2f}" % aurc_rank["RuleFit"],
        r"\newcommand{\SigNmethods}{%d}" % len(acc_piv.columns),
    ]
    by_name = {r[0]: (r, i) for i, r in enumerate(rows)}
    for key, macro in (("RuleFit", "RuleFit"), ("LogReg", "LogReg")):
        (m, da, pa, du, pu, na), i = by_name[key]
        macros += [r"\newcommand{\SigVs%sAccP}{%s}" % (macro, _pval(holm[i])),
                   r"\newcommand{\SigVs%sAurcP}{%s}" % (macro, _pval(holm_aurc[i])),
                   r"\newcommand{\SigVs%sAccD}{%+.2f}" % (macro, da),
                   r"\newcommand{\SigVs%sAurcD}{%+.2f}" % (macro, du)]
    if "FERL-deep-tuned" in piv_acc.columns:
        for other, macro in (("RuleFit", "RuleFit"), ("FERL-deep", "Deep"), ("LogReg", "LogReg")):
            da, pa, _ = wilcox(piv_acc, "FERL-deep-tuned", other, better_low=False)
            du, pu, _ = wilcox(piv_aur, "FERL-deep-tuned", other, better_low=True)
            macros += [r"\newcommand{\SigTunedVs%sAccP}{%s}" % (macro, _pval(pa)),
                       r"\newcommand{\SigTunedVs%sAurcP}{%s}" % (macro, _pval(pu)),
                       r"\newcommand{\SigTunedVs%sAccD}{%+.2f}" % (macro, da),
                       r"\newcommand{\SigTunedVs%sAurcD}{%+.2f}" % (macro, du)]
    rule_learners = [i for i, r in enumerate(rows)
                     if r[0] not in ("RuleFit", "LogReg", "FERL-compact", "FERL-medium")]
    macros.append(r"\newcommand{\SigMaxRuleLearnerP}{%s}"
                  % _pval(max(holm[i] for i in rule_learners)))
    (GEN / "sig_summary.tex").write_text("\n".join(macros) + "\n")

    print(f"wrote cd_accuracy.pdf, tab_significance.tex, sig_summary.tex "
          f"(n={len(acc_piv)} datasets; acc Friedman p={acc_p:.1e}, "
          f"FERL-deep acc rank {acc_rank['FERL-deep']:.2f})")


# --- shift robustness: native evidential sets vs split conformal ------------

def shift_assets(
    input_path="results/credal_shift.csv",
    summary_shift=1.0,
    generated_dir=GEN,
    figure_dir=FIG,
    stats_dir="results/submission_stats",
):
    """Write the submission's two-panel shift figure and its quoted numbers."""
    path = Path(input_path)
    if not path.exists():
        fallback = Path("results/credal_shift_synth.csv")
        if not fallback.exists():
            print("no credal shift data; skipping shift assets")
            return None
        path = fallback

    d = pd.read_csv(path)
    required = {"k", "acc", "ds_cov", "ds_size", "gl_cov", "gl_size"}
    missing = required.difference(d.columns)
    if missing:
        raise ValueError(f"{path} lacks shift columns: {sorted(missing)}")

    metrics = ["acc", "ds_cov", "ds_size", "gl_cov", "gl_size"]
    # Seeds are repeated runs within a dataset, not independent benchmark
    # observations. Average them first so uncertainty bands reflect variation
    # across datasets rather than treating every dataset-seed row independently.
    units = d.groupby(["ds", "k"], as_index=False)[metrics].mean()
    grouped = units.groupby("k")[metrics]
    means = grouped.mean().sort_index()
    sems = grouped.sem().reindex(means.index).fillna(0.0)
    summary = means.join(sems, lsuffix="_mean", rsuffix="_sem").reset_index()
    generated_dir = Path(generated_dir)
    figure_dir = Path(figure_dir)
    stats_dir = Path(stats_dir)
    generated_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(stats_dir / "shift_summary.csv", index=False)

    x = means.index.to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), sharex=True)
    ax = axes[0]
    for key, colour, label in (("ds_cov", C_FERL, "FERL plausibility conformal"),
                               ("gl_cov", C_INTERP, "split conformal")):
        y = means[key].to_numpy(dtype=float)
        ci = 1.96 * sems[key].to_numpy(dtype=float)
        ax.plot(x, y, "-o", color=colour, ms=3.5, lw=1.8, label=label)
        ax.fill_between(x, y - ci, y + ci, color=colour, alpha=0.13, linewidth=0)
    ax.plot(x, means.acc, "--", color=C_CEIL, lw=1.4, label="accuracy")
    ax.axhline(0.9, color="black", ls=":", lw=1.0, label="90% target")
    ax.set_ylabel("coverage / accuracy")
    ax.set_ylim(0.25, 1.0)
    ax.set_title("Coverage under shift")

    ax = axes[1]
    for key, colour, label in (("ds_size", C_FERL, "FERL plausibility conformal"),
                               ("gl_size", C_INTERP, "split conformal")):
        y = means[key].to_numpy(dtype=float)
        ci = 1.96 * sems[key].to_numpy(dtype=float)
        ax.plot(x, y, "-o", color=colour, ms=3.5, lw=1.8, label=label)
        ax.fill_between(x, np.maximum(0.0, y - ci), y + ci,
                        color=colour, alpha=0.13, linewidth=0)
    ax.set_ylabel("mean prediction-set size")
    ax.set_title("Set adaptation under shift")

    for ax in axes:
        ax.set_xlabel("shift magnitude")
        ax.grid(True, ls=":", alpha=0.4)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04),
               ncol=4, fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(figure_dir / "shift_coverage.pdf", bbox_inches="tight")
    plt.close(fig)

    available = means.index.to_numpy(dtype=float)
    index = int(np.argmin(np.abs(available - summary_shift)))
    selected_shift = float(available[index])
    if not np.isclose(selected_shift, summary_shift):
        raise ValueError(f"summary shift {summary_shift} is unavailable in {path}")
    row = means.loc[selected_shift]
    macro = (
        f"\\newcommand{{\\ShiftLevel}}{{{selected_shift:g}}}\n"
        f"\\newcommand{{\\ShiftConformalCoverage}}{{{100 * row.gl_cov:.1f}\\%}}\n"
        f"\\newcommand{{\\ShiftNativeCoverage}}{{{100 * row.ds_cov:.1f}\\%}}\n"
        f"\\newcommand{{\\ShiftNativeSetSize}}{{{row.ds_size:.2f}}}\n"
        f"\\newcommand{{\\ShiftConformalSetSize}}{{{row.gl_size:.2f}}}\n"
        f"\\newcommand{{\\ShiftAccuracy}}{{{100 * row.acc:.1f}\\%}}\n"
        f"\\newcommand{{\\ShiftZeroNativeCoverage}}{{{100 * means.loc[0.0].ds_cov:.1f}\\%}}\n"
        f"\\newcommand{{\\ShiftZeroConformalCoverage}}{{{100 * means.loc[0.0].gl_cov:.1f}\\%}}\n"
        f"\\newcommand{{\\ShiftZeroNativeSetSize}}{{{means.loc[0.0].ds_size:.2f}}}\n"
    )
    (generated_dir / "shift_summary.tex").write_text(macro, encoding="ascii")
    print(f"wrote shift_coverage.pdf and shift_summary.tex from {path}")
    return row


# --- CBM figures (intervention, risk-coverage, corruption robustness) --------

# one panel per intervention policy; models compared against each other inside
E5_POLICIES = [("random", "Random order"), ("importance", "Global importance"),
               ("path", "Rule path (per-sample)")]
E5_MODELS = [("decision_tree", "CART", C_INTERP, "--"),
             ("ferl-deep", "FERL-deep", C_FERL, "-"),
             ("logistic_regression", "Logistic reg.", C_CEIL, ":"),
             ("logistic_l1", "Sparse LR (concept-matched)", C_MODERN, "-.")]
# LR has no rule paths; its best available policy is global importance.
E5_BEST_POLICY = {"logistic_regression": "importance", "logistic_l1": "importance"}


def _load_e5(perf_dir, subset):
    """Calibrated-detector E5 rows for one subset. Rows predating the `concepts`
    column were run on raw detector scores and are dropped."""
    e5 = pd.read_csv(Path(perf_dir) / "e5_intervention.csv")
    e5 = e5[e5.subset.astype(str) == subset]
    if "concepts" not in e5:
        raise SystemExit("e5_intervention.csv has no calibrated rows; rerun E5")
    e5 = e5[e5.concepts == "calibrated"]
    # the runner appends, so re-runs leave duplicate (seed, method, policy, k) rows
    e5 = e5.drop_duplicates(["detector_seed", "method", "policy", "k"], keep="last")
    if e5.empty:
        raise SystemExit(f"no calibrated E5 rows for subset={subset}; rerun E5")
    return e5


def _e5_deltas(e5):
    """Accuracy gain over each run's own k=0 baseline, in points. Paired within
    (seed, method, policy) so the baseline spread never enters the delta."""
    base = (e5[e5.k == 0]
            .set_index(["detector_seed", "method", "policy"]).accuracy)
    idx = pd.MultiIndex.from_frame(e5[["detector_seed", "method", "policy"]])
    out = e5.copy()
    out["delta"] = (e5.accuracy.to_numpy() - base.reindex(idx).to_numpy()) * 100
    return out


def intervention_figure(perf_dir="results/cub_cbm_perf", tag="cub", subset="full"):
    """Model comparison at each intervention budget k, one panel per policy.
    LR has no rule paths, so it is absent from the `path` panel by construction.
    """
    e5 = _e5_deltas(_load_e5(perf_dir, subset))

    def _curve(ax, gg, name, c, ls, col):
        stat = gg.groupby("k")[col].agg(["mean", "std"])
        if col == "accuracy":
            stat *= 100
        ax.plot(stat.index, stat["mean"], ls, marker="o", ms=3, c=c, label=name)
        ax.fill_between(stat.index, stat["mean"] - stat["std"],
                        stat["mean"] + stat["std"], color=c, alpha=0.15, lw=0)

    def _panels(col, ylabel, fname):
        fig, axes = plt.subplots(1, len(E5_POLICIES) + 1, figsize=(13.6, 3.2), sharey=True)
        for ax, (pol, pretty) in zip(axes, E5_POLICIES):
            sub = e5[e5.policy == pol]
            for method, name, c, ls in E5_MODELS:
                gg = sub[sub.method == method]
                if not gg.empty:
                    _curve(ax, gg, name, c, ls, col)
            ax.set_title(pretty, fontsize=10)

        # Best policy each model can actually run: the trees get per-sample rule
        # paths, LR has no paths and so tops out at global importance.
        ax = axes[-1]
        for method, name, c, ls in E5_MODELS:
            pol = E5_BEST_POLICY.get(method, "path")
            gg = e5[(e5.method == method) & (e5.policy == pol)]
            if not gg.empty:
                _curve(ax, gg, f"{name} ({pol})", c, ls, col)
        ax.set_title("Best available policy", fontsize=10)

        kmax = int(e5.k.max())
        for ax in axes:
            ax.set_xscale("symlog", base=2)
            ax.set_xlim(0, kmax * 1.1)  # symlog margins otherwise autoscale negative
            ax.set_xlabel("# concepts intervened")
            ax.grid(True, ls=":", alpha=0.4)
            ax.legend(fontsize=7, loc="upper left")
        axes[0].set_ylabel(ylabel)
        fig.tight_layout()
        fig.savefig(FIG / fname)
        plt.close(fig)

    _panels("accuracy", "accuracy (%)", f"intervention_{tag}.pdf")
    _panels("delta", "accuracy gain over $k{=}0$ (pts)", f"intervention_delta_{tag}.pdf")
    _intervention_table(e5, tag, subset)
    print(f"wrote intervention_{tag}.pdf + intervention_delta_{tag}.pdf ({subset}, "
          f"{e5.detector_seed.nunique()} seeds)")


def _intervention_table(e5, tag, subset):
    """Accuracy and gain-over-baseline at each model's best available policy."""
    ks = [0, 8, 32, 64, 112]
    lines = [r"\begin{tabular}{ll" + "r" * len(ks) + "}", r"\toprule",
             r"Model & Policy & " + " & ".join(f"$k{{=}}{k}$" for k in ks) + r" \\",
             r"\midrule"]
    for method, name, _c, _ls in E5_MODELS:
        pol = E5_BEST_POLICY.get(method, "path")
        gg = e5[(e5.method == method) & (e5.policy == pol)]
        if gg.empty:
            continue
        acc = gg.groupby("k").accuracy.agg(["mean", "std"])
        dlt = gg.groupby("k").delta.agg(["mean", "std"])
        cells = [_pm(acc["mean"][k], acc["std"][k]) if k in acc.index else "--"
                 for k in ks]
        lines.append(f"{name} & {pol} & " + " & ".join(cells) + r"\\")
        cells = [("--" if k == 0 else
                  "$+$" + _pm(dlt["mean"][k], dlt["std"][k], scale=1.0))
                 if k in dlt.index else "--" for k in ks]
        lines.append(r"\quad $\Delta$ vs $k{=}0$ & & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out = GEN / f"tab_intervention_{tag}.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


def cbm_figures(perf_dir="results/awa2_cbm_perf", tag="awa2"):
    perf = Path(perf_dir)
    styles = {"random": ":", "importance": "--", "path": "-"}

    # intervention efficiency
    e5 = pd.read_csv(perf / "e5_intervention.csv")
    e5 = e5[e5.method == "ferl-deep"]
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for pol, gg in e5.groupby("policy"):
        m = gg.groupby("k").accuracy.mean()
        ax.plot(m.index, m.values * 100, styles.get(pol, "-"), marker="o", ms=3,
                c=C_FERL if pol == "path" else C_INTERP if pol == "importance" else C_CEIL,
                label=pol)
    ax.set_xlabel("# concepts intervened"); ax.set_ylabel("accuracy (%)")
    ax.set_title("Rule-path intervention (FERL-deep)")
    ax.grid(True, ls=":", alpha=0.4); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / f"intervention_{tag}.pdf"); plt.close(fig)

    # corruption robustness: set size vs concept-noise level
    e4 = pd.read_csv(perf / "e4_robustness.csv")
    cn = e4[e4.degradation == "concept_noise"]
    if not cn.empty:
        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        m = cn.groupby("level")[["avg_set_size", "conformal_coverage"]].mean()
        ax.plot(m.index, m.avg_set_size, "-o", c=C_FERL, ms=4, label="avg.\\ set size")
        ax.set_xlabel("concept-noise level"); ax.set_ylabel("evidential set size")
        ax.set_title("Set inflation under concept noise")
        ax.grid(True, ls=":", alpha=0.4)
        fig.tight_layout(); fig.savefig(FIG / f"robustness_{tag}.pdf"); plt.close(fig)

    # risk-coverage (selective risk) for ferl-deep scores
    e3 = pd.read_csv(perf / "e3_risk_coverage.csv")
    rc = e3[(e3.method == "ferl-deep") & (e3.source == "predicted")]
    if not rc.empty:
        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        for sc, gg in rc.groupby("score"):
            m = gg.groupby("coverage").risk.mean()
            ax.plot(m.index, m.values * 100, "-", label=sc, lw=1.4)
        ax.set_xlabel("coverage"); ax.set_ylabel("selective risk (%)")
        ax.set_title("Risk--coverage (FERL-deep)")
        ax.grid(True, ls=":", alpha=0.4); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(FIG / f"risk_coverage_{tag}.pdf"); plt.close(fig)
    print(f"wrote CBM figures for {tag}")


if __name__ == "__main__":
    tabular_assets()
    significance_assets()
    shift_assets()
    cbm_assets()
    adaptive_cbm_assets()
    ablation_table()
    residual_ood_table()
    cbm_figures()
    intervention_figure()
    print("done")

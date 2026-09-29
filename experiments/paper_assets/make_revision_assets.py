"""Tables and macros added for the MLJ revision of the FERL paper.

Reads the result CSVs written by

* experiments/benchmark2 (results/benchmark2_per_fold.csv) -- credal sets;
* experiments/reliability/ds_ood_residual_variants.py -- tabular near-OOD;
* experiments/reliability/ds_decision_rule_ablation.py -- read-out ablation;
* experiments/reliability/ds_width_ablation.py -- band-width ablation;
* experiments/reliability/ds_stability_constants.py -- stability constants;
* experiments/reliability/ds_ignorance_semantics.py -- ignorance semantics;
* experiments/cub_cbm/concept_error_robustness.py -- concept-error robustness;

and writes LaTeX snippets to paper/generated/ (``FERL_GEN_DIR`` overrides),
plus ``rev_macros.tex`` with the numbers quoted in the prose. Missing inputs
are skipped with a message. Run from the repository root:

    python experiments/paper_assets/make_revision_assets.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

GEN = Path(os.environ.get("FERL_GEN_DIR", "paper/generated"))
GEN.mkdir(parents=True, exist_ok=True)
MACROS: dict[str, str] = {}


_DIGITS = dict(zip("0123456789", ["Zero", "One", "Two", "Three", "Four", "Five",
                                  "Six", "Seven", "Eight", "Nine"]))


def _pval(p):
    """p-value for math mode: two significant digits, powers of ten when tiny."""
    if p >= 0.001:
        return f"{p:.2g}"
    mant, exp = f"{p:.1e}".split("e")
    return f"{mant}\\times10^{{{int(exp)}}}"


def _macro(name, value, fmt="{:.2f}"):
    """Store a prose macro; LaTeX control words may only contain letters."""
    name = "".join(_DIGITS.get(ch, ch) for ch in name if ch.isalnum())
    if fmt == "p":
        MACROS[name] = _pval(value)
        return
    MACROS[name] = fmt.format(value) if not isinstance(value, str) else value


def _pm(mean, std, scale=100.0, dp=1):
    return f"{mean * scale:.{dp}f}\\,$\\pm$\\,{std * scale:.{dp}f}"


def _holm(pvalues):
    order = np.argsort(pvalues)
    adjusted, running, k = [0.0] * len(pvalues), 0.0, len(pvalues)
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, pvalues[idx] * (k - rank)))
        adjusted[idx] = running
    return adjusted


def _wilcoxon(a, b):
    diff = np.asarray(a) - np.asarray(b)
    if np.allclose(diff, 0):
        return 1.0
    try:
        return float(wilcoxon(a, b).pvalue)
    except ValueError:
        return 1.0


def _wtl(a, b, tol=1e-9):
    diff = np.asarray(a) - np.asarray(b)
    return int((diff > tol).sum()), int((np.abs(diff) <= tol).sum()), int((diff < -tol).sum())


def _stars(p):
    return "$^{***}$" if p < 0.001 else "$^{**}$" if p < 0.01 else "$^{*}$" if p < 0.05 else ""


def _write(name, lines):
    (GEN / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {GEN / name}")


def _need(path):
    if not Path(path).exists():
        print(f"skip: {path} missing")
        return False
    return True


# --- credal sets --------------------------------------------------------------

NATIVE = [("FERL-deep", "FERL-deep"), ("FERL-deep-tuned", "FERL-deep, tuned width"),
          ("FERL-credal", "FERL-compact"), ("FuzzyUCS-DS", "FUCS (DS)"), ("NCC", "NCC"),
          ("ICDT", "CDT")]
CONFORMAL = [("FERL-deep", "FERL-deep + APS"), ("RuleFit", "RuleFit + APS"),
             ("LogReg", "LR + APS"), ("MLP", "MLP + APS")]


def credal_table():
    from ferl.pipeline.run_configs import SELECTED_30
    d = pd.read_csv("results/benchmark2_per_fold.csv")
    d = d[d.dataset.isin(SELECTED_30) & (d.status == "ok")].copy()
    det, cov, sacc = d.determinacy, d.set_cov, d.set_acc.fillna(0.0)
    d["single_risk"] = 1.0 - (cov - (1.0 - det) * sacc) / det.where(det > 0)
    g = d.groupby(["model", "dataset"]).mean(numeric_only=True).reset_index()

    def metric_frame(model, conformal):
        s = g[g.model == model].set_index("dataset")
        if conformal:
            return pd.DataFrame({"determinacy": s.conf_determinacy, "coverage": s.conf_cov,
                                 "size": s.conf_size, "single_risk": np.nan,
                                 "u65": s.conf_u65, "u80": s.conf_u80})
        return pd.DataFrame({"determinacy": s.determinacy, "coverage": s.set_cov,
                             "size": s.mean_size, "single_risk": s.single_risk,
                             "u65": s.u65, "u80": s.u80})

    present = set(g.model)
    frames = {name: metric_frame(m, False) for m, name in NATIVE if m in present}
    frames.update({name: metric_frame(m, True) for m, name in CONFORMAL})
    ref = frames["FERL-deep"]
    others = [n for n in frames if n != "FERL-deep"]
    tests = {}
    for metric in ("u65", "u80", "coverage"):
        ps = [_wilcoxon(ref[metric], frames[n].loc[ref.index, metric]) for n in others]
        for n, p in zip(others, _holm(ps)):
            tests[(n, metric)] = p

    def row(name):
        f = frames[name]
        risk = "--" if f.single_risk.isna().all() else f"{100 * f.single_risk.mean():.1f}"
        cells = [f"{100 * f.determinacy.mean():.1f}",
                 _pm(f.coverage.mean(), f.coverage.std()) + ("" if name == "FERL-deep" else _stars(tests[(name, 'coverage')])),
                 f"{f['size'].mean():.2f}", risk,
                 _pm(f.u65.mean(), f.u65.std()) + ("" if name == "FERL-deep" else _stars(tests[(name, 'u65')])),
                 _pm(f.u80.mean(), f.u80.std()) + ("" if name == "FERL-deep" else _stars(tests[(name, 'u80')]))]
        return f"{name} & " + " & ".join(cells) + r"\\"

    lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\caption{\textbf{Native and conformal set-valued prediction} on the 30 tabular "
             r"datasets (0--100 scale except set size; mean, or mean\,$\pm$\,std over datasets). "
             r"Determinacy is the singleton rate, coverage the rate at which the set contains "
             r"the true class, singleton risk the error rate among singleton outputs; "
             r"$u_{65}$/$u_{80}$ are utility-discounted accuracies. Conformal rows use adaptive "
             r"prediction sets (APS) at $1-\alpha=0.9$ on the 25\% calibration split. Stars mark "
             r"a Wilcoxon signed-rank difference from FERL-deep over datasets, Holm-corrected "
             r"within each metric: $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$.}",
             r"\label{tab:credal}", r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
             r"Method & Determ. & Coverage & Size & Sgl.\ risk $\downarrow$ & $u_{65}$ & $u_{80}$\\",
             r"\midrule", r"\multicolumn{7}{@{}l}{\emph{Native set-valued predictors}}\\"]
    lines += [row(n) for _, n in NATIVE if n in frames]
    lines += [r"\midrule", r"\multicolumn{7}{@{}l}{\emph{Post-hoc conformal predictors}}\\"]
    lines += [row(n) for _, n in CONFORMAL]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write("tab_credal.tex", lines)

    for n, key in (("NCC", "Ncc"), ("CDT", "Cdt"), ("MLP + APS", "Mlp"), ("FUCS (DS)", "Fucs"),
                   ("RuleFit + APS", "RuleFitAps"), ("LR + APS", "LrAps")):
        for metric in ("u65", "u80", "coverage"):
            _macro(f"Cred{key}{metric.capitalize()}P", tests[(n, metric)], "p")
        w, t, l = _wtl(ref.u65, frames[n].loc[ref.index, "u65"])
        _macro(f"Cred{key}UWins", f"{w}/{t}/{l}")
    for metric in ("u65", "u80", "coverage", "determinacy", "single_risk"):
        _macro(f"CredDeep{metric.replace('_', '').capitalize()}", 100 * ref[metric].mean(), "{:.1f}")
    _macro("CredDeepSize", ref["size"].mean())
    _macro("CredNccSingleRisk", 100 * frames["NCC"].single_risk.mean(), "{:.1f}")
    for n, key in (("RuleFit + APS", "RuleFitAps"), ("LR + APS", "LrAps"), ("MLP + APS", "Mlp")):
        _macro(f"Cred{key}USixFive", 100 * frames[n].u65.mean(), "{:.1f}")
        _macro(f"Cred{key}Coverage", 100 * frames[n].coverage.mean(), "{:.1f}")
        _macro(f"Cred{key}Size", frames[n]["size"].mean())
    if "FERL-deep, tuned width" in frames:
        t = frames["FERL-deep, tuned width"]
        ncc = frames["NCC"].loc[t.index]
        for metric in ("u65", "u80"):
            _macro(f"CredTunedVsNcc{metric.capitalize()}P", _wilcoxon(t[metric], ncc[metric]), "p")
        for metric in ("u65", "u80", "coverage", "single_risk"):
            _macro(f"CredTuned{metric.replace('_', '').capitalize()}", 100 * t[metric].mean(), "{:.1f}")
        _macro("CredTunedSize", t["size"].mean())
    _macro("CredCdtSingleRisk", 100 * frames["CDT"].single_risk.mean(), "{:.1f}")


# --- tabular near-OOD ------------------------------------------------------------

OOD_CFGS = [("ferl-compact", "FERL-compact"), ("ferl-medium", "FERL-medium"),
            ("ferl-deep", "FERL-deep")]
DETECTORS = [("maha", "Mahalanobis"), ("knn", "$k$NN distance"),
             ("isoforest", "Isolation Forest"), ("edl_vacuity", "EDL vacuity"),
             ("edl_entropy", "EDL entropy")]


def ood_table():
    paths = {cfg: Path(f"results/ds_ood_residual_{cfg}.csv") for cfg, _ in OOD_CFGS}
    if not all(_need(p) for p in paths.values()):
        return
    per = {}
    for cfg, _ in OOD_CFGS:
        df = pd.read_csv(paths[cfg])
        if "residual_fpr95" not in df.columns:
            print(f"warning: {paths[cfg]} is an old run without FPR95/AUPR columns")
        per[cfg] = df.groupby("dataset").mean(numeric_only=True)
    deep = per["ferl-deep"]
    rows_spec = [(cfg, "residual", f"{name} residual") for cfg, name in OOD_CFGS]
    rows_spec += [("ferl-deep", "ignorance", "FERL-deep ignorance"),
                  ("ferl-deep", "softmax", "FERL-deep soft-vote entropy")]
    rows_spec += [("ferl-deep", key, name) for key, name in DETECTORS]
    detector_ps = _holm([_wilcoxon(deep.residual, deep[k]) for k, _ in DETECTORS])
    pmap = {k: p for (k, _), p in zip(DETECTORS, detector_ps)}

    def cell(frame, col, lower=False):
        if col not in frame.columns:
            return "--"
        return _pm(frame[col].mean(), frame[col].std())

    lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\caption{\textbf{Tabular near-OOD detection} (leave-one-class-out on the "
             f"{len(deep)} multiclass benchmarks; 0--100 scale, mean\\,$\\pm$\\,std over datasets "
             r"after averaging three seeds and all held-out classes). AUPR-Out treats the novel "
             r"class as positive; FPR95 is the in-distribution false-alarm rate when 95\% of novel "
             r"inputs are flagged; TNR@95 is the share of novel inputs rejected when 95\% of "
             r"in-distribution inputs are accepted (both are ROC operating points computed on the "
             r"evaluation set, not deployed thresholds). Dedicated detectors and EDL are fitted on "
             r"the same retained-class training split. Stars mark a Wilcoxon difference in AUROC "
             r"from the FERL-deep residual over datasets (Holm-corrected): "
             r"$^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$.}",
             r"\label{tab:ood}", r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
             r"Score & AUROC $\uparrow$ & AUPR-Out $\uparrow$ & FPR95 $\downarrow$ & TNR@95 $\uparrow$\\",
             r"\midrule", r"\multicolumn{5}{@{}l}{\emph{Native FERL scores}}\\"]
    for i, (cfg, col, label) in enumerate(rows_spec):
        if i == 5:
            lines += [r"\midrule", r"\multicolumn{5}{@{}l}{\emph{Dedicated detectors and EDL}}\\"]
        f = per[cfg]
        star = _stars(pmap[col]) if col in pmap else ""
        lines.append(f"{label} & {cell(f, col)}{star} & {cell(f, col + '_aupr')} & "
                     f"{cell(f, col + '_fpr95')} & {cell(f, col + '_rej')}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write("tab_ood.tex", lines)

    _macro("OodNds", len(deep), "{:d}")
    for key, name in [("residual", "Res"), ("knn", "Knn"), ("maha", "Maha"),
                      ("isoforest", "Iso"), ("edl_vacuity", "EdlVac"), ("edl_entropy", "EdlEnt"),
                      ("ignorance", "Ign"), ("softmax", "Soft")]:
        _macro(f"OodDeep{name}Auroc", 100 * deep[key].mean(), "{:.1f}")
        _macro(f"OodDeep{name}Fpr", 100 * deep[key + "_fpr95"].mean(), "{:.1f}")
        _macro(f"OodDeep{name}Rej", 100 * deep[key + "_rej"].mean(), "{:.1f}")
        _macro(f"OodDeep{name}Aupr", 100 * deep[key + "_aupr"].mean(), "{:.1f}")
    for key, name in [("knn", "Knn"), ("maha", "Maha"), ("isoforest", "Iso"),
                      ("edl_vacuity", "EdlVac"), ("edl_entropy", "EdlEnt")]:
        _macro(f"OodResVs{name}P", pmap[key], "p")
        w, t, l = _wtl(deep.residual, deep[key])
        _macro(f"OodResVs{name}Wtl", f"{w}/{t}/{l}")
    for cfg, name in (("ferl-compact", "Compact"), ("ferl-medium", "Medium")):
        _macro(f"Ood{name}ResAuroc", 100 * per[cfg].residual.mean(), "{:.1f}")


# --- read-out and width ablations ---------------------------------------------------

READOUTS = [("dempster_all", "Dempster, all nodes"),
            ("dempster_leaves", "Dempster, leaves (sets, ignorance)"),
            ("cautious_leaves", "Cautious rule, leaves"),
            ("average_leaves", "Average of leaves (point label)"),
            ("mixture_leaves", "Support-discounted average, leaves")]


def decision_rule_table():
    if not _need("results/decision_rule_ablation.csv"):
        return
    r = pd.read_csv("results/decision_rule_ablation.csv")
    per = r.groupby(["dataset", "readout"]).mean(numeric_only=True).reset_index()
    s = per.groupby("readout").mean(numeric_only=True)
    lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{2.5pt}",
             r"\caption{\textbf{Evidence sources and combination rule for FERL-deep} "
             f"({per.dataset.nunique()} datasets, {r.fold.nunique()} folds; one fitted tree per fold, "
             r"only the read-out changes). All entries except set size are on a 0--100 scale. "
             r"Accuracy, AURC and ECE use each read-out's pignistic probabilities. The average of "
             r"the leaf masses is the normalised soft vote that FERL reports as its point "
             r"prediction; Dempster's rule over the leaves gives its sets and ignorance. "
             r"Abst.\ is the rate of full abstention $S(x)=\Theta$ (for binary tasks every "
             r"non-singleton set is a full abstention); Sgl.\ risk is the error rate of "
             r"singleton outputs.}",
             r"\label{tab:decision-rule-ablation}", r"\begin{tabular}{@{}lcccccccccc@{}}", r"\toprule",
             r"Read-out & Acc. & AURC & ECE & Determ. & Cov. & Size & $u_{65}$ & Ign. & Abst. & Sgl.\ risk\\",
             r"\midrule"]
    for key, label in READOUTS:
        x = s.loc[key]
        lines.append(f"{label} & {100*x.accuracy:.1f} & {100*x.aurc:.1f} & {100*x.ece:.1f} & "
                     f"{100*x.determinacy:.1f} & {100*x.set_coverage:.1f} & {x.set_size:.2f} & "
                     f"{100*x.u65:.1f} & {100*x.ignorance:.1f} & {100*x.full_abstention:.1f} & "
                     f"{100*x.singleton_risk:.1f}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write("tab_decision_rule_ablation.tex", lines)
    for key, name in (("dempster_leaves", "Dl"), ("dempster_all", "Da"), ("cautious_leaves", "Cl"),
                      ("average_leaves", "Av"), ("mixture_leaves", "Mx")):
        x = s.loc[key]
        for col in ("accuracy", "aurc", "ece", "determinacy", "set_coverage", "u65",
                    "ignorance", "full_abstention", "singleton_risk"):
            _macro(f"Dr{name}{col.replace('_', '').capitalize()}", 100 * x[col], "{:.1f}")
        _macro(f"Dr{name}Size", x.set_size)
    pa = per.pivot(index="dataset", columns="readout", values="accuracy")
    _macro("DrAvVsDlAccP", _wilcoxon(pa.average_leaves, pa.dempster_leaves), "p")
    from scipy.stats import spearmanr
    dl = per[per.readout == "dempster_leaves"].set_index("dataset")
    rho, p = spearmanr(dl.full_abstention, 1 - dl.accuracy)
    _macro("DrAbstErrRho", rho)
    _macro("DrAbstErrP", p, "p")


WIDTHS = [("bootstrap", "Bootstrap std (used)"), ("fixed_0.5", "$0.5\\,\\sigma_{o,f}$"),
          ("fixed_0.25", "$0.25\\,\\sigma_{o,f}$"), ("fixed_0.1", "$0.1\\,\\sigma_{o,f}$"),
          ("fixed_0", "Width floor only (near-crisp)")]


def width_table():
    if not _need("results/width_ablation.csv"):
        return
    r = pd.read_csv("results/width_ablation.csv")
    per = r.groupby(["dataset", "width"]).mean(numeric_only=True).reset_index()
    s = per.groupby("width").mean(numeric_only=True)
    acc = per.pivot(index="dataset", columns="width", values="accuracy")
    aurc = per.pivot(index="dataset", columns="width", values="aurc")
    others = [k for k, _ in WIDTHS if k != "bootstrap"]
    pacc = dict(zip(others, _holm([_wilcoxon(acc.bootstrap, acc[k]) for k in others])))
    paurc = dict(zip(others, _holm([_wilcoxon(aurc.bootstrap, aurc[k]) for k in others])))
    lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\caption{\textbf{Band-width ablation for FERL-deep} "
             f"({per.dataset.nunique()} datasets, {r.fold.nunique()} folds). Only the rule that sets "
             r"each split's half-width $h$ changes; $\sigma_{o,f}$ is the node-weighted standard "
             r"deviation of the split feature. The last row keeps only the width floor, giving an "
             r"almost crisp tree grown by the same code. Accuracy, AURC and ECE use the soft vote; "
             r"the set columns use leaves-only Dempster. W/T/L counts datasets where the bootstrap "
             r"width is more/equally/less accurate; stars mark Holm-corrected Wilcoxon "
             r"differences from the bootstrap row (accuracy and AURC).}",
             r"\label{tab:width-ablation}", r"\begin{tabular}{@{}lccccccc@{}}", r"\toprule",
             r"Half-width rule & Acc. & AURC & ECE & Ign. & Cov. & Size & W/T/L\\", r"\midrule"]
    for key, label in WIDTHS:
        x = s.loc[key]
        sa = "" if key == "bootstrap" else _stars(pacc[key])
        su = "" if key == "bootstrap" else _stars(paurc[key])
        wtl = "--" if key == "bootstrap" else "/".join(map(str, _wtl(acc.bootstrap, acc[key])))
        lines.append(f"{label} & {100*x.accuracy:.2f}{sa} & {100*x.aurc:.2f}{su} & {100*x.ece:.1f} & "
                     f"{100*x.ignorance:.1f} & {100*x.set_coverage:.1f} & {x.set_size:.2f} & {wtl}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write("tab_width_ablation.tex", lines)
    for key, name in (("bootstrap", "Boot"), ("fixed_0", "Crisp"), ("fixed_0.25", "Q"),
                      ("fixed_0.5", "H"), ("fixed_0.1", "T")):
        _macro(f"W{name}Acc", 100 * s.loc[key].accuracy)
        _macro(f"W{name}Aurc", 100 * s.loc[key].aurc)
        _macro(f"W{name}Ign", 100 * s.loc[key].ignorance, "{:.1f}")
    _macro("WCrispAccP", pacc["fixed_0"], "p")
    _macro("WCrispWtl", "/".join(map(str, _wtl(acc.bootstrap, acc.fixed_0))))


# --- stability constants and explanation size ---------------------------------------

def stability_table():
    if not _need("results/stability_constants.csv"):
        return
    r = pd.read_csv("results/stability_constants.csv")
    per = r.groupby("dataset").mean(numeric_only=True)
    per["prop1"] = 2 * per.K * per.depth * per.lam_std / per.tau_q05

    def q(col, fmt="{:.2f}"):
        v = per[col]
        return (f"{fmt.format(v.median())} [{fmt.format(v.quantile(0.25))}, "
                f"{fmt.format(v.quantile(0.75))}]")

    spec = [
        (r"\emph{Constants of the global bounds}", None, None),
        (r"Leaves $R$", "n_leaves", "{:.0f}"),
        (r"Maximum depth $D$", "depth", "{:.1f}"),
        (r"Largest ramp slope $\lambda$ (per std)", "lam_std", "{:.1f}"),
        (r"Largest number of firing leaves $K$", "K", "{:.1f}"),
        (r"5\% quantile of the normaliser $Z$ ($\tau$)", "tau_q05", "{:.2f}"),
        (r"Mass-Lipschitz scale $2KD\lambda/\tau$ ($D$ = depth; lower bound for Prop.~\ref{prop:lipschitz})", "prop1", "{:.0f}"),
        (r"Certificate constant $L_T$ (per std, Prop.~\ref{prop:certificate})", "LT_std", "{:.0f}"),
        (r"\emph{Certified and empirical radii (std units)}", None, None),
        (r"Median soft-vote margin $\Delta(x)$", "margin_median", "{:.2f}"),
        (r"Median global radius $\Delta(x)/L_T$", "r_global_median", "{:.3f}"),
        (r"Median local radius", "r_local_median", "{:.3f}"),
        (r"Share of test points with local radius $\ge 0.05$", "r_local_ge_0.05", "{:.2f}"),
        (r"Share of test points with local radius $\ge 0.1$", "r_local_ge_0.1", "{:.2f}"),
        (r"Median flip distance found by random search", "r_attack_median", "{:.2f}"),
        (r"\emph{Routing overlap and explanation size}", None, None),
        (r"Mean training overlap per level $\omega$", "overlap_mean", "{:.2f}"),
        (r"Leaves firing $\ge 0.05$ per test point", "leaves_ge05", "{:.2f}"),
        (r"Leaves covering 90\% of the firing mass", "leaves_mass90", "{:.2f}"),
        (r"Conditions on the most-firing leaf", "dominant_depth", "{:.1f}"),
    ]
    lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
             r"\caption{\textbf{Stability constants, certificates and explanation size of "
             r"FERL-deep} (median [interquartile range] over the "
             f"{len(per)} datasets of the per-dataset means over five folds). Perturbations "
             r"are measured in training standard deviations ($\ell_\infty$). The local radius is "
             r"the largest $\epsilon$ with $\epsilon\,L_x(\epsilon)<\Delta(x)$, where $L_x$ counts "
             r"only the bands and subtrees an $\epsilon$-ball around $x$ can reach and the ball "
             r"stays inside every reached support gate; it is a valid lower bound on the flip "
             r"distance on every evaluated point, while random search gives an upper bound.}",
             r"\label{tab:stability}", r"\begin{tabular}{@{}lc@{}}", r"\toprule",
             r"Quantity & Median [IQR]\\", r"\midrule"]
    for label, col, fmt in spec:
        if col is None:
            lines.append(f"\\multicolumn{{2}}{{@{{}}l}}{{{label}}}\\\\")
        else:
            lines.append(f"{label} & {q(col, fmt)}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write("tab_stability.tex", lines)
    for col, name, fmt in (("prop1", "Prop", "{:.0f}"), ("LT_std", "Lt", "{:.0f}"),
                           ("lam_std", "Lam", "{:.0f}"), ("r_global_median", "Rglob", "{:.3f}"),
                           ("r_local_median", "Rloc", "{:.3f}"), ("r_local_ge_0.05", "RlocFive", "{:.2f}"),
                           ("r_local_ge_0.1", "RlocTen", "{:.2f}"),
                           ("r_attack_median", "Ratt", "{:.2f}"), ("overlap_mean", "Omega", "{:.2f}"),
                           ("overlap_max", "OmegaMax", "{:.2f}"),
                           ("leaves_ge05", "Active", "{:.1f}"), ("leaves_mass90", "Mass", "{:.1f}"),
                           ("dominant_depth", "Dom", "{:.1f}"), ("K", "K", "{:.0f}"),
                           ("depth", "D", "{:.0f}"), ("tau_q05", "Tau", "{:.2f}")):
        _macro(f"St{name}", per[col].median(), fmt)
    _macro("StValid", "all" if r.certificate_valid.min() == 1 else "not all", "{}")
    prop = per.prop1.median()
    _macro("StProp", f"{round(prop, -3):,.0f}".replace(",", "{,}") if prop >= 10000 else f"{prop:.0f}")
    assert r.certificate_valid.min() == 1, "local certificate exceeded a found flip distance"


# --- ignorance semantics ------------------------------------------------------------

TAXONOMY = [("correct_singleton", "Correct singleton"), ("wrong_singleton", "Wrong singleton"),
            ("cautious_correct", r"Set, point label correct"),
            ("useful_set", r"Set, point wrong, truth in set"), ("wrong_set", r"Set missing the truth")]


def ignorance_tables():
    if _need("results/ignorance_noop.csv"):
        r = pd.read_csv("results/ignorance_noop.csv")
        per = r.groupby("dataset").mean(numeric_only=True)
        m = per.mean()
        lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
                 r"\caption{\textbf{Representation dependence of the leaves-only Dempster read-out} "
                 f"({len(per)} datasets, five folds). Every leaf of the fitted FERL-deep tree is "
                 r"replaced by two children that keep its consequent, split on the leaf's "
                 r"highest-variance feature at its weighted median with a one-standard-deviation "
                 r"band. The classifier is unchanged---the soft vote differs by at most "
                 f"{r.soft_vote_max_abs_change.max():.1e}"
                 r"---but the evidential read-out is not. Upper block: 0--100 scale except set size.}",
                 r"\label{tab:noop}", r"\begin{tabular}{@{}lcc@{}}", r"\toprule",
                 r"Leaves-only Dempster read-out & Fitted tree & With no-op splits\\", r"\midrule",
                 f"Mean ignorance $m(\\Theta)$ & {100*m.ignorance:.1f} & {100*m.ignorance_noop:.1f}\\\\",
                 f"Set coverage & {100*m.coverage:.1f} & {100*m.coverage_noop:.1f}\\\\",
                 f"Mean set size & {m.set_size:.2f} & {m.set_size_noop:.2f}\\\\",
                 f"Determinacy & {100*m.determinacy:.1f} & {100*m.determinacy_noop:.1f}\\\\",
                 f"$u_{{65}}$ & {100*m.u65:.1f} & {100*m.u65_noop:.1f}\\\\",
                 f"Pignistic accuracy & {100*m.accuracy_betp:.1f} & {100*m.accuracy_betp_noop:.1f}\\\\",
                 r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        _write("tab_noop.tex", lines)
        _macro("NoopIgn", 100 * m.ignorance, "{:.1f}")
        _macro("NoopIgnAfter", 100 * m.ignorance_noop, "{:.1f}")
        _macro("NoopSize", m.set_size)
        _macro("NoopSizeAfter", m.set_size_noop)
        _macro("NoopCov", 100 * m.coverage, "{:.1f}")
        _macro("NoopCovAfter", 100 * m.coverage_noop, "{:.1f}")
        _macro("NoopIgnUp", int((per.ignorance_noop > per.ignorance).sum()), "{:d}")
        for key, name in (("auroc_err_ignorance", "Ign"), ("auroc_err_maxprob", "Maxp"),
                          ("auroc_err_setsize", "Size")):
            _macro(f"ErrAuroc{name}", 100 * per[key].mean(), "{:.1f}")
            _macro(f"ErrAuroc{name}Sd", 100 * per[key].std(), "{:.1f}")

        tax = [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
               r"\caption{\textbf{In-distribution failure taxonomy of FERL-deep's native sets} "
               f"({len(per)} datasets, five folds; means over datasets of per-fold means). "
               r"Conflict is $1-Z$, the mass Dempster's rule discards; support loss is "
               r"$1-\sum_\ell\phi_\ell(x)$, the routing mass removed by the support gates; "
               r"active leaves fire at $\ge 0.05$.}",
               r"\label{tab:taxonomy}", r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
               r"Outcome & Share & Ignorance & Conflict & Support loss & Active leaves\\", r"\midrule"]
        for key, label in TAXONOMY:
            tax.append(f"{label} & {100*m['frac_' + key]:.1f} & {100*per['ign_' + key].mean():.1f} & "
                       f"{100*per['conflict_' + key].mean():.1f} & "
                       f"{100*per['supportloss_' + key].mean():.2f} & {per['active_' + key].mean():.2f}\\\\")
        tax += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        _write("tab_taxonomy.tex", tax)
        for key, name in (("correct_singleton", "Cs"), ("wrong_singleton", "Ws"),
                          ("cautious_correct", "Cc"), ("useful_set", "Us"), ("wrong_set", "Wset")):
            _macro(f"Tax{name}Frac", 100 * m["frac_" + key], "{:.1f}")
            _macro(f"Tax{name}Ign", 100 * per["ign_" + key].mean(), "{:.1f}")


# --- concept-error robustness --------------------------------------------------------

CONDITIONS = [("oracle_pipeline", "Annotated concepts (train and test)"),
              ("oracle_test", "Annotated concepts at test only"),
              ("calibrated", "Calibrated detector (reference)"),
              ("raw", "Uncalibrated detector at test"),
              ("flip_0.05", "5\\% of concepts flipped"), ("flip_0.1", "10\\% flipped"),
              ("flip_0.2", "20\\% flipped"), ("ambiguous_0.1", "10\\% set to 0.5"),
              ("ambiguous_0.2", "20\\% set to 0.5"), ("ambiguous_0.3", "30\\% set to 0.5")]


def concept_error_table():
    blocks = []
    for subset, label in (("full", "CUB-200"), ("20", "CUB-20")):
        p = Path(f"results/cub_cbm_perf/concept_error_robustness_{subset}.csv")
        if p.exists():
            blocks.append((label, pd.read_csv(p).groupby("condition").mean(numeric_only=True),
                           pd.read_csv(p).detector_seed.nunique()))
    if not blocks:
        print("skip: concept-error robustness results missing")
        return
    lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{2.5pt}",
             r"\caption{\textbf{Concept-error propagation in the CUB concept bottleneck} (means over "
             r"detector seeds; 0--100 scale except set size). Heads are fitted on per-concept "
             r"isotonic-calibrated detector outputs and evaluated under the listed test-time "
             r"conditions; the first row refits them on annotated concepts. FERL columns use the "
             r"leaves-only Dempster read-out. Route agr.\ is the share of images whose most-firing "
             r"leaf is unchanged from the calibrated reference; Res.\ alarm is the share flagged by "
             r"the residual novelty score at the threshold that accepts 95\% of clean validation "
             r"images.}",
             r"\label{tab:concept-error}", r"\begin{tabular}{@{}lcccccccc@{}}", r"\toprule",
             r"Test-time concepts & FERL & CART & LR & Ign. & Size & Sgl.\ risk & Route agr. & Res.\ alarm\\"]
    for label, s, n_seeds in blocks:
        lines += [r"\midrule", f"\\multicolumn{{9}}{{@{{}}l}}{{\\emph{{{label} ({n_seeds} detector seeds)}}}}\\\\"]
        for key, name in CONDITIONS:
            if key not in s.index:
                continue
            x = s.loc[key]

            def f(col, scale=100, fmt="{:.1f}"):
                v = x.get(col, np.nan)
                return "--" if pd.isna(v) else fmt.format(scale * v)
            lines.append(f"{name} & {f('acc_FERL-deep')} & {f('acc_CART')} & {f('acc_LR')} & "
                         f"{f('ignorance')} & {f('set_size', 1, '{:.2f}')} & {f('singleton_risk')} & "
                         f"{f('route_agreement')} & {f('residual_false_alarm')}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write("tab_concept_error.tex", lines)
    s = blocks[0][1]
    for key, name in (("calibrated", "Cal"), ("flip_0.05", "FlipFive"), ("flip_0.1", "FlipTen"),
                      ("ambiguous_0.2", "AmbTwenty"), ("oracle_test", "OracleTest"), ("raw", "Raw")):
        if key in s.index:
            for col, cname in (("acc_FERL-deep", "Ferl"), ("acc_LR", "Lr"), ("acc_CART", "Cart"),
                               ("ignorance", "Ign"), ("set_size", "Size"),
                               ("singleton_risk", "Risk"), ("route_agreement", "Route"),
                               ("residual_false_alarm", "Alarm")):
                scale, fmt = (1, "{:.2f}") if col == "set_size" else (100, "{:.1f}")
                if col == "set_size" and s.loc[key, col] >= 10:
                    fmt = "{:.0f}"
                _macro(f"Ce{name}{cname}", scale * s.loc[key, col], fmt)


RUNTIME_NAMES = {"FERL-compact": "FERL-compact", "FERL-medium": "FERL-medium",
                 "FERL-deep": "FERL-deep", "CART": "CART", "C45": "C4.5", "FIGS": "FIGS",
                 "FURIA": "FURIA", "RuleFit": "RuleFit"}


def runtime_summary():
    """\\RuntimeSummary prose from results/runtime_scaling/wallclock_scaling.csv."""
    p = Path("results/runtime_scaling/wallclock_scaling.csv")
    if not p.exists():
        print("skip: runtime results missing")
        return
    d = pd.read_csv(p)
    d = d[(d.status == "ok") & d.model.isin(RUNTIME_NAMES)]
    if "FERL-deep" not in set(d.model):
        print("skip runtime summary: FERL-deep not timed yet")
        return
    per = d.groupby(["model", "dataset"]).train_s.median().groupby("model")
    med, mx = per.median(), per.max()
    n = d.dataset.nunique()
    fmt = lambda v: f"{v:.3f}" if v < 0.1 else f"{v:.2f}"
    text = (
        f"Median over the {n} datasets, FERL-compact fits in {fmt(med['FERL-compact'])}\\,s, "
        f"FERL-medium in {fmt(med['FERL-medium'])}\\,s and FERL-deep in "
        f"{fmt(med['FERL-deep'])}\\,s (at most {fmt(mx['FERL-deep'])}\\,s). "
        f"CART takes {fmt(med['CART'])}\\,s, FIGS {fmt(med['FIGS'])}\\,s, C4.5 "
        f"{fmt(med['C45'])}\\,s"
        + (f", FURIA {fmt(med['FURIA'])}\\,s and RuleFit {fmt(med['RuleFit'])}\\,s"
           if "RuleFit" in med else f" and FURIA {fmt(med['FURIA'])}\\,s")
        + ".")
    from ferl.pipeline.run_configs import SELECTED_30
    b = pd.read_csv("results/benchmark2_per_fold.csv")
    b = b[b.dataset.isin(SELECTED_30) & (b.status == "ok")]
    bt = b.groupby(["model", "dataset"]).train_s.median().groupby("model").median()
    bp = b.groupby(["model", "dataset"]).pred_s.median().groupby("model").median()
    text += (
        f" In the main benchmark (Table~\\ref{{tab:frontier}}), FERL-deep's median fit time "
        f"({fmt(bt['FERL-deep'])}\\,s) is below that of RRL ({fmt(bt['RRL'])}\\,s), "
        f"NeuRules ({fmt(bt['NeuRules'])}\\,s), RL-Net ({bt['RL-Net']:.0f}\\,s) and FUCS "
        f"({bt['FuzzyUCS-DS']:.0f}\\,s), and it scores a test fold in "
        f"{1000 * bp['FERL-deep']:.1f}\\,ms against {1000 * bp['FURIA']:.0f}\\,ms for "
        f"FURIA and {1000 * bp['FuzzyUCS-DS']:.0f}\\,ms for FUCS, the other evidential fuzzy "
        f"rule learner.")
    (GEN / "runtime_summary.tex").write_text("\\newcommand{\\RuntimeSummary}{" + text + "}\n")
    print(f"wrote {GEN / 'runtime_summary.tex'}")


def benchmark_macros():
    """Per-method accuracy and size from the main benchmark, for the prose."""
    from ferl.pipeline.run_configs import SELECTED_30
    d = pd.read_csv("results/benchmark2_per_fold.csv")
    d = d[d.dataset.isin(SELECTED_30) & (d.status == "ok")]
    g = d.groupby(["model", "dataset"]).mean(numeric_only=True).groupby("model").mean()
    for key, name in (("FERL-compact", "Compact"), ("FERL-credal", "CompactDs"),
                      ("FERL-medium", "Medium"), ("FERL-deep", "Deep"),
                      ("CART", "Cart"), ("C45", "Cfour"), ("NeuRules", "NeuRules"),
                      ("RuleFit", "RuleFit"), ("FIGS", "Figs"), ("FERL-deep-tuned", "Tuned")):
        if key not in g.index:
            continue
        _macro(f"Bm{name}Acc", 100 * g.loc[key, "acc"], "{:.1f}")
        _macro(f"Bm{name}Aurc", 100 * g.loc[key, "aurc"], "{:.2f}")
        _macro(f"Bm{name}Size", g.loc[key, "complexity"], "{:.0f}")


def per_dataset_table():
    """Appendix D: per-dataset accuracy of FERL-deep, FERL-medium, LR and FIGS,
    sorted by the FERL-deep - LR gap, with the statistics quoted in the text."""
    from scipy.stats import mannwhitneyu, spearmanr
    from ferl.pipeline.run_configs import SELECTED_30
    d = pd.read_csv("results/benchmark2_per_fold.csv")
    d = d[d.dataset.isin(SELECTED_30) & (d.status == "ok")]
    acc = d.pivot_table(index="dataset", columns="model", values="acc", aggfunc="mean") * 100
    meta = {}
    for line in (GEN / "tab_datasets.tex").read_text().splitlines():
        parts = [p.strip().rstrip("\\").strip() for p in line.split("&")]
        if len(parts) == 4 and parts[1].isdigit():
            meta[parts[0]] = tuple(int(x) for x in parts[1:])
    cols = ["FERL-deep", "FERL-medium", "LogReg", "FIGS"]
    t = acc[cols].copy()
    t["delta"] = t["FERL-deep"] - t["LogReg"]
    t = t.sort_values("delta", ascending=False)
    rows = []
    for ds, r in t.iterrows():
        n, dd, C = meta[ds]
        best = r[cols].max()
        cells = [(f"\\textbf{{{r[c]:.1f}}}" if np.isclose(r[c], best) else f"{r[c]:.1f}") for c in cols]
        rows.append(f"{ds} & {n} & {dd} & {C} & " + " & ".join(cells) + f" & {r.delta:+.1f}\\\\")
    m = t.mean()
    lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\caption{\textbf{Per-dataset accuracy on the 30 tabular benchmarks} for four "
             r"representative methods (mean over the five folds; $n$, $d$ and $C$ are the numbers "
             r"of samples, features and classes). Rows are sorted by "
             r"$\Delta=\text{FERL-deep}-\text{LR}$ ($>0$ favours FERL-deep).}",
             r"\label{tab:supp-frontier-nuance}", r"\begin{tabular}{@{}lrrrccccr@{}}", r"\toprule",
             r"Dataset & $n$ & $d$ & $C$ & FERL-deep & FERL-medium & LR & FIGS & $\Delta$\\", r"\midrule",
             *rows, r"\midrule",
             f"Mean & & & & {m['FERL-deep']:.1f} & {m['FERL-medium']:.1f} & {m['LogReg']:.1f} & "
             f"{m['FIGS']:.1f} & {m['delta']:+.1f}\\\\",
             r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write("tab_per_dataset.tex", lines)
    w, ti, l = _wtl(t["FERL-deep"], t["LogReg"], tol=0.05)
    _macro("PdWtl", f"{w}/{ti}/{l}")
    dims = np.array([meta[ds][1] for ds in t.index]); ns = np.array([meta[ds][0] for ds in t.index])
    cs = np.array([meta[ds][2] for ds in t.index])
    for name, x in (("Dim", dims), ("N", ns), ("C", cs)):
        rho, p = spearmanr(t.delta, x)
        _macro(f"PdRho{name}", rho)
        _macro(f"PdRho{name}P", p, "p")
    multi, binary = t.delta[cs > 2], t.delta[cs == 2]
    _macro("PdMultiDelta", multi.mean(), "{:+.1f}")
    _macro("PdBinDelta", binary.mean(), "{:+.1f}")
    _macro("PdMwP", mannwhitneyu(multi, binary).pvalue, "p")


def tuned_width_selection():
    """How often inner cross-validation picks each band-width rule (manifests of
    experiments/benchmark2/harness.py --models FERL-deep-tuned)."""
    import glob
    import json
    picks = []
    for path in glob.glob("results/bench_tuned/*/FERL-deep-tuned/fold*.json"):
        sel = json.load(open(path)).get("resolved_params", {}).get("selected_width")
        if sel is not None:
            picks.append(str(sel))
    if not picks:
        print("skip: no tuned-width manifests")
        return
    counts = pd.Series(picks).value_counts(normalize=True)
    for key, name in (("bootstrap", "Boot"), ("0.25", "Q"), ("0.5", "H"), ("1.0", "One")):
        _macro(f"TunedPick{name}", 100 * counts.get(key, 0.0), "{:.0f}")
    _macro("TunedNfits", len(picks), "{:d}")


ENC_DATASETS = ["australian", "crx", "german"]
ENC_NAMES = {"FERL-compact": "FERL-compact", "FERL-medium": "FERL-medium", "FERL-deep": "FERL-deep",
             "CART": "CART", "C45": "C4.5", "FIGS": "FIGS", "RuleFit": "RuleFit",
             "FURIA": "FURIA", "FuzzyUCS-DS": "FUCS (DS)", "NeuRules": "NeuRules",
             "SampledRuleList": "SamRuLe", "NCC": "NCC", "ICDT": "CDT", "LogReg": "Logistic reg.",
             "RF": "RF", "GBDT": "GB", "EDL": "EDL"}


def encoding_table():
    """Appendix: accuracy on the three datasets with multi-level nominal attributes
    under the benchmark's integer codes and under one-hot encoding
    (harness.py --out-dir results/bench_oh with FERL_NOMINAL=onehot, scored with
    score.py --bench-dir results/bench_oh --per-fold-csv results/benchmark2_onehot_per_fold.csv)."""
    p = Path("results/benchmark2_onehot_per_fold.csv")
    if not p.exists():
        print("skip: one-hot sensitivity results missing")
        return
    oh = pd.read_csv(p)
    base = pd.read_csv("results/benchmark2_per_fold.csv")
    rows, deltas = [], {}
    for key, name in ENC_NAMES.items():
        a = base[(base.model == key) & base.dataset.isin(ENC_DATASETS) & (base.status == "ok")]
        b = oh[(oh.model == key) & oh.dataset.isin(ENC_DATASETS) & (oh.status == "ok")]
        if a.dataset.nunique() < 3 or b.dataset.nunique() < 3:
            continue
        ao = a.groupby("dataset").acc.mean() * 100
        bo = b.groupby("dataset").acc.mean() * 100
        d = bo - ao
        deltas[key] = d.mean()
        rows.append(f"{name} & " + " & ".join(f"{ao[x]:.1f} / {bo[x]:.1f}" for x in ENC_DATASETS)
                    + f" & {d.mean():+.1f}\\\\")
    lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
             r"\caption{\textbf{Sensitivity to the encoding of nominal attributes.} Accuracy "
             r"(0--100, mean over the five folds) with the benchmark's integer codes / with one-hot "
             r"encoding, on the three datasets with multi-level nominal attributes, and the mean "
             r"change. RRL and RL-Net could not be rerun and are omitted.}",
             r"\label{tab:encoding}", r"\begin{tabular}{@{}lcccr@{}}", r"\toprule",
             r"Method & australian & crx & german & $\Delta$\\", r"\midrule",
             *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write("tab_encoding.tex", lines)
    s = pd.Series(deltas)
    _macro("EncDeepDelta", s.get("FERL-deep", np.nan), "{:+.1f}")
    _macro("EncLrDelta", s.get("LogReg", np.nan), "{:+.1f}")
    _macro("EncNgain", int((s > 0.05).sum()), "{:d}")
    _macro("EncNlose", int((s < -0.05).sum()), "{:d}")
    _macro("EncNmethods", len(s), "{:d}")
    _macro("EncMeanDelta", s.mean(), "{:+.1f}")


def write_macros():
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(MACROS.items())]
    _write("rev_macros.tex", lines)


if __name__ == "__main__":
    for step in (credal_table, ood_table, decision_rule_table, width_table,
                 stability_table, ignorance_tables, concept_error_table, runtime_summary,
                 benchmark_macros, per_dataset_table, tuned_width_selection, encoding_table):
        try:
            step()
        except Exception as exc:  # keep going: each table is independent
            print(f"FAILED {step.__name__}: {exc}")
    write_macros()

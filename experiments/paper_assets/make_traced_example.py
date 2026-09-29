"""Traced FERL-deep predictions on the wine data for the paper's worked example.

Two deterministic cases, both from outer fold 0 of the main tabular protocol:

(a) In distribution: the first test wine whose native prediction set has two
    classes. Reports every leaf firing >= 0.01 (its conditions, firing and
    consequent), the combined Dempster masses, belief/plausibility, the set and
    the soft-vote label.
(b) Near-OOD: a tree trained without cultivar 3. Among cultivar-3 test wines
    whose leaves fire at full in-support strength, the one with the largest
    residual score. Reports the dominant route, the native ignorance, the
    residual score against the 95th percentile of training scores, and the
    three most atypical free attributes of the dominant leaf.

Writes paper/generated/tab_traced_example.tex and results/traced_example.json.
Run from the repository root:
    python experiments/paper_assets/make_traced_example.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from ferl.core.learned_tree import LearnedFuzzyTree
from ferl.pipeline.run_configs import load_filtered

sys.path.insert(0, "experiments/reliability")
from ds_stability_constants import protocol_folds  # noqa: E402
from ds_ood_residual_variants import fit_residual_learned, residual_score  # noqa: E402

GEN = Path(os.environ.get("FERL_GEN_DIR", "paper/generated"))
KEEL = Path(os.environ.get("KEEL_DIR", "../keel_datasets"))
CLASS = ["cultivar 1", "cultivar 2", "cultivar 3"]


def feature_names(dataset="wine"):
    lines = (KEEL / dataset / f"{dataset}.dat").read_text().splitlines()
    return [l.split()[1] for l in lines
            if l.lower().startswith("@attribute") and not l.split()[1].lower() == "class"]


def leaf_paths(node, path=(), name="r"):
    """{leaf name: [(feature, '<=' | '>=', centre, half-width), ...]}."""
    if node["leaf"]:
        return {name: list(path)}
    cond = (node["f"], node["center"], node["h"])
    out = leaf_paths(node["L"], path + ((cond[0], "<=", cond[1], cond[2]),), name + "_0")
    out.update(leaf_paths(node["R"], path + ((cond[0], ">=", cond[1], cond[2]),), name + "_1"))
    return out


def describe(path, names):
    """Merge repeated tests on one feature into a (fuzzy) interval."""
    bounds = {}
    for f, d, c, _ in path:
        lo, hi = bounds.get(f, (-np.inf, np.inf))
        bounds[f] = (max(lo, c), hi) if d == ">=" else (lo, min(hi, c))
    parts = []
    for f, (lo, hi) in bounds.items():
        if np.isfinite(lo) and np.isfinite(hi):
            parts.append(f"{lo:.3g} $\\lesssim$ {names[f]} $\\lesssim$ {hi:.3g}")
        elif np.isfinite(hi):
            parts.append(f"{names[f]} $\\lesssim$ {hi:.3g}")
        else:
            parts.append(f"{names[f]} $\\gtrsim$ {lo:.3g}")
    return ", ".join(parts)


def dempster(M, cons):
    q_theta = np.prod(1 - M)
    q_c = np.prod(1 - M[:, None] + M[:, None] * cons, axis=0)
    m_c = np.clip(q_c - q_theta, 0, None)
    Z = m_c.sum() + q_theta
    return m_c / Z, q_theta / Z


def case_in_distribution(X, y, names):
    fold, train, test = next(iter(protocol_folds(X, y)))
    model = LearnedFuzzyTree(random_state=0).fit(X[train], y[train])
    sets = model.predict_set(X[test])
    pick = int(np.flatnonzero(sets.sum(1) == 2)[0])
    x = X[test][pick:pick + 1]
    M, cons, lnames, _ = model.node_activation_matrix(x)
    leaves = np.flatnonzero(model.leaf_mask(lnames))
    paths = leaf_paths(model.root_)
    fired = [(lnames[i], float(M[0, i]), cons[i]) for i in leaves if M[0, i] >= 0.01]
    fired.sort(key=lambda t: -t[1])
    m_c, m_theta = dempster(M[0, leaves], cons[leaves])
    soft = model.predict_proba(x)[0]
    return {
        "test_position": pick, "truth": CLASS[int(y[test][pick])],
        "leaves": [{"conditions": describe(paths[n], names), "depth": len(paths[n]),
                    "firing": phi, "consequent": [float(v) for v in c]}
                   for n, phi, c in fired],
        "mass": [float(v) for v in m_c], "ignorance": float(m_theta),
        "belief": [float(v) for v in m_c], "plausibility": [float(v + m_theta) for v in m_c],
        "set": [CLASS[c] for c in np.flatnonzero(sets[pick])],
        "soft_vote": [float(v) for v in soft], "soft_label": CLASS[int(soft.argmax())],
    }


def case_near_ood(X, y, names):
    fold, train, test = next(iter(protocol_folds(X, y)))
    keep = train[y[train] != 2]
    model = LearnedFuzzyTree(random_state=0).fit(X[keep], y[keep])
    stats, _ = fit_residual_learned(model, X[keep], X.shape[1])
    threshold = float(np.quantile(residual_score(model, X[keep], stats), 0.95))
    novel = test[y[test] == 2]
    M, cons, lnames, _ = model.node_activation_matrix(X[novel])
    leaves = np.flatnonzero(model.leaf_mask(lnames))
    firing = M[:, leaves].sum(1)
    scores = residual_score(model, X[novel], stats)
    full = np.flatnonzero(firing >= 0.99)
    pick = int(full[np.argmax(scores[full])])
    x = X[novel][pick:pick + 1]
    Ml = M[pick, leaves]
    dom = lnames[leaves[int(Ml.argmax())]]
    free, mu, var = stats[dom]
    z2 = (x[0, free] - mu) ** 2 / var
    top = np.argsort(-z2)[:3]
    _, bel, pl, ign = model.predict_ds(x, leaves_only=True)
    paths = leaf_paths(model.root_)
    return {
        "truth": CLASS[2], "predicted": CLASS[int(model.predict_proba(x)[0].argmax())],
        "dominant_route": describe(paths[dom], names), "dominant_firing": float(Ml.max()),
        "total_firing": float(firing[pick]), "ignorance": float(ign[0]),
        "residual": float(scores[pick]), "threshold": threshold,
        "anomalous": [{"attribute": names[int(free[i])], "value": float(x[0, free[i]]),
                       "leaf_mean": float(mu[i]), "leaf_sd": float(np.sqrt(var[i])),
                       "z2": float(z2[i])} for i in top],
    }


def to_latex(a, b):
    rows = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
            r"\caption{\textbf{Two traced FERL-deep predictions on the wine data} "
            r"(outer fold~0). (a)~A test wine whose native set contains two "
            r"cultivars: every leaf firing at least $0.01$, its conditions "
            r"($\lesssim$/$\gtrsim$ mark a fuzzy threshold; repeated tests on one "
            r"feature are merged into an interval), firing and consequent, "
            r"then the combined Dempster masses. (b)~A cultivar-3 wine scored by a "
            r"tree trained without cultivar~3: the rules fire at full strength and "
            r"ignorance stays low, but the residual score exceeds the 95th "
            r"percentile of training scores and names the atypical free "
            r"attributes of the dominant leaf.}",
            r"\label{tab:traced}", r"\setlength{\tabcolsep}{3pt}",
            r"\begin{tabular}{@{}p{0.52\linewidth}rp{0.36\linewidth}@{}}", r"\toprule",
            r"\multicolumn{3}{@{}l}{\emph{(a) In distribution; true class: "
            + a["truth"] + r"}}\\",
            r"Fired leaf (conditions) & $\phi_\ell$ & consequent $p_\ell$\\", r"\midrule"]
    for leaf in a["leaves"]:
        cons = "/".join(f"{v:.2f}" for v in leaf["consequent"])
        rows.append(f"{leaf['conditions']} & {leaf['firing']:.2f} & ({cons})\\\\")
    mass = "/".join(f"{v:.2f}" for v in a["mass"])
    pl = "/".join(f"{v:.2f}" for v in a["plausibility"])
    soft = "/".join(f"{v:.2f}" for v in a["soft_vote"])
    rows += [r"\midrule",
             f"Dempster masses $m(\\{{c\\}})$ & \\multicolumn{{2}}{{p{{0.42\\linewidth}}}}{{({mass}), $m(\\Theta)={a['ignorance']:.2f}$}}\\\\",
             f"Plausibility $\\mathrm{{Pl}}(c)$ & \\multicolumn{{2}}{{p{{0.42\\linewidth}}}}{{({pl})}}\\\\",
             f"Set $S(x)$; soft vote & \\multicolumn{{2}}{{p{{0.42\\linewidth}}}}{{\\{{{', '.join(a['set'])}\\}}; ({soft}) $\\to$ {a['soft_label']}}}\\\\",
             r"\midrule",
             r"\multicolumn{3}{@{}l}{\emph{(b) Near-OOD; true class: " + b["truth"]
             + r" (unseen in training)}}\\",
             f"Dominant route: {b['dominant_route']} & {b['dominant_firing']:.2f} & $\\to$ {b['predicted']}\\\\",
             f"Total leaf firing; ignorance & \\multicolumn{{2}}{{p{{0.42\\linewidth}}}}{{{b['total_firing']:.2f}; {b['ignorance']:.2f}}}\\\\",
             f"Residual score (threshold) & \\multicolumn{{2}}{{p{{0.42\\linewidth}}}}{{{b['residual']:.1f} ({b['threshold']:.1f})}}\\\\"]
    for an in b["anomalous"]:
        rows.append(f"\\quad {an['attribute']}: value {an['value']:.3g} vs.\\ leaf "
                    f"{an['leaf_mean']:.3g}\\,$\\pm$\\,{an['leaf_sd']:.2g} & "
                    f"\\multicolumn{{2}}{{p{{0.42\\linewidth}}}}{{$z^2={an['z2']:.1f}$}}\\\\")
    rows += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(rows) + "\n"


def main():
    X, y = load_filtered("wine")
    X = np.asarray(X, float)
    names = feature_names("wine")
    assert len(names) == X.shape[1], (names, X.shape)
    a, b = case_in_distribution(X, y, names), case_near_ood(X, y, names)
    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / "tab_traced_example.tex").write_text(to_latex(a, b))
    Path("results/traced_example.json").write_text(json.dumps({"a": a, "b": b}, indent=1))
    print(json.dumps({"a": a, "b": b}, indent=1))


if __name__ == "__main__":
    main()

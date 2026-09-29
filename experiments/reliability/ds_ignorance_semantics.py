"""What does FERL-deep's leaves-only Dempster ignorance measure?

Same protocol as the main tabular benchmark (five seeded stratified outer
folds, 25% of each outer-training fold held out and unused). Three studies:

1. ``--study noop``: representation dependence. Every leaf of a fitted tree is
   replaced by a *no-op split*: two children that keep the leaf's consequent,
   split on the leaf's highest-variance feature at its weighted median with a
   band of one weighted standard deviation, and no support gate. The classifier is unchanged (the soft vote is identical), so any change
   in the Dempster read-out is due to representation alone. Also reports the
   in-distribution failure taxonomy and how well ignorance flags errors.
2. ``--study depth``: depth sweep (max_depth 2..12) of FERL-deep, reporting
   accuracy, leaf count and the Dempster read-out, next to depth-matched CART,
   leaf-matched CART (``max_leaf_nodes`` = FERL's leaf count) and leaf-matched
   FIGS (``max_rules`` = FERL's leaf count - 1, only up to 128 rules because
   FIGS's fit time grows quickly with its rule budget).

Run from the repository root:
    python experiments/reliability/ds_ignorance_semantics.py --study noop
    python experiments/reliability/ds_ignorance_semantics.py --study depth
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from ferl.core.learned_tree import LearnedFuzzyTree
from ferl.pipeline.run_configs import SELECTED_30, load_filtered

sys.path.insert(0, "experiments/reliability")
from ds_stability_constants import protocol_folds  # noqa: E402

DEPTHS = [2, 4, 6, 8, 10, 12]
FIGS_MAX_RULES = 128
EPS = 1e-12


def dempster_leaves(model, X):
    """Leaves-only Dempster read-out plus the raw quantities behind it."""
    M, cons, names, _ = model.node_activation_matrix(X)
    leaves = np.flatnonzero(model.leaf_mask(names))
    Ml, cl = M[:, leaves], cons[leaves]
    q_theta = np.prod(1.0 - Ml, axis=1)
    q_c = np.prod(1.0 - Ml[:, :, None] + Ml[:, :, None] * cl[None], axis=1)
    m_c = np.clip(q_c - q_theta[:, None], 0.0, None)
    Z = m_c.sum(1) + q_theta
    m_c, ign = m_c / Z[:, None], q_theta / Z
    bel, pl = m_c, m_c + ign[:, None]
    S = pl >= bel.max(1, keepdims=True) - EPS
    return {"betp": m_c + ign[:, None] / m_c.shape[1], "ign": ign, "set": S,
            "conflict": 1.0 - Z, "firing": Ml.sum(1),
            "active": (Ml >= 0.05).sum(1)}


def set_metrics(S, y):
    sizes = S.sum(1).astype(float)
    hit = S[np.arange(len(y)), y]
    u65 = np.where(hit, 1.6 / sizes - 0.6 / sizes ** 2, 0.0)
    return {"coverage": float(hit.mean()), "set_size": float(sizes.mean()),
            "determinacy": float((sizes == 1).mean()), "u65": float(u65.mean())}


def _auroc(label, score):
    return float(roc_auc_score(label, score)) if 0 < label.mean() < 1 else float("nan")


def _noop_leaf(leaf, Xtr, m):
    """Replace one leaf by a no-op split (identical consequent on both sides)."""
    region = m > 1e-6
    if region.sum() < 4:
        return leaf
    w = m[region] / m[region].sum()
    Xr = Xtr[region]
    mean = w @ Xr
    std = np.sqrt(w @ (Xr - mean) ** 2)
    f = int(np.argmax(std))
    if std[f] <= 0:
        return leaf
    order = np.argsort(Xr[:, f])
    center = float(Xr[order, f][np.searchsorted(np.cumsum(w[order]), 0.5)])
    child = {"leaf": True, "dist": leaf["dist"].copy(), "support": leaf["support"] / 2}
    # No support gate on the added split (range pushed far out), so the two
    # children's firings always sum to the parent's and the soft vote is unchanged.
    far = 1e6 * (float(np.ptp(Xtr[:, f])) + 1.0)
    return {"leaf": False, "f": f, "center": center, "h": float(std[f]),
            "lo": float(Xtr[:, f].min()) - far, "hi": float(Xtr[:, f].max()) + far,
            "dist": leaf["dist"].copy(), "support": leaf["support"],
            "L": child, "R": copy.deepcopy(child)}


def noop_tree(model, Xtr):
    new = copy.deepcopy(model)

    def walk(node, m):
        if node["leaf"]:
            return _noop_leaf(node, Xtr, m)
        left, right = model._split(node, Xtr)
        node["L"] = walk(node["L"], m * left)
        node["R"] = walk(node["R"], m * right)
        return node

    new.root_ = walk(new.root_, np.ones(len(Xtr)))
    return new


def taxonomy(out, point, y):
    """In-distribution failure taxonomy of the native set."""
    size = out["set"].sum(1)
    hit = out["set"][np.arange(len(y)), y]
    correct = point == y
    groups = {
        "correct_singleton": (size == 1) & hit,
        "wrong_singleton": (size == 1) & ~hit,
        "cautious_correct": (size > 1) & correct,
        "useful_set": (size > 1) & ~correct & hit,
        "wrong_set": (size > 1) & ~hit,
    }
    rows = {}
    for g, mask in groups.items():
        rows[f"frac_{g}"] = float(mask.mean())
        for key in ("ign", "conflict", "active"):
            rows[f"{key}_{g}"] = float(out[key][mask].mean()) if mask.any() else np.nan
        rows[f"supportloss_{g}"] = (float((1 - out["firing"][mask]).mean())
                                    if mask.any() else np.nan)
    return rows


def noop_fold(X, y, train, test):
    model = LearnedFuzzyTree(random_state=0).fit(X[train], y[train])
    noop = noop_tree(model, X[train])
    Xte, yte = X[test], y[test]
    P, P_noop = model.predict_proba(Xte), noop.predict_proba(Xte)
    base, alt = dempster_leaves(model, Xte), dempster_leaves(noop, Xte)
    point = P.argmax(1)
    err = (point != yte).astype(int)
    row = {"soft_vote_max_abs_change": float(np.abs(P - P_noop).max()),
           "accuracy_soft": float((point == yte).mean()),
           "accuracy_betp": float((base["betp"].argmax(1) == yte).mean()),
           "accuracy_betp_noop": float((alt["betp"].argmax(1) == yte).mean()),
           "ignorance": float(base["ign"].mean()),
           "ignorance_noop": float(alt["ign"].mean()),
           "auroc_err_ignorance": _auroc(err, base["ign"]),
           "auroc_err_maxprob": _auroc(err, -P.max(1)),
           "auroc_err_setsize": _auroc(err, base["set"].sum(1))}
    row.update({k: v for k, v in set_metrics(base["set"], yte).items()})
    row.update({f"{k}_noop": v for k, v in set_metrics(alt["set"], yte).items()})
    row.update(taxonomy(base, point, yte))
    return row


def depth_fold(X, y, train, test, depth):
    from imodels import FIGSClassifier
    Xtr, ytr, Xte, yte = X[train], y[train], X[test], y[test]
    model = LearnedFuzzyTree(max_depth=depth, random_state=0).fit(Xtr, ytr)
    out = dempster_leaves(model, Xte)
    n_leaves = model.n_leaves_
    cart_d = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(Xtr, ytr)
    row = {"n_leaves": n_leaves,
           "accuracy": float((model.predict_proba(Xte).argmax(1) == yte).mean()),
           "ignorance": float(out["ign"].mean()),
           **set_metrics(out["set"], yte),
           "cart_depth_accuracy": float((cart_d.predict(Xte) == yte).mean()),
           "cart_depth_leaves": int(cart_d.get_n_leaves())}
    if n_leaves >= 2:
        cart_l = DecisionTreeClassifier(max_leaf_nodes=n_leaves, random_state=0).fit(Xtr, ytr)
        row["cart_leafmatch_accuracy"] = float((cart_l.predict(Xte) == yte).mean())
        row["figs_leafmatch_accuracy"] = np.nan
        if n_leaves - 1 <= FIGS_MAX_RULES:   # FIGS cost grows fast with its rule budget
            try:
                figs = FIGSClassifier(max_rules=max(n_leaves - 1, 1), random_state=0).fit(Xtr, ytr)
                row["figs_leafmatch_accuracy"] = float((figs.predict(Xte) == yte).mean())
            except Exception as exc:  # FIGS can fail on degenerate folds
                print(f"    FIGS failed: {str(exc)[:80]}", flush=True)
    return row


def run(study, datasets, output):
    rows = []
    for dataset in datasets:
        try:
            X, y = load_filtered(dataset)
        except Exception as exc:
            print(f"{dataset}: SKIP load ({exc})", flush=True)
            continue
        X = np.asarray(X, float)
        for fold, train, test in protocol_folds(X, y):
            if study == "noop":
                rows.append({"dataset": dataset, "fold": fold,
                             **noop_fold(X, y, train, test)})
            else:
                for depth in DEPTHS:
                    rows.append({"dataset": dataset, "fold": fold, "max_depth": depth,
                                 **depth_fold(X, y, train, test, depth)})
        print(f"{dataset}: done", flush=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output, index=False)
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=["noop", "depth"], required=True)
    parser.add_argument("--datasets", nargs="+", default=SELECTED_30)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    out = args.output or Path(f"results/ignorance_{args.study}.csv")
    frame = run(args.study, args.datasets, out)
    keys = ["dataset", "max_depth"] if args.study == "depth" else ["dataset"]
    per_ds = frame.groupby(keys).mean(numeric_only=True).reset_index()
    group = per_ds.groupby("max_depth") if args.study == "depth" else per_ds
    print(group.mean(numeric_only=True).round(4).to_string())

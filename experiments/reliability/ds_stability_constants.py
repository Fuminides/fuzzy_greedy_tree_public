"""Empirical stability constants, local certificates, routing overlap and
explanation size for FERL-deep (``LearnedFuzzyTree``).

Same protocol as the main tabular benchmark: five seeded stratified outer folds,
25% of each outer-training fold held out (unused here), tree fitted on the rest.
All perturbation sizes are in feature-standard-deviation units (l_inf ball on
x / sigma, sigma = training std), so datasets with different scales compare.

Per dataset and fold it reports

* ``LT_std`` -- the global constant of the local-certificate proposition,
  L_T = max over root-to-leaf paths of sum_o sigma_{f_o} / h_o;
* ``r_global_*`` -- the certified radius Delta(x) / L_T it implies, where
  Delta(x) is the top-1 minus top-2 soft-vote margin;
* ``r_local_*`` -- a local certificate: the largest eps such that
  eps * L_x(eps) < Delta(x), where L_x(eps) counts only the fuzzy bands the
  eps-ball around x reaches and only the subtrees it can route to. The ball must
  also stay inside every reached node's support gate (gate = 1), which is the
  proposition's in-support assumption;
* ``r_attack_*`` -- an empirical upper bound: the smallest flip distance found
  along 16 random sign directions (bisection, up to 5 std) for <= 200 test points;
* ``lam_std``, ``depth``, ``K``, ``tau_q05`` -- the quantities in the global
  mass-Lipschitz proposition (largest ramp slope, maximum depth, largest number
  of simultaneously firing leaves, 5% quantile of the Dempster normaliser Z);
* ``overlap_mean`` / ``overlap_max`` -- training routing overlap per depth
  level, sum_o n_o / N with n_o the samples of positive membership at node o;
* ``leaves_ge05`` / ``leaves_mass90`` / ``dominant_depth`` -- explanation size:
  leaves firing >= 0.05, leaves needed to cover 90% of firing mass, and the
  number of conditions on the most-firing leaf.

Run from the repository root:
    python experiments/reliability/ds_stability_constants.py
    python experiments/reliability/ds_stability_constants.py --datasets wine
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from ferl.core.learned_tree import LearnedFuzzyTree
from ferl.pipeline.run_configs import SELECTED_30, load_filtered

N_FOLDS = 5
CAL_FRAC = 0.25
EPS_MAX = 5.0
N_ATTACK_POINTS = 200
N_DIRECTIONS = 16
DEFAULT_OUTPUT = Path("results/stability_constants.csv")


def protocol_folds(X, y, n_folds=N_FOLDS):
    """Yield (fold, train_idx, test_idx) exactly as the benchmark harness."""
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=33)
    for fold, (outer_train, test) in enumerate(splitter.split(X, y)):
        try:
            train, _ = train_test_split(outer_train, test_size=CAL_FRAC,
                                        random_state=fold, stratify=y[outer_train])
        except ValueError:
            train, _ = train_test_split(outer_train, test_size=CAL_FRAC,
                                        random_state=fold)
        yield fold, train, test


def _internal(node, depth=0):
    if node["leaf"]:
        return
    yield node, depth
    yield from _internal(node["L"], depth + 1)
    yield from _internal(node["R"], depth + 1)


def path_sums(node, sigma, acc=0.0, depth=0):
    """(sum_o sigma_f/h_o, depth) for every root-to-leaf path."""
    if node["leaf"]:
        return [(acc, depth)]
    term = sigma[node["f"]] / node["h"]
    return (path_sums(node["L"], sigma, acc + term, depth + 1)
            + path_sums(node["R"], sigma, acc + term, depth + 1))


def local_constant(node, X, sigma, eps):
    """Vectorised local Lipschitz bound over an l_inf ball of radius eps (std
    units) around each row of X. Returns (L, ok): L bounds the l1 change of the
    soft vote per unit of eps; ok is False where the ball leaves the support
    gate of a node it can reach."""
    n = len(X)
    if node["leaf"]:
        return np.zeros(n), np.ones(n, bool)
    f, c, h = node["f"], node["center"], node["h"]
    lo, hi = node["lo"], node["hi"]
    xf, rad = X[:, f], eps * sigma[f]
    in_band = (xf + rad > c - h) & (xf - rad < c + h)
    reach_l = xf - rad < c + h
    reach_r = xf + rad > c - h
    ok = (xf - rad >= lo) & (xf + rad <= hi)
    L_l, ok_l = local_constant(node["L"], X, sigma, eps)
    L_r, ok_r = local_constant(node["R"], X, sigma, eps)
    L = in_band * (sigma[f] / h) + np.maximum(np.where(reach_l, L_l, 0.0),
                                              np.where(reach_r, L_r, 0.0))
    ok &= np.where(reach_l, ok_l, True) & np.where(reach_r, ok_r, True)
    return L, ok


def local_radius(root, X, sigma, margin, iters=30):
    """Largest eps in [0, EPS_MAX] with ok and eps * L_x(eps) < margin."""
    lo = np.zeros(len(X))
    hi = np.full(len(X), EPS_MAX)
    L, ok = local_constant(root, X, sigma, hi)
    full = ok & (hi * L < margin)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        L, ok = local_constant(root, X, sigma, mid)
        good = ok & (mid * L < margin)
        lo = np.where(good, mid, lo)
        hi = np.where(good, hi, mid)
    return np.where(full, EPS_MAX, lo)


def attack_radius(model, X, sigma, rng):
    """Upper bound on the robust radius from random sign-direction bisection."""
    base = model.predict_proba(X).argmax(1)
    n, d = X.shape
    dirs = rng.choice([-1.0, 1.0], size=(N_DIRECTIONS, d))
    best = np.full(n, np.inf)
    for u in dirs:
        step = u * sigma
        flipped_far = model.predict_proba(X + EPS_MAX * step).argmax(1) != base
        lo, hi = np.zeros(n), np.full(n, EPS_MAX)
        for _ in range(15):
            mid = 0.5 * (lo + hi)
            flip = model.predict_proba(X + mid[:, None] * step).argmax(1) != base
            hi = np.where(flip, mid, hi)
            lo = np.where(flip, lo, mid)
        best = np.minimum(best, np.where(flipped_far, hi, np.inf))
    return best


def routing_overlap(root, model, X):
    """sum_o n_o / N per depth level on the training data."""
    counts = {}

    def walk(node, m, depth):
        counts[depth] = counts.get(depth, 0) + int((m > 1e-6).sum())
        if not node["leaf"]:
            left, right = model._split(node, X)
            walk(node["L"], m * left, depth + 1)
            walk(node["R"], m * right, depth + 1)

    walk(root, np.ones(len(X)), 0)
    levels = np.array([counts[k] for k in sorted(counts)], float) / len(X)
    return float(levels.mean()), float(levels.max())


def dempster_normaliser(M, cons):
    q_theta = np.prod(1.0 - M, axis=1)
    q_c = np.prod(1.0 - M[:, :, None] + M[:, :, None] * cons[None], axis=1)
    return (q_c - q_theta[:, None]).sum(1) + q_theta


def _q(v, q):
    v = v[np.isfinite(v)]
    return float(np.quantile(v, q)) if v.size else float("nan")


def evaluate_fold(X, y, train, test, rng):
    model = LearnedFuzzyTree(random_state=0).fit(X[train], y[train])
    root = model.root_
    sigma = X[train].std(0)
    sigma = np.where(sigma > 0, sigma, 1.0)
    Xte = X[test]

    paths = path_sums(root, sigma)
    LT = max(p for p, _ in paths) if paths else 0.0
    depth = max(d for _, d in paths) if paths else 0
    lam = max((sigma[n["f"]] / (2 * n["h"]) for n, _ in _internal(root)), default=0.0)

    P = model.predict_proba(Xte)
    top2 = np.sort(P, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    r_global = margin / LT if LT > 0 else np.full(len(test), np.inf)
    r_local = local_radius(root, Xte, sigma, margin)
    pick = rng.choice(len(test), min(N_ATTACK_POINTS, len(test)), replace=False)
    r_attack = attack_radius(model, Xte[pick], sigma, rng)

    M, cons, names, _ = model.node_activation_matrix(Xte)
    leaves = np.flatnonzero(model.leaf_mask(names))
    Ml, cl = M[:, leaves], cons[leaves]
    Z = dempster_normaliser(Ml, cl)
    order = -np.sort(-Ml, axis=1)
    cum = np.cumsum(order, axis=1) / np.clip(order.sum(1, keepdims=True), 1e-12, None)
    leaf_depth = np.array([names[i].count("_") for i in leaves])
    overlap_mean, overlap_max = routing_overlap(root, model, X[train])

    return {
        "n_leaves": int(len(leaves)),
        "depth": int(depth),
        "LT_std": float(LT),
        "lam_std": float(lam),
        "K": int((Ml > 0).sum(1).max()),
        "tau_q05": _q(Z, 0.05),
        "tau_min": float(Z.min()),
        "margin_median": float(np.median(margin)),
        "r_global_median": _q(r_global, 0.5),
        "r_local_median": _q(r_local, 0.5),
        "r_local_q25": _q(r_local, 0.25),
        "r_local_ge_0.05": float((r_local >= 0.05).mean()),
        "r_local_ge_0.1": float((r_local >= 0.1).mean()),
        "r_local_attack_pairs": float(np.median(r_local[pick])),
        "r_attack_median": _q(r_attack, 0.5),
        "attack_found": float(np.isfinite(r_attack).mean()),
        "certificate_valid": float(np.all(r_local[pick] <= r_attack + 1e-9)),
        "overlap_mean": overlap_mean,
        "overlap_max": overlap_max,
        "leaves_ge05": float((Ml >= 0.05).sum(1).mean()),
        "leaves_mass90": float(((cum < 0.9).sum(1) + 1).mean()),
        "dominant_depth": float(leaf_depth[Ml.argmax(1)].mean()),
        "accuracy": float((P.argmax(1) == y[test]).mean()),
    }


def run(datasets, output=DEFAULT_OUTPUT):
    rows = []
    for dataset in datasets:
        try:
            X, y = load_filtered(dataset)
        except Exception as exc:
            print(f"{dataset}: SKIP load ({exc})", flush=True)
            continue
        X = np.asarray(X, float)
        for fold, train, test in protocol_folds(X, y):
            rng = np.random.default_rng(fold)
            rows.append({"dataset": dataset, "fold": fold,
                         **evaluate_fold(X, y, train, test, rng)})
        print(f"{dataset}: done", flush=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output, index=False)
    result = pd.DataFrame(rows)
    result.to_csv(output, index=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=SELECTED_30)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    frame = run(args.datasets, args.output)
    per_ds = frame.groupby("dataset").mean(numeric_only=True)
    print(per_ds.median().round(4).to_string())

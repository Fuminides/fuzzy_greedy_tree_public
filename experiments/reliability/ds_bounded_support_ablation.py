"""Inference-only ablation of FERL-deep's bounded-support gates.

Same protocol as the main tabular benchmark (five seeded stratified outer folds,
25% of each outer-training fold held out and unused). One tree is fitted per
fold; the gate is switched off, or its margin changed, at inference only.

* In distribution: soft-vote accuracy and the native leaves-only Dempster set
  (coverage, mean size) with the gate off and on (default margin).
* Geometric OOD, two constructions: ``all`` moves every feature of a test input
  outside the training range, ``one`` moves a single randomly chosen feature;
  a moved value is max + u*range or min - u*range with u ~ U(0.5, 2) and a random
  side. AUROC of the routing-mass loss (1 - total leaf firing) and of the
  ignorance for separating these inputs from the unmodified test inputs, and the
  mean total firing on both, for the gate off and margins of 0.5, 1 (default)
  and 2 node ranges.

Writes results/bounded_support_id.csv and results/bounded_support_ood.csv,
read by experiments/paper_assets/make_bounded_support_ablation.py.
Run from the repository root:
    python experiments/reliability/ds_bounded_support_ablation.py
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ferl.core.learned_tree import LearnedFuzzyTree
from ferl.pipeline.run_configs import SELECTED_30, load_filtered

sys.path.insert(0, "experiments/reliability")
from ds_stability_constants import protocol_folds  # noqa: E402

MARGINS = {"unbounded": None, "0.5": 0.5, "1.0": 1.0, "2.0": 2.0}


def readout(model, X):
    M, _, names, _ = model.node_activation_matrix(X)
    firing = M[:, np.flatnonzero(model.leaf_mask(names))].sum(1)
    _, bel, pl, ign = model.predict_ds(X, rule="dempster", leaves_only=True)
    S = pl >= bel.max(1, keepdims=True) - 1e-12
    return model.predict_proba(X), S, firing, ign


def shift_off_support(Xte, Xtr, rng, mode):
    """Move one feature ('one') or every feature ('all') of each test input
    outside the training range."""
    lo, hi = Xtr.min(0), Xtr.max(0)
    span = hi - lo
    usable = np.flatnonzero(span > 0)
    Xood = Xte.copy()
    n = len(Xte)
    if mode == "one":
        mask = np.zeros(Xte.shape, bool)
        mask[np.arange(n), rng.choice(usable, size=n)] = True
    else:
        mask = np.zeros(Xte.shape, bool)
        mask[:, usable] = True
    u = rng.uniform(0.5, 2.0, size=Xte.shape)
    up = rng.random(Xte.shape) < 0.5
    moved = np.where(up, hi + u * span, lo - u * span)
    return np.where(mask, moved, Xood)


def run(datasets, id_path="results/bounded_support_id.csv",
        ood_path="results/bounded_support_ood.csv"):
    id_rows, ood_rows = [], []
    for ds in datasets:
        try:
            X, y = load_filtered(ds)
        except Exception as exc:
            print(f"{ds}: SKIP load ({exc})", flush=True)
            continue
        X = np.asarray(X, float)
        for fold, train, test in protocol_folds(X, y):
            model = LearnedFuzzyTree(random_state=0).fit(X[train], y[train])
            Xte, yte = X[test], y[test]
            rng = np.random.default_rng(fold)
            shifted = {mode: shift_off_support(Xte, X[train], rng, mode) for mode in ("all", "one")}
            row = {"ds": ds, "fold": fold}
            for margin_label, margin in MARGINS.items():
                model.bounded_support = margin is not None
                model.oob_margin = 1.0 if margin is None else margin
                P, S, phi_id, ign_id = readout(model, Xte)
                for mode, Xood in shifted.items():
                    _, _, phi_ood, ign_ood = readout(model, Xood)
                    label = np.r_[np.zeros(len(Xte)), np.ones(len(Xood))]
                    ood_rows.append({
                        "ds": ds, "seed": fold, "shift": mode, "margin": margin_label,
                        "auroc_firing": roc_auc_score(label, np.r_[1 - phi_id, 1 - phi_ood]),
                        "auroc_ignorance": roc_auc_score(label, np.r_[ign_id, ign_ood]),
                        "phi_id": float(phi_id.mean()), "phi_ood": float(phi_ood.mean())})
                if margin_label in ("unbounded", "1.0"):
                    key = "unb" if margin is None else "bnd"
                    row[f"acc_{key}"] = float((P.argmax(1) == yte).mean())
                    row[f"cov_{key}"] = float(S[np.arange(len(yte)), yte].mean())
                    row[f"size_{key}"] = float(S.sum(1).mean())
            id_rows.append(row)
        print(f"{ds}: done", flush=True)
        pd.DataFrame(id_rows).to_csv(id_path, index=False)
        pd.DataFrame(ood_rows).to_csv(ood_path, index=False)
    return pd.DataFrame(id_rows), pd.DataFrame(ood_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=SELECTED_30)
    args = parser.parse_args()
    ids, oods = run(args.datasets)
    print(ids.mean(numeric_only=True).round(4).to_string())
    print(oods.groupby(["shift", "margin"]).mean(numeric_only=True).round(4).to_string())

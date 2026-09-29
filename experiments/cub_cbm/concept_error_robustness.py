"""How concept-detector errors propagate through a FERL concept-bottleneck head.

For each detector seed, concepts are calibrated per concept with isotonic
regression on the validation split against the annotated concepts (as in E7).
FERL-deep (``LearnedFuzzyTree``), CART and logistic regression are fitted on the
calibrated training concepts, then evaluated on the test split under:

* ``oracle_pipeline`` -- heads refitted and tested on annotated binary concepts
  (a perfect detector; upper reference);
* ``oracle_test`` -- the calibrated-trained heads tested on annotated concepts;
* ``calibrated`` -- calibrated detector outputs (the paper's setting);
* ``raw`` -- uncalibrated detector outputs at test time only;
* ``flip_p`` -- calibrated values with a random fraction p of concept values
  flipped (v -> 1 - v);
* ``ambiguous_p`` -- a random fraction p of concept values set to 0.5.

For FERL it also reports the native leaves-only Dempster read-out (ignorance,
set size, singleton rate, coverage, singleton risk), route agreement (share of
test images whose most-firing leaf equals the one under clean calibrated
concepts), and the residual novelty false-alarm rate: the share of test images
whose residual score exceeds the 95th percentile of clean validation scores.

Run from the repository root (local artifacts, CPU only):
    python experiments/cub_cbm/concept_error_robustness.py
    python experiments/cub_cbm/concept_error_robustness.py --subset 20 --depth 12 --detector-seeds 0 1 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from ferl.core.learned_tree import LearnedFuzzyTree

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/cub_cbm")
sys.path.insert(0, "experiments/reliability")
from run_cub_cbm_extras import load_pair, predict_set_learned  # noqa: E402
from ds_ood_residual_variants import fit_residual_learned, residual_score  # noqa: E402

FLIPS = [0.05, 0.1, 0.2]
AMBIGUOUS = [0.1, 0.2, 0.3]


def calibrate(oracle_val, pred_val, *arrays):
    out = [a.copy() for a in arrays]
    for j in range(pred_val.shape[1]):
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(pred_val[:, j], (oracle_val[:, j] >= 0.5).astype(float))
        for o, a in zip(out, arrays):
            o[:, j] = iso.predict(a[:, j])
    return out


def fit_heads(Xtr, ytr, seed, depth):
    return {"FERL-deep": LearnedFuzzyTree(max_depth=depth, random_state=seed).fit(Xtr, ytr),
            "CART": DecisionTreeClassifier(min_samples_split=5, min_samples_leaf=2,
                                           random_state=seed).fit(Xtr, ytr),
            "LR": LogisticRegression(max_iter=2000, random_state=seed).fit(Xtr, ytr)}


def dominant_leaf(model, X):
    M, _, names, _ = model.node_activation_matrix(X)
    leaves = np.flatnonzero(model.leaf_mask(names))
    return leaves[M[:, leaves].argmax(1)]


def ferl_readout(model, X, y, stats, threshold, clean_route):
    sets, ign = predict_set_learned(model, X)
    sizes = sets.sum(1)
    hit = sets[np.arange(len(y)), y]
    single = sizes == 1
    return {"ignorance": float(ign.mean()), "set_size": float(sizes.mean()),
            "singleton_rate": float(single.mean()), "set_coverage": float(hit.mean()),
            "singleton_risk": float(1 - hit[single].mean()) if single.any() else np.nan,
            "route_agreement": float((dominant_leaf(model, X) == clean_route).mean()),
            "residual_false_alarm": float((residual_score(model, X, stats) > threshold).mean())}


def run(artifact_dir, subset, seeds, depth, output):
    rows = []
    for det in seeds:
        (otr, ova, ote), (ptr, pva, pte) = load_pair(artifact_dir, subset, det)
        Xtr, Xva, Xte = calibrate(ova.C, pva.C, ptr.C, pva.C, pte.C)
        heads = fit_heads(Xtr, ptr.y, det, depth)
        ferl = heads["FERL-deep"]
        stats, _ = fit_residual_learned(ferl, Xtr, Xtr.shape[1])
        threshold = np.quantile(residual_score(ferl, Xva, stats), 0.95)
        clean_route = dominant_leaf(ferl, Xte)
        rng = np.random.default_rng(det)
        conditions = {"oracle_test": ote.C.astype(float), "calibrated": Xte, "raw": pte.C}
        for p in FLIPS:
            mask = rng.random(Xte.shape) < p
            conditions[f"flip_{p}"] = np.where(mask, 1.0 - Xte, Xte)
        for p in AMBIGUOUS:
            mask = rng.random(Xte.shape) < p
            conditions[f"ambiguous_{p}"] = np.where(mask, 0.5, Xte)
        for cond, X in conditions.items():
            row = {"subset": subset, "detector_seed": det, "condition": cond}
            for name, model in heads.items():
                row[f"acc_{name}"] = float((model.predict(X) == pte.y).mean())
            row.update(ferl_readout(ferl, X, pte.y, stats, threshold, clean_route))
            rows.append(row)
            print(f"[{subset} det={det}] {cond}: FERL={row['acc_FERL-deep']:.3f} "
                  f"LR={row['acc_LR']:.3f} ign={row['ignorance']:.3f}", flush=True)
        oracle_heads = fit_heads(otr.C.astype(float), otr.y, det, depth)
        row = {"subset": subset, "detector_seed": det, "condition": "oracle_pipeline"}
        for name, model in oracle_heads.items():
            row[f"acc_{name}"] = float((model.predict(ote.C.astype(float)) == ote.y).mean())
        rows.append(row)
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output, index=False)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="full")
    parser.add_argument("--depth", type=int, default=60)
    parser.add_argument("--detector-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    art = Path(f"results/cub_cbm_artifacts/cub_koh112_{args.subset}")
    out = args.output or Path(f"results/cub_cbm_perf/concept_error_robustness_{args.subset}.csv")
    frame = run(art, args.subset, args.detector_seeds, args.depth, out)
    print(frame.groupby("condition").mean(numeric_only=True).round(4).to_string())

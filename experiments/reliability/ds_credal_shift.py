"""Prediction-set coverage of FERL-deep under graded covariate shift.

For each dataset and seed, a stratified 60/20/20 train/calibration/test split
is drawn. FERL-deep is fitted on the training part. Two split-conformal
predictors are calibrated in distribution at alpha = 0.1:

* ``ds``: score 1 - Pl(y | x), the true-class plausibility of the leaves-only
  Dempster read-out, so sets widen when plausibility spreads across classes;
* ``gl``: global split conformal on 1 - p_y with p the normalised soft vote.

Test inputs are then perturbed feature by feature as x + k * sigma * eps, with
sigma the training standard deviation and eps ~ N(0, 1), for
k in {0, 0.25, 0.5, 1, 1.5, 2}. Writes results/credal_shift.csv
(ds, seed, k, acc, ds_cov, ds_size, gl_cov, gl_size), read by
make_results_assets.shift_assets. Run from the repository root:
    python experiments/reliability/ds_credal_shift.py
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ferl.core.learned_tree import LearnedFuzzyTree
from ferl.pipeline.run_configs import load_filtered

DATASETS = ["magic", "optdigits", "penbased", "phoneme", "ring", "satimage", "texture"]
SEEDS = [0, 1, 2]
SHIFTS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
ALPHA = 0.1


def conformal_threshold(scores, alpha=ALPHA):
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def plausibility(model, X):
    return model.predict_ds(X, rule="dempster", leaves_only=True)[2]


def run(datasets, seeds):
    rows = []
    for ds in datasets:
        X, y = load_filtered(ds)
        X = np.asarray(X, float)
        for seed in seeds:
            Xtr, Xrest, ytr, yrest = train_test_split(X, y, test_size=0.4, random_state=seed,
                                                      stratify=y)
            Xcal, Xte, ycal, yte = train_test_split(Xrest, yrest, test_size=0.5,
                                                    random_state=seed, stratify=yrest)
            model = LearnedFuzzyTree(random_state=0).fit(Xtr, ytr)
            idx = np.arange(len(ycal))
            q_ds = conformal_threshold(1 - plausibility(model, Xcal)[idx, ycal])
            q_gl = conformal_threshold(1 - model.predict_proba(Xcal)[idx, ycal])
            sigma = Xtr.std(0)
            rng = np.random.default_rng(seed)
            eps = rng.standard_normal(Xte.shape)
            for k in SHIFTS:
                Xs = Xte + k * sigma * eps
                P, Pl = model.predict_proba(Xs), plausibility(model, Xs)
                S_ds, S_gl = (1 - Pl) <= q_ds, (1 - P) <= q_gl
                t = np.arange(len(yte))
                rows.append({"ds": ds, "seed": seed, "k": k,
                             "acc": float((P.argmax(1) == yte).mean()),
                             "ds_cov": float(S_ds[t, yte].mean()),
                             "ds_size": float(S_ds.sum(1).mean()),
                             "gl_cov": float(S_gl[t, yte].mean()),
                             "gl_size": float(S_gl.sum(1).mean())})
            print(f"{ds} seed {seed}: done", flush=True)
        pd.DataFrame(rows).to_csv("results/credal_shift.csv", index=False)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()
    frame = run(args.datasets, args.seeds)
    print(frame.groupby("k").mean(numeric_only=True).round(3).to_string())

"""Band-width ablation for FERL-deep (``LearnedFuzzyTree``).

Same protocol as the main tabular benchmark. Only the rule that sets each
split's half-width h changes:

* ``bootstrap`` -- the default: h is the standard deviation of the optimal cut
  over 25 weighted bootstrap resamples, centred at their mean;
* ``fixed_c`` -- h = c times the node-weighted standard deviation of the split
  feature, centred at the optimal cut (c in 0.1, 0.25, 0.5);
* ``fixed_0`` -- c = 0: h collapses to the width floor 1e-3 x node range, i.e.
  an almost crisp tree grown by the same code.

Reports the soft-vote accuracy/AURC/ECE and the leaves-only Dempster read-out.

Run from the repository root:
    python experiments/reliability/ds_width_ablation.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ferl.core.learned_tree import LearnedFuzzyTree
from ferl.pipeline.run_configs import SELECTED_30, load_filtered

sys.path.insert(0, "experiments/reliability")
from ds_decision_rule_ablation import _aurc, _ece, _set_metrics  # noqa: E402
from ds_stability_constants import protocol_folds  # noqa: E402

WIDTHS = {"bootstrap": "bootstrap", "fixed_0.5": 0.5, "fixed_0.25": 0.25,
          "fixed_0.1": 0.1, "fixed_0": 0.0}
DEFAULT_OUTPUT = Path("results/width_ablation.csv")


def evaluate(model, X, y):
    proba = model.predict_proba(X)
    _, bel, pl, ign = model.predict_ds(X, rule="dempster", leaves_only=True)
    S = pl >= bel.max(1, keepdims=True) - 1e-12
    return {"accuracy": float((proba.argmax(1) == y).mean()),
            "aurc": _aurc(proba, y), "ece": _ece(proba, y),
            "ignorance": float(ign.mean()), "n_leaves": model.n_leaves_,
            **_set_metrics(S, y)}


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
            for label, width in WIDTHS.items():
                model = LearnedFuzzyTree(width=width, random_state=0).fit(X[train], y[train])
                rows.append({"dataset": dataset, "fold": fold, "width": label,
                             **evaluate(model, X[test], y[test])})
        print(f"{dataset}: done", flush=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output, index=False)
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=SELECTED_30)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    frame = run(args.datasets, args.output)
    per_ds = frame.groupby(["dataset", "width"]).mean(numeric_only=True)
    print(per_ds.groupby("width").mean().loc[list(WIDTHS)].round(4).to_string())

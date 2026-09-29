"""Same-tree ablation of FERL's evidence sources and combination rule.

The comparison matches the main tabular protocol: five seeded stratified outer
folds, with 25% of each outer-training fold reserved as calibration data.  The
calibration split is intentionally unused here because the native read-outs are
calibration-free.  Each FERL-deep tree is fitted once per fold and evaluated
under three read-outs:

* ``dempster_all`` -- Dempster's rule over all non-root nodes;
* ``dempster_leaves`` -- Dempster's rule over leaves only (used by FERL-deep);
* ``cautious_leaves`` -- Denoeux's idempotent rule over the same leaves;
* ``average_leaves`` -- the firing-weighted average of the leaf masses
  (m(c) = sum_l phi_l p_l(c), m(Theta) = 1 - sum_l phi_l). Its pignistic
  transform is the normalised soft vote that ``LearnedFuzzyTree.predict_proba``
  returns, i.e. the point read-out reported in the main accuracy table;
* ``mixture_leaves`` -- the same average with each leaf discounted by its
  training support, rho = n_l / (n_l + 10) (``_ds_combine(rule='mixture')``).

Set metrics also report the full-abstention rate (S(x) = Theta) and the error
rate on accepted singletons.

Run from the repository root, for example:

    python experiments/reliability/ds_decision_rule_ablation.py
    python experiments/reliability/ds_decision_rule_ablation.py --datasets wine
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
EPS = 1e-12
READOUTS = {
    "dempster_all": {"rule": "dempster", "leaves_only": False},
    "dempster_leaves": {"rule": "dempster", "leaves_only": True},
    "cautious_leaves": {"rule": "cautious", "leaves_only": True},
    "average_leaves": {"rule": "average", "leaves_only": True},
    "mixture_leaves": {"rule": "mixture", "leaves_only": True},
}
DEFAULT_OUTPUT = Path("results/decision_rule_ablation.csv")


def _ece(proba: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    confidence = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    result = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            result += selected.mean() * abs(
                correct[selected].mean() - confidence[selected].mean()
            )
    return float(result)


def _aurc(proba: np.ndarray, y: np.ndarray, n_levels: int = 20) -> float:
    order = np.argsort(-proba.max(axis=1))
    correct = (proba.argmax(axis=1) == y).astype(float)[order]
    n_samples = len(correct)
    risks = [
        1.0 - correct[: max(int(round(coverage * n_samples)), 1)].mean()
        for coverage in np.linspace(1.0 / n_levels, 1.0, n_levels)
    ]
    return float(np.mean(risks))


def _set_metrics(
    prediction_set: np.ndarray, y: np.ndarray
) -> dict[str, float]:
    sizes = prediction_set.sum(axis=1).astype(float)
    hit = prediction_set[np.arange(len(y)), y].astype(float)
    rewarded = hit > 0
    u65 = np.zeros(len(y), dtype=float)
    u80 = np.zeros(len(y), dtype=float)
    u65[rewarded] = (
        1.6 / sizes[rewarded] - 0.6 / sizes[rewarded] ** 2
    )
    u80[rewarded] = (
        2.2 / sizes[rewarded] - 1.2 / sizes[rewarded] ** 2
    )
    singleton = sizes == 1
    return {
        "full_abstention": float((sizes == prediction_set.shape[1]).mean()),
        "singleton_risk": (
            float(1.0 - hit[singleton].mean()) if singleton.any() else float("nan")
        ),
        "determinacy": float((sizes == 1).mean()),
        "set_coverage": float(hit.mean()),
        "set_size": float(sizes.mean()),
        "u65": float(u65.mean()),
        "u80": float(u80.mean()),
    }


def _evaluate(
    model: LearnedFuzzyTree,
    X: np.ndarray,
    y: np.ndarray,
    *,
    rule: str,
    leaves_only: bool,
) -> dict[str, float]:
    if rule == "average":
        M, cons, names, _support = model.node_activation_matrix(X)
        keep = np.flatnonzero(model.leaf_mask(names))
        belief = M[:, keep] @ cons[keep]
        ignorance = np.clip(1.0 - belief.sum(axis=1), 0.0, 1.0)
        plausibility = belief + ignorance[:, None]
        betp = belief + ignorance[:, None] / belief.shape[1]
    else:
        betp, belief, plausibility, ignorance = model.predict_ds(
            X, rule=rule, leaves_only=leaves_only
        )
    prediction_set = (
        plausibility >= belief.max(axis=1, keepdims=True) - EPS
    )
    return {
        "accuracy": float((betp.argmax(axis=1) == y).mean()),
        "aurc": _aurc(betp, y),
        "ece": _ece(betp, y),
        "ignorance": float(ignorance.mean()),
        **_set_metrics(prediction_set, y),
    }


def run(
    datasets: list[str],
    n_folds: int = N_FOLDS,
    output: Path = DEFAULT_OUTPUT,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for dataset in datasets:
        try:
            X, y = load_filtered(dataset)
        except Exception as exc:
            print(f"{dataset}: SKIP load ({exc})", flush=True)
            continue

        splitter = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=33
        )
        for fold, (outer_train, test) in enumerate(splitter.split(X, y)):
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

            model = LearnedFuzzyTree(random_state=0).fit(X[train], y[train])
            for readout, config in READOUTS.items():
                rows.append(
                    {
                        "dataset": dataset,
                        "fold": fold,
                        "readout": readout,
                        **config,
                        **_evaluate(model, X[test], y[test], **config),
                    }
                )
            print(f"{dataset}: fold {fold + 1}/{n_folds}", flush=True)

        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output, index=False)

    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("No decision-rule ablation results were produced")
    result.to_csv(output, index=False)
    return result


def _print_summary(result: pd.DataFrame) -> None:
    per_dataset = (
        result.groupby(["dataset", "readout"], as_index=False)
        .mean(numeric_only=True)
    )
    columns = [
        "accuracy",
        "aurc",
        "ece",
        "determinacy",
        "set_coverage",
        "set_size",
        "u65",
        "u80",
        "ignorance",
        "full_abstention",
        "singleton_risk",
    ]
    print("\nDataset-averaged results")
    print(
        per_dataset.groupby("readout")[columns]
        .mean()
        .loc[list(READOUTS)]
        .round(4)
        .to_string()
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=SELECTED_30,
        help="Dataset names (default: the selected 30-paper benchmark)",
    )
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    frame = run(args.datasets, n_folds=args.folds, output=args.output)
    _print_summary(frame)

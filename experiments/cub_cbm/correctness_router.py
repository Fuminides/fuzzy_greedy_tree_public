"""Learned logistic router between FERL and a logistic CBM head.

The router implements the policy described in the paper: a binary logistic
regression predicts whether FERL will classify an input correctly.  FERL is
used when that prediction is positive; otherwise the ordinary multiclass
logistic-regression head is used.  Routing never depends on whether the two
heads agree or disagree.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.cub_cbm.ferl_lr_hybrid import HybridHeadOutputs


@dataclass
class CorrectnessRouter:
    """Binary FERL-correctness predictor with constant-target support."""

    model: object | None = None
    constant: bool = False

    def predicts_ferl_correct(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.model is None:
            return np.full(len(X), self.constant, dtype=bool)
        return np.asarray(self.model.predict(X), dtype=bool)


def fit_correctness_router(
    X: np.ndarray,
    ferl_prediction: np.ndarray,
    y: np.ndarray,
    *,
    random_state: int = 0,
) -> tuple[CorrectnessRouter, dict[str, float | int]]:
    """Fit LR to the binary target ``FERL prediction is correct``.

    ``X`` should be an honest router-training split that was not used to fit
    FERL.  In the CBM experiments this is the retained-class validation split.
    """
    X = np.asarray(X, dtype=float)
    ferl_prediction = np.asarray(ferl_prediction)
    y = np.asarray(y)
    if X.ndim != 2 or len(X) != len(y) or len(ferl_prediction) != len(y):
        raise ValueError("router features, predictions, and labels must align")
    if not np.isfinite(X).all():
        raise ValueError("router features contain non-finite values")

    target = ferl_prediction == y
    positives = int(target.sum())
    if np.unique(target).size < 2:
        router = CorrectnessRouter(model=None, constant=bool(target[0]))
    else:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=random_state),
        )
        model.fit(X, target.astype(int))
        router = CorrectnessRouter(model=model)

    fitted = router.predicts_ferl_correct(X)
    return router, {
        "train_size": int(len(y)),
        "train_ferl_correct_count": positives,
        "train_ferl_correct_rate": float(target.mean()),
        "train_router_accuracy": float(np.mean(fitted == target)),
        "train_ferl_route_rate": float(fitted.mean()),
    }


def correctness_router_metrics(
    outputs: HybridHeadOutputs,
    y: np.ndarray,
    use_ferl: np.ndarray,
) -> dict[str, float | int]:
    """Evaluate routing, head accuracy, and changes relative to the LR head."""
    y = np.asarray(y)
    use_ferl = np.asarray(use_ferl, dtype=bool)
    lr_prediction = np.asarray(outputs.lr_prediction)
    ferl_prediction = np.asarray(outputs.ferl_prediction)
    if len(y) != len(use_ferl) or len(lr_prediction) != len(y):
        raise ValueError("router decisions, head predictions, and labels must align")

    lr_correct = lr_prediction == y
    ferl_correct = ferl_prediction == y
    prediction = np.where(use_ferl, ferl_prediction, lr_prediction)
    benefit = use_ferl & ferl_correct & ~lr_correct
    harm = use_ferl & lr_correct & ~ferl_correct
    deferred = ~use_ferl
    true_positive = use_ferl & ferl_correct

    return {
        "accuracy": float(np.mean(prediction == y)),
        "lr_accuracy": float(lr_correct.mean()),
        "ferl_accuracy": float(ferl_correct.mean()),
        "ferl_route_count": int(use_ferl.sum()),
        "ferl_route_rate": float(use_ferl.mean()),
        "routed_ferl_accuracy": (
            float(ferl_correct[use_ferl].mean()) if use_ferl.any() else np.nan
        ),
        "deferred_lr_accuracy": (
            float(lr_correct[deferred].mean()) if deferred.any() else np.nan
        ),
        "router_accuracy": float(np.mean(use_ferl == ferl_correct)),
        "router_precision": (
            float(true_positive.sum() / use_ferl.sum()) if use_ferl.any() else np.nan
        ),
        "router_recall": (
            float(true_positive.sum() / ferl_correct.sum())
            if ferl_correct.any()
            else np.nan
        ),
        "benefit_count": int(benefit.sum()),
        "harm_count": int(harm.sum()),
        "net_corrections": int(benefit.sum() - harm.sum()),
        "disagreement_rate": float(np.mean(lr_prediction != ferl_prediction)),
    }

import numpy as np
import pytest

from experiments.cub_cbm.correctness_router import (
    correctness_router_metrics,
    fit_correctness_router,
)
from experiments.cub_cbm.ferl_lr_hybrid import HybridHeadOutputs


def _outputs(lr_prediction, ferl_prediction):
    n = len(lr_prediction)
    return HybridHeadOutputs(
        lr_prediction=np.asarray(lr_prediction),
        ferl_prediction=np.asarray(ferl_prediction),
        lr_probability=np.full((n, 2), 0.5),
        ferl_probability=np.full((n, 2), 0.5),
        gate_features=np.zeros((n, 10)),
        routes=np.full(n, "r_0", dtype=object),
    )


def test_router_learns_ferl_correctness_without_lr_predictions():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    ferl_prediction = np.array([1, 1, 0, 0])
    y = np.array([0, 0, 0, 0])
    router, fit = fit_correctness_router(X, ferl_prediction, y, random_state=3)
    decision = router.predicts_ferl_correct(X)
    assert np.array_equal(decision, [False, False, True, True])
    assert fit["train_ferl_correct_rate"] == pytest.approx(0.5)


def test_router_constant_target_and_metrics():
    X = np.arange(6, dtype=float).reshape(-1, 1)
    router, _ = fit_correctness_router(X, np.zeros(6), np.zeros(6))
    assert router.predicts_ferl_correct(X).all()

    outputs = _outputs(
        lr_prediction=[0, 0, 1, 1],
        ferl_prediction=[0, 1, 0, 1],
    )
    metrics = correctness_router_metrics(
        outputs,
        y=np.array([0, 1, 0, 0]),
        use_ferl=np.array([True, True, False, False]),
    )
    assert metrics["ferl_route_rate"] == pytest.approx(0.5)
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["lr_accuracy"] == pytest.approx(0.25)
    assert metrics["benefit_count"] == 1
    assert metrics["harm_count"] == 0

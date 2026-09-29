import numpy as np

from experiments.cub_cbm.ferl_lr_hybrid import (
    HybridHeadOutputs,
    HybridPolicy,
    exception_metrics,
    fit_rule_evidence,
    route_prefixes,
)


def _outputs(lr, ferl, routes):
    n = len(lr)
    probability = np.full((n, 2), 0.5)
    return HybridHeadOutputs(
        lr_prediction=np.asarray(lr),
        ferl_prediction=np.asarray(ferl),
        lr_probability=probability,
        ferl_probability=probability,
        gate_features=np.zeros((n, 10)),
        routes=np.asarray(routes, dtype=object),
    )


def test_route_prefixes_exclude_root_and_preserve_order():
    assert route_prefixes("r_0_1_0") == ("r_0", "r_0_1", "r_0_1_0")
    assert route_prefixes("r_0_1_0", min_depth=2) == ("r_0_1", "r_0_1_0")
    assert route_prefixes("unsupported") == ()


def test_rule_policy_uses_deepest_supported_positive_exception():
    lr = np.array([0, 0, 0, 1, 1, 1])
    ferl = np.array([1, 1, 1, 0, 0, 0])
    y = np.array([1, 1, 1, 1, 1, 1])
    routes = np.array([
        "r_0_1", "r_0_1", "r_0_1", "r_1_0", "r_1_0", "r_1_0",
    ])
    evidence = fit_rule_evidence(routes, lr, ferl, y)
    policy = HybridPolicy(
        kind="rule_local", threshold=0.7, min_support=2,
        rule_evidence=evidence,
    )
    override, matched = policy.overrides(_outputs(lr, ferl, routes))

    assert override.tolist() == [True, True, True, False, False, False]
    assert matched[:3].tolist() == ["r_0_1", "r_0_1", "r_0_1"]
    assert evidence["r_0_1"].benefit == 3
    assert evidence["r_1_0"].harm == 3


def test_exception_metrics_count_benefits_harms_and_neutral_overrides():
    lr = np.array([0, 0, 0, 1])
    ferl = np.array([1, 1, 1, 0])
    y = np.array([1, 0, 2, 1])
    override = np.array([True, True, True, False])

    metrics = exception_metrics(lr, ferl, y, override)

    assert metrics["benefit_count"] == 1
    assert metrics["harm_count"] == 1
    assert metrics["neutral_wrong_count"] == 1
    assert metrics["net_corrections"] == 0
    assert metrics["override_precision"] == 0.5
    assert metrics["accuracy"] == metrics["lr_accuracy"]

import numpy as np

from experiments.cub_cbm.ferl_lr_hybrid import HybridHeadOutputs, HybridPolicy
from experiments.cub_cbm.rule_coverage_cascade import (
    CoverageCandidate,
    cascade_metrics,
    certify_coverage_frontier,
    clopper_pearson_upper,
    ferl_use_mask,
)


def _outputs(lr, ferl):
    lr = np.asarray(lr)
    ferl = np.asarray(ferl)
    n = len(lr)
    probability = np.full((n, 2), 0.5)
    features = np.zeros((n, 10))
    features[:, 0] = 1.0
    return HybridHeadOutputs(
        lr_prediction=lr,
        ferl_prediction=ferl,
        lr_probability=probability,
        ferl_probability=probability,
        gate_features=features,
        routes=np.full(n, "r_0", dtype=object),
    )


def test_agreement_only_maximizes_exactly_safe_rule_coverage():
    outputs = _outputs([0, 1, 0, 1], [0, 0, 0, 1])
    eligible = np.array([True, True, False, True])
    candidate = CoverageCandidate("agreement_only", "agreement_only")

    use, additional = ferl_use_mask(candidate, outputs, eligible)

    assert use.tolist() == [True, False, False, True]
    assert not additional.any()
    metrics = cascade_metrics(candidate, outputs, np.array([0, 1, 0, 1]), eligible)
    assert metrics["ferl_coverage"] == 0.5
    assert metrics["accuracy"] == metrics["lr_accuracy"]
    assert metrics["rule_accuracy"] == 1.0
    assert metrics["deferred_accuracy"] == 1.0


def test_zero_tolerance_selects_structural_agreement_policy():
    n = 1000
    outputs = _outputs(np.zeros(n, int), np.ones(n, int))
    outputs.gate_features[:, 0] = 1.0
    eligible = np.ones(n, dtype=bool)
    agreement = CoverageCandidate("agreement_only", "agreement_only")
    override = CoverageCandidate(
        "all_disagreements", "confidence",
        HybridPolicy(kind="confidence", threshold=0.5, confidence_constant=1.0),
    )
    y = np.ones(n, dtype=int)  # overriding is beneficial and has zero harms

    frontier = certify_coverage_frontier(
        (agreement, override), outputs, y, eligible,
        epsilons=(0.0, 0.01), delta=0.05,
    )

    assert frontier.selected[0.0].name == "agreement_only"
    assert frontier.selected[0.01].name == "all_disagreements"
    extended = next(x for x in frontier.candidates if x.candidate.name == "all_disagreements")
    assert 0.0 < extended.calibration_metrics["harm_rate_upper"] < 0.01


def test_harmful_high_coverage_candidate_is_not_certified():
    n = 1000
    outputs = _outputs(np.zeros(n, int), np.ones(n, int))
    eligible = np.ones(n, dtype=bool)
    agreement = CoverageCandidate("agreement_only", "agreement_only")
    override = CoverageCandidate(
        "harmful", "confidence",
        HybridPolicy(kind="confidence", threshold=0.5, confidence_constant=1.0),
    )
    y = np.zeros(n, dtype=int)

    frontier = certify_coverage_frontier(
        (agreement, override), outputs, y, eligible,
        epsilons=(0.01,), delta=0.05,
    )

    assert frontier.selected[0.01].name == "agreement_only"
    assert clopper_pearson_upper(n, n, 0.025) == 1.0

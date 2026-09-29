import numpy as np
import pandas as pd

from experiments.cub_cbm.adaptive_intervention import (
    adaptive_lr_query_trajectory,
    adaptive_query_trajectory,
    adaptive_reliability_query_trajectory,
    confidence_acceptance,
    confidence_threshold_for_coverage,
    fit_feature_intervention_utility,
    fit_route_reliability,
    fixed_order_lr_query_trajectory,
    fixed_order_query_trajectory,
    lr_expected_information_queries,
    predict_credal,
    suggest_support_guided_query,
    suggest_reliability_purity_query,
    trace_dominant_route,
)
from experiments.cub_cbm.run_cub_cbm_extras import (
    _selective_intervention_metrics,
    _upsert_csv,
)
from ferl.core.learned_tree import LearnedFuzzyTree


def _leaf(distribution, support=10.0):
    return {
        "leaf": True,
        "dist": np.asarray(distribution, dtype=float),
        "support": support,
    }


def _zero_support_tree():
    """x=[0,.003] reaches root.L, then falls outside that node's support."""
    model = LearnedFuzzyTree(max_depth=2, random_state=0, bounded_support=True)
    model.C = 2
    model.classes_ = np.array([0, 1])
    model.root_ = {
        "leaf": False,
        "f": 0,
        "center": 0.5,
        "h": 0.001,
        "lo": 0.0,
        "hi": 1.0,
        "dist": np.array([0.5, 0.5]),
        "support": 30.0,
        "L": {
            "leaf": False,
            "f": 1,
            "center": 0.0,
            "h": 0.001,
            "lo": 0.0,
            "hi": 0.001,
            "dist": np.array([0.5, 0.5]),
            "support": 20.0,
            "L": _leaf([0.2, 0.8]),
            "R": _leaf([0.3, 0.7]),
        },
        "R": _leaf([0.9, 0.1]),
    }
    return model


class _BinaryLR:
    """Small sklearn-compatible linear head for query-policy tests."""

    classes_ = np.array([0, 1])
    coef_ = np.array([[4.0, 0.0]])
    intercept_ = np.array([-2.0])

    def predict_proba(self, X):
        score = np.asarray(X) @ self.coef_[0] + self.intercept_[0]
        positive = 1.0 / (1.0 + np.exp(-score))
        return np.column_stack([1.0 - positive, positive])


def test_support_failure_backtracks_to_nearest_repairing_split():
    model = _zero_support_tree()
    x = np.array([0.0, 0.003])
    trace = trace_dominant_route(model, x)

    assert [step.feature for step in trace.steps] == [0]
    assert trace.failure_feature == 1

    suggestion = suggest_support_guided_query(
        model,
        x,
        queried=np.zeros(2, dtype=bool),
        value_absent=np.zeros(2),
        value_present=np.ones(2),
    )
    assert suggestion.feature == 0
    assert suggestion.reason == "support_backtrack"


def test_adaptive_policy_reroutes_and_stops_after_resolution():
    model = _zero_support_tree()
    predicted = np.array([[0.0, 0.003]])
    verified = np.array([[1.0, 0.0]])

    initial_sets, _, initial_ignorance = predict_credal(model, predicted)
    assert initial_sets.sum() == 2
    assert initial_ignorance[0] == 1.0

    snapshots = adaptive_query_trajectory(
        model,
        predicted,
        verified,
        value_absent=np.zeros(2),
        value_present=np.ones(2),
        max_queries=2,
    )
    final_sets, _, final_ignorance = predict_credal(model, snapshots[2].X)

    assert snapshots[1].query_counts.tolist() == [1]
    assert snapshots[2].query_counts.tolist() == [1]
    assert snapshots[2].reason_counts == {"support_backtrack": 1}
    assert final_sets.sum() == 1
    assert final_ignorance[0] == 0.0


def test_reliability_purity_policy_prefers_high_belief_route_repair():
    model = _zero_support_tree()
    x = np.array([0.0, 0.003])

    suggestion = suggest_reliability_purity_query(
        model,
        x,
        queried=np.zeros(2, dtype=bool),
        value_absent=np.zeros(2),
        value_present=np.ones(2),
        reliability=np.ones(2),
    )

    # Unlike nearest-sibling backtracking (feature 0), lookahead sees that
    # verifying the failed split itself has the better expected singleton
    # belief under a reliable detector score.
    assert suggestion.feature == 1
    assert suggestion.reason == "reliability_purity_support"
    assert suggestion.expected_correctness > 0.5


def test_reliability_policy_selection_does_not_see_verified_answer():
    model = _zero_support_tree()
    predicted = np.array([[0.0, 0.003]])
    trajectories = []
    for verified in (np.full((1, 2), 0.2), np.full((1, 2), 0.8)):
        trajectories.append(adaptive_reliability_query_trajectory(
            model,
            predicted,
            verified,
            value_absent=np.zeros(2),
            value_present=np.ones(2),
            reliability=np.ones(2),
            max_queries=1,
        ))

    changed = [
        np.flatnonzero(trajectory[1].X[0] != predicted[0]).tolist()
        for trajectory in trajectories
    ]
    assert changed == [[1], [1]]


def test_validation_reliability_models_are_finite_and_route_scoped():
    model = _zero_support_tree()
    validation = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0005]])
    labels = model.predict(validation)
    verified = validation.copy()
    verified[:, 0] = 1.0 - verified[:, 0]

    route = fit_route_reliability(model, validation, labels)
    utility = fit_feature_intervention_utility(
        model, validation, labels, verified,
    )

    assert 0.0 < route.global_accuracy < 1.0
    assert route.leaf_accuracy
    assert utility.net_correction.shape == (2,)
    assert utility.post_accuracy.shape == (2,)
    assert utility.support.shape == (2,)
    assert np.isfinite(utility.net_correction).all()
    assert utility.support[0] == len(validation)


def test_selective_metrics_report_harm_as_well_as_recovery():
    metrics = _selective_intervention_metrics(
        prediction=np.array([0, 0, 1, 0]),
        y=np.array([0, 1, 0, 1]),
        accepted=np.array([True, True, True, False]),
        query_counts=np.array([1, 1, 2, 0]),
        initial_rejected=np.ones(4, dtype=bool),
        initial_prediction=np.array([1, 1, 0, 0]),
    )

    assert metrics["wrong_to_correct_rate"] == 0.25
    assert metrics["correct_to_wrong_rate"] == 0.5
    assert metrics["net_correction_rate"] == -0.25
    assert metrics["wrong_singleton_rate"] == 0.5
    assert metrics["wrong_to_correct_given_initially_wrong"] == 0.5
    assert metrics["correct_to_wrong_given_initially_correct"] == 1.0
    assert metrics["net_corrections_per_query"] == -0.25


def test_result_upsert_normalizes_numeric_subset_labels(tmp_path):
    path = tmp_path / "frontier.csv"
    key = ["subset", "detector_seed", "method", "concept_budget"]
    pd.DataFrame([dict(
        subset=20,
        detector_seed=0,
        method="ferl-deep",
        concept_budget=8,
        accuracy=0.5,
    )]).to_csv(path, index=False)

    _upsert_csv(path, pd.DataFrame([dict(
        subset="20",
        detector_seed=0,
        method="ferl-deep",
        concept_budget=8,
        accuracy=0.7,
    )]), key)

    result = pd.read_csv(path)
    assert len(result) == 1
    assert result.loc[0, "accuracy"] == 0.7


def test_fixed_order_baseline_uses_the_same_early_stopping_rule():
    model = _zero_support_tree()
    predicted = np.array([[0.0, 0.003]])
    verified = np.array([[1.0, 0.0]])
    orders = np.array([[0, 1]])

    snapshots = fixed_order_query_trajectory(
        model, predicted, verified, orders, max_queries=2,
    )

    assert snapshots[1].query_counts.tolist() == [1]
    assert snapshots[2].query_counts.tolist() == [1]


def test_confidence_threshold_matches_requested_validation_coverage():
    confidence = np.array([0.9, 0.8, 0.7, 0.6])
    threshold = confidence_threshold_for_coverage(confidence, 0.5)

    assert 0.7 < threshold < 0.8
    assert (confidence >= threshold).tolist() == [True, True, False, False]


def test_lr_expected_information_gain_selects_the_informative_concept():
    model = _BinaryLR()
    X = np.array([[0.5, 0.5]])

    feature = lr_expected_information_queries(
        model,
        X,
        queried=np.zeros_like(X, dtype=bool),
        value_absent=np.zeros(2),
        value_present=np.ones(2),
    )

    assert feature.tolist() == [0]


def test_adaptive_lr_stops_after_crossing_validation_threshold():
    model = _BinaryLR()
    predicted = np.array([[0.5, 0.5]])
    verified = np.array([[1.0, 0.0]])

    snapshots = adaptive_lr_query_trajectory(
        model,
        predicted,
        verified,
        value_absent=np.zeros(2),
        value_present=np.ones(2),
        confidence_threshold=0.8,
        max_queries=2,
    )
    _, confidence, accepted = confidence_acceptance(
        model, snapshots[2].X, threshold=0.8,
    )

    assert snapshots[1].query_counts.tolist() == [1]
    assert snapshots[2].query_counts.tolist() == [1]
    assert snapshots[2].reason_counts == {"expected_information_gain": 1}
    assert confidence[0] > 0.8
    assert accepted.tolist() == [True]


def test_fixed_lr_order_uses_the_same_early_stopping_rule():
    model = _BinaryLR()
    predicted = np.array([[0.5, 0.5]])
    verified = np.array([[1.0, 0.0]])
    orders = np.array([[0, 1]])

    snapshots = fixed_order_lr_query_trajectory(
        model,
        predicted,
        verified,
        orders,
        confidence_threshold=0.8,
        max_queries=2,
    )

    assert snapshots[1].query_counts.tolist() == [1]
    assert snapshots[2].query_counts.tolist() == [1]

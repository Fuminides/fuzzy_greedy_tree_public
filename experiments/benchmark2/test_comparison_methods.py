"""Contract and smoke tests for comparison-paper benchmark adapters."""
from __future__ import annotations

import os
import sys
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ferl.uncertainty.conformal import ConformalFERL

import artifact_schema
import comparison_methods as cmp
import fuzzy_ucs
import harness
import metrics
import neurules
import validate_results


def _iris_split():
    X, y = load_iris(return_X_y=True)
    return train_test_split(X, y, test_size=0.2, random_state=7, stratify=y)


def _assert_classifier_contract(estimator):
    X_train, X_test, y_train, _ = _iris_split()
    estimator.fit(X_train, y_train)
    proba = estimator.predict_proba(X_test)
    pred = estimator.predict(X_test)

    assert proba.shape == (len(X_test), 3)
    assert pred.shape == (len(X_test),)
    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert set(np.unique(pred)).issubset(set(estimator.classes_))
    assert np.isfinite(estimator.complexity_)


def test_quantile_binarizer_is_train_only_and_binary():
    X_train = np.array([[0.0, 1.0], [1.0, np.nan], [2.0, 3.0], [3.0, 4.0]])
    X_test = np.array([[-100.0, 2.0], [100.0, 5.0]])
    binarizer = cmp.QuantileBinarizer(n_bins=2).fit(X_train)

    transformed = binarizer.transform(X_test)
    assert transformed.shape == (2, binarizer.n_features_out_)
    assert set(np.unique(transformed)).issubset({0.0, 1.0})


def test_sampled_rule_list_classifier_contract():
    _assert_classifier_contract(
        cmp.SampledGreedyRuleList(max_samples=90, n_bins=3, max_depth=3, random_state=3)
    )


def test_sampled_rule_list_is_reproducible():
    X_train, X_test, y_train, _ = _iris_split()
    kwargs = dict(max_samples=60, n_bins=3, max_depth=3, random_state=11)
    first = cmp.SampledGreedyRuleList(**kwargs).fit(X_train, y_train).predict_proba(X_test)
    second = cmp.SampledGreedyRuleList(**kwargs).fit(X_train, y_train).predict_proba(X_test)
    np.testing.assert_allclose(first, second)


def test_neurules_soft_predicate_matches_paper_equation():
    import torch

    layer = neurules._DiscretizingLayer(
        n_features=1,
        n_rules=1,
        limits=torch.tensor([[-2.0, 2.0]]),
        temperature=0.2,
    )
    with torch.no_grad():
        layer.cut_points[:, 0, :] = -0.5
        layer.cut_points[:, 1, :] = 0.75
    values = torch.tensor([[-1.0], [0.0], [1.0]])
    observed = layer(values).detach().numpy()[:, 0, 0]
    expected = 1.0 / (
        1.0
        + np.exp((-0.5 - values.numpy()[:, 0]) / 0.2)
        + np.exp((values.numpy()[:, 0] - 0.75) / 0.2)
    )
    np.testing.assert_allclose(observed, expected, rtol=1e-5, atol=1e-7)


def test_neurules_hard_list_uses_priority_and_first_match():
    X = np.arange(6, dtype=float)[:, None]
    y = np.array([0, 0, 0, 1, 1, 1])
    estimator = neurules.NeuRules(
        n_rules=2, epochs=1, batch_size=6, random_state=4
    ).fit(X, y)
    transformed = estimator._transform(X)

    estimator.cut_points_ = np.empty((1, 2, 2), dtype=float)
    estimator.cut_points_[:, 0, 0] = -10.0
    estimator.cut_points_[:, 1, 0] = 10.0
    estimator.cut_points_[:, 0, 1] = transformed[0, 0] - 0.1
    estimator.cut_points_[:, 1, 1] = transformed[0, 0] + 0.1
    estimator.active_features_ = np.ones((2, 1), dtype=bool)
    estimator.rule_indices_ = np.array([1, 0])
    estimator.rule_logits_ = np.array([[8.0, 0.0], [0.0, 8.0]])

    np.testing.assert_array_equal(estimator.predict(X[[0, -1]]), [1, 0])


def test_neurules_classifier_contract_and_rule_budget():
    estimator = neurules.NeuRules(epochs=6, batch_size=2048, random_state=3)
    _assert_classifier_contract(estimator)
    assert estimator.n_rules_ == 15
    assert len(estimator.rules_) == estimator.complexity_
    assert estimator.condition_complexity_ >= estimator.complexity_


def test_neurules_is_reproducible():
    X_train, X_test, y_train, _ = _iris_split()
    kwargs = dict(epochs=4, batch_size=2048, random_state=11)
    first = neurules.NeuRules(**kwargs).fit(X_train, y_train)
    second = neurules.NeuRules(**kwargs).fit(X_train, y_train)
    np.testing.assert_allclose(first.loss_curve_, second.loss_curve_)
    np.testing.assert_allclose(first.predict_proba(X_test), second.predict_proba(X_test))


def test_fuzzy_ucs_membership_partition_and_dempster_combination():
    terms = np.eye(5, dtype=bool)
    np.testing.assert_allclose(
        [fuzzy_ucs.cnf_membership(term, 0.125) for term in terms],
        [0.5, 0.5, 0.0, 0.0, 0.0],
    )
    combined, conflict = fuzzy_ucs.combine_dempster(
        np.array([0.6, 0.1, 0.3]), np.array([0.2, 0.5, 0.3])
    )
    np.testing.assert_allclose(combined, [0.52941176, 0.33823529, 0.13235294])
    assert conflict == pytest.approx(0.32)


def test_fuzzy_ucs_covering_conditions_match_jucs_boundaries():
    estimator = fuzzy_ucs.FuzzyUCSDS(wildcard_probability=0, random_state=0)
    estimator.parameters_ = estimator._parameters()
    estimator.rng_ = np.random.RandomState(0)
    condition = estimator._new_condition(np.array([0.25, 0.5, 0.75]))
    np.testing.assert_array_equal(condition, np.eye(5, dtype=bool)[[1, 2, 3]])


def test_fuzzy_ucs_classifier_contract():
    _assert_classifier_contract(
        fuzzy_ucs.FuzzyUCSDS(
            epochs=2,
            population_size=100,
            exploitation_threshold=0,
            random_state=3,
        )
    )


def test_fuzzy_ucs_is_reproducible():
    X_train, X_test, y_train, _ = _iris_split()
    kwargs = dict(epochs=2, population_size=100, exploitation_threshold=0, random_state=9)
    first = fuzzy_ucs.FuzzyUCSDS(**kwargs).fit(X_train, y_train).predict_proba(X_test)
    second = fuzzy_ucs.FuzzyUCSDS(**kwargs).fit(X_train, y_train).predict_proba(X_test)
    np.testing.assert_allclose(first, second)


def test_fuzzy_ucs_native_prediction_sets():
    X_train, X_test, y_train, _ = _iris_split()
    estimator = fuzzy_ucs.FuzzyUCSDS(
        epochs=1, population_size=80, exploitation_threshold=0, random_state=5
    ).fit(X_train, y_train)
    prediction_set = estimator.predict_set(X_test)
    assert prediction_set.shape == (len(X_test), 3)
    assert prediction_set.dtype == bool
    assert prediction_set.any(axis=1).all()


def test_evidential_metrics_use_ignorance_as_native_uncertainty():
    mass = np.array([[0.8, 0.1, 0.1], [0.1, 0.2, 0.7]])
    result = metrics.evidential_metrics(mass, np.array([0, 1]), np.array([0, 1]))
    assert result["ignorance"] == pytest.approx(0.4)
    assert result["true_belief"] == pytest.approx(0.5)
    assert result["true_plausibility"] == pytest.approx(0.9)
    assert np.isfinite(result["evidence_aurc"])


def test_aps_inversion_retains_labels_tied_at_the_threshold():
    # Pure CART leaves commonly emit one-hot probabilities.  All candidates
    # then have APS score one: keeping only the argmax would under-cover, while
    # candidate-wise conformal inversion correctly retains every tied label.
    proba_cal = np.tile([1.0, 0.0, 0.0], (12, 1))
    y_cal = np.array([0] * 9 + [1] * 2 + [2])
    proba_test = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    prediction_sets = metrics.aps_sets(
        proba_cal, y_cal, proba_test, np.array([0, 1, 2]), alpha=0.1
    )

    np.testing.assert_array_equal(prediction_sets, np.ones((2, 3), dtype=bool))


def test_aps_candidate_scores_are_permutation_invariant_under_ties():
    probabilities = np.array([[0.45, 0.45, 0.10]])
    permuted = probabilities[:, [1, 0, 2]]

    observed = metrics._aps_candidate_scores(probabilities)
    reordered = metrics._aps_candidate_scores(permuted)[:, [1, 0, 2]]

    np.testing.assert_allclose(observed, [[0.90, 0.90, 1.00]])
    np.testing.assert_allclose(observed, reordered)


def test_conformal_ferl_aps_uses_the_same_tie_conservative_inversion():
    class StaticProbabilityModel:
        classes_ = np.array([0, 1, 2])

        def predict_proba(self, probabilities):
            return np.asarray(probabilities)

        def firing_strength(self, probabilities):
            return np.ones(len(probabilities))

    proba_cal = np.tile([1.0, 0.0, 0.0], (12, 1))
    y_cal = np.array([0] * 9 + [1] * 2 + [2])
    wrapper = ConformalFERL(StaticProbabilityModel(), score="aps").calibrate(
        proba_cal, y_cal, alpha=0.1
    )

    prediction_sets = wrapper.predict_set(
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    )

    np.testing.assert_array_equal(prediction_sets, np.ones((2, 3), dtype=bool))


def test_harness_retries_incomplete_artifacts_and_writes_full_schema(tmp_path, monkeypatch):
    X, y = load_iris(return_X_y=True)
    monkeypatch.setattr(harness, "OUT", str(tmp_path))
    monkeypatch.setattr(harness, "N_FOLDS", 2)
    monkeypatch.setattr(harness, "load_filtered", lambda _: (X, y))
    monkeypatch.setattr(validate_results, "load_filtered", lambda _: (X, y))

    failed_path = harness.artifact_path("iris", "SampledRuleList", 0)
    failed_path = Path(failed_path)
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(failed_path, status="dnf", err="intentional test artifact", C=3)
    legacy_path = Path(harness.artifact_path("iris", "SampledRuleList", 1))
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(legacy_path, status="ok", C=3)

    assert harness.run(["iris"], ["SampledRuleList"]) == 0
    assert validate_results.validate(
        str(tmp_path), ["iris"], ["SampledRuleList"], n_folds=2, check_models=True
    ) == []
    for fold in range(2):
        path = harness.artifact_path("iris", "SampledRuleList", fold)
        with np.load(path, allow_pickle=False) as artifact:
            assert str(artifact["status"]) == "ok"
            assert artifact_schema.REQUIRED_ARRAY_FIELDS.issubset(artifact.files)
            assert int(artifact["schema_version"]) == artifact_schema.SCHEMA_VERSION
            assert not np.intersect1d(artifact["train_idx"], artifact["cal_idx"]).size
            assert not np.intersect1d(artifact["train_idx"], artifact["test_idx"]).size
            assert not np.intersect1d(artifact["cal_idx"], artifact["test_idx"]).size
        with open(artifact_schema.metadata_path(path), encoding="utf-8") as stream:
            manifest = json.load(stream)
        assert manifest["dataset"] == "iris"
        assert manifest["model"] == "SampledRuleList"
        assert manifest["fold"] == fold
        restored = artifact_schema.load_estimator(path)
        assert restored.classes_.shape == (3,)

    corrupt_path = harness.artifact_path("iris", "SampledRuleList", 0)
    Path(artifact_schema.model_path(corrupt_path)).write_bytes(b"not-a-joblib-archive")
    assert not artifact_schema.artifact_complete(
        corrupt_path, "iris", "SampledRuleList", 0
    )


@pytest.mark.skipif(
    not Path(os.environ.get("FERL_RRL_REPO", "external/rrl")).exists(),
    reason="public RRL checkout is not installed",
)
def test_rrl_public_adapter_contract():
    _assert_classifier_contract(cmp.RRLExternal(epochs=1, batch_size=32, hidden_width=8))


@pytest.mark.skipif(
    not Path(os.environ.get("FERL_RLNET_REPO", "external/RLNet")).exists(),
    reason="public RL-Net checkout is not installed",
)
def test_rlnet_public_adapter_contract():
    _assert_classifier_contract(
        cmp.RLNetExternal(epochs=2, batch_size=16, n_rules=4, n_bins=3)
    )


@pytest.mark.skipif(
    not (
        Path(os.environ.get("FERL_SAMRULE_REPO", "external/SamRuLe")) / "src" / "corels"
    ).exists(),
    reason="public SamRuLe checkout is not installed and built",
)
def test_samrule_public_adapter_binary_and_ovr_contracts():
    X, y = load_iris(return_X_y=True)
    binary = y < 2
    binary_estimator = cmp.SamRuLeExternal(theta=0.05, random_state=2)
    binary_estimator.fit(X[binary], y[binary])
    binary_proba = binary_estimator.predict_proba(X[binary][:8])
    assert binary_proba.shape == (8, 2)
    np.testing.assert_allclose(binary_proba.sum(axis=1), 1.0)
    np.testing.assert_array_equal(
        binary_estimator.predict(X[binary][:8]),
        binary_estimator.classes_[binary_estimator.models_[0].hard_predict(binary_estimator._transform(X[binary][:8]))],
    )
    repeated = cmp.SamRuLeExternal(theta=0.05, random_state=2).fit(X[binary], y[binary])
    np.testing.assert_allclose(
        binary_estimator.predict_proba(X[binary][:20]),
        repeated.predict_proba(X[binary][:20]),
    )

    _assert_classifier_contract(
        cmp.SamRuLeExternal(
            theta=0.1, allow_multiclass=True, max_rules=3, random_state=2
        )
    )


def test_samrule_sample_bound_matches_public_implementation():
    assert cmp.SamRuLeExternal._sample_size(32, 5, 1, 1.0, 0.05, 0.05) == 5112

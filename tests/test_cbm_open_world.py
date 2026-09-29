import json

import numpy as np
import pytest

from experiments.cub_cbm.open_world import (
    ConceptOODBaselines,
    id_calibrated_rejection_metrics,
    make_class_folds,
    ood_ranking_metrics,
    read_and_validate_provenance,
    write_provenance,
)
from experiments.cub_cbm.probe_open_world_cbm import run_fold


def test_class_folds_are_deterministic_disjoint_and_complete():
    first = make_class_folds(np.arange(11), n_folds=4, seed=7)
    second = make_class_folds(np.arange(11), n_folds=4, seed=7)
    assert first == second
    flattened = [label for fold in first for label in fold]
    assert sorted(flattened) == list(range(11))
    assert len(flattened) == len(set(flattened))
    assert max(map(len, first)) - min(map(len, first)) <= 1


def test_provenance_rejects_detector_class_leakage(tmp_path):
    prefix = "openworld_fold0_seed0"
    path = write_provenance(
        tmp_path,
        prefix=prefix,
        dataset="cub",
        fold=0,
        detector_seed=0,
        heldout_classes=[3],
        detector_train_classes=[0, 1, 2],
    )
    payload = read_and_validate_provenance(
        tmp_path,
        prefix=prefix,
        observed_classes=[0, 1, 2, 3],
        fold=0,
        detector_seed=0,
    )
    assert payload["heldout_classes"] == [3]

    corrupted = json.loads(path.read_text())
    corrupted["detector_train_classes"].append(3)
    path.write_text(json.dumps(corrupted))
    with pytest.raises(ValueError, match="leaked"):
        read_and_validate_provenance(
            tmp_path,
            prefix=prefix,
            observed_classes=[0, 1, 2, 3],
        )


def test_ood_metrics_and_id_threshold_have_expected_direction():
    validation = np.linspace(0.0, 1.0, 101)
    test_id = np.linspace(0.0, 1.0, 50)
    test_ood = np.linspace(2.0, 3.0, 20)
    ranking = ood_ranking_metrics(test_id, test_ood)
    assert ranking["auroc"] == pytest.approx(1.0)
    assert ranking["aupr_out"] == pytest.approx(1.0)
    assert ranking["fpr95"] == pytest.approx(0.0)

    threshold, rejection, accepted = id_calibrated_rejection_metrics(
        validation,
        test_id,
        test_ood,
        target_id_acceptance=0.95,
    )
    assert threshold >= 0.95
    assert rejection["id_acceptance"] >= 0.94
    assert rejection["ood_rejection"] == pytest.approx(1.0)
    assert accepted.dtype == bool


def test_dedicated_detectors_return_finite_scores():
    rng = np.random.default_rng(2)
    X = np.r_[rng.normal(-1, 0.2, (30, 4)), rng.normal(1, 0.2, (30, 4))]
    y = np.repeat([0, 1], 30)
    detector = ConceptOODBaselines(random_state=2).fit(X, y)
    scores = detector.scores(rng.normal(4, 0.2, (5, 4)))
    assert set(scores) == {"mahalanobis", "knn", "isolation_forest"}
    assert all(values.shape == (5,) for values in scores.values())
    assert all(np.isfinite(values).all() for values in scores.values())


def test_open_world_fold_smoke(tmp_path):
    rng = np.random.default_rng(4)
    n_classes, n_concepts = 4, 8
    prototypes = np.array(
        [
            [1, 1, 0, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 0, 1, 1, 0],
            [0, 1, 1, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 1],
        ],
        dtype=float,
    )
    (tmp_path / "concept_names.json").write_text(
        json.dumps([f"concept_{index}" for index in range(n_concepts)])
    )

    prefix = "openworld_fold0_seed0"
    next_id = 0
    for split, per_class in (("train", 18), ("val", 10), ("test", 12)):
        y = np.repeat(np.arange(n_classes), per_class)
        oracle = prototypes[y]
        logits = 3.0 * (2.0 * oracle - 1.0) + rng.normal(
            0.0, 1.0, (len(y), n_concepts)
        )
        predicted = 1.0 / (1.0 + np.exp(-logits))
        ids = np.arange(next_id, next_id + len(y))
        next_id += len(y)
        np.savez(tmp_path / f"oracle_{split}.npz", C=oracle, y=y, ids=ids)
        np.savez(
            tmp_path / f"{prefix}_{split}.npz",
            C=predicted,
            y=y,
            ids=ids,
        )

    write_provenance(
        tmp_path,
        prefix=prefix,
        dataset="synthetic",
        fold=0,
        detector_seed=0,
        heldout_classes=[3],
        detector_train_classes=[0, 1, 2],
    )
    router, cascade, candidates, ood = run_fold(
        artifact_dir=tmp_path,
        dataset="cub",
        subset="smoke",
        fold=0,
        detector_seed=0,
        learned_depth=4,
        epsilons=(0.0,),
        delta=0.05,
        target_id_acceptance=0.95,
        confidence_thresholds=(0.5,),
        rule_supports=(1,),
        rule_thresholds=(0.6,),
    )
    assert len(router) == 1
    assert router.iloc[0]["router_features"] == "calibrated_concepts"
    assert 0.0 <= router.iloc[0]["test_ferl_route_rate"] <= 1.0
    assert len(cascade) == 1
    assert cascade.iloc[0]["selected_kind"] == "agreement_only"
    assert cascade.iloc[0]["test_accuracy"] == pytest.approx(
        cascade.iloc[0]["test_lr_accuracy"]
    )
    assert not candidates.empty
    assert set(ood["signal"]) == {
        "ferl_residual",
        "ferl_ignorance",
        "ferl_set_size",
        "ferl_one_minus_maxp",
        "lr_one_minus_maxp",
        "lr_entropy",
        "mahalanobis",
        "knn",
        "isolation_forest",
    }

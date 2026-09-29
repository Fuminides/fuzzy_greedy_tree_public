"""E16: open-world calibrated CBM system on class-disjoint detector folds.

For every fold, the held-out species/classes must be absent from the image
detector's training images.  The required provenance sidecar is validated
before any model is fit.  FERL and LR are then trained on the retained classes:

* FERL novelty signals and dedicated concept-space OOD detectors separate ID
  test images from detector-unseen classes;
* OOD rejection thresholds are fixed from ID validation scores only.

Learned and agreement-based FERL/LR routing outputs are retained only as
unpublished diagnostics for experimental provenance.

Expected artifact names for fold ``f`` and detector seed ``s`` are
``openworld_fold<f>_seed<s>_{train,val,test}.npz`` plus the matching
``..._provenance.json``.  CUB and AwA2 exporters in this directory produce
that contract.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from experiments.cub_cbm.correctness_router import (
    correctness_router_metrics,
    fit_correctness_router,
)
from experiments.cub_cbm.ferl_lr_hybrid import HybridHeadOutputs, hybrid_head_outputs
from experiments.cub_cbm.open_world import (
    ConceptOODBaselines,
    entropy,
    fit_learned_residual,
    id_calibrated_rejection_metrics,
    learned_residual_score,
    ood_ranking_metrics,
    read_and_validate_provenance,
)
from experiments.cub_cbm.rule_coverage_cascade import (
    cascade_metrics,
    certify_coverage_frontier,
    ferl_use_mask,
    fit_coverage_candidates,
)
from experiments.cub_cbm.run_cub_cbm import (
    ArtifactBundle,
    SplitData,
    _read_split,
    read_bundle,
)
from experiments.cub_cbm.run_cub_cbm_extras import (
    _apply_calibrators,
    _fit_concept_calibrators,
    _upsert_csv,
    predict_set_learned,
)
from ferl.core.learned_tree import EPS, LearnedFuzzyTree


def read_prefixed_bundle(
    artifact_dir: Path,
    *,
    prefix: str,
    subset: str,
) -> ArtifactBundle:
    concept_names = json.loads(
        (artifact_dir / "concept_names.json").read_text(encoding="utf-8")
    )
    splits = {
        split: _read_split(
            artifact_dir / f"{prefix}_{split}.npz",
            len(concept_names),
        )
        for split in ("train", "val", "test")
    }
    return ArtifactBundle(
        artifact_dir=artifact_dir,
        subset=subset,
        source=prefix,
        detector_seed=None,
        concept_names=list(concept_names),
        train=splits["train"],
        val=splits["val"],
        test=splits["test"],
    )


def _aligned_oracle_and_prediction(
    artifact_dir: Path,
    *,
    prefix: str,
    subset: str,
) -> tuple[ArtifactBundle, ArtifactBundle]:
    oracle = read_bundle(artifact_dir, source="oracle", subset=subset)
    predicted = read_prefixed_bundle(artifact_dir, prefix=prefix, subset=subset)
    for name in ("train", "val", "test"):
        left = getattr(oracle, name)
        right = getattr(predicted, name)
        if not np.array_equal(left.ids, right.ids):
            raise ValueError(f"{prefix}: oracle/predicted {name} ids are not aligned")
        if not np.array_equal(left.y, right.y):
            raise ValueError(f"{prefix}: oracle/predicted {name} labels are not aligned")
    return oracle, predicted


def _rows(split: SplitData, mask: np.ndarray) -> SplitData:
    return SplitData(C=split.C[mask], y=split.y[mask], ids=split.ids[mask])


def _take_outputs(outputs: HybridHeadOutputs, rows: np.ndarray) -> HybridHeadOutputs:
    return HybridHeadOutputs(
        lr_prediction=outputs.lr_prediction[rows],
        ferl_prediction=outputs.ferl_prediction[rows],
        lr_probability=outputs.lr_probability[rows],
        ferl_probability=outputs.ferl_probability[rows],
        gate_features=outputs.gate_features[rows],
        routes=outputs.routes[rows],
    )


def _strict_eligibility(
    ferl: LearnedFuzzyTree,
    X: np.ndarray,
    outputs: HybridHeadOutputs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sets, ignorance = predict_set_learned(ferl, X)
    supported = (outputs.routes != "unsupported") & (outputs.gate_features[:, 7] > EPS)
    singleton = sets.sum(axis=1) == 1
    return supported & singleton, sets, ignorance


def _score_bundle(
    *,
    ferl: LearnedFuzzyTree,
    lr: LogisticRegression,
    residual,
    dedicated: ConceptOODBaselines,
    X: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    ferl_probability = np.asarray(ferl.predict_proba(X), dtype=float)
    lr_probability = np.asarray(lr.predict_proba(X), dtype=float)
    sets, ignorance = predict_set_learned(ferl, X)
    scores = {
        "ferl_residual": learned_residual_score(ferl, X, residual),
        "ferl_ignorance": np.asarray(ignorance, dtype=float),
        "ferl_set_size": sets.sum(axis=1).astype(float),
        "ferl_one_minus_maxp": 1.0 - ferl_probability.max(axis=1),
        "lr_one_minus_maxp": 1.0 - lr_probability.max(axis=1),
        "lr_entropy": entropy(lr_probability),
        **dedicated.scores(X),
    }
    return scores, sets


def run_fold(
    *,
    artifact_dir: Path,
    dataset: str,
    subset: str,
    fold: int,
    detector_seed: int,
    learned_depth: int,
    epsilons: tuple[float, ...],
    delta: float,
    target_id_acceptance: float,
    confidence_thresholds: tuple[float, ...],
    rule_supports: tuple[int, ...],
    rule_thresholds: tuple[float, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prefix = f"openworld_fold{fold}_seed{detector_seed}"
    oracle, predicted = _aligned_oracle_and_prediction(
        artifact_dir, prefix=prefix, subset=subset
    )
    all_labels = np.concatenate(
        [predicted.train.y, predicted.val.y, predicted.test.y]
    )
    provenance = read_and_validate_provenance(
        artifact_dir,
        prefix=prefix,
        observed_classes=all_labels,
        fold=fold,
        detector_seed=detector_seed,
    )
    heldout = np.asarray(provenance["heldout_classes"], dtype=int)

    train_id = ~np.isin(predicted.train.y, heldout)
    validation_id = ~np.isin(predicted.val.y, heldout)
    test_id = ~np.isin(predicted.test.y, heldout)
    test_ood = ~test_id
    if not train_id.any() or not validation_id.any() or not test_id.any() or not test_ood.any():
        raise ValueError(f"{prefix}: every ID/OOD split must be non-empty")

    otr = _rows(oracle.train, train_id)
    ova = _rows(oracle.val, validation_id)
    ptr = _rows(predicted.train, train_id)
    pva = _rows(predicted.val, validation_id)
    pte_id = _rows(predicted.test, test_id)
    pte_ood = _rows(predicted.test, test_ood)

    label_encoder = LabelEncoder().fit(ptr.y)
    y_train = label_encoder.transform(ptr.y)
    y_validation = label_encoder.transform(pva.y)
    y_test_id = label_encoder.transform(pte_id.y)

    validation_rows = np.arange(len(pva.y))
    selector_rows, safety_rows = train_test_split(
        validation_rows,
        test_size=0.5,
        random_state=detector_seed,
    )
    calibrators = _fit_concept_calibrators(
        ova.C[selector_rows],
        pva.C[selector_rows],
    )
    X_train = _apply_calibrators(calibrators, ptr.C)
    X_validation = _apply_calibrators(calibrators, pva.C)
    X_test_id = _apply_calibrators(calibrators, pte_id.C)
    X_test_ood = _apply_calibrators(calibrators, pte_ood.C)

    print(
        f"[cbm:e16] dataset={dataset} fold={fold} detector={detector_seed} "
        f"heldout={heldout.tolist()} ID train/val/test="
        f"{len(X_train)}/{len(X_validation)}/{len(X_test_id)} OOD={len(X_test_ood)}",
        flush=True,
    )
    started = time.perf_counter()
    ferl = LearnedFuzzyTree(
        max_depth=learned_depth,
        random_state=detector_seed,
    ).fit(X_train, y_train)
    lr = LogisticRegression(max_iter=2000, random_state=detector_seed).fit(
        X_train, y_train
    )
    fit_seconds = time.perf_counter() - started

    validation_outputs = hybrid_head_outputs(lr, ferl, X_validation)
    router_started = time.perf_counter()
    router, router_fit = fit_correctness_router(
        X_validation,
        validation_outputs.ferl_prediction,
        y_validation,
        random_state=detector_seed,
    )
    router_fit_seconds = time.perf_counter() - router_started
    validation_eligible, _, _ = _strict_eligibility(
        ferl, X_validation, validation_outputs
    )
    candidates = fit_coverage_candidates(
        _take_outputs(validation_outputs, selector_rows),
        y_validation[selector_rows],
        confidence_thresholds=confidence_thresholds,
        rule_supports=rule_supports,
        rule_thresholds=rule_thresholds,
        random_state=detector_seed,
    )
    frontier = certify_coverage_frontier(
        candidates,
        _take_outputs(validation_outputs, safety_rows),
        y_validation[safety_rows],
        validation_eligible[safety_rows],
        epsilons=epsilons,
        delta=delta,
    )

    candidate_rows = []
    selected_names = {
        epsilon: candidate.name for epsilon, candidate in frontier.selected.items()
    }
    for item in frontier.candidates:
        candidate_rows.append(
            {
                "dataset": dataset,
                "subset": subset,
                "fold": fold,
                "detector_seed": detector_seed,
                "heldout_classes": ",".join(str(value) for value in heldout),
                "candidate": item.candidate.name,
                "kind": item.candidate.kind,
                "selected_epsilons": ",".join(
                    f"{epsilon:g}"
                    for epsilon, name in selected_names.items()
                    if name == item.candidate.name
                ),
                **{
                    f"calibration_{key}": value
                    for key, value in item.calibration_metrics.items()
                },
            }
        )

    test_outputs = hybrid_head_outputs(lr, ferl, X_test_id)
    test_eligible, test_sets, test_ignorance = _strict_eligibility(
        ferl, X_test_id, test_outputs
    )
    router_use_ferl = router.predicts_ferl_correct(X_test_id)
    router_metrics = correctness_router_metrics(
        test_outputs,
        y_test_id,
        router_use_ferl,
    )
    router_prediction = np.where(
        router_use_ferl,
        test_outputs.ferl_prediction,
        test_outputs.lr_prediction,
    )
    router_row = {
        "dataset": dataset,
        "subset": subset,
        "fold": fold,
        "detector_seed": detector_seed,
        "heldout_classes": ",".join(str(value) for value in heldout),
        "router_features": "calibrated_concepts",
        "head_fit_seconds": fit_seconds,
        "router_fit_seconds": router_fit_seconds,
        "test_native_abstention_rate": float((test_sets.sum(axis=1) > 1).mean()),
        "test_mean_ignorance": float(test_ignorance.mean()),
        **{f"router_{key}": value for key, value in router_fit.items()},
        **{f"test_{key}": value for key, value in router_metrics.items()},
    }
    print(
        f"[cbm:e16] learned router: FERL route="
        f"{router_metrics['ferl_route_rate']:.3f} accuracy FERL/router/LR="
        f"{router_metrics['ferl_accuracy']:.3f}/"
        f"{router_metrics['accuracy']:.3f}/"
        f"{router_metrics['lr_accuracy']:.3f}",
        flush=True,
    )

    # Retain both routing variants only as unpublished diagnostics.  The
    # learned router above never observes whether the two heads agree.
    cascade_rows = []
    exact_prediction = None
    exact_use_ferl = None
    for epsilon in epsilons:
        candidate = frontier.selected[float(epsilon)]
        calibration_item = next(
            item for item in frontier.candidates
            if item.candidate.name == candidate.name
        )
        metrics = cascade_metrics(candidate, test_outputs, y_test_id, test_eligible)
        use_ferl, _ = ferl_use_mask(candidate, test_outputs, test_eligible)
        prediction = np.where(
            use_ferl,
            test_outputs.ferl_prediction,
            test_outputs.lr_prediction,
        )
        if float(epsilon) == 0.0:
            exact_prediction = prediction
            exact_use_ferl = use_ferl
        cascade_rows.append(
            {
                "dataset": dataset,
                "subset": subset,
                "fold": fold,
                "detector_seed": detector_seed,
                "heldout_classes": ",".join(str(value) for value in heldout),
                "epsilon": float(epsilon),
                "delta": float(delta),
                "selected_candidate": candidate.name,
                "selected_kind": candidate.kind,
                "selector_size": len(selector_rows),
                "safety_size": len(safety_rows),
                "fit_seconds": fit_seconds,
                "calibration_harm_rate_upper": calibration_item.calibration_metrics[
                    "harm_rate_upper"
                ],
                "test_native_abstention_rate": float((test_sets.sum(axis=1) > 1).mean()),
                "test_mean_ignorance": float(test_ignorance.mean()),
                **{f"test_{key}": value for key, value in metrics.items()},
            }
        )
    if exact_prediction is None or exact_use_ferl is None:
        raise ValueError("epsilons must include 0.0 for the exact system policy")

    residual = fit_learned_residual(ferl, X_train)
    dedicated = ConceptOODBaselines(random_state=detector_seed).fit(X_train, y_train)
    validation_scores, _ = _score_bundle(
        ferl=ferl,
        lr=lr,
        residual=residual,
        dedicated=dedicated,
        X=X_validation,
    )
    test_id_scores, _ = _score_bundle(
        ferl=ferl,
        lr=lr,
        residual=residual,
        dedicated=dedicated,
        X=X_test_id,
    )
    test_ood_scores, test_ood_sets = _score_bundle(
        ferl=ferl,
        lr=lr,
        residual=residual,
        dedicated=dedicated,
        X=X_test_ood,
    )

    ood_rows = []
    for signal in validation_scores:
        ranking = ood_ranking_metrics(
            test_id_scores[signal],
            test_ood_scores[signal],
        )
        threshold, rejection, accepted_id = id_calibrated_rejection_metrics(
            validation_scores[signal],
            test_id_scores[signal],
            test_ood_scores[signal],
            target_id_acceptance=target_id_acceptance,
        )
        ood_rows.append(
            {
                "dataset": dataset,
                "subset": subset,
                "fold": fold,
                "detector_seed": detector_seed,
                "heldout_classes": ",".join(str(value) for value in heldout),
                "signal": signal,
                "target_id_acceptance": target_id_acceptance,
                "validation_threshold": threshold,
                **ranking,
                **rejection,
                "accepted_id_accuracy": float(
                    (router_prediction[accepted_id] == y_test_id[accepted_id]).mean()
                )
                if accepted_id.any()
                else np.nan,
                "accepted_id_rule_coverage": float(
                    router_use_ferl[accepted_id].mean()
                )
                if accepted_id.any()
                else np.nan,
                "native_ood_abstention_rate": float(
                    (test_ood_sets.sum(axis=1) > 1).mean()
                ),
            }
        )
        print(
            f"[cbm:e16] {signal}: AUROC={ranking['auroc']:.3f} "
            f"FPR95={ranking['fpr95']:.3f} "
            f"IDaccept/OODreject={rejection['id_acceptance']:.3f}/"
            f"{rejection['ood_rejection']:.3f}",
            flush=True,
        )
    return (
        pd.DataFrame([router_row]),
        pd.DataFrame(cascade_rows),
        pd.DataFrame(candidate_rows),
        pd.DataFrame(ood_rows),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=("cub", "awa2"), required=True)
    parser.add_argument("--subset", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/cbm_open_world"))
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--folds", nargs="+", type=int)
    parser.add_argument("--detector-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--learned-depth", type=int, default=60)
    parser.add_argument(
        "--epsilons",
        nargs="+",
        type=float,
        default=[0.0, 0.005, 0.01, 0.02],
    )
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--target-id-acceptance", type=float, default=0.95)
    parser.add_argument(
        "--confidence-thresholds",
        nargs="+",
        type=float,
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    parser.add_argument("--rule-supports", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument(
        "--rule-thresholds",
        nargs="+",
        type=float,
        default=[0.6, 0.7, 0.8],
    )
    args = parser.parse_args()

    folds = args.folds if args.folds is not None else list(range(args.n_folds))
    router_parts, cascade_parts, candidate_parts, ood_parts = [], [], [], []
    for fold in folds:
        for detector_seed in args.detector_seeds:
            router, cascade, candidates, ood = run_fold(
                artifact_dir=args.artifact_dir,
                dataset=args.dataset,
                subset=args.subset,
                fold=fold,
                detector_seed=detector_seed,
                learned_depth=args.learned_depth,
                epsilons=tuple(args.epsilons),
                delta=args.delta,
                target_id_acceptance=args.target_id_acceptance,
                confidence_thresholds=tuple(args.confidence_thresholds),
                rule_supports=tuple(args.rule_supports),
                rule_thresholds=tuple(args.rule_thresholds),
            )
            router_parts.append(router)
            cascade_parts.append(cascade)
            candidate_parts.append(candidates)
            ood_parts.append(ood)

    router = pd.concat(router_parts, ignore_index=True)
    cascade = pd.concat(cascade_parts, ignore_index=True)
    candidates = pd.concat(candidate_parts, ignore_index=True)
    ood = pd.concat(ood_parts, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _upsert_csv(
        args.output_dir / "e16_open_world_router.csv",
        router,
        key=["dataset", "subset", "fold", "detector_seed"],
    )
    _upsert_csv(
        args.output_dir / "e16_open_world_cascade.csv",
        cascade,
        key=["dataset", "subset", "fold", "detector_seed", "epsilon"],
    )
    _upsert_csv(
        args.output_dir / "e16_open_world_cascade_candidates.csv",
        candidates,
        key=["dataset", "subset", "fold", "detector_seed", "candidate"],
    )
    _upsert_csv(
        args.output_dir / "e16_open_world_ood.csv",
        ood,
        key=["dataset", "subset", "fold", "detector_seed", "signal"],
    )
    print(f"[cbm:e16] wrote open-world results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

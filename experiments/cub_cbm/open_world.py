"""Shared machinery for leakage-safe open-world CBM experiments.

The paper-facing protocol is stricter than ordinary leave-one-class-out at the
``C -> Y`` head: every held-out class must also be absent from the image
detector's training images.  Detector exporters write one provenance JSON per
fold/seed and the evaluator refuses artifacts that do not make that exclusion
explicit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


PROTOCOL = "detector-class-disjoint-v1"


def make_class_folds(
    classes: np.ndarray | list[int],
    *,
    n_folds: int,
    seed: int,
) -> tuple[tuple[int, ...], ...]:
    """Partition class labels into deterministic, approximately equal folds."""
    labels = np.unique(np.asarray(classes, dtype=int))
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if n_folds > len(labels):
        raise ValueError("n_folds cannot exceed the number of classes")
    shuffled = labels.copy()
    np.random.default_rng(seed).shuffle(shuffled)
    return tuple(
        tuple(sorted(int(label) for label in fold))
        for fold in np.array_split(shuffled, n_folds)
    )


def provenance_path(artifact_dir: Path, prefix: str) -> Path:
    return Path(artifact_dir) / f"{prefix}_provenance.json"


def write_provenance(
    artifact_dir: Path,
    *,
    prefix: str,
    dataset: str,
    fold: int,
    detector_seed: int,
    heldout_classes: tuple[int, ...] | list[int],
    detector_train_classes: tuple[int, ...] | list[int],
    class_names: dict[int, str] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write the auditable detector-exclusion record consumed by E16."""
    heldout = sorted(int(value) for value in heldout_classes)
    trained = sorted(int(value) for value in detector_train_classes)
    if not heldout:
        raise ValueError("heldout_classes cannot be empty")
    if set(heldout) & set(trained):
        raise ValueError("held-out classes appear in detector_train_classes")
    payload: dict[str, Any] = {
        "protocol": PROTOCOL,
        "strict_detector_holdout": True,
        "dataset": str(dataset),
        "prefix": str(prefix),
        "fold": int(fold),
        "detector_seed": int(detector_seed),
        "heldout_classes": heldout,
        "detector_train_classes": trained,
    }
    if class_names:
        payload["heldout_class_names"] = [
            str(class_names[label]) for label in heldout
        ]
    if extra:
        payload.update(extra)
    path = provenance_path(artifact_dir, prefix)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def read_and_validate_provenance(
    artifact_dir: Path,
    *,
    prefix: str,
    observed_classes: np.ndarray | list[int],
    fold: int | None = None,
    detector_seed: int | None = None,
) -> dict[str, Any]:
    """Load provenance and reject artifacts that cannot support a strict claim."""
    path = provenance_path(artifact_dir, prefix)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is required: open-world results must prove that held-out "
            "classes were excluded from detector training"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != PROTOCOL:
        raise ValueError(f"{path}: unsupported protocol {payload.get('protocol')!r}")
    if payload.get("strict_detector_holdout") is not True:
        raise ValueError(f"{path}: strict_detector_holdout must be true")
    if payload.get("prefix") != prefix:
        raise ValueError(f"{path}: prefix does not match {prefix!r}")
    if fold is not None and int(payload.get("fold", -1)) != int(fold):
        raise ValueError(f"{path}: fold does not match {fold}")
    if detector_seed is not None and int(payload.get("detector_seed", -1)) != int(
        detector_seed
    ):
        raise ValueError(f"{path}: detector_seed does not match {detector_seed}")

    heldout = {int(value) for value in payload.get("heldout_classes", [])}
    trained = {int(value) for value in payload.get("detector_train_classes", [])}
    observed = {int(value) for value in np.asarray(observed_classes)}
    if not heldout:
        raise ValueError(f"{path}: heldout_classes cannot be empty")
    if heldout - observed:
        raise ValueError(f"{path}: held-out classes are absent from the artifact labels")
    if heldout & trained:
        raise ValueError(f"{path}: held-out classes leaked into detector training")
    if trained != observed - heldout:
        raise ValueError(
            f"{path}: detector_train_classes must equal all observed non-held-out classes"
        )
    return payload


@dataclass(frozen=True)
class ResidualStatistics:
    """Per-node density summaries over features absent from the active path."""

    by_node: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]


def fit_learned_residual(model: Any, X_train: np.ndarray) -> ResidualStatistics:
    """Fit FERL's path-conditional residual concept-density summaries."""
    X_train = np.asarray(X_train, dtype=float)
    dimension = X_train.shape[1]
    stats: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def walk(node, name: str, path_features: set[int], membership: np.ndarray) -> None:
        if name != "r" and len(path_features) < dimension:
            free = np.array(
                [feature for feature in range(dimension) if feature not in path_features],
                dtype=int,
            )
            weight = float(membership.sum())
            if len(free) and weight > 1e-6:
                mean = (membership[:, None] * X_train[:, free]).sum(axis=0) / weight
                variance = (
                    membership[:, None] * (X_train[:, free] - mean) ** 2
                ).sum(axis=0) / weight
                stats[name] = (free, mean, variance + 1e-6)
        if not node["leaf"]:
            left, right = model._split(node, X_train)
            feature = int(node["f"])
            walk(
                node["L"],
                name + "_0",
                path_features | {feature},
                membership * left,
            )
            walk(
                node["R"],
                name + "_1",
                path_features | {feature},
                membership * right,
            )

    walk(model.root_, "r", set(), np.ones(len(X_train), dtype=float))
    return ResidualStatistics(stats)


def learned_residual_score(
    model: Any,
    X: np.ndarray,
    statistics: ResidualStatistics,
) -> np.ndarray:
    """Return the firing-weighted residual novelty score (higher is stranger)."""
    X = np.asarray(X, dtype=float)
    activation, _, names, _ = model.node_activation_matrix(X)
    numerator = np.zeros(len(X), dtype=float)
    denominator = np.zeros(len(X), dtype=float)
    for column, name in enumerate(names):
        if name not in statistics.by_node:
            continue
        free, mean, variance = statistics.by_node[name]
        distance = (((X[:, free] - mean) ** 2) / variance).mean(axis=1)
        numerator += activation[:, column] * distance
        denominator += activation[:, column]
    return numerator / np.clip(denominator, 1e-9, None)


class ConceptOODBaselines:
    """Dedicated concept-space novelty detectors with a shared fit interface."""

    def __init__(self, *, random_state: int = 0, n_neighbors: int = 20):
        self.random_state = int(random_state)
        self.n_neighbors = int(n_neighbors)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ConceptOODBaselines":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if len(X) < 3:
            raise ValueError("at least three ID training samples are required")
        self.scaler_ = StandardScaler().fit(X)
        transformed = self.scaler_.transform(X)
        covariance = LedoitWolf().fit(transformed)
        self.precision_ = covariance.precision_
        self.class_means_ = np.stack(
            [transformed[y == label].mean(axis=0) for label in np.unique(y)]
        )
        neighbors = min(self.n_neighbors, len(transformed))
        self.neighbors_ = NearestNeighbors(n_neighbors=neighbors).fit(transformed)
        self.isolation_ = IsolationForest(random_state=self.random_state).fit(transformed)
        return self

    def scores(self, X: np.ndarray) -> dict[str, np.ndarray]:
        transformed = self.scaler_.transform(np.asarray(X, dtype=float))
        distances = [
            np.einsum(
                "ij,jk,ik->i",
                transformed - mean,
                self.precision_,
                transformed - mean,
            )
            for mean in self.class_means_
        ]
        return {
            "mahalanobis": np.min(distances, axis=0),
            "knn": self.neighbors_.kneighbors(transformed)[0][:, -1],
            "isolation_forest": -self.isolation_.score_samples(transformed),
        }


def entropy(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    return -(
        probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))
    ).sum(axis=1)


def ood_ranking_metrics(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
) -> dict[str, float]:
    """AUROC/AUPR/FPR95 for scores whose larger values mean more OOD."""
    id_scores = np.asarray(id_scores, dtype=float)
    ood_scores = np.asarray(ood_scores, dtype=float)
    labels = np.r_[np.zeros(len(id_scores), dtype=int), np.ones(len(ood_scores), dtype=int)]
    scores = np.r_[id_scores, ood_scores]
    if len(id_scores) == 0 or len(ood_scores) == 0 or np.ptp(scores) <= 1e-12:
        return {"auroc": np.nan, "aupr_out": np.nan, "fpr95": np.nan}
    false_positive, true_positive, _ = roc_curve(labels, scores)
    eligible = false_positive[true_positive >= 0.95]
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "aupr_out": float(average_precision_score(labels, scores)),
        "fpr95": float(eligible.min()) if len(eligible) else 1.0,
    }


def id_calibrated_rejection_metrics(
    validation_id_scores: np.ndarray,
    test_id_scores: np.ndarray,
    test_ood_scores: np.ndarray,
    *,
    target_id_acceptance: float,
) -> tuple[float, dict[str, float], np.ndarray]:
    """Choose a label-free ID quantile and evaluate rejection on untouched test."""
    if not 0.0 < target_id_acceptance < 1.0:
        raise ValueError("target_id_acceptance must lie strictly between 0 and 1")
    validation_id_scores = np.asarray(validation_id_scores, dtype=float)
    if not len(validation_id_scores):
        raise ValueError("validation_id_scores cannot be empty")
    threshold = float(
        np.quantile(validation_id_scores, target_id_acceptance, method="higher")
    )
    accepted_id = np.asarray(test_id_scores, dtype=float) <= threshold
    rejected_ood = np.asarray(test_ood_scores, dtype=float) > threshold
    return (
        threshold,
        {
            "id_acceptance": float(accepted_id.mean()),
            "ood_rejection": float(rejected_ood.mean()),
        },
        accepted_id,
    )

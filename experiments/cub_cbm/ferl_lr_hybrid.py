"""Validation-gated FERL exceptions on top of a logistic-regression CBM head.

Logistic regression remains the default prediction.  FERL is allowed to
override it only when a gate, learned without test labels, predicts that the
local fuzzy rule is more likely to correct LR than to damage it.  Two gates are
compared by cross-fitting on validation:

``rule_local``
    Jeffreys-smoothed benefit/harm evidence attached to prefixes of the
    dominant FERL path.  The deepest sufficiently supported positive prefix is
    used as the exception rule.

``confidence``
    A regularised logistic gate over label-free confidence, entropy, route
    membership, support and depth features from the two heads.

The no-override LR policy participates in selection and wins every tie.  Test
labels are used only after the policy kind and its hyperparameters are fixed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ferl.core.learned_tree import EPS, LearnedFuzzyTree


GATE_FEATURE_NAMES = (
    "lr_max_probability",
    "lr_margin",
    "lr_entropy",
    "ferl_max_probability",
    "ferl_margin",
    "ferl_entropy",
    "ferl_minus_lr_confidence",
    "dominant_leaf_membership",
    "log_leaf_support",
    "route_depth",
)


@dataclass(frozen=True)
class HybridHeadOutputs:
    lr_prediction: np.ndarray
    ferl_prediction: np.ndarray
    lr_probability: np.ndarray
    ferl_probability: np.ndarray
    gate_features: np.ndarray
    routes: np.ndarray


@dataclass(frozen=True)
class RuleEvidence:
    benefit: int
    harm: int
    neutral_wrong: int

    @property
    def exclusive_support(self) -> int:
        return self.benefit + self.harm

    @property
    def posterior_benefit(self) -> float:
        """Jeffreys posterior mean P(FERL helps | exactly one head is right)."""
        return float((self.benefit + 0.5) / (self.exclusive_support + 1.0))


@dataclass
class HybridPolicy:
    kind: str = "none"
    threshold: float = 0.5
    min_support: int = 0
    min_depth: int = 1
    rule_evidence: dict[str, RuleEvidence] | None = None
    confidence_model: object | None = None
    confidence_constant: float | None = None

    def overrides(self, outputs: HybridHeadOutputs) -> tuple[np.ndarray, np.ndarray]:
        disagreement = outputs.lr_prediction != outputs.ferl_prediction
        override = np.zeros(len(disagreement), dtype=bool)
        matched = np.full(len(disagreement), "", dtype=object)
        if self.kind == "none":
            return override, matched
        if self.kind == "rule_local":
            evidence = self.rule_evidence or {}
            for i in np.flatnonzero(disagreement):
                route = str(outputs.routes[i])
                for prefix in reversed(route_prefixes(route, self.min_depth)):
                    record = evidence.get(prefix)
                    if record is None or record.exclusive_support < self.min_support:
                        continue
                    if record.posterior_benefit >= self.threshold:
                        override[i] = True
                        matched[i] = prefix
                        break
            return override, matched
        if self.kind == "confidence":
            if self.confidence_model is not None:
                probability = self.confidence_model.predict_proba(
                    outputs.gate_features,
                )[:, 1]
            else:
                probability = np.full(
                    len(disagreement), float(self.confidence_constant or 0.0),
                )
            override = disagreement & (probability >= self.threshold)
            matched[override] = outputs.routes[override]
            return override, matched
        raise ValueError(f"Unknown hybrid policy kind: {self.kind}")


@dataclass(frozen=True)
class HybridSelection:
    policy: HybridPolicy
    candidates: tuple[dict, ...]
    validation_outputs: HybridHeadOutputs


def _aligned_probability(model, X: np.ndarray, classes: np.ndarray) -> np.ndarray:
    probability = np.asarray(model.predict_proba(X), dtype=float)
    model_classes = np.asarray(model.classes_)
    if np.array_equal(model_classes, classes):
        return probability
    out = np.zeros((len(X), len(classes)), dtype=float)
    columns = {label: i for i, label in enumerate(classes)}
    for source, label in enumerate(model_classes):
        if label in columns:
            out[:, columns[label]] = probability[:, source]
    total = out.sum(axis=1, keepdims=True)
    out[total[:, 0] <= EPS] = 1.0 / len(classes)
    return out / out.sum(axis=1, keepdims=True)


def _probability_shape(probability: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if probability.shape[1] < 2:
        maximum = probability[:, 0]
        return maximum, maximum.copy(), np.zeros(len(probability))
    top = -np.partition(-probability, 1, axis=1)[:, :2]
    maximum = top[:, 0]
    margin = top[:, 0] - top[:, 1]
    entropy = -np.sum(probability * np.log(np.clip(probability, 1e-12, None)), axis=1)
    entropy /= np.log(probability.shape[1])
    return maximum, margin, entropy


def dominant_route_descriptors(
    model: LearnedFuzzyTree,
    X: np.ndarray,
    *,
    batch: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return dominant leaf name, firing, support and depth for each sample."""
    X = np.asarray(X, dtype=float)
    routes = np.full(len(X), "unsupported", dtype=object)
    membership = np.zeros(len(X), dtype=float)
    support_out = np.zeros(len(X), dtype=float)
    depth = np.zeros(len(X), dtype=float)
    for start in range(0, len(X), batch):
        stop = min(start + batch, len(X))
        activation, _, names, support = model.node_activation_matrix(X[start:stop])
        if activation.shape[1] == 0:
            continue
        keep = np.flatnonzero(model.leaf_mask(names))
        if keep.size == 0:
            continue
        leaf_activation = activation[:, keep]
        best = leaf_activation.argmax(axis=1)
        firing = leaf_activation[np.arange(stop - start), best]
        leaf_names = np.asarray([names[k] for k in keep], dtype=object)
        valid = firing > EPS
        rows = np.arange(start, stop)[valid]
        chosen = best[valid]
        routes[rows] = leaf_names[chosen]
        membership[rows] = firing[valid]
        support_out[rows] = support[keep][chosen]
        depth[rows] = np.asarray(
            [str(name).count("_") for name in leaf_names[chosen]], dtype=float,
        )
    return routes, membership, support_out, depth


def hybrid_head_outputs(
    lr_model,
    ferl_model: LearnedFuzzyTree,
    X: np.ndarray,
) -> HybridHeadOutputs:
    """Compute predictions and label-free gate features for both CBM heads."""
    X = np.asarray(X, dtype=float)
    classes = np.asarray(lr_model.classes_)
    lr_probability = _aligned_probability(lr_model, X, classes)
    ferl_probability = _aligned_probability(ferl_model, X, classes)
    lr_prediction = classes[lr_probability.argmax(axis=1)]
    ferl_prediction = classes[ferl_probability.argmax(axis=1)]
    lr_max, lr_margin, lr_entropy = _probability_shape(lr_probability)
    ferl_max, ferl_margin, ferl_entropy = _probability_shape(ferl_probability)
    routes, membership, support, depth = dominant_route_descriptors(ferl_model, X)
    features = np.column_stack([
        lr_max,
        lr_margin,
        lr_entropy,
        ferl_max,
        ferl_margin,
        ferl_entropy,
        ferl_max - lr_max,
        membership,
        np.log1p(support),
        depth,
    ])
    if not np.isfinite(features).all():
        raise ValueError("hybrid gate features contain non-finite values")
    return HybridHeadOutputs(
        lr_prediction=lr_prediction,
        ferl_prediction=ferl_prediction,
        lr_probability=lr_probability,
        ferl_probability=ferl_probability,
        gate_features=features,
        routes=routes,
    )


def route_prefixes(route: str, min_depth: int = 1) -> tuple[str, ...]:
    if not route.startswith("r_"):
        return ()
    pieces = route.split("_")
    first = max(int(min_depth), 1)
    return tuple("_".join(pieces[:depth + 1]) for depth in range(first, len(pieces)))


def fit_rule_evidence(
    routes: np.ndarray,
    lr_prediction: np.ndarray,
    ferl_prediction: np.ndarray,
    y: np.ndarray,
    *,
    min_depth: int = 1,
) -> dict[str, RuleEvidence]:
    """Count FERL benefit and harm on every dominant-route prefix."""
    counts: dict[str, list[int]] = {}
    lr_correct = np.asarray(lr_prediction) == y
    ferl_correct = np.asarray(ferl_prediction) == y
    disagreement = np.asarray(lr_prediction) != np.asarray(ferl_prediction)
    for i in np.flatnonzero(disagreement):
        benefit = bool(ferl_correct[i] and not lr_correct[i])
        harm = bool(lr_correct[i] and not ferl_correct[i])
        neutral = bool(not lr_correct[i] and not ferl_correct[i])
        for prefix in route_prefixes(str(routes[i]), min_depth):
            record = counts.setdefault(prefix, [0, 0, 0])
            record[0] += int(benefit)
            record[1] += int(harm)
            record[2] += int(neutral)
    return {
        key: RuleEvidence(*record)
        for key, record in counts.items()
    }


def fit_confidence_gate(
    features: np.ndarray,
    lr_prediction: np.ndarray,
    ferl_prediction: np.ndarray,
    y: np.ndarray,
    *,
    random_state: int = 0,
) -> tuple[object | None, float | None]:
    """Fit P(FERL helps | exactly one disagreeing head is correct)."""
    lr_correct = np.asarray(lr_prediction) == y
    ferl_correct = np.asarray(ferl_prediction) == y
    exclusive = lr_correct != ferl_correct
    target = ferl_correct[exclusive].astype(int)
    if target.size == 0:
        return None, 0.0
    if np.unique(target).size < 2:
        return None, float(target[0])
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=random_state),
    )
    model.fit(features[exclusive], target)
    return model, None


def exception_metrics(
    lr_prediction: np.ndarray,
    ferl_prediction: np.ndarray,
    y: np.ndarray,
    override: np.ndarray,
) -> dict[str, float | int]:
    """Accuracy and correctness-transition accounting for an override mask."""
    lr_prediction = np.asarray(lr_prediction)
    ferl_prediction = np.asarray(ferl_prediction)
    y = np.asarray(y)
    override = np.asarray(override, dtype=bool)
    lr_correct = lr_prediction == y
    ferl_correct = ferl_prediction == y
    benefit_mask = override & ferl_correct & ~lr_correct
    harm_mask = override & lr_correct & ~ferl_correct
    neutral_mask = override & ~lr_correct & ~ferl_correct
    prediction = np.where(override, ferl_prediction, lr_prediction)
    benefit = int(benefit_mask.sum())
    harm = int(harm_mask.sum())
    exclusive = benefit + harm
    return {
        "accuracy": float(np.mean(prediction == y)),
        "lr_accuracy": float(lr_correct.mean()),
        "ferl_accuracy": float(ferl_correct.mean()),
        "disagreement_rate": float(np.mean(lr_prediction != ferl_prediction)),
        "override_count": int(override.sum()),
        "override_rate": float(override.mean()),
        "benefit_count": benefit,
        "harm_count": harm,
        "neutral_wrong_count": int(neutral_mask.sum()),
        "net_corrections": benefit - harm,
        "net_correction_rate": float((benefit - harm) / len(y)),
        "override_precision": float(benefit / exclusive) if exclusive else np.nan,
        "ferl_only_opportunity": int((ferl_correct & ~lr_correct).sum()),
        "lr_only_opportunity": int((lr_correct & ~ferl_correct).sum()),
        "oracle_switch_accuracy": float(np.mean(lr_correct | ferl_correct)),
    }


def _candidate_record(kind: str, label: str, metrics: dict, **parameters) -> dict:
    return {
        "kind": kind,
        "candidate": label,
        "selected": False,
        **parameters,
        **{f"validation_{key}": value for key, value in metrics.items()},
    }


def _cv_splits(y: np.ndarray, n_splits: int, random_state: int):
    _, encoded = np.unique(y, return_inverse=True)
    minimum = int(np.bincount(encoded).min())
    folds = min(int(n_splits), minimum)
    if folds < 2:
        raise ValueError("at least two validation examples per class are required")
    return list(StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=random_state,
    ).split(np.zeros(len(y)), y))


def fit_validation_selected_hybrid(
    lr_model,
    ferl_model: LearnedFuzzyTree,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    n_splits: int = 3,
    rule_supports: tuple[int, ...] = (2, 4, 8),
    rule_thresholds: tuple[float, ...] = (0.6, 0.7, 0.8),
    confidence_thresholds: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8),
    min_rule_depth: int = 1,
    random_state: int = 0,
) -> HybridSelection:
    """Cross-fit candidate gates on validation, select, then refit on all val."""
    y_validation = np.asarray(y_validation)
    outputs = hybrid_head_outputs(lr_model, ferl_model, X_validation)
    folds = _cv_splits(y_validation, n_splits, random_state)
    n = len(y_validation)

    candidates: list[tuple[dict, np.ndarray]] = []
    never = np.zeros(n, dtype=bool)
    base_metrics = exception_metrics(
        outputs.lr_prediction, outputs.ferl_prediction, y_validation, never,
    )
    candidates.append((_candidate_record("none", "LR default", base_metrics), never))

    rule_masks = {
        (support, threshold): np.zeros(n, dtype=bool)
        for support in rule_supports for threshold in rule_thresholds
    }
    gate_probability = np.zeros(n, dtype=float)
    for fold, (train, held_out) in enumerate(folds):
        evidence = fit_rule_evidence(
            outputs.routes[train], outputs.lr_prediction[train],
            outputs.ferl_prediction[train], y_validation[train],
            min_depth=min_rule_depth,
        )
        for support, threshold in rule_masks:
            policy = HybridPolicy(
                kind="rule_local", threshold=threshold, min_support=support,
                min_depth=min_rule_depth, rule_evidence=evidence,
            )
            mask, _ = policy.overrides(HybridHeadOutputs(
                lr_prediction=outputs.lr_prediction[held_out],
                ferl_prediction=outputs.ferl_prediction[held_out],
                lr_probability=outputs.lr_probability[held_out],
                ferl_probability=outputs.ferl_probability[held_out],
                gate_features=outputs.gate_features[held_out],
                routes=outputs.routes[held_out],
            ))
            rule_masks[(support, threshold)][held_out] = mask

        gate, constant = fit_confidence_gate(
            outputs.gate_features[train], outputs.lr_prediction[train],
            outputs.ferl_prediction[train], y_validation[train],
            random_state=random_state + fold,
        )
        if gate is not None:
            gate_probability[held_out] = gate.predict_proba(
                outputs.gate_features[held_out],
            )[:, 1]
        else:
            gate_probability[held_out] = float(constant or 0.0)

    disagreement = outputs.lr_prediction != outputs.ferl_prediction
    for (support, threshold), mask in rule_masks.items():
        metrics = exception_metrics(
            outputs.lr_prediction, outputs.ferl_prediction, y_validation, mask,
        )
        label = f"support>={support},posterior>={threshold:g}"
        candidates.append((_candidate_record(
            "rule_local", label, metrics, min_support=support,
            threshold=threshold, min_rule_depth=min_rule_depth,
        ), mask))
    for threshold in confidence_thresholds:
        mask = disagreement & (gate_probability >= threshold)
        metrics = exception_metrics(
            outputs.lr_prediction, outputs.ferl_prediction, y_validation, mask,
        )
        candidates.append((_candidate_record(
            "confidence", f"probability>={threshold:g}", metrics,
            min_support=np.nan, threshold=threshold,
            min_rule_depth=np.nan,
        ), mask))

    # Accuracy is LR accuracy plus net corrections / n.  Baseline appears first
    # and wins exact ties; among equal improvements prefer fewer overrides.
    best_index = 0
    for index in range(1, len(candidates)):
        current, current_mask = candidates[index]
        best, best_mask = candidates[best_index]
        current_key = (
            current["validation_accuracy"],
            current["validation_net_corrections"],
            -int(current_mask.sum()),
        )
        best_key = (
            best["validation_accuracy"],
            best["validation_net_corrections"],
            -int(best_mask.sum()),
        )
        if current_key > best_key:
            best_index = index
    records = [record for record, _ in candidates]
    records[best_index]["selected"] = True
    selected = records[best_index]

    if selected["kind"] == "rule_local":
        evidence = fit_rule_evidence(
            outputs.routes, outputs.lr_prediction, outputs.ferl_prediction,
            y_validation, min_depth=min_rule_depth,
        )
        policy = HybridPolicy(
            kind="rule_local", threshold=float(selected["threshold"]),
            min_support=int(selected["min_support"]), min_depth=min_rule_depth,
            rule_evidence=evidence,
        )
    elif selected["kind"] == "confidence":
        gate, constant = fit_confidence_gate(
            outputs.gate_features, outputs.lr_prediction,
            outputs.ferl_prediction, y_validation, random_state=random_state,
        )
        policy = HybridPolicy(
            kind="confidence", threshold=float(selected["threshold"]),
            confidence_model=gate, confidence_constant=constant,
        )
    else:
        policy = HybridPolicy(kind="none")
    return HybridSelection(policy, tuple(records), outputs)


def hybrid_prediction(
    policy: HybridPolicy,
    outputs: HybridHeadOutputs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return hybrid class/probability, override mask, and matched rule names."""
    override, matched = policy.overrides(outputs)
    prediction = np.where(
        override, outputs.ferl_prediction, outputs.lr_prediction,
    )
    probability = np.where(
        override[:, None], outputs.ferl_probability, outputs.lr_probability,
    )
    return prediction, probability, override, matched


def path_features(model: LearnedFuzzyTree, route: str) -> list[int]:
    """Concept indices along a complete or prefix FERL route."""
    if not route.startswith("r_"):
        return []
    node = model.root_
    features: list[int] = []
    for branch in route.split("_")[1:]:
        if node["leaf"]:
            break
        features.append(int(node["f"]))
        node = node["L"] if branch == "0" else node["R"]
    return features


def format_path_rule(
    model: LearnedFuzzyTree,
    route: str,
    concept_names: list[str] | None = None,
) -> str:
    """Human-readable fuzzy conditions for an override path/prefix."""
    if not route.startswith("r_"):
        return route
    node = model.root_
    conditions = []
    for branch in route.split("_")[1:]:
        if node["leaf"]:
            break
        feature = int(node["f"])
        name = concept_names[feature] if concept_names is not None else f"c{feature}"
        relation = "low" if branch == "0" else "high"
        conditions.append(
            f"{name} is {relation} around {float(node['center']):.3f}",
        )
        node = node["L"] if branch == "0" else node["R"]
    return " AND ".join(conditions)

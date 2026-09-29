"""Adaptive concept interventions for FERL and linear CBM heads.

The test-time policy never uses the current instance's oracle concept values to
choose a query. Oracle values enter only after a query, standing in for a human
answer during evaluation. The utility-aware variant may use annotated
validation concepts to estimate a fixed per-feature intervention prior, just as
concept calibration uses validation annotations; it never sees a test answer
before committing to that test query.

The original support-guided policy handles two cases:

* If bounded support vanishes, backtrack from the failure and query the nearest
  preceding split whose alternative branch restores a supported credal output.
  This is the smallest local edit to the model's current reasoning route.
* If FERL abstains while support remains, use one-step lookahead over concepts
  on the dominant route and query the one most likely to yield a singleton
  credal set (ties: expected set-size and ignorance reduction).

After every answer the route is recomputed.  Thus this is an adaptive policy,
not a static ordering of concepts.

The reliability/purity policy uses the same adaptive protocol but ranks every
unqueried concept on the active route jointly.  Its answer probabilities are
the detector scores shrunk toward 0.5 according to validation-estimated concept
reliability. Candidate outcomes are then scored by validation-estimated route
purity (falling back to singleton belief when no validation model is supplied),
and validation-estimated wrong-to-correct minus correct-to-wrong transitions,
so a support repair that merely produces a brittle singleton is disfavoured.
Neither selector sees the current test oracle answer before it commits.

For logistic regression, which has no native credal abstention state, a
confidence threshold is selected on validation data to match FERL's validation
acceptance rate.  Rejected samples query the concept with greatest one-step
expected reduction in predictive entropy and stop once confidence crosses that
fixed threshold.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from ferl.core.learned_tree import LearnedFuzzyTree, _ds_combine


@dataclass(frozen=True)
class RouteStep:
    feature: int
    branch: str
    low_membership: float
    high_membership: float
    node: dict = field(repr=False, compare=False)


@dataclass(frozen=True)
class RouteTrace:
    steps: tuple[RouteStep, ...]
    failure_feature: int | None = None
    failure_node: dict | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class QuerySuggestion:
    feature: int
    reason: str
    expected_singleton: float
    expected_set_size: float
    expected_ignorance: float
    expected_correctness: float = np.nan
    expected_flip: float = np.nan


@dataclass
class AdaptiveSnapshot:
    X: np.ndarray
    query_counts: np.ndarray
    reason_counts: dict[str, int]


@dataclass(frozen=True)
class RouteReliability:
    """Validation-only correctness estimates for dominant FERL leaves/classes."""

    global_accuracy: float
    class_accuracy: dict[object, float]
    leaf_accuracy: dict[str, float]

    def score(self, model: LearnedFuzzyTree, x: np.ndarray, prediction) -> float:
        trace = trace_dominant_route(model, x)
        if trace.failure_node is None:
            key = "r" + "".join(
                "_0" if step.branch == "low" else "_1" for step in trace.steps
            )
            if key in self.leaf_accuracy:
                return self.leaf_accuracy[key]
        return self.class_accuracy.get(prediction, self.global_accuracy)


def fit_route_reliability(
    model: LearnedFuzzyTree,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
) -> RouteReliability:
    """Estimate route correctness on validation data with Jeffreys smoothing.

    This is intentionally a correctness model rather than another confidence
    transformation: FERL leaf beliefs can be nearly constant when training
    leaves are pure.  The validation labels reveal whether a dominant leaf (or,
    as fallback, a predicted class) transfers to detector-produced concepts.
    """
    X_validation = np.asarray(X_validation, dtype=float)
    y_validation = np.asarray(y_validation)
    if len(X_validation) != len(y_validation) or not len(y_validation):
        raise ValueError("validation inputs and labels must be non-empty and aligned")
    prediction = np.asarray(model.predict(X_validation))
    correct = prediction == y_validation

    def smoothed(n_correct: int, n_total: int) -> float:
        return float((n_correct + 0.5) / (n_total + 1.0))

    global_accuracy = smoothed(int(correct.sum()), len(correct))
    class_accuracy: dict[object, float] = {}
    for label in np.asarray(model.classes_):
        mask = prediction == label
        if mask.any():
            class_accuracy[label.item() if hasattr(label, "item") else label] = smoothed(
                int(correct[mask].sum()), int(mask.sum()),
            )

    leaf_total: Counter[str] = Counter()
    leaf_correct: Counter[str] = Counter()
    for x, is_correct in zip(X_validation, correct):
        trace = trace_dominant_route(model, x)
        if trace.failure_node is not None:
            continue
        key = "r" + "".join(
            "_0" if step.branch == "low" else "_1" for step in trace.steps
        )
        leaf_total[key] += 1
        leaf_correct[key] += int(is_correct)
    leaf_accuracy = {
        key: smoothed(leaf_correct[key], total)
        for key, total in leaf_total.items()
    }
    return RouteReliability(global_accuracy, class_accuracy, leaf_accuracy)


@dataclass(frozen=True)
class FeatureInterventionUtility:
    """Validation-estimated benefit of verifying each active-route concept."""

    net_correction: np.ndarray
    post_accuracy: np.ndarray
    support: np.ndarray


def fit_feature_intervention_utility(
    model: LearnedFuzzyTree,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    X_verified_validation: np.ndarray,
) -> FeatureInterventionUtility:
    """Estimate feature-specific correction benefit without test-oracle access.

    A feature is evaluated only on validation samples whose current dominant
    route contains it. ``net_correction`` counts wrong-to-correct as +1 and
    correct-to-wrong as -1, divided by ``n + 2`` to shrink small route samples
    toward zero. ``post_accuracy`` uses Jeffreys smoothing. These quantities
    become fixed before test-time querying.
    """
    X_validation = np.asarray(X_validation, dtype=float)
    X_verified_validation = np.asarray(X_verified_validation, dtype=float)
    y_validation = np.asarray(y_validation)
    if X_validation.shape != X_verified_validation.shape:
        raise ValueError("validation and verified validation inputs must align")
    if len(X_validation) != len(y_validation) or not len(y_validation):
        raise ValueError("validation inputs and labels must be non-empty and aligned")
    n_features = X_validation.shape[1]
    eligible = np.zeros(X_validation.shape, dtype=bool)
    for row, x in enumerate(X_validation):
        trace = trace_dominant_route(model, x)
        features = [step.feature for step in trace.steps]
        if trace.failure_feature is not None:
            features.append(trace.failure_feature)
        eligible[row, list(dict.fromkeys(features))] = True

    baseline_correct = np.asarray(model.predict(X_validation)) == y_validation
    global_accuracy = float((baseline_correct.sum() + 0.5) / (len(y_validation) + 1.0))
    net_correction = np.zeros(n_features, dtype=float)
    post_accuracy = np.full(n_features, global_accuracy, dtype=float)
    support = eligible.sum(0).astype(int)
    for feature in np.flatnonzero(support):
        mask = eligible[:, feature]
        intervened = X_validation[mask].copy()
        intervened[:, feature] = X_verified_validation[mask, feature]
        after_correct = np.asarray(model.predict(intervened)) == y_validation[mask]
        before_correct = baseline_correct[mask]
        gain = after_correct.astype(int) - before_correct.astype(int)
        net_correction[feature] = float(gain.sum() / (mask.sum() + 2.0))
        post_accuracy[feature] = float(
            (after_correct.sum() + 0.5) / (mask.sum() + 1.0)
        )
    return FeatureInterventionUtility(net_correction, post_accuracy, support)


def confidence_threshold_for_coverage(
    confidence: np.ndarray,
    target_acceptance_rate: float,
) -> float:
    """Choose a fixed threshold giving the requested validation coverage.

    The threshold is placed between adjacent confidence scores when possible.
    Ties can make exact matching impossible; callers should therefore record
    the realised validation acceptance rate as well as the target.
    """
    confidence = np.asarray(confidence, dtype=float)
    if confidence.ndim != 1 or len(confidence) == 0:
        raise ValueError("confidence must be a non-empty one-dimensional array")
    if not np.isfinite(confidence).all():
        raise ValueError("confidence must contain only finite values")
    target = float(np.clip(target_acceptance_rate, 0.0, 1.0))
    n_accept = int(np.rint(target * len(confidence)))
    if n_accept <= 0:
        return float("inf")
    if n_accept >= len(confidence):
        return float("-inf")
    ordered = np.sort(confidence)[::-1]
    accepted_edge = float(ordered[n_accept - 1])
    rejected_edge = float(ordered[n_accept])
    if accepted_edge > rejected_edge:
        return accepted_edge - 0.5 * (accepted_edge - rejected_edge)
    return accepted_edge


def confidence_acceptance(
    model,
    X: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return point predictions, maximum probabilities, and accept decisions."""
    probabilities = np.asarray(model.predict_proba(X), dtype=float)
    best = probabilities.argmax(axis=1)
    prediction = np.asarray(model.classes_)[best]
    confidence = probabilities[np.arange(len(probabilities)), best]
    return prediction, confidence, confidence >= threshold


def _linear_parameters(model) -> tuple[np.ndarray, np.ndarray]:
    """Return class-by-feature weights whose softmax matches sklearn LR."""
    weights = np.asarray(model.coef_, dtype=float)
    intercept = np.asarray(model.intercept_, dtype=float)
    classes = np.asarray(model.classes_)
    if weights.ndim != 2 or weights.shape[1] == 0:
        raise ValueError("model must expose a fitted two-dimensional coef_")
    if weights.shape[0] == 1 and len(classes) == 2:
        weights = np.vstack([np.zeros_like(weights), weights])
        intercept = np.concatenate([np.zeros(1, dtype=float), intercept])
    if weights.shape[0] != len(classes) or intercept.shape != (len(classes),):
        raise ValueError("model coefficients are incompatible with model.classes_")
    return weights, intercept


def _entropy_from_logits(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    normalizer = exponentiated.sum(axis=-1, keepdims=True)
    probabilities = exponentiated / normalizer
    return np.log(normalizer[..., 0]) - np.sum(
        probabilities * shifted, axis=-1,
    )


def lr_expected_information_queries(
    model,
    X: np.ndarray,
    queried: np.ndarray,
    value_absent: np.ndarray,
    value_present: np.ndarray,
    *,
    batch: int = 128,
) -> np.ndarray:
    """Select one query per row by expected predictive-entropy reduction.

    The probability of a positive human answer is the current detector score.
    Verified values are deliberately absent from this function: they enter only
    after the selected query, preserving the no-oracle-selection protocol.
    """
    X = np.asarray(X, dtype=float)
    queried = np.asarray(queried, dtype=bool)
    value_absent = np.asarray(value_absent, dtype=float)
    value_present = np.asarray(value_present, dtype=float)
    if X.shape != queried.shape:
        raise ValueError("X and queried must have the same shape")
    if value_absent.shape != (X.shape[1],) or value_present.shape != (X.shape[1],):
        raise ValueError("intervention values must have one entry per feature")

    weights, intercept = _linear_parameters(model)
    if weights.shape[1] != X.shape[1]:
        raise ValueError("X has a different number of features than the model")
    weight_by_feature = weights.T
    chosen = np.full(len(X), -1, dtype=int)
    for start in range(0, len(X), batch):
        stop = min(start + batch, len(X))
        values = X[start:stop]
        unavailable = queried[start:stop]
        base_logits = values @ weights.T + intercept
        delta_absent = value_absent[None, :] - values
        delta_present = value_present[None, :] - values
        absent_logits = (
            base_logits[:, None, :]
            + delta_absent[:, :, None] * weight_by_feature[None, :, :]
        )
        present_logits = (
            base_logits[:, None, :]
            + delta_present[:, :, None] * weight_by_feature[None, :, :]
        )
        probability_present = np.clip(values, 0.0, 1.0)
        expected_entropy = (
            (1.0 - probability_present) * _entropy_from_logits(absent_logits)
            + probability_present * _entropy_from_logits(present_logits)
        )
        expected_entropy[unavailable] = np.inf
        best = expected_entropy.argmin(axis=1)
        exhausted = np.isinf(expected_entropy[np.arange(len(values)), best])
        best[exhausted] = -1
        chosen[start:stop] = best
    return chosen


def adaptive_lr_query_trajectory(
    model,
    X_predicted: np.ndarray,
    X_verified: np.ndarray,
    value_absent: np.ndarray,
    value_present: np.ndarray,
    *,
    confidence_threshold: float,
    max_queries: int,
    batch: int = 128,
) -> dict[int, AdaptiveSnapshot]:
    """Adaptively query rejected LR samples and stop when they are accepted."""
    X = np.asarray(X_predicted, dtype=float).copy()
    X_verified = np.asarray(X_verified, dtype=float)
    if X.shape != X_verified.shape:
        raise ValueError("X_predicted and X_verified must have the same shape")
    max_queries = min(int(max_queries), X.shape[1])
    queried = np.zeros(X.shape, dtype=bool)
    query_counts = np.zeros(len(X), dtype=int)
    reasons: Counter[str] = Counter()
    snapshots = {0: AdaptiveSnapshot(X.copy(), query_counts.copy(), {})}
    _, _, accepted = confidence_acceptance(model, X, confidence_threshold)
    unresolved = np.flatnonzero(~accepted)

    for budget in range(1, max_queries + 1):
        if len(unresolved):
            features = lr_expected_information_queries(
                model,
                X[unresolved],
                queried[unresolved],
                value_absent,
                value_present,
                batch=batch,
            )
            valid = features >= 0
            rows = unresolved[valid]
            cols = features[valid]
            queried[rows, cols] = True
            X[rows, cols] = X_verified[rows, cols]
            query_counts[rows] += 1
            reasons["expected_information_gain"] += int(valid.sum())
        snapshots[budget] = AdaptiveSnapshot(
            X.copy(), query_counts.copy(), dict(reasons),
        )
        if budget < max_queries and len(unresolved):
            _, _, updated_accepted = confidence_acceptance(
                model, X[unresolved], confidence_threshold,
            )
            unresolved = unresolved[~updated_accepted]
    return snapshots


def fixed_order_lr_query_trajectory(
    model,
    X_predicted: np.ndarray,
    X_verified: np.ndarray,
    orders: np.ndarray,
    *,
    confidence_threshold: float,
    max_queries: int,
) -> dict[int, AdaptiveSnapshot]:
    """Apply a static LR query order with the same rejection and stopping rule."""
    X = np.asarray(X_predicted, dtype=float).copy()
    X_verified = np.asarray(X_verified, dtype=float)
    orders = np.asarray(orders, dtype=int)
    if X.shape != X_verified.shape or orders.shape != X.shape:
        raise ValueError("X_predicted, X_verified, and orders must have the same shape")
    max_queries = min(int(max_queries), X.shape[1])
    query_counts = np.zeros(len(X), dtype=int)
    snapshots = {0: AdaptiveSnapshot(X.copy(), query_counts.copy(), {})}
    _, _, accepted = confidence_acceptance(model, X, confidence_threshold)
    unresolved = np.flatnonzero(~accepted)
    for budget in range(1, max_queries + 1):
        positions = query_counts[unresolved]
        available = positions < X.shape[1]
        rows = unresolved[available]
        cols = orders[rows, positions[available]]
        X[rows, cols] = X_verified[rows, cols]
        query_counts[rows] += 1
        snapshots[budget] = AdaptiveSnapshot(X.copy(), query_counts.copy(), {})
        if budget < max_queries and len(unresolved):
            _, _, updated_accepted = confidence_acceptance(
                model, X[unresolved], confidence_threshold,
            )
            unresolved = unresolved[~updated_accepted]
    return snapshots


def predict_credal(
    model: LearnedFuzzyTree,
    X: np.ndarray,
    *,
    batch: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(sets, belief, ignorance)`` using the paper's leaf-only rule."""
    X = np.asarray(X, dtype=float)
    set_parts, belief_parts, ignorance_parts = [], [], []
    for start in range(0, len(X), batch):
        membership, consequents, names, support = model.node_activation_matrix(
            X[start:start + batch]
        )
        keep = np.flatnonzero(model.leaf_mask(names))
        _, belief, plausibility, ignorance = _ds_combine(
            membership[:, keep],
            consequents[keep],
            [names[k] for k in keep],
            model.C,
            rule="dempster",
            support=support[keep],
        )
        set_parts.append(plausibility >= belief.max(1, keepdims=True) - 1e-12)
        belief_parts.append(belief)
        ignorance_parts.append(ignorance)
    return (
        np.concatenate(set_parts, axis=0),
        np.concatenate(belief_parts, axis=0),
        np.concatenate(ignorance_parts, axis=0),
    )


def trace_dominant_route(
    model: LearnedFuzzyTree,
    x: np.ndarray,
    *,
    zero_tol: float = 1e-12,
) -> RouteTrace:
    """Follow the stronger fuzzy branch and stop when both branches have zero support."""
    row = np.asarray(x, dtype=float).reshape(1, -1)
    node = model.root_
    steps: list[RouteStep] = []
    while not node["leaf"]:
        feature = int(node["f"])
        low, high = model._split(node, row)
        low, high = float(low[0]), float(high[0])
        if max(low, high) <= zero_tol:
            return RouteTrace(tuple(steps), feature, node)
        take_low = low >= high
        steps.append(RouteStep(
            feature=feature,
            branch="low" if take_low else "high",
            low_membership=low,
            high_membership=high,
            node=node,
        ))
        node = node["L"] if take_low else node["R"]
    return RouteTrace(tuple(steps))


def _single_state(model: LearnedFuzzyTree, x: np.ndarray) -> tuple[int, float]:
    sets, _, ignorance = predict_credal(model, np.asarray(x)[None, :])
    return int(sets[0].sum()), float(ignorance[0])


def _simulate_feature(
    model: LearnedFuzzyTree,
    x: np.ndarray,
    feature: int,
    value_absent: np.ndarray,
    value_present: np.ndarray,
) -> tuple[tuple[int, float], tuple[int, float]]:
    absent = np.asarray(x, dtype=float).copy()
    present = absent.copy()
    absent[feature] = value_absent[feature]
    present[feature] = value_present[feature]
    return _single_state(model, absent), _single_state(model, present)


def _route_candidates(trace: RouteTrace, queried: np.ndarray) -> list[int]:
    """Unique, unqueried concepts that can affect the current dominant route."""
    candidates: list[int] = []
    if trace.failure_feature is not None:
        candidates.append(trace.failure_feature)
    candidates.extend(step.feature for step in reversed(trace.steps))
    return list(dict.fromkeys(f for f in candidates if not queried[f]))


def _simulated_candidate_rows(
    x: np.ndarray,
    candidates: list[int],
    value_absent: np.ndarray,
    value_present: np.ndarray,
) -> np.ndarray:
    """Stack absent outcomes followed by present outcomes for each candidate."""
    absent = np.broadcast_to(x, (len(candidates), len(x))).copy()
    present = absent.copy()
    rows = np.arange(len(candidates))
    absent[rows, candidates] = value_absent[candidates]
    present[rows, candidates] = value_present[candidates]
    return np.concatenate([absent, present], axis=0)


def _simulated_route_outcomes(
    model: LearnedFuzzyTree,
    x: np.ndarray,
    candidates: list[int],
    value_absent: np.ndarray,
    value_present: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Credal and point predictions for absent/present candidate outcomes."""
    simulated = _simulated_candidate_rows(
        x, candidates, value_absent, value_present,
    )
    simulated_sets, simulated_belief, simulated_ignorance = predict_credal(
        model, simulated,
    )
    simulated_probability = np.asarray(model.predict_proba(simulated), dtype=float)
    return (
        simulated_sets,
        simulated_belief,
        simulated_ignorance,
        simulated_probability,
    )


def suggest_support_guided_query(
    model: LearnedFuzzyTree,
    x: np.ndarray,
    queried: np.ndarray,
    value_absent: np.ndarray,
    value_present: np.ndarray,
    *,
    global_order: np.ndarray | None = None,
    current_state: tuple[int, float] | None = None,
) -> QuerySuggestion | None:
    """Choose the next concept without consulting its oracle/human value."""
    x = np.asarray(x, dtype=float)
    queried = np.asarray(queried, dtype=bool)
    if current_state is None:
        current_size, current_ignorance = _single_state(model, x)
    else:
        current_size, current_ignorance = current_state
    if current_size == 1:
        return None

    trace = trace_dominant_route(model, x)

    # Geometric failure: preserve as much of the current explanation as
    # possible by trying supported sibling branches from deepest to shallowest.
    if trace.failure_node is not None:
        # Short-circuiting matters for deep trees: the nearest sibling usually
        # repairs support, so evaluating every shallower alternative would turn
        # one local query decision into dozens of needless credal predictions.
        for step in reversed(trace.steps):
            feature = step.feature
            if queried[feature]:
                continue
            alternative = x.copy()
            if step.branch == "low":
                alternative[feature] = value_present[feature]
            else:
                alternative[feature] = value_absent[feature]
            alt_size, alt_ignorance = _single_state(model, alternative)
            if alt_size < current_size or alt_ignorance < current_ignorance - 1e-12:
                return QuerySuggestion(
                    feature=step.feature,
                    reason="support_backtrack",
                    expected_singleton=float(alt_size == 1),
                    expected_set_size=float(alt_size),
                    expected_ignorance=float(alt_ignorance),
                )

    # In-support ambiguity (or no repairing sibling): one-step value of
    # information over the concepts that actually participate in this route.
    candidates = _route_candidates(trace, queried)

    sizes0 = sizes1 = np.empty(0, dtype=int)
    ignorance0 = ignorance1 = np.empty(0, dtype=float)
    if candidates:
        absent = np.broadcast_to(x, (len(candidates), len(x))).copy()
        present = absent.copy()
        rows = np.arange(len(candidates))
        absent[rows, candidates] = value_absent[candidates]
        present[rows, candidates] = value_present[candidates]
        simulated_sets, _, simulated_ignorance = predict_credal(
            model, np.concatenate([absent, present], axis=0),
        )
        sizes0 = simulated_sets[:len(candidates)].sum(1)
        sizes1 = simulated_sets[len(candidates):].sum(1)
        ignorance0 = simulated_ignorance[:len(candidates)]
        ignorance1 = simulated_ignorance[len(candidates):]

    best: QuerySuggestion | None = None
    best_key: tuple[float, float, float, float] | None = None
    for rank, (feature, size0, size1, ign0, ign1) in enumerate(zip(
        candidates, sizes0, sizes1, ignorance0, ignorance1,
    )):
        probability_present = float(np.clip(x[feature], 0.0, 1.0))
        expected_singleton = (
            (1.0 - probability_present) * float(size0 == 1)
            + probability_present * float(size1 == 1)
        )
        expected_size = (1.0 - probability_present) * size0 + probability_present * size1
        expected_ignorance = (
            (1.0 - probability_present) * ign0 + probability_present * ign1
        )
        key = (
            expected_singleton,
            float(current_size) - expected_size,
            current_ignorance - expected_ignorance,
            -float(rank),
        )
        if best_key is None or key > best_key:
            best_key = key
            best = QuerySuggestion(
                feature=feature,
                reason="expected_uncertainty",
                expected_singleton=expected_singleton,
                expected_set_size=expected_size,
                expected_ignorance=expected_ignorance,
            )
    if best is not None:
        return best

    # Once route concepts are exhausted, fall back to the model's global order
    # so the policy remains defined under any query budget.
    if global_order is None:
        global_order = np.arange(len(x))
    for feature in np.asarray(global_order, dtype=int):
        if not queried[feature]:
            return QuerySuggestion(
                feature=int(feature),
                reason="global_fallback",
                expected_singleton=0.0,
                expected_set_size=float(current_size),
                expected_ignorance=current_ignorance,
            )
    return None


def suggest_reliability_purity_query(
    model: LearnedFuzzyTree,
    x: np.ndarray,
    queried: np.ndarray,
    value_absent: np.ndarray,
    value_present: np.ndarray,
    reliability: np.ndarray,
    *,
    route_reliability: RouteReliability | None = None,
    feature_utility: FeatureInterventionUtility | None = None,
    global_order: np.ndarray | None = None,
    current_state: tuple[int, float] | None = None,
) -> QuerySuggestion | None:
    """Choose a route query using reliability-weighted singleton purity.

    For concept ``j``, the probability of a positive answer is

    ``0.5 + reliability[j] * (detector_score[j] - 0.5)``.

    Thus a validated detector keeps its score while an uninformative detector
    is treated as genuinely uncertain. Each possible answer is simulated and
    its correctness proxy is the validation-estimated accuracy of the
    resulting dominant route *only when* the credal set is a singleton. If no
    route model is supplied, the largest Dempster--Shafer belief is used as a
    fallback. Hidden verified values are not an input.
    """
    x = np.asarray(x, dtype=float)
    queried = np.asarray(queried, dtype=bool)
    reliability = np.asarray(reliability, dtype=float)
    if reliability.shape != x.shape:
        raise ValueError("reliability must have one entry per feature")
    if not np.isfinite(reliability).all():
        raise ValueError("reliability must contain only finite values")
    reliability = np.clip(reliability, 0.0, 1.0)
    if feature_utility is not None:
        for values in (
            feature_utility.net_correction,
            feature_utility.post_accuracy,
            feature_utility.support,
        ):
            if np.asarray(values).shape != x.shape:
                raise ValueError("feature utility must have one entry per feature")
    if current_state is None:
        current_size, current_ignorance = _single_state(model, x)
    else:
        current_size, current_ignorance = current_state
    if current_size == 1:
        return None

    trace = trace_dominant_route(model, x)
    candidates = _route_candidates(trace, queried)
    if candidates:
        sets, belief, ignorance, probability = _simulated_route_outcomes(
            model, x, candidates, value_absent, value_present,
        )
        n_candidates = len(candidates)
        size0 = sets[:n_candidates].sum(1)
        size1 = sets[n_candidates:].sum(1)
        ign0 = ignorance[:n_candidates]
        ign1 = ignorance[n_candidates:]
        singleton_belief0 = belief[:n_candidates].max(1) * (size0 == 1)
        singleton_belief1 = belief[n_candidates:].max(1) * (size1 == 1)
        if route_reliability is None:
            singleton_quality0 = singleton_belief0
            singleton_quality1 = singleton_belief1
        else:
            simulated_rows = _simulated_candidate_rows(
                x, candidates, value_absent, value_present,
            )
            simulated_prediction = np.asarray(model.classes_)[probability.argmax(1)]
            quality = np.array([
                route_reliability.score(model, row, pred)
                for row, pred in zip(simulated_rows, simulated_prediction)
            ])
            singleton_quality0 = quality[:n_candidates] * (size0 == 1)
            singleton_quality1 = quality[n_candidates:] * (size1 == 1)
        effect = 0.5 * np.abs(
            probability[:n_candidates] - probability[n_candidates:]
        ).sum(1)

        best: QuerySuggestion | None = None
        best_key: tuple[float, ...] | None = None
        reason = (
            "reliability_purity_support"
            if trace.failure_node is not None
            else "reliability_purity_uncertainty"
        )
        for rank, feature in enumerate(candidates):
            probability_present = float(
                0.5 + reliability[feature] * (np.clip(x[feature], 0.0, 1.0) - 0.5)
            )
            probability_absent = 1.0 - probability_present
            expected_correctness = float(
                probability_absent * singleton_quality0[rank]
                + probability_present * singleton_quality1[rank]
            )
            expected_singleton_belief = float(
                probability_absent * singleton_belief0[rank]
                + probability_present * singleton_belief1[rank]
            )
            expected_singleton = float(
                probability_absent * (size0[rank] == 1)
                + probability_present * (size1[rank] == 1)
            )
            expected_size = float(
                probability_absent * size0[rank]
                + probability_present * size1[rank]
            )
            expected_ignorance = float(
                probability_absent * ign0[rank]
                + probability_present * ign1[rank]
            )
            expected_flip = float(
                probability_present if x[feature] < 0.5 else probability_absent
            )
            # Lexicographic ordering avoids dataset-specific mixing weights.
            # Singleton belief is the correctness proxy; later terms resolve
            # ties by resolution, likely correction impact, and uncertainty.
            key = (
                (
                    float(feature_utility.net_correction[feature])
                    if feature_utility is not None else 0.0
                ),
                (
                    float(feature_utility.post_accuracy[feature])
                    if feature_utility is not None else 0.0
                ),
                expected_correctness,
                expected_singleton,
                expected_singleton_belief,
                expected_flip * float(effect[rank]),
                float(current_size) - expected_size,
                current_ignorance - expected_ignorance,
                -float(rank),
            )
            if best_key is None or key > best_key:
                best_key = key
                best = QuerySuggestion(
                    feature=feature,
                    reason=reason,
                    expected_singleton=expected_singleton,
                    expected_set_size=expected_size,
                    expected_ignorance=expected_ignorance,
                    expected_correctness=expected_correctness,
                    expected_flip=expected_flip,
                )
        if best is not None:
            return best

    if global_order is None:
        global_order = np.arange(len(x))
    for feature in np.asarray(global_order, dtype=int):
        if not queried[feature]:
            return QuerySuggestion(
                feature=int(feature),
                reason="reliability_purity_fallback",
                expected_singleton=0.0,
                expected_set_size=float(current_size),
                expected_ignorance=current_ignorance,
            )
    return None


def adaptive_query_trajectory(
    model: LearnedFuzzyTree,
    X_predicted: np.ndarray,
    X_verified: np.ndarray,
    value_absent: np.ndarray,
    value_present: np.ndarray,
    *,
    max_queries: int,
    global_order: np.ndarray | None = None,
) -> dict[int, AdaptiveSnapshot]:
    """Run at most one new query per unresolved sample and round.

    ``X_verified`` supplies the simulated human answers.  It is never passed to
    the query selector; the selector sees an answer only after choosing the
    corresponding feature.
    """
    X = np.asarray(X_predicted, dtype=float).copy()
    X_verified = np.asarray(X_verified, dtype=float)
    if X.shape != X_verified.shape:
        raise ValueError("X_predicted and X_verified must have the same shape")
    max_queries = min(int(max_queries), X.shape[1])
    queried = np.zeros(X.shape, dtype=bool)
    query_counts = np.zeros(len(X), dtype=int)
    reasons: Counter[str] = Counter()
    snapshots = {
        0: AdaptiveSnapshot(X.copy(), query_counts.copy(), dict(reasons)),
    }

    sets, _, ignorance = predict_credal(model, X)
    sizes = sets.sum(1)
    unresolved = np.flatnonzero(sizes > 1)

    for budget in range(1, max_queries + 1):
        for sample in unresolved:
            suggestion = suggest_support_guided_query(
                model,
                X[sample],
                queried[sample],
                value_absent,
                value_present,
                global_order=global_order,
                current_state=(int(sizes[sample]), float(ignorance[sample])),
            )
            if suggestion is None:
                continue
            feature = suggestion.feature
            queried[sample, feature] = True
            X[sample, feature] = X_verified[sample, feature]
            query_counts[sample] += 1
            reasons[suggestion.reason] += 1
        snapshots[budget] = AdaptiveSnapshot(
            X.copy(), query_counts.copy(), dict(reasons),
        )
        if budget < max_queries and len(unresolved):
            updated_sets, _, updated_ignorance = predict_credal(
                model, X[unresolved],
            )
            updated_sizes = updated_sets.sum(1)
            sizes[unresolved] = updated_sizes
            ignorance[unresolved] = updated_ignorance
            unresolved = unresolved[updated_sizes > 1]
    return snapshots


def adaptive_reliability_query_trajectory(
    model: LearnedFuzzyTree,
    X_predicted: np.ndarray,
    X_verified: np.ndarray,
    value_absent: np.ndarray,
    value_present: np.ndarray,
    reliability: np.ndarray,
    *,
    route_reliability: RouteReliability | None = None,
    feature_utility: FeatureInterventionUtility | None = None,
    max_queries: int,
    global_order: np.ndarray | None = None,
) -> dict[int, AdaptiveSnapshot]:
    """Run reliability/purity-aware FERL queries with native early stopping."""
    X = np.asarray(X_predicted, dtype=float).copy()
    X_verified = np.asarray(X_verified, dtype=float)
    reliability = np.asarray(reliability, dtype=float)
    if X.shape != X_verified.shape:
        raise ValueError("X_predicted and X_verified must have the same shape")
    if reliability.shape != (X.shape[1],):
        raise ValueError("reliability must have one entry per feature")
    max_queries = min(int(max_queries), X.shape[1])
    queried = np.zeros(X.shape, dtype=bool)
    query_counts = np.zeros(len(X), dtype=int)
    reasons: Counter[str] = Counter()
    snapshots = {
        0: AdaptiveSnapshot(X.copy(), query_counts.copy(), dict(reasons)),
    }

    sets, _, ignorance = predict_credal(model, X)
    sizes = sets.sum(1)
    unresolved = np.flatnonzero(sizes > 1)
    for budget in range(1, max_queries + 1):
        for sample in unresolved:
            suggestion = suggest_reliability_purity_query(
                model,
                X[sample],
                queried[sample],
                value_absent,
                value_present,
                reliability,
                route_reliability=route_reliability,
                feature_utility=feature_utility,
                global_order=global_order,
                current_state=(int(sizes[sample]), float(ignorance[sample])),
            )
            if suggestion is None:
                continue
            feature = suggestion.feature
            queried[sample, feature] = True
            X[sample, feature] = X_verified[sample, feature]
            query_counts[sample] += 1
            reasons[suggestion.reason] += 1
        snapshots[budget] = AdaptiveSnapshot(
            X.copy(), query_counts.copy(), dict(reasons),
        )
        if budget < max_queries and len(unresolved):
            updated_sets, _, updated_ignorance = predict_credal(
                model, X[unresolved],
            )
            updated_sizes = updated_sets.sum(1)
            sizes[unresolved] = updated_sizes
            ignorance[unresolved] = updated_ignorance
            unresolved = unresolved[updated_sizes > 1]
    return snapshots


def fixed_order_query_trajectory(
    model: LearnedFuzzyTree,
    X_predicted: np.ndarray,
    X_verified: np.ndarray,
    orders: np.ndarray,
    *,
    max_queries: int,
) -> dict[int, AdaptiveSnapshot]:
    """Apply a static order fairly: query unresolved samples only and stop on acceptance."""
    X = np.asarray(X_predicted, dtype=float).copy()
    X_verified = np.asarray(X_verified, dtype=float)
    orders = np.asarray(orders, dtype=int)
    if X.shape != X_verified.shape or orders.shape != X.shape:
        raise ValueError("X_predicted, X_verified, and orders must have the same shape")
    max_queries = min(int(max_queries), X.shape[1])
    query_counts = np.zeros(len(X), dtype=int)
    snapshots = {
        0: AdaptiveSnapshot(X.copy(), query_counts.copy(), {}),
    }
    sets, _, _ = predict_credal(model, X)
    unresolved = np.flatnonzero(sets.sum(1) > 1)
    for budget in range(1, max_queries + 1):
        for sample in unresolved:
            position = query_counts[sample]
            if position >= X.shape[1]:
                continue
            feature = int(orders[sample, position])
            X[sample, feature] = X_verified[sample, feature]
            query_counts[sample] += 1
        snapshots[budget] = AdaptiveSnapshot(X.copy(), query_counts.copy(), {})
        if budget < max_queries and len(unresolved):
            updated_sets, _, _ = predict_credal(model, X[unresolved])
            unresolved = unresolved[updated_sets.sum(1) > 1]
    return snapshots

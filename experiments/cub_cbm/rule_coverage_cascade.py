"""Maximum-rule-coverage LR/FERL cascade with finite-sample safety control.

FERL is used whenever both heads agree and the FERL output passes a strict
eligibility check supplied by the caller (supported route + singleton credal
set in E15).  These agreement decisions are *exactly* prediction-equivalent to
LR and therefore incur zero excess error without a statistical assumption.

Candidate policies may additionally use FERL on disagreements.  They are fit
on a selector split and evaluated on an independent calibration split.  For
each fixed candidate, a one-sided Clopper--Pearson bound controls

    P(cascade is wrong AND LR is correct),

which upper-bounds the cascade's excess risk relative to LR because beneficial
switches can only reduce error.  Bonferroni simultaneous bounds permit choosing
the maximum-coverage certified candidate after seeing all calibration results.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import beta as beta_distribution

from experiments.cub_cbm.ferl_lr_hybrid import (
    HybridHeadOutputs,
    HybridPolicy,
    fit_confidence_gate,
    fit_rule_evidence,
)


@dataclass(frozen=True)
class CoverageCandidate:
    name: str
    kind: str
    policy: HybridPolicy | None = None

    def additional_overrides(self, outputs: HybridHeadOutputs) -> np.ndarray:
        if self.kind == "agreement_only":
            return np.zeros(len(outputs.lr_prediction), dtype=bool)
        if self.policy is None:
            raise ValueError("a disagreement candidate requires a fitted policy")
        override, _ = self.policy.overrides(outputs)
        return override


@dataclass(frozen=True)
class CertifiedCandidate:
    candidate: CoverageCandidate
    calibration_metrics: dict


@dataclass(frozen=True)
class CoverageFrontier:
    candidates: tuple[CertifiedCandidate, ...]
    selected: dict[float, CoverageCandidate]


def ferl_use_mask(
    candidate: CoverageCandidate,
    outputs: HybridHeadOutputs,
    eligible: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return all FERL decisions and the disagreement-only subset."""
    eligible = np.asarray(eligible, dtype=bool)
    if eligible.shape != (len(outputs.lr_prediction),):
        raise ValueError("eligible must have one value per observation")
    agreement = outputs.lr_prediction == outputs.ferl_prediction
    additional = candidate.additional_overrides(outputs) & ~agreement & eligible
    use_ferl = eligible & agreement
    use_ferl |= additional
    return use_ferl, additional


def cascade_metrics(
    candidate: CoverageCandidate,
    outputs: HybridHeadOutputs,
    y: np.ndarray,
    eligible: np.ndarray,
) -> dict[str, float | int]:
    """Rule coverage, accuracy and correctness transitions for one policy."""
    y = np.asarray(y)
    use_ferl, additional = ferl_use_mask(candidate, outputs, eligible)
    lr_correct = outputs.lr_prediction == y
    ferl_correct = outputs.ferl_prediction == y
    prediction = np.where(
        use_ferl, outputs.ferl_prediction, outputs.lr_prediction,
    )
    harm = additional & lr_correct & ~ferl_correct
    benefit = additional & ferl_correct & ~lr_correct
    neutral = additional & ~lr_correct & ~ferl_correct
    agreement_use = use_ferl & ~additional
    deferred = ~use_ferl
    return {
        "accuracy": float(np.mean(prediction == y)),
        "lr_accuracy": float(lr_correct.mean()),
        "ferl_accuracy": float(ferl_correct.mean()),
        "ferl_coverage": float(use_ferl.mean()),
        "ferl_count": int(use_ferl.sum()),
        "agreement_ferl_count": int(agreement_use.sum()),
        "disagreement_override_count": int(additional.sum()),
        "benefit_count": int(benefit.sum()),
        "harm_count": int(harm.sum()),
        "neutral_wrong_count": int(neutral.sum()),
        "net_corrections": int(benefit.sum() - harm.sum()),
        "excess_error": float(np.mean(harm) - np.mean(benefit)),
        "rule_accuracy": float(ferl_correct[use_ferl].mean()) if use_ferl.any() else np.nan,
        "deferred_accuracy": float(lr_correct[deferred].mean()) if deferred.any() else np.nan,
        "eligible_rate": float(np.mean(eligible)),
    }


def clopper_pearson_upper(harm_count: int, n: int, alpha: float) -> float:
    """One-sided exact upper bound for a binomial harmful-switch rate."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    harm_count = int(harm_count)
    if harm_count < 0 or harm_count > n:
        raise ValueError("harm_count must lie in [0, n]")
    if harm_count == n:
        return 1.0
    return float(beta_distribution.ppf(
        1.0 - alpha, harm_count + 1, n - harm_count,
    ))


def fit_coverage_candidates(
    selector_outputs: HybridHeadOutputs,
    y_selector: np.ndarray,
    *,
    confidence_thresholds: tuple[float, ...] = (
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
    ),
    rule_supports: tuple[int, ...] = (1, 2, 4),
    rule_thresholds: tuple[float, ...] = (0.6, 0.7, 0.8),
    random_state: int = 0,
) -> tuple[CoverageCandidate, ...]:
    """Fit fixed disagreement policies on the selector half of validation."""
    y_selector = np.asarray(y_selector)
    candidates = [CoverageCandidate("agreement_only", "agreement_only")]

    gate, constant = fit_confidence_gate(
        selector_outputs.gate_features,
        selector_outputs.lr_prediction,
        selector_outputs.ferl_prediction,
        y_selector,
        random_state=random_state,
    )
    for threshold in confidence_thresholds:
        candidates.append(CoverageCandidate(
            name=f"confidence_p>={threshold:g}",
            kind="confidence",
            policy=HybridPolicy(
                kind="confidence", threshold=threshold,
                confidence_model=gate, confidence_constant=constant,
            ),
        ))

    evidence = fit_rule_evidence(
        selector_outputs.routes,
        selector_outputs.lr_prediction,
        selector_outputs.ferl_prediction,
        y_selector,
    )
    for support in rule_supports:
        for threshold in rule_thresholds:
            candidates.append(CoverageCandidate(
                name=f"route_support>={support}_posterior>={threshold:g}",
                kind="rule_local",
                policy=HybridPolicy(
                    kind="rule_local", threshold=threshold,
                    min_support=support, rule_evidence=evidence,
                ),
            ))
    return tuple(candidates)


def certify_coverage_frontier(
    candidates: tuple[CoverageCandidate, ...],
    calibration_outputs: HybridHeadOutputs,
    y_calibration: np.ndarray,
    eligible_calibration: np.ndarray,
    *,
    epsilons: tuple[float, ...] = (0.0, 0.005, 0.01, 0.02),
    delta: float = 0.05,
) -> CoverageFrontier:
    """Simultaneously certify candidates and maximize coverage for each epsilon."""
    if not candidates:
        raise ValueError("at least one candidate is required")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    n = len(y_calibration)
    simultaneous_alpha = delta / len(candidates)
    certified: list[CertifiedCandidate] = []
    for candidate in candidates:
        metrics = cascade_metrics(
            candidate, calibration_outputs, y_calibration, eligible_calibration,
        )
        if candidate.kind == "agreement_only":
            upper = 0.0  # structural equality with LR, not an estimated bound
        else:
            upper = clopper_pearson_upper(
                int(metrics["harm_count"]), n, simultaneous_alpha,
            )
        metrics = {
            **metrics,
            "harm_rate_upper": upper,
            "simultaneous_alpha": simultaneous_alpha,
        }
        certified.append(CertifiedCandidate(candidate, metrics))

    selected: dict[float, CoverageCandidate] = {}
    for epsilon in epsilons:
        valid = [
            item for item in certified
            if item.calibration_metrics["harm_rate_upper"] <= float(epsilon) + 1e-15
        ]
        if not valid:
            raise RuntimeError("agreement-only candidate should always be certified")
        # Coverage is the objective. Then prefer the smaller risk bound, fewer
        # disagreement switches, and earlier (agreement-only wins exact ties).
        best = max(enumerate(valid), key=lambda pair: (
            pair[1].calibration_metrics["ferl_coverage"],
            -pair[1].calibration_metrics["harm_rate_upper"],
            -pair[1].calibration_metrics["disagreement_override_count"],
            -pair[0],
        ))[1]
        selected[float(epsilon)] = best.candidate
    return CoverageFrontier(tuple(certified), selected)

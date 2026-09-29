"""
Stage B metric functions. Operate on saved proba arrays (model-agnostic). Set-
valued metrics use conformal-APS sets so every model gets a comparable credal
evaluation; native predict_set models (ICDT/NCC, our credal) plug in the same
way in M2.
"""
import numpy as np


def _cols(y, classes):
    idx = {c: i for i, c in enumerate(classes)}
    return np.array([idx[v] for v in y])


def accuracy(proba, y, classes):
    return float((classes[proba.argmax(1)] == y).mean())


def aurc(proba, y, classes, n_levels=20):
    """Area under risk-coverage curve (lower = better selective ranking)."""
    return aurc_from_confidence(proba.max(1), proba.argmax(1), y, classes, n_levels)


def aurc_from_confidence(confidence, predictions, y, classes, n_levels=20):
    """AURC for an externally supplied confidence/uncertainty ranking."""
    conf = np.asarray(confidence); cols = _cols(y, classes)
    predictions = np.asarray(predictions)
    correct = (predictions == cols).astype(float)[np.argsort(-conf)]
    N = len(correct)
    return float(np.mean([1 - correct[:max(int(round(c * N)), 1)].mean()
                          for c in np.linspace(1 / n_levels, 1, n_levels)]))


def acc_at_coverage(proba, y, classes, cov):
    conf = proba.max(1); cols = _cols(y, classes)
    sel = np.argsort(-conf)[:max(int(cov * len(y)), 1)]
    return float((proba.argmax(1)[sel] == cols[sel]).mean())


def _qhat(scores, alpha):
    n = len(scores); k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else float(np.sort(scores)[k - 1])


def _aps_candidate_scores(proba):
    """Deterministic APS score for every candidate label.

    The score of candidate ``c`` is the probability mass assigned to labels
    that are at least as likely as ``c``.  Using ``>=`` makes the inversion
    conservative and permutation-invariant when a predictor emits tied
    probabilities (notably the zero/one probabilities of pure CART leaves).
    """
    proba = np.asarray(proba, dtype=float)
    if proba.ndim != 2:
        raise ValueError("proba must be a two-dimensional array")
    scores = np.empty_like(proba)
    for i, row in enumerate(proba):
        scores[i] = [row[row >= candidate].sum() for candidate in row]
    return scores


def ece(proba, y, classes, n_bins=15):
    """Expected calibration error of the top-class confidence."""
    cols = _cols(y, classes)
    conf = proba.max(1); pred = proba.argmax(1)
    correct = (pred == cols).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for b in range(n_bins):
        m = (conf > edges[b]) & (conf <= edges[b + 1])
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def isotonic_cal(proba_cal, y_cal, proba_test, classes):
    """OvR isotonic calibration of saved probabilities (Stage-B transform)."""
    from sklearn.isotonic import IsotonicRegression
    cols = _cols(y_cal, classes)
    out = np.zeros_like(proba_test)
    for c in range(len(classes)):
        try:
            ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
            ir.fit(proba_cal[:, c], (cols == c).astype(float))
            out[:, c] = ir.transform(proba_test[:, c])
        except Exception:
            out[:, c] = proba_test[:, c]
    out = np.clip(out, 1e-8, None)
    return out / out.sum(1, keepdims=True)


def venn_abers_cal(proba_cal, y_cal, proba_test, classes):
    """Inductive Venn-Abers calibration of saved probabilities (Stage-B transform)."""
    from venn_abers import VennAbersCalibrator
    cols = _cols(y_cal, classes)
    P = VennAbersCalibrator().predict_proba(p_cal=proba_cal, y_cal=cols, p_test=proba_test)
    P = np.asarray(P[0] if isinstance(P, tuple) else P)
    P = np.clip(P, 1e-8, None)
    return P / P.sum(1, keepdims=True)


def _raps_true_scores(P, cols, k_reg, lam):
    """RAPS nonconformity of the true class (APS = the lam=0 special case)."""
    s = np.zeros(len(cols))
    order = np.argsort(-P, axis=1)
    for i in range(len(cols)):
        cum = 0.0
        for rank, c in enumerate(order[i], start=1):
            cum += P[i, c] + lam * max(0, rank - k_reg)
            if c == cols[i]:
                s[i] = cum; break
    return s


def raps_sets(proba_cal, y_cal, proba_test, classes, alpha=0.1, k_reg=1, lam=0.01):
    """Regularized APS (Angelopoulos et al.): APS plus a rank penalty for leaner sets."""
    s = _raps_true_scores(proba_cal, _cols(y_cal, classes), k_reg, lam)
    q = _qhat(s, alpha)
    sets = np.zeros_like(proba_test, dtype=bool)
    order = np.argsort(-proba_test, axis=1)
    for i in range(len(proba_test)):
        cum = 0.0
        for rank, c in enumerate(order[i], start=1):
            cum += proba_test[i, c] + lam * max(0, rank - k_reg)
            sets[i, c] = True
            if cum >= q:
                break
    return sets


def mondrian_sets(proba_cal, y_cal, proba_test, classes, alpha=0.1):
    """Class-conditional (Mondrian) LAC conformal: per-class threshold -> group-
    conditional coverage. Include class c iff 1 - p_c <= qhat_c."""
    cols = _cols(y_cal, classes)
    q = np.array([_qhat(1.0 - proba_cal[cols == c, c], alpha) for c in range(len(classes))])
    return (1.0 - proba_test) <= q[None, :]


def aps_sets(proba_cal, y_cal, proba_test, classes, alpha=0.1):
    """Deterministic Adaptive Prediction Sets at miscoverage level ``alpha``.

    Candidate-wise inversion is essential here.  Stopping at the first class
    whose cumulative probability crosses the conformal threshold drops other
    labels with the same score and can invalidate coverage for discrete
    predictors such as CART.
    """
    cols = _cols(y_cal, classes)
    calibration_scores = _aps_candidate_scores(proba_cal)
    q = _qhat(calibration_scores[np.arange(len(y_cal)), cols], alpha)
    return _aps_candidate_scores(proba_test) <= q + 1e-12


def credal_utils(sets, y, classes):
    """Imprecise-classification metrics over a set-valued prediction."""
    cols = _cols(y, classes); n = len(y)
    sizes = sets.sum(1).astype(float)
    hit = sets[np.arange(n), cols].astype(float)
    det = sizes == 1
    indet = ~det
    discounted = np.zeros(n, dtype=float)
    u65 = np.zeros(n, dtype=float)
    u80 = np.zeros(n, dtype=float)
    rewarded = (hit > 0) & (sizes > 0)
    discounted[rewarded] = 1.0 / sizes[rewarded]
    u65[rewarded] = 1.6 / sizes[rewarded] - 0.6 / sizes[rewarded] ** 2
    u80[rewarded] = 2.2 / sizes[rewarded] - 1.2 / sizes[rewarded] ** 2
    return dict(
        coverage=float(hit.mean()),
        determinacy=float(det.mean()),
        single_acc=float(hit[det].mean()) if det.any() else np.nan,
        set_acc=float(hit[indet].mean()) if indet.any() else np.nan,
        mean_size=float(sizes.mean()),
        disc_acc=float(discounted.mean()),
        u65=float(u65.mean()),
        u80=float(u80.mean()),
    )


def evidential_metrics(mass, y, classes):
    """Metrics for singleton-plus-frame Dempster-Shafer mass functions."""
    mass = np.asarray(mass, dtype=float)
    n_classes = len(classes)
    if mass.ndim != 2 or mass.shape[1] != n_classes + 1:
        raise ValueError("mass must contain C singleton columns plus the full frame")
    singleton = mass[:, :n_classes]
    ignorance = mass[:, -1]
    pignistic = singleton + ignorance[:, None] / n_classes
    cols = _cols(y, classes)
    rows = np.arange(len(y))
    return dict(
        ignorance=float(ignorance.mean()),
        true_belief=float(singleton[rows, cols].mean()),
        true_plausibility=float((singleton[rows, cols] + ignorance).mean()),
        evidence_aurc=aurc_from_confidence(
            1.0 - ignorance, pignistic.argmax(axis=1), y, classes
        ),
    )

"""
Split (inductive) conformal prediction for FERL, with fuzzy-firing-aware scores.

Given a fitted FuzzyCART and a held-out calibration set, produce prediction sets
C(x) with marginal coverage P(y in C(x)) >= 1 - alpha. Four nonconformity scores:

  - 'lac'     : least-ambiguous, s = 1 - p_hat(y|x)               (baseline)
  - 'aps'     : adaptive prediction sets, cumulative sorted probs  (baseline)
  - 'firing'  : firing-aware, s = (1 - p_hat(y|x)) / Phi(x)^gamma  (proposed)
  - 'mondrian': group-conditional LAC, calibrated per firing bin   (proposed)
  - 'plausibility' : DS-native, s = 1 - Pl(y|x) from the belief    (proposed)
                function; conformalises the credal set's upper probability

The firing-aware / Mondrian scores use FERL's total rule-firing Phi(x) so that
prediction sets grow where the rule base barely fires (extrapolation) and stay
tight where strong rules fire. Abstention = set is not a singleton.

Coverage guarantees: marginal (lac/aps/firing); per-firing-bin (mondrian).
"""
from __future__ import annotations

import numpy as np


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample conformal quantile: ceil((n+1)(1-alpha))/n empirical quantile."""
    n = len(scores)
    if n == 0:
        return np.inf
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return np.inf  # not enough calibration data for this alpha -> full sets
    return float(np.sort(scores)[k - 1])


def _aps_candidate_scores(proba: np.ndarray) -> np.ndarray:
    """Tie-conservative cumulative APS score for every candidate class."""
    proba = np.asarray(proba, dtype=float)
    if proba.ndim != 2:
        raise ValueError("proba must be a two-dimensional array")
    scores = np.empty_like(proba)
    for i, row in enumerate(proba):
        scores[i] = [row[row >= candidate].sum() for candidate in row]
    return scores


class ConformalFERL:
    """Split-conformal wrapper around a fitted FuzzyCART classifier."""

    def __init__(self, model, score: str = "firing", gamma: float = 1.0, n_bins: int = 5,
                 rule: str = "dempster"):
        self.model = model
        self.score = score
        self.gamma = gamma
        self.n_bins = n_bins
        self.rule = rule          # DS combination rule for the 'plausibility' score
        self.classes_ = model.classes_
        self._cls_idx = {c: i for i, c in enumerate(self.classes_)}

    # --- helpers -----------------------------------------------------------
    def _proba(self, X):
        return self.model.predict_proba(X)

    def _firing(self, X):
        # Strictly positive, for safe division.
        return np.maximum(self.model.firing_strength(X), 1e-8)

    def _plausibility(self, X, observed_mask=None):
        # Per-class plausibility Pl(c|x) from the DS belief function.
        kw = {} if observed_mask is None else {"observed_mask": observed_mask}
        _, _, pl, _ = self.model.predict_ds(X, rule=self.rule, **kw)
        return pl

    def _true_class_cols(self, y):
        return np.array([self._cls_idx[c] for c in y])

    # --- calibration -------------------------------------------------------
    def calibrate(self, X_cal, y_cal, alpha: float = 0.1):
        self.alpha = alpha
        P = self._proba(X_cal)
        cols = self._true_class_cols(y_cal)
        p_true = P[np.arange(len(y_cal)), cols]

        if self.score == "lac":
            s = 1.0 - p_true
            self.qhat = _conformal_quantile(s, alpha)

        elif self.score == "plausibility":
            pl = self._plausibility(X_cal)
            s = 1.0 - pl[np.arange(len(y_cal)), cols]
            self.qhat = _conformal_quantile(s, alpha)

        elif self.score == "aps":
            # Sum of probabilities of classes at least as likely as the true class.
            candidate_scores = _aps_candidate_scores(P)
            s = candidate_scores[np.arange(len(y_cal)), cols]
            self.qhat = _conformal_quantile(s, alpha)

        elif self.score == "firing":
            # Normalize firing by a calibration reference (95th pct) into (0, 1],
            # so dividing inflates nonconformity for low-firing (extrapolated)
            # points without exploding the score scale.
            phi = self._firing(X_cal)
            self._phi_ref = max(np.quantile(phi, 0.95), 1e-8)
            phin = np.clip(phi / self._phi_ref, 1e-3, 1.0)
            s = (1.0 - p_true) / (phin ** self.gamma)
            self.qhat = _conformal_quantile(s, alpha)

        elif self.score == "mondrian":
            # Per-firing-bin LAC thresholds (group-conditional coverage).
            phi = self._firing(X_cal)
            self._bin_edges = np.quantile(phi, np.linspace(0, 1, self.n_bins + 1))
            self._bin_edges[0] = -np.inf
            self._bin_edges[-1] = np.inf
            s = 1.0 - p_true
            bins = np.digitize(phi, self._bin_edges[1:-1])
            self.qhat_bins = np.array([
                _conformal_quantile(s[bins == b], alpha) for b in range(self.n_bins)
            ])
        else:
            raise ValueError(f"unknown score {self.score!r}")
        return self

    # --- prediction --------------------------------------------------------
    def predict_set(self, X):
        """Return a boolean (n_samples, n_classes) membership matrix for C(x)."""
        P = self._plausibility(X) if self.score == "plausibility" else self.model.predict_proba(X)
        return self._sets_from(P, self._firing(X))

    def predict_set_masked(self, X, observed_mask):
        """Prediction sets with missing features (observed_mask marks observed)."""
        if self.score == "plausibility":
            P = self._plausibility(X, observed_mask=observed_mask)
        else:
            P = self.model.predict_proba(X, observed_mask=observed_mask)
        phi = np.maximum(self.model.firing_strength(X, observed_mask=observed_mask), 1e-8)
        return self._sets_from(P, phi)

    def _sets_from(self, P, phi):
        if self.score in ("lac", "plausibility"):
            return P >= 1.0 - self.qhat
        if self.score == "aps":
            return self._aps_sets(P)
        if self.score == "firing":
            phin = np.clip(phi / self._phi_ref, 1e-3, 1.0)[:, None]
            return (1.0 - P) / (phin ** self.gamma) <= self.qhat
        if self.score == "mondrian":
            bins = np.digitize(phi, self._bin_edges[1:-1])
            thr = self.qhat_bins[bins][:, None]
            return P >= 1.0 - thr
        raise ValueError(self.score)

    def _aps_sets(self, P):
        # Invert the same candidate score used during calibration.  A prefix
        # construction is not equivalent when probabilities tie at the
        # threshold and can under-cover for pure tree leaves.
        return _aps_candidate_scores(P) <= self.qhat + 1e-12


# --- evaluation metrics ----------------------------------------------------
def coverage(sets, y, classes):
    idx = {c: i for i, c in enumerate(classes)}
    cols = np.array([idx[c] for c in y])
    return float(sets[np.arange(len(y)), cols].mean())


def avg_set_size(sets):
    return float(sets.sum(axis=1).mean())


def conditional_coverage_by_firing(sets, y, classes, firing, n_bins=5):
    """Coverage within firing-strength quantile bins; returns (bin_cov, worst)."""
    idx = {c: i for i, c in enumerate(classes)}
    cols = np.array([idx[c] for c in y])
    hit = sets[np.arange(len(y)), cols]
    edges = np.quantile(firing, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    bins = np.digitize(firing, edges[1:-1])
    cov = [float(hit[bins == b].mean()) if np.any(bins == b) else np.nan
           for b in range(n_bins)]
    return cov, float(np.nanmin(cov))


def abstention_rate(sets):
    """Fraction of samples where the model abstains (set is not a singleton)."""
    sizes = sets.sum(axis=1)
    return float((sizes != 1).mean())


def worst_slab_coverage(sets, y, classes, axis, delta=0.1):
    """Worst-slab coverage along a 1-D axis (e.g. firing strength).

    Slides a contiguous window holding a delta-fraction of the (axis-sorted)
    samples and returns the minimum coverage over all windows -- the worst local
    under-coverage. A rigorous conditional-coverage metric (vs fixed bins).
    """
    idx = {c: i for i, c in enumerate(classes)}
    cols = np.array([idx[c] for c in y])
    hit = sets[np.arange(len(y)), cols].astype(float)
    order = np.argsort(axis)
    hit = hit[order]
    n = len(hit)
    w = max(int(delta * n), 5)
    if w >= n:
        return float(hit.mean())
    return float(min(hit[s:s + w].mean() for s in range(0, n - w + 1)))


def selective_risk_coverage(proba, confidence, y, classes, n_levels=20):
    """Risk-coverage curve + AURC for a confidence score (lower AURC = better).

    Point prediction = argmax(proba). Accept the most-confident `cov` fraction;
    risk = error rate among accepted. AURC = mean risk over coverage levels.
    """
    idx = {c: i for i, c in enumerate(classes)}
    cols = np.array([idx[c] for c in y])
    pred = np.argmax(proba, axis=1)
    correct = (pred == cols).astype(float)
    order = np.argsort(-confidence)        # most confident first
    correct = correct[order]
    n = len(correct)
    covs = np.linspace(1.0 / n_levels, 1.0, n_levels)
    risks = []
    for cov in covs:
        k = max(int(round(cov * n)), 1)
        risks.append(1.0 - correct[:k].mean())
    return float(np.mean(risks)), covs, np.array(risks)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    from sklearn.model_selection import train_test_split
    from ferl.core.tree_learning import FuzzyCART
    from ferl.fuzzification.fuzzification_mdlp import learn_partitions_mdlp
    import ferl.eval._eval_accuracy as e

    def load_filtered(name, min_per_class=12):
        X, y = e.load_keel(name)
        keep_classes = [c for c in np.unique(y) if np.sum(y == c) >= min_per_class]
        mask = np.isin(y, keep_classes)
        X, y = X[mask], y[mask]
        # re-encode labels to 0..C-1 (FuzzyCART/metrics assume contiguous)
        _, y = np.unique(y, return_inverse=True)
        return X, y

    ALPHA = 0.1
    # larger datasets -> calibration sets big enough for clean conformal sets
    DATASETS = ["wine", "wisconsin", "pima", "vehicle", "balance"]
    scores = ["lac", "aps", "firing", "mondrian", "plausibility"]
    print(f"Split conformal (50/25/25 train/cal/test), target coverage = {1 - ALPHA:.0%}")
    print("cells = marginal-cover / avg-size / worst-firing-bin-cover\n")
    header = "dataset".ljust(12) + "".join(s.ljust(24) for s in scores)
    print(header)
    agg = {s: {"cov": [], "size": [], "worst": []} for s in scores}
    for name in DATASETS:
        X, y = load_filtered(name)
        Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.5, random_state=0, stratify=y)
        Xcal, Xte, ycal, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=0, stratify=ytmp)
        parts = learn_partitions_mdlp(Xtr, ytr)
        clf = FuzzyCART(parts, max_rules=20)
        clf.fit(Xtr, ytr)
        firing_te = clf.firing_strength(Xte)
        row = name.ljust(12)
        for sc in scores:
            cp = ConformalFERL(clf, score=sc).calibrate(Xcal, ycal, alpha=ALPHA)
            S = cp.predict_set(Xte)
            cov = coverage(S, yte, clf.classes_)
            size = avg_set_size(S)
            _, worst = conditional_coverage_by_firing(S, yte, clf.classes_, firing_te)
            agg[sc]["cov"].append(cov); agg[sc]["size"].append(size); agg[sc]["worst"].append(worst)
            row += f"{cov:.2f}/{size:.2f}/{worst:.2f}".ljust(24)
        print(row)
    print("-" * len(header))
    mrow = "MEAN".ljust(12)
    for sc in scores:
        mrow += (f"{np.mean(agg[sc]['cov']):.2f}/{np.mean(agg[sc]['size']):.2f}/"
                 f"{np.nanmean(agg[sc]['worst']):.2f}").ljust(24)
    print(mrow)

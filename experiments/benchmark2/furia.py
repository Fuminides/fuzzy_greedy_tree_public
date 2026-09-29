"""
FURIA (Huhn & Hullermeier 2009) -- Python reimplementation, the fuzzy-rule SOTA
baseline for FERL. Three additions over RIPPER:

  1. Base learner: RIPPER (via `wittgenstein`), one unordered rule set per class
     (one-vs-rest, no default rule).
  2. Fuzzification: each crisp interval antecedent becomes a trapezoid -- the core
     stays the crisp bound, the support is extended outward to the value that
     maximises fuzzified rule purity on the data covered by the rule's *other*
     antecedents.
  3. Rule stretching: an instance covered by no rule generalises each rule by
     dropping the unsatisfied antecedents; the generalised rule's certainty is
     recomputed on the training set.

Classification: rule activation = product of antecedent memberships x certainty
factor; a class's score is its best rule's activation; predict = argmax.

Exposes the benchmark2 protocol: fit / predict / predict_proba / classes_ /
complexity_ (number of fuzzy rules).
"""
import re
import warnings
import numpy as np

warnings.filterwarnings("ignore")
INF = np.inf


def _parse_interval(val):
    """wittgenstein Cond.val -> (lo, hi) in raw feature scale."""
    v = str(val).strip()
    nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", v)]
    if "<" in v:
        return -INF, nums[0]
    if ">" in v:
        return nums[0], INF
    if len(nums) == 1:                           # single-value bin -> point interval
        return nums[0], nums[0]
    return nums[0], nums[1]                      # "a - b"


def _tri(v, lo, hi, slo, shi):
    """Trapezoidal membership: core [lo,hi] = 1, linear ramps down to [slo,shi]."""
    mu = np.zeros_like(v, dtype=float)
    mu[(v >= lo) & (v <= hi)] = 1.0
    if slo > -INF and lo > slo:                  # lower ramp
        m = (v >= slo) & (v < lo)
        mu[m] = (v[m] - slo) / (lo - slo)
    if shi < INF and shi > hi:                   # upper ramp
        m = (v > hi) & (v <= shi)
        mu[m] = (shi - v[m]) / (shi - hi)
    return mu


class FURIA:
    def __init__(self, random_state=0):
        self.random_state = random_state

    # --- membership --------------------------------------------------------
    def _mu_rule(self, ants, X):
        mu = np.ones(len(X))
        for (j, lo, hi, slo, shi) in ants:
            mu = mu * _tri(X[:, j], lo, hi, slo, shi)
        return mu

    def _crisp_cover(self, ants, X, skip=None):
        """Boolean crisp coverage using the crisp cores of all antecedents but `skip`."""
        m = np.ones(len(X), dtype=bool)
        for i, (j, lo, hi, _, _) in enumerate(ants):
            if i == skip:
                continue
            m &= (X[:, j] >= lo) & (X[:, j] <= hi)
        return m

    # --- fuzzification of one antecedent -----------------------------------
    def _fuzzify(self, ants, i, X, yc):
        """Return (slo, shi) support for antecedent i, maximising fuzzified purity
        on the instances covered by the rule's other (crisp) antecedents."""
        j, lo, hi, _, _ = ants[i]
        cover = self._crisp_cover(ants, X, skip=i)
        Xj, yj = X[cover, j], yc[cover]
        slo, shi = lo, hi
        if Xj.size == 0:
            return slo, shi

        def purity(a, b, s_lo, s_hi):
            mu = _tri(Xj, a, b, s_lo, s_hi)
            tot = mu.sum()
            return (mu[yj].sum() / tot) if tot > 1e-12 else 0.0

        base = purity(lo, hi, lo, hi)
        if hi < INF:                             # extend upper support outward
            cands = np.unique(Xj[Xj > hi])
            best_p, best_s = base, hi
            for s in cands:
                p = purity(lo, hi, lo, s)
                if p > best_p + 1e-9:            # strictly purer -> generalise
                    best_p, best_s = p, s
            shi = best_s
        if lo > -INF:                            # extend lower support outward
            cands = np.unique(Xj[Xj < lo])[::-1]
            best_p, best_s = base, lo
            for s in cands:
                p = purity(lo, hi, s, hi)
                if p > best_p + 1e-9:
                    best_p, best_s = p, s
            slo = best_s
        return slo, shi

    # --- fit ---------------------------------------------------------------
    def fit(self, X, y):
        import pandas as pd
        import wittgenstein as lw
        X = np.asarray(X, float)
        self.classes_, yi = np.unique(y, return_inverse=True)
        self.Xtr_, self.ytr_ = X, yi
        self.prior_ = np.bincount(yi, minlength=len(self.classes_)) / len(yi)
        Xdf = pd.DataFrame(X, columns=[f"f{k}" for k in range(X.shape[1])])
        fidx = {f"f{k}": k for k in range(X.shape[1])}

        self.rules_ = []                         # each: (class_idx, ants, cf)
        for c in range(len(self.classes_)):
            yb = (yi == c).astype(int)
            if yb.sum() < 2:
                continue
            rip = lw.RIPPER(random_state=self.random_state)
            try:
                rip.fit(Xdf, yb)
            except Exception:
                continue
            for rule in rip.ruleset_.rules:
                ants = []
                for cond in rule.conds:
                    lo, hi = _parse_interval(cond.val)
                    ants.append([fidx[cond.feature], lo, hi, lo, hi])
                if not ants:
                    continue
                for i in range(len(ants)):        # fuzzify each antecedent
                    slo, shi = self._fuzzify(ants, i, X, yb.astype(bool))
                    ants[i][3], ants[i][4] = slo, shi
                ants = [tuple(a) for a in ants]
                cf = self._certainty(ants, c)
                self.rules_.append((c, ants, cf))
        self.complexity_ = float(len(self.rules_))
        return self

    def _certainty(self, ants, c, cover_mask=None):
        mu = self._mu_rule(ants, self.Xtr_)
        if cover_mask is not None:
            mu = mu * cover_mask
        reach = mu.sum()
        pos = mu[self.ytr_ == c].sum()
        return (2.0 * self.prior_[c] + pos) / (2.0 + reach)

    # --- predict -----------------------------------------------------------
    def predict_proba(self, X):
        X = np.asarray(X, float)
        C = len(self.classes_)
        scores = np.zeros((len(X), C))
        for c, ants, cf in self.rules_:
            act = self._mu_rule(ants, X) * cf
            scores[:, c] = np.maximum(scores[:, c], act)

        dead = np.where(scores.sum(1) <= 0)[0]    # covered by no rule -> stretch
        for idx in dead:
            x = X[idx:idx + 1]
            best = np.zeros(C)
            for c, ants, _ in self.rules_:
                mems = np.array([_tri(x[:, j], lo, hi, slo, shi)[0]
                                 for (j, lo, hi, slo, shi) in ants])
                keep = mems > 0
                if 0 < keep.sum() < len(ants):    # a proper generalisation
                    sub = [ants[i] for i in range(len(ants)) if keep[i]]
                    m = mems[keep].prod()
                    cf_s = self._certainty(sub, c)
                    best[c] = max(best[c], m * cf_s)
            scores[idx] = best

        rs = scores.sum(1)
        out = np.where(rs[:, None] > 0, scores / np.clip(rs[:, None], 1e-12, None),
                       self.prior_[None, :])
        return out

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]

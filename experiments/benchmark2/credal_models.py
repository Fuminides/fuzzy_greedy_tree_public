"""
Credal / imprecise classifiers (Python reimplementations; no WEKA), behind the
common protocol: fit / predict / predict_proba (precise point estimate, for the
proba-based metrics) / predict_set (native set-valued output) / classes_ /
complexity_.

  CredalTree(imprecise=False) = Credal-C4.5 : tree split by Imprecise Information
      Gain (max entropy over the IDM credal set); point prediction.
  CredalTree(imprecise=True)  = ICDT        : same tree, leaves return the set of
      credal-non-dominated classes (interval dominance on the leaf IDM intervals).
  NCC = Naive Credal Classifier             : naive Bayes with IDM + interval
      dominance -> set-valued.

Shared crisp discretization (quantile KBins) so all credal methods use the same
preprocessing (see BENCHMARK_PLAN discretization protocol).
"""
import warnings
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

warnings.filterwarnings("ignore")
S = 1.0          # IDM hyperparameter
N_BINS = 4


def _discretize_fit(X, n_bins=N_BINS):
    disc = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile",
                            subsample=None)
    Xd = disc.fit_transform(X).astype(int)
    return disc, Xd


def idm_max_entropy(counts, s=S):
    """Maximum entropy over the IDM credal set (water-fill the s mass to flatten)."""
    n = np.asarray(counts, float); N = n.sum()
    if N <= 0:
        return float(np.log(len(n)))
    a = n.copy(); rem = float(s)
    for _ in range(2 * len(n) + 5):
        if rem <= 1e-12:
            break
        m = a.min(); cand = a <= m + 1e-12; k = int(cand.sum())
        higher = a[a > m + 1e-12]
        nxt = higher.min() if higher.size else m + rem / k
        step = min(nxt - m, rem / k)
        if step <= 1e-15:
            a += rem / len(a); rem = 0; break
        a[cand] += step; rem -= step * k
    p = a / (N + s)
    p = p / p.sum()
    return float(-(p * np.log(np.clip(p, 1e-12, 1.0))).sum())


def nondominated_intervals(lo_log, hi_log):
    """Interval (credal) dominance in log space: keep c unless some c2 has
    lower-bound > c's upper-bound."""
    K = len(lo_log); keep = np.ones(K, bool)
    for c in range(K):
        for c2 in range(K):
            if c2 != c and lo_log[c2] > hi_log[c] + 1e-9:
                keep[c] = False; break
    return keep


class CredalTree:
    def __init__(self, s=S, n_bins=N_BINS, max_depth=6, min_samples=10, imprecise=False):
        self.s = s; self.n_bins = n_bins; self.max_depth = max_depth
        self.min_samples = min_samples; self.imprecise = imprecise

    def _counts(self, y_idx):
        return np.bincount(y_idx, minlength=self.K).astype(float)

    def fit(self, X, y):
        self.disc, Xd = _discretize_fit(X, self.n_bins)
        self.classes_, yi = np.unique(y, return_inverse=True)
        self.K = len(self.classes_); self.D = X.shape[1]
        self._n_leaves = 0
        self.root_ = self._build(Xd, yi, 0)
        self.complexity_ = self._n_leaves
        return self

    def _build(self, Xd, yi, depth):
        counts = self._counts(yi)
        node = {"leaf": True, "counts": counts, "pred": int(counts.argmax())}
        if depth >= self.max_depth or len(yi) < self.min_samples or (counts > 0).sum() <= 1:
            self._n_leaves += 1; return node
        parentH = idm_max_entropy(counts, self.s)
        best_gain, best_j = 0.0, -1
        for j in range(self.D):
            vals = np.unique(Xd[:, j])
            if len(vals) < 2:
                continue
            childH = 0.0
            for v in vals:
                m = Xd[:, j] == v
                childH += (m.sum() / len(yi)) * idm_max_entropy(self._counts(yi[m]), self.s)
            gain = parentH - childH
            if gain > best_gain + 1e-9:
                best_gain, best_j = gain, j
        if best_j < 0:
            self._n_leaves += 1; return node
        node = {"leaf": False, "feature": best_j, "counts": counts, "children": {}}
        for v in np.unique(Xd[:, best_j]):
            m = Xd[:, best_j] == v
            node["children"][int(v)] = self._build(Xd[m], yi[m], depth + 1)
        return node

    def _leaf(self, x):
        node = self.root_
        while not node["leaf"]:
            v = int(x[node["feature"]])
            node = node["children"].get(v)
            if node is None:               # unseen bin value -> stop at parent
                return None
        return node

    def _leaf_counts(self, X):
        Xd = self.disc.transform(X).astype(int)
        out = []
        for i in range(len(X)):
            leaf = self._leaf(Xd[i])
            out.append(leaf["counts"] if leaf is not None else self.root_["counts"])
        return np.array(out)

    def predict_proba(self, X):
        c = self._leaf_counts(X) + 1.0
        return c / c.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]

    def predict_set(self, X):
        sets = np.zeros((len(X), self.K), bool)
        for i, counts in enumerate(self._leaf_counts(X)):
            N = counts.sum()
            lo = np.log(np.clip(counts, 1e-12, None)) - np.log(N + self.s)
            hi = np.log(counts + self.s) - np.log(N + self.s)
            sets[i] = nondominated_intervals(lo, hi)
        return sets


class NCC:
    def __init__(self, s=S, n_bins=N_BINS):
        self.s = s; self.n_bins = n_bins

    def fit(self, X, y):
        self.disc, Xd = _discretize_fit(X, self.n_bins)
        self.classes_, yi = np.unique(y, return_inverse=True)
        self.K = len(self.classes_); self.D = X.shape[1]
        self.prior_ = np.bincount(yi, minlength=self.K).astype(float)
        self.nvals_ = [int(Xd[:, j].max()) + 1 for j in range(self.D)]
        self.cond_ = []
        for j in range(self.D):
            t = np.zeros((self.K, self.nvals_[j]))
            for c in range(self.K):
                t[c] = np.bincount(Xd[yi == c, j], minlength=self.nvals_[j])
            self.cond_.append(t)
        self.complexity_ = self.K * int(sum(self.nvals_))      # #conditional params
        return self

    def _bounds(self, xd):
        N = self.prior_.sum(); s = self.s
        lo = np.log(np.clip(self.prior_, 1e-12, None)) - np.log(N + s)
        hi = np.log(self.prior_ + s) - np.log(N + s)
        for j in range(self.D):
            v = min(int(xd[j]), self.nvals_[j] - 1)
            ncv = self.cond_[j][:, v]
            lo = lo + np.log(np.clip(ncv, 1e-12, None)) - np.log(self.prior_ + s)
            hi = hi + np.log(ncv + s) - np.log(self.prior_ + s)
        return lo, hi

    def predict_proba(self, X):
        Xd = self.disc.transform(X).astype(int)
        N = self.prior_.sum()
        out = np.zeros((len(X), self.K))
        for i in range(len(X)):
            lp = np.log(self.prior_ + 1.0) - np.log(N + self.K)
            for j in range(self.D):
                v = min(int(Xd[i, j]), self.nvals_[j] - 1)
                lp = lp + np.log(self.cond_[j][:, v] + 1.0) - np.log(self.prior_ + self.nvals_[j])
            lp -= lp.max(); p = np.exp(lp); out[i] = p / p.sum()
        return out

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]

    def predict_set(self, X):
        Xd = self.disc.transform(X).astype(int)
        sets = np.zeros((len(X), self.K), bool)
        for i in range(len(X)):
            lo, hi = self._bounds(Xd[i])
            sets[i] = nondominated_intervals(lo, hi)
        return sets

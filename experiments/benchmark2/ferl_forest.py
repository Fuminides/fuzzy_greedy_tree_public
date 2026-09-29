"""
FERLForest -- bagging ensemble of FERL-compact trees (the decision-tree -> random-
forest analogue for fuzzy rule trees). Diversity from stratified bootstrap +
random feature subsets; members combine in one of two ways:

  combine='avg'       : average the members' pignistic/soft probabilities (the
                        RandomForest analogue).
  combine='evidence'  : Dempster-combine the members' DS masses *across members*.
                        Unlike the within-tree case (nested, dependent rules ->
                        double-counting, see diary §7), separate bootstrap members
                        are genuinely independent sources, so the product rule is
                        justified. Yields a native credal set + ignorance.

Exposes the common benchmark2 protocol: fit / predict / predict_proba /
predict_set (evidence mode) / classes_ / complexity_.
"""
import numpy as np

EPS = 1e-12


class FERLForest:
    def __init__(self, n_estimators=25, max_features=0.7, combine="avg",
                 top_p=None, random_state=0):
        self.n_estimators = n_estimators
        self.max_features = max_features          # fraction of features per member
        self.combine = combine
        self.top_p = top_p                        # optional per-member top-p routing
        self.random_state = random_state

    # --- fit ---------------------------------------------------------------
    def fit(self, X, y):
        from ferl.pipeline.ferl_pipeline import make
        from sklearn.utils import resample
        X = np.asarray(X, float)
        self.classes_ = np.unique(y)
        self._cidx = {c: i for i, c in enumerate(self.classes_)}
        D = X.shape[1]
        m = max(1, int(round(self.max_features * D)))
        rng = np.random.RandomState(self.random_state)
        self.members_, self.feats_ = [], []
        for b in range(self.n_estimators):
            feats = rng.choice(D, m, replace=False)
            # stratified bootstrap -> every class present in every member
            Xb, yb = resample(X[:, feats], y, replace=True, stratify=y,
                              random_state=self.random_state + b)
            tree = make("ferl-compact", random_state=self.random_state + b).fit(Xb, yb).tree_
            self.members_.append(tree)
            self.feats_.append(feats)
        self.complexity_ = float(sum(t.tree_rules for t in self.members_))
        return self

    # --- per-member read-outs aligned to the global class set --------------
    def _aligned(self, tree, Xi, masses):
        """Return per-member arrays aligned to self.classes_.
        masses=False -> proba (N,C); masses=True -> (Q_c (N,C), Q_theta (N,))."""
        cols = np.array([self._cidx[c] for c in tree.classes_])
        if not masses:
            P = tree.predict_proba(Xi)
            out = np.zeros((Xi.shape[0], len(self.classes_)))
            out[:, cols] = P
            return out
        _, bel, pl, _ = tree.predict_ds(Xi, top_p=self.top_p)
        mtheta = np.clip((pl - bel), 0.0, 1.0).mean(1)          # singleton+Theta -> equal cols
        Qc = np.full((Xi.shape[0], len(self.classes_)), 0.0)
        Qc[:, cols] = pl                                        # commonality Q({c}) = pl(c)
        # classes absent from this member: fully ignorant -> Q({c}) = Q(Theta)
        absent = np.setdiff1d(np.arange(len(self.classes_)), cols)
        if absent.size:
            Qc[:, absent] = mtheta[:, None]
        return Qc, mtheta

    # --- combination -------------------------------------------------------
    def _combined_masses(self, X):
        """Dempster product across members -> (m_c (N,C), m_theta (N,)).

        Accumulated in log space: the product of ~n_estimators sub-1 commonalities
        underflows to 0 in linear space. Dempster is invariant to per-source scaling,
        so a per-sample shift before exp is exact, not an approximation."""
        X = np.asarray(X, float)
        logQc = np.zeros((X.shape[0], len(self.classes_)))
        logQt = np.zeros(X.shape[0])
        for tree, feats in zip(self.members_, self.feats_):
            qc, qt = self._aligned(tree, X[:, feats], masses=True)
            logQc += np.log(np.clip(qc, EPS, None))
            logQt += np.log(np.clip(qt, EPS, None))
        shift = np.maximum(logQc.max(1), logQt)                 # scale cancels in norm
        Qc = np.exp(logQc - shift[:, None])
        Qt = np.exp(logQt - shift)
        m_c = np.clip(Qc - Qt[:, None], 0.0, None)
        tot = np.where(m_c.sum(1) + Qt <= 0, 1.0, m_c.sum(1) + Qt)
        return m_c / tot[:, None], Qt / tot

    def predict_proba(self, X):
        if self.combine == "avg":
            X = np.asarray(X, float)
            acc = np.zeros((X.shape[0], len(self.classes_)))
            for tree, feats in zip(self.members_, self.feats_):
                acc += self._aligned(tree, X[:, feats], masses=False)
            P = acc / len(self.members_)
            return P / np.clip(P.sum(1, keepdims=True), EPS, None)
        m_c, m_theta = self._combined_masses(X)              # evidence: pignistic betp
        return m_c + m_theta[:, None] / len(self.classes_)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]

    def predict_set(self, X):
        """Native credal set (evidence mode): interval dominance on [Bel, Pl]."""
        m_c, m_theta = self._combined_masses(X)
        bel = m_c
        pl = m_c + m_theta[:, None]
        return pl >= bel.max(1, keepdims=True) - EPS

"""LearnedFuzzyTree -- the max-accuracy, performance-oriented sibling of FuzzyCART.

Where FuzzyCART fuzzifies on a fixed partition (compact, ~5 rules) and combines
by soft-vote-over-all-nodes (tuned for shallow trees), this grows a deep binary
fuzzy tree whose split *locations are learned* (CART-style, weighted-Gini optimal
per node) and rendered as soft ramps whose half-width is the bootstrap uncertainty
of the cut (stable cut -> near-crisp, unstable cut -> wide/fuzzy). Inference is a
leaf-only soft vote.

Validated (30 KEEL datasets): beats CART at ~half the leaves and generalizes
better than crisp CART at equal depth (the soft boundary acts as a regularizer).
Trades FuzzyCART's compactness (~90 vs ~5 rules) for accuracy; use FuzzyCART's
``split_mode='learned'`` performance config instead when the native credal / UQ
output and compactness matter more than the last accuracy points.

sklearn-style: ``fit`` / ``predict`` / ``predict_proba`` / ``classes_``.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

EPS = 1e-9


def _best_cut(x, y_oh, w, parent, W, criterion="gini"):
    """Optimal crisp threshold on one feature. Returns (gain, thr).

    ``criterion='gini'`` (default) maximises weighted-Gini impurity reduction.
    ``criterion='cci'`` maximises the Complete Classification Index, i.e. the
    accuracy improvement of the split's two majority-class predictions over the
    parent's single majority. Because the parent baseline is constant within a
    node, CCI selection reduces to maximising ``max_c l_c + max_c r_c`` (the two
    children's majority mass); a tiny Gini term breaks ties so the tree still
    grows through nodes where no single split flips the majority -- the
    many-class regime CCI is meant to help. Vectorized over candidate cuts."""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    cum = np.cumsum(y_oh[order] * w[order, None], axis=0)      # (n, C)
    Wl_arr = np.cumsum(w[order])                              # (n,)
    tot = cum[-1]                                             # (C,)
    bnd = np.flatnonzero(xs[:-1] != xs[1:])                   # candidate split rows
    if bnd.size == 0:
        return 0.0, None
    Wl = Wl_arr[bnd]
    Wr = W - Wl
    ok = (Wl > EPS) & (Wr > EPS)
    if not ok.any():
        return 0.0, None
    bnd, Wl, Wr = bnd[ok], Wl[ok], Wr[ok]
    l = cum[bnd]                                              # (m, C)
    r = tot[None, :] - l
    gini_l = 1.0 - ((l / Wl[:, None]) ** 2).sum(1)
    gini_r = 1.0 - ((r / Wr[:, None]) ** 2).sum(1)
    gini_gain = parent - (Wl / W * gini_l + Wr / W * gini_r)  # (m,)
    if criterion == "cci":
        cci = (l.max(1) + r.max(1) - tot.max()) / W          # >= 0 always
        gain = cci + 1e-6 * np.clip(gini_gain, 0.0, None)
    else:
        gain = gini_gain
    k = int(np.argmax(gain))
    if gain[k] <= 0.0:
        return 0.0, None
    i = bnd[k]
    return float(gain[k]), 0.5 * (xs[i] + xs[i + 1])


def _ramp(x, center, h):
    """Membership of the 'below' branch: 1 at center-h, 0 at center+h."""
    return np.clip((center + h - x) / (2.0 * h), 0.0, 1.0)


def _ds_combine(M, cons, names, C, rule="dempster", top_p=None, support=None,
                beta=10.0, reliability=None):
    """Dempster-Shafer combination over rule nodes -- the same singleton+Theta
    mass math as FuzzyCART.predict_ds, so the combination-rule study transfers to
    the deep learned tree. Each node n is a mass m_n({c}) = mu_n p_n(c),
    m_n(Theta) = 1 - mu_n. Chain structure (for 'hybrid'/'incremental') is read
    from the '_'-delimited node names. ``reliability`` (per-node rho in [0,1],
    e.g. from finetune_reliability) discounts firing toward Theta: mu -> rho*mu.
    ``rule='mixture'`` is a convex (additive, not multiplicative) combination:
    m(c) = sum_n mu_n rho_n cons_n(c), valid only when the combined nodes form a
    true partition of unity (leaves_only=True) -- it is immune to the
    shared-ancestor double-counting that Dempster's independent-source product
    incurs even among leaves, since it never multiplies one node's evidence
    against another's.
    Returns (betp, bel, pl, ignorance)."""
    N = M.shape[0]
    if M.shape[1] == 0:
        return (np.full((N, C), 1.0 / C), np.zeros((N, C)), np.ones((N, C)), np.ones(N))

    if top_p is not None:
        order = np.argsort(-cons, axis=1)
        sc = np.take_along_axis(cons, order, axis=1)
        before = np.cumsum(sc, axis=1) - sc
        kept = np.zeros_like(cons)
        np.put_along_axis(kept, order, np.where(before < top_p, sc, 0.0), axis=1)
        rtp = kept.sum(1)
        cons = kept / np.clip(rtp[:, None], 1e-12, None)
        M = M * rtp[None, :]

    if reliability is not None:
        M = M * np.clip(np.asarray(reliability, float), 0.0, 1.0)[None, :]

    M_raw = M.copy()
    one_minus = 1.0 - M
    term = M[:, :, None] * cons[None, :, :] + one_minus[:, :, None]        # (N,K,C)

    def _cautious_mass(cols):
        w_nc = one_minus[:, cols, None] / np.clip(term[:, cols, :], 1e-12, None)
        w_c = np.clip(w_nc.min(axis=1), 1e-12, 1.0)
        inv = 1.0 / w_c
        S = (1.0 - C) + inv.sum(axis=1)
        return (inv - 1.0) / S[:, None], 1.0 / S

    if rule in ("incremental", "incremental_local"):
        # Residual (incremental) evidence: keep each chain-root's full belief and
        # replace each descendant by its commonality residual over its immediate
        # parent, so a clean refinement counts the shared ancestor ONCE instead of
        # double-counting it (Dempster). Contradicting children keep full mass, so
        # the rule interpolates specificity <-> Dempster. 'incremental_local'
        # additionally discounts a residual node by the conditional firing
        # mu_k/mu_parent so the parent's firing is not re-counted in the increment.
        K = M.shape[1]
        local = (rule == "incremental_local")
        supp = np.ones(K) if support is None else np.asarray(support, float)
        r_inc = supp / (supp + beta)                                       # reliability rho_n
        t = np.clip(1.0 - r_inc, 1e-9, 1.0)                                # (K,) Theta mass
        a = r_inc[:, None] * cons                                          # (K,C) singleton mass
        g = a + t[:, None]                                                 # (K,C) base commonality
        # immediate active parent of each node (longest name that is a prefix)
        parent = np.full(K, -1, dtype=int)
        for kk, nk in enumerate(names):
            blen = -1
            for j, nj in enumerate(names):
                if j != kk and nk.startswith(nj + "_") and len(nj) > blen:
                    parent[kk], blen = j, len(nj)
        a_eff, t_eff = a.copy(), t.copy()
        is_res = np.zeros(K, dtype=bool)
        for kk in range(K):
            p = parent[kk]
            if p < 0:
                continue                                                   # chain-root: full mass
            q_res = g[kk] / np.clip(g[p], 1e-12, None)
            q_res_th = float(np.clip(t[kk] / t[p], 0.0, 1.0))
            a_res = q_res - q_res_th
            if np.all(a_res >= -1e-9):                                      # additive refinement
                a_eff[kk] = np.clip(a_res, 0.0, None)
                t_eff[kk] = q_res_th
                is_res[kk] = True                                          # else keep full base mass
        mu = M_raw.copy()
        if local and is_res.any():
            pr = parent[is_res]
            mu[:, is_res] = np.clip(M_raw[:, is_res] / np.clip(M_raw[:, pr], 1e-12, None), 0.0, 1.0)
        f_th = 1.0 - mu * (1.0 - t_eff[None, :])                           # (N,K)
        f_c = mu[:, :, None] * a_eff[None, :, :] + f_th[:, :, None]        # (N,K,C)
        Qc = np.prod(f_c, axis=1)
        Qt = np.prod(f_th, axis=1)
        m_c_un = np.clip(Qc - Qt[:, None], 0.0, None)
        total = np.where(m_c_un.sum(1) + Qt <= 0, 1.0, m_c_un.sum(1) + Qt)
        m_c, m_theta = m_c_un / total[:, None], Qt / total
    elif rule == "mixture":
        supp = np.ones(M.shape[1]) if support is None else np.asarray(support, float)
        rho = supp / (supp + beta)
        m_c = M @ (rho[:, None] * cons)                                    # (N,C)
        m_theta = np.clip(1.0 - m_c.sum(1), 0.0, 1.0)                      # unassigned -> Theta
    elif rule == "cautious":
        m_c, m_theta = _cautious_mass(np.arange(M.shape[1]))
    elif rule == "hybrid":
        leaves = [i for i, n in enumerate(names)
                  if not any(o != n and o.startswith(n + "_") for o in names)]
        Qc, Qt = np.ones((N, C)), np.ones(N)
        for li in leaves:
            ln = names[li]
            anc = [j for j, n in enumerate(names) if ln == n or ln.startswith(n + "_")]
            mc, mt = _cautious_mass(anc)
            Qc *= (mc + mt[:, None])
            Qt *= mt
        m_c_un = np.clip(Qc - Qt[:, None], 0.0, None)
        total = np.where(m_c_un.sum(1) + Qt <= 0, 1.0, m_c_un.sum(1) + Qt)
        m_c, m_theta = m_c_un / total[:, None], Qt / total
    else:                                                                  # dempster
        Qc = np.prod(term, axis=1)
        Qtheta = np.prod(one_minus, axis=1)
        m_c_un = np.clip(Qc - Qtheta[:, None], 0.0, None)
        total = np.where(m_c_un.sum(1) + Qtheta <= 0, 1.0, m_c_un.sum(1) + Qtheta)
        m_c, m_theta = m_c_un / total[:, None], Qtheta / total

    return m_c + m_theta[:, None] / C, m_c, m_c + m_theta[:, None], m_theta


class LearnedFuzzyTree(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=12, min_leaf_w=2.0, n_boot=25,
                 width="bootstrap", random_state=0, bounded_support=True, oob_margin=1.0,
                 criterion="gini"):
        self.max_depth = max_depth
        self.min_leaf_w = min_leaf_w
        self.n_boot = n_boot
        self.width = width            # 'bootstrap' or a float c (h = c * feature-std)
        self.criterion = criterion    # 'gini' (default) or 'cci' (many-class aware)
        self.random_state = random_state
        # Bounded-support inference (DEFAULT): gate each split's membership to the
        # node's training-data range [lo,hi] on that feature (extended by
        # oob_margin * range on each side), decaying to 0 beyond. In-distribution
        # points (inside [lo,hi]) are unaffected -> accuracy/probabilities/conformal
        # sets identical to unbounded (verified 30 ds: acc delta -4e-5, proba L1
        # 1.4e-3); far-OOD points fall outside -> zero firing -> ignorance/Phi
        # returns a strong geometric-OOD signal (AUROC ~0.99) and the credal set
        # inflates (OOD-aware abstention). Tree STRUCTURE is identical either way
        # (gate applies at inference only), so a fitted tree can be toggled.
        # Set bounded_support=False for the pre-bounding (OOD-blind) behaviour.
        self.bounded_support = bounded_support
        self.oob_margin = oob_margin

    def fit(self, X, y):
        X = np.asarray(X, float)
        self.classes_, yi = np.unique(y, return_inverse=True)
        self.C = len(self.classes_)
        self._y_oh = np.eye(self.C)[yi]
        self._X = X
        self._rng = np.random.default_rng(self.random_state)
        self.n_leaves_ = 0
        self.root_ = self._build(np.ones(len(X)), 0)
        self.complexity_ = float(self.n_leaves_)
        # release training refs
        self._X = None
        self._y_oh = None
        return self

    def _leaf(self, w):
        self.n_leaves_ += 1
        cnt = (self._y_oh * w[:, None]).sum(0) + 1.0          # Laplace
        return {"leaf": True, "dist": cnt / cnt.sum(), "support": float(w.sum())}

    def _build(self, w, depth):
        W = w.sum()
        if depth >= self.max_depth or W < 2 * self.min_leaf_w:
            return self._leaf(w)
        p = (self._y_oh * w[:, None]).sum(0) / W
        parent = 1.0 - (p ** 2).sum()
        if parent < 1e-9:
            return self._leaf(w)
        region = w > 1e-6
        n_eff = int(region.sum())
        if n_eff < 4:
            return self._leaf(w)
        Xr, yr, wr = self._X[region], self._y_oh[region], w[region]
        Wr = wr.sum()
        pr = 1.0 - (((yr * wr[:, None]).sum(0) / Wr) ** 2).sum()
        # best feature by weighted-Gini gain
        bf, bg, bthr = -1, 0.0, None
        for f in range(self._X.shape[1]):
            g, thr = _best_cut(Xr[:, f], yr, wr, pr, Wr, self.criterion)
            if thr is not None and g > bg:
                bf, bg, bthr = f, g, thr
        if bf < 0:
            return self._leaf(w)
        xr = Xr[:, bf]
        if self.width == "bootstrap":
            pw = wr / Wr
            thetas = []
            for _ in range(self.n_boot):
                idx = self._rng.choice(n_eff, n_eff, p=pw)
                _g, tb = _best_cut(xr[idx], yr[idx], np.ones(n_eff), 1.0, float(n_eff), self.criterion)
                if tb is not None:
                    thetas.append(tb)
            if len(thetas) < 2:
                return self._leaf(w)
            center, h = float(np.mean(thetas)), float(np.std(thetas))
        else:
            mean = (wr * xr).sum() / Wr
            std = float(np.sqrt((wr * (xr - mean) ** 2).sum() / Wr))
            center, h = bthr, float(self.width) * std
        h = max(h, 1e-3 * (float(xr.max() - xr.min()) + EPS))
        ml = _ramp(self._X[:, bf], center, h)
        wl, wR = w * ml, w * (1.0 - ml)
        if wl.sum() < self.min_leaf_w or wR.sum() < self.min_leaf_w:
            return self._leaf(w)
        cnt = (self._y_oh * w[:, None]).sum(0) + 1.0          # internal-node consequent
        return {"leaf": False, "f": bf, "center": center, "h": h,
                "lo": float(xr.min()), "hi": float(xr.max()),      # node data range on bf
                "dist": cnt / cnt.sum(), "support": float(W),
                "L": self._build(wl, depth + 1), "R": self._build(wR, depth + 1)}

    def _split(self, node, X):
        """Return (left, right) routing weights for a node, gated to the node's
        training-data range on its split feature when bounded_support is on."""
        xf = X[:, node["f"]]
        ml = _ramp(xf, node["center"], node["h"])
        if not self.bounded_support:
            return ml, 1.0 - ml
        lo, hi = node["lo"], node["hi"]
        g = self.oob_margin * (hi - lo) + EPS
        gate = np.clip((xf - (lo - g)) / g, 0.0, 1.0) * np.clip(((hi + g) - xf) / g, 0.0, 1.0)
        return ml * gate, (1.0 - ml) * gate       # OOD (outside [lo-g,hi+g]) -> both ~0

    def _accumulate(self, node, X, m, out):
        if node["leaf"]:
            out += m[:, None] * node["dist"]
            return
        left, right = self._split(node, X)
        self._accumulate(node["L"], X, m * left, out)
        self._accumulate(node["R"], X, m * right, out)

    def predict_proba(self, X):
        X = np.asarray(X, float)
        out = np.zeros((len(X), self.C))
        self._accumulate(self.root_, X, np.ones(len(X)), out)
        s = out.sum(1)
        out[s <= EPS] = 1.0 / self.C
        return out / out.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]

    def n_rules(self):
        return self.n_leaves_

    # --- credal / Dempster-Shafer read-out (parity with FuzzyCART.predict_ds) ---
    def _collect(self, node, X, m, name, acc):
        """Accumulate path membership to every non-root node; acc gets
        (name, membership_vector, consequent, support). Top node is excluded root."""
        if name != "r":
            acc.append((name, m, node["dist"], node["support"]))
        if not node["leaf"]:
            left, right = self._split(node, X)
            self._collect(node["L"], X, m * left, name + "_0", acc)
            self._collect(node["R"], X, m * right, name + "_1", acc)

    def node_activation_matrix(self, X):
        """(M, cons, names, support) over all non-root nodes -- M[i,k] path
        membership, cons[k] consequent, names[k] '_'-delimited hierarchical id,
        support[k] fuzzy training support (for the incremental rule's reliability)."""
        X = np.asarray(X, float)
        acc = []
        self._collect(self.root_, X, np.ones(len(X)), "r", acc)
        if not acc:
            return np.zeros((len(X), 0)), np.zeros((0, self.C)), [], np.zeros(0)
        names = [a[0] for a in acc]
        M = np.stack([a[1] for a in acc], axis=1)
        cons = np.stack([a[2] for a in acc], axis=0)
        support = np.array([a[3] for a in acc], float)
        return M, cons, names, support

    def leaf_mask(self, names):
        """Boolean mask selecting leaf nodes (no descendant) from a names list."""
        return np.array([not any(o != n and o.startswith(n + "_") for o in names)
                         for n in names])

    def node_depths(self, names):
        """Path length (number of splits) of each node from its name."""
        return np.array([n.count("_") for n in names], float)

    def predict_ds(self, X, rule="dempster", top_p=None, leaves_only=False, reliability_vec=None):
        """Dempster-Shafer credal read-out over the tree's nodes. ``rule`` in
        {dempster, cautious, hybrid, incremental, incremental_local, mixture};
        ``top_p``
        applies nucleus routing first; ``leaves_only`` combines only leaf rules;
        ``reliability_vec`` (aligned to the post-filter node order) discounts each
        node's firing toward Theta. Returns (betp, bel, pl, ignorance)."""
        M, cons, names, support = self.node_activation_matrix(X)
        if leaves_only and M.shape[1] > 0:
            keep = np.flatnonzero(self.leaf_mask(names))
            M, cons, support = M[:, keep], cons[keep], support[keep]
            names = [names[i] for i in keep]
        return _ds_combine(M, cons, names, self.C, rule=rule, top_p=top_p,
                           support=support, reliability=reliability_vec)

    def predict_dirichlet(self, X, u_floor=1e-3, **ds_kwargs):
        """Subjective-Logic Dirichlet params: alpha_c = C*Bel(c)/m(Theta)+1."""
        _, bel, _, m_theta = self.predict_ds(X, **ds_kwargs)
        u = np.clip(m_theta, u_floor, 1.0)
        return self.C * bel / u[:, None] + 1.0

    def predict_set(self, X):
        """Native (calibration-free) credal set: leaves-only Dempster, kept by
        interval dominance (plausibility >= max belief). Singleton = confident,
        |set|>1 = abstain. With bounded_support (default) the set inflates on
        low-density / OOD inputs. Boolean (n_samples, n_classes)."""
        _, bel, pl, _ = self.predict_ds(X, rule="dempster", leaves_only=True)
        return pl >= bel.max(1, keepdims=True) - 1e-12


class LearnedFuzzyTreeCV(LearnedFuzzyTree):
    """FERL-deep with its band-width rule chosen by inner cross-validation.

    Each candidate in ``width_grid`` (by default the bootstrap rule and half-widths
    of 0.25, 0.5 and 1 node-weighted standard deviations) is scored by the mean
    Brier score of the soft vote over ``inner_folds`` stratified folds of the
    training data. Ties go to the earlier candidate, i.e. to the bootstrap
    default. The tree is then refitted on all training data with the selected
    rule (``selected_width_``; inner scores in ``width_scores_``)."""

    def __init__(self, max_depth=12, min_leaf_w=2.0, n_boot=25, random_state=0,
                 bounded_support=True, oob_margin=1.0, criterion="gini",
                 width_grid=("bootstrap", 0.25, 0.5, 1.0), inner_folds=3):
        super().__init__(max_depth=max_depth, min_leaf_w=min_leaf_w, n_boot=n_boot,
                         width="bootstrap", random_state=random_state,
                         bounded_support=bounded_support, oob_margin=oob_margin,
                         criterion=criterion)
        self.width_grid = width_grid
        self.inner_folds = inner_folds

    def fit(self, X, y):
        from sklearn.model_selection import StratifiedKFold
        X, y = np.asarray(X, float), np.asarray(y)
        classes, yi = np.unique(y, return_inverse=True)
        folds = min(self.inner_folds, int(np.bincount(yi).min()))
        scores = []
        if folds >= 2:
            splitter = StratifiedKFold(folds, shuffle=True, random_state=self.random_state)
            for width in self.width_grid:
                briers = []
                for tr, va in splitter.split(X, yi):
                    m = LearnedFuzzyTree(max_depth=self.max_depth, min_leaf_w=self.min_leaf_w,
                                         n_boot=self.n_boot, width=width,
                                         random_state=self.random_state,
                                         bounded_support=self.bounded_support,
                                         oob_margin=self.oob_margin,
                                         criterion=self.criterion).fit(X[tr], y[tr])
                    P = np.zeros((len(va), len(classes)))
                    P[:, np.searchsorted(classes, m.classes_)] = m.predict_proba(X[va])
                    briers.append(((P - np.eye(len(classes))[yi[va]]) ** 2).sum(1).mean())
                scores.append(float(np.mean(briers)))
            best = int(np.argmin(np.round(scores, 12)))
        else:
            best = 0
        self.width_scores_ = scores
        self.selected_width_ = self.width_grid[best]
        self.width = self.selected_width_
        return super().fit(X, y)

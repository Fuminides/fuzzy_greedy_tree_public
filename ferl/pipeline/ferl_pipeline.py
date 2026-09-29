"""
Config-driven FERL pipeline + a registry of named configurations.

FERL is treated as ONE of several configurations (base, base-enhanced, credal
slot, ...). Each config is a self-contained classifier: it manages its own
partition learning, growth options, and post-hoc calibration (with an internal
calibration split), exposing the usual fit / predict / predict_proba so it drops
into the same benchmark loop as the sklearn baselines.

Add new variants by adding entries to CONFIGS. The credal variant is a stub
extension point (predict_set) to be implemented later.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from ex_fuzzy import utils, fuzzy_sets as fs
from ferl.core.tree_learning import FuzzyCART
from ferl.fuzzification.fuzzification_mdlp import learn_partitions_mdlp
from ferl.uncertainty.recalibrate import soft_predict, finetune_consequents

try:
    from ferl_fast import FuzzyCARTFast, learn_partitions_mdlp_fast
    _FAST_AVAILABLE = True
except Exception:
    FuzzyCARTFast = None
    learn_partitions_mdlp_fast = None
    _FAST_AVAILABLE = False

EPS = 1e-8


def _softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# --- post-hoc calibrators (fit on a held-out split, then transform) ----------
class _Identity:
    def fit(self, P, y): return self
    def transform(self, P): return P


class _Temperature:
    def fit(self, P, y):
        logit = np.log(np.clip(P, EPS, 1.0))
        best_T, best_nll = 1.0, np.inf
        for T in np.exp(np.linspace(np.log(0.05), np.log(10), 60)):
            Q = _softmax(logit / T)
            nll = -np.log(np.clip(Q[np.arange(len(y)), y], EPS, 1)).mean()
            if nll < best_nll:
                best_nll, best_T = nll, T
        self.T = best_T
        return self

    def transform(self, P):
        return _softmax(np.log(np.clip(P, EPS, 1.0)) / self.T)


class _Isotonic:
    def fit(self, P, y):
        self.irs = []
        for c in range(P.shape[1]):
            try:
                ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
                ir.fit(P[:, c], (y == c).astype(float))
            except Exception:
                ir = None
            self.irs.append(ir)
        return self

    def transform(self, P):
        out = np.zeros_like(P)
        for c, ir in enumerate(self.irs):
            out[:, c] = ir.transform(P[:, c]) if ir is not None else P[:, c]
        out = np.clip(out, EPS, None)
        return out / out.sum(axis=1, keepdims=True)


_CALIBRATORS = {None: _Identity, "temp": _Temperature, "isotonic": _Isotonic}


class FERLPipeline:
    """A single named FERL configuration as a drop-in classifier.

    Parameters mirror the configurable axes we've validated; ``credal`` is a
    not-yet-implemented extension point.
    """

    def __init__(self, partition="mdlp", overlap_frac=1.2, max_rules=20,
                 coverage_weight=0.0, consistent_cci=True, prediction_mode="soft",
                 calibration=None, cal_frac=0.25, credal=False, random_state=0,
                 split_mode="fixed", learned_width="bootstrap",
                 max_depth=5, min_improvement=0.01, patience=3,
                 use_fast=True, max_threads=None, exact_low_gain_ties=False):
        self.partition = partition
        self.overlap_frac = overlap_frac
        self.max_rules = max_rules
        self.coverage_weight = coverage_weight
        self.consistent_cci = consistent_cci
        self.prediction_mode = prediction_mode
        self.calibration = calibration
        self.cal_frac = cal_frac
        self.credal = credal
        self.random_state = random_state
        # Performance mode (learned fuzzification) + growth controls.
        self.split_mode = split_mode
        self.learned_width = learned_width
        self.max_depth = max_depth
        self.min_improvement = min_improvement
        self.patience = patience
        self.use_fast = use_fast
        self.max_threads = max_threads
        self.exact_low_gain_ties = exact_low_gain_ties

    def _make_partitions(self, X, y):
        if self.partition == "mdlp":
            if self.use_fast and _FAST_AVAILABLE:
                return learn_partitions_mdlp_fast(X, y, overlap_frac=self.overlap_frac)
            return learn_partitions_mdlp(X, y, overlap_frac=self.overlap_frac)
        return utils.construct_partitions(X, fs.FUZZY_SETS.t1)

    def fit(self, X, y):
        needs_cal = self.calibration is not None or self.calibration == "consequent_ft"
        if needs_cal:
            try:
                Xf, Xc, yf, yc = train_test_split(
                    X, y, test_size=self.cal_frac, random_state=self.random_state, stratify=y)
            except ValueError:
                Xf, Xc, yf, yc = train_test_split(
                    X, y, test_size=self.cal_frac, random_state=self.random_state)
        else:
            Xf, yf, Xc, yc = X, y, None, None

        tree_cls = FuzzyCARTFast if (self.use_fast and _FAST_AVAILABLE) else FuzzyCART
        self.tree_ = tree_cls(self._make_partitions(Xf, yf), max_rules=self.max_rules,
                              max_depth=self.max_depth, min_improvement=self.min_improvement)
        if self.max_threads is not None and hasattr(self.tree_, "max_threads"):
            self.tree_.max_threads = self.max_threads
        if hasattr(self.tree_, "exact_low_gain_ties"):
            self.tree_.exact_low_gain_ties = self.exact_low_gain_ties
        self.implementation_ = "fast" if tree_cls is FuzzyCARTFast else "python"
        self.tree_.coverage_weight = self.coverage_weight
        self.tree_.consistent_cci = self.consistent_cci
        self.tree_.split_mode = self.split_mode
        self.tree_.learned_width = self.learned_width
        # 'ds' is a pipeline-level inference mode; the tree itself grows/predicts
        # with 'soft' internally.
        self.tree_.prediction_mode = "soft" if self.prediction_mode == "ds" else self.prediction_mode
        self.tree_.fit(Xf, yf, patience=self.patience)
        self.classes_ = self.tree_.classes_

        if self.calibration == "consequent_ft":
            M_cal, cons, _ = self.tree_.node_activation_matrix(Xc)
            self._cons_ft = finetune_consequents(M_cal, yc, cons, len(self.classes_))
            self._cal = None
        else:
            self._cons_ft = None
            self._cal = _CALIBRATORS[self.calibration]()
            if Xc is not None:
                self._cal.fit(self.tree_.predict_proba(Xc), yc)
        return self

    def predict_proba(self, X):
        if self._cons_ft is not None:
            M, _, _ = self.tree_.node_activation_matrix(X)
            return soft_predict(M, self._cons_ft)
        _ds_rule = {"ds": "dempster", "ds_cautious": "cautious", "ds_hybrid": "hybrid"}
        if self.prediction_mode in _ds_rule:
            return self.tree_.predict_ds(X, rule=_ds_rule[self.prediction_mode])[0]
        P = self.tree_.predict_proba(X)
        return self._cal.transform(P) if self._cal is not None else P

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def n_rules(self):
        return self.tree_.tree_rules

    def predict_credal(self, X):
        """Return (betp, belief, plausibility, ignorance) from DS combination."""
        return self.tree_.predict_ds(X)

    def predict_set(self, X):
        """Credal set per sample via interval dominance: keep classes whose
        plausibility >= max belief. Singletons = confident; |set|>1 = abstain."""
        if not self.credal:
            raise ValueError("predict_set requires credal=True")
        rule = {"ds_cautious": "cautious", "ds_hybrid": "hybrid"}.get(self.prediction_mode, "dempster")
        _, bel, pl, _ = self.tree_.predict_ds(X, rule=rule)
        return pl >= bel.max(axis=1, keepdims=True) - 1e-12


# --- the configuration registry ---------------------------------------------
# Edit/extend here; the benchmark iterates over whatever is enabled.
CONFIGS = {
    # the pre-improvement starting point (hard winner-take-all, quantile parts)
    "ferl-original": dict(partition="quantile", coverage_weight=0.0,
                          consistent_cci=False, prediction_mode="winner", calibration=None),
    # core method: soft inference + consistent CCI, quantile partitions
    "ferl-compact": dict(partition="quantile", coverage_weight=0.0,
                      consistent_cci=True, prediction_mode="soft", calibration=None),
    # performance mode: learned-threshold fuzzification (data-chosen soft cuts,
    # bootstrap ramp width) + deeper growth. Trades compactness (~50-90 rules)
    # for accuracy (beats CART); same soft inference + credal/UQ machinery.
    "ferl-medium": dict(partition="quantile", coverage_weight=0.0,
                             consistent_cci=True, prediction_mode="soft", calibration=None,
                             split_mode="learned", learned_width="bootstrap",
                             max_depth=12, min_improvement=0.0, max_rules=150, patience=16),
    # isolates the fuzzification axis: MDLP partitions, no coverage-aware growth.
    "ferl-mdlp": dict(partition="mdlp", overlap_frac=1.2, coverage_weight=0.0,
                      consistent_cci=True, prediction_mode="soft", calibration=None),
    # base-enhanced: + MDLP fuzzification + coverage-aware growth.
    # Calibration is applied uniformly to ALL methods as an eval axis (see the
    # benchmark), not baked into the model, so accuracy comparisons stay fair.
    "ferl-enhanced": dict(partition="mdlp", overlap_frac=1.2, coverage_weight=0.25,
                          consistent_cci=True, prediction_mode="soft", calibration=None),
    # credal variant: base model + Dempster-Shafer evidence combination.
    # Point prediction (pignistic) ~ ties soft; adds set-valued / abstaining
    # predictions via belief/plausibility intervals and an ignorance signal.
    "ferl-credal": dict(partition="quantile", coverage_weight=0.0,
                        consistent_cci=True, prediction_mode="ds", calibration=None,
                        credal=True),
    # cautious (Denoeux) combination: idempotent, respects the tree's dependent
    # rules -> wider, better-calibrated credal intervals at no accuracy cost.
    "ferl-credal-cautious": dict(partition="quantile", coverage_weight=0.0,
                                 consistent_cci=True, prediction_mode="ds_cautious",
                                 calibration=None, credal=True),
}


def make(config_name, random_state=0, use_fast=True, max_threads=None,
         exact_low_gain_ties=False):
    """Instantiate a pipeline from a registered config name.

    ``'ferl-deep'`` is special-cased: it returns the standalone
    ``LearnedFuzzyTree`` (deep learned-threshold performance sibling), not an
    ``FERLPipeline``. For the compact, credal-capable performance mode inside
    ``FuzzyCART`` use the ``'ferl-medium'`` config instead.
    """
    if config_name == "ferl-deep":
        from ferl.core.learned_tree import LearnedFuzzyTree
        return LearnedFuzzyTree(random_state=random_state)
    if config_name not in CONFIGS:
        raise KeyError(f"unknown config {config_name!r}; have {list(CONFIGS)}")
    return FERLPipeline(random_state=random_state, use_fast=use_fast,
                        max_threads=max_threads,
                        exact_low_gain_ties=exact_low_gain_ties,
                        **CONFIGS[config_name])
